from common.imports import *


TOKEN_PAD = 0
TOKEN_CLS = 1
TOKEN_GLOBAL = 2
TOKEN_GENERATION = 3
TOKEN_LOAD = 4
TOKEN_LINE_LOADING = 5
TOKEN_LINE_STATUS = 6
TOKEN_TOPOLOGY = 7
TOKEN_SUBSTATION_COOLDOWN = 8
TOKEN_LINE_COOLDOWN = 9
TOKEN_MAINTENANCE = 10
TOKEN_REDISPATCH = 11
TOKEN_CURTAILMENT = 12
TOKEN_LOCAL_DOMAIN = 13
TOKEN_GENERATOR_ENTITY = 14
TOKEN_LOAD_ENTITY = 15
TOKEN_LINE_ENTITY = 16
TOKEN_SUBSTATION_ENTITY = 17
TOKEN_BUSBAR_ENTITY = 18

ENTITY_PAD = 0
ENTITY_GLOBAL = 1
ENTITY_GENERATOR = 2
ENTITY_LOAD = 3
ENTITY_LINE = 4
ENTITY_SUBSTATION = 5
ENTITY_BUSBAR = 6


class GridTokenBuilder:
    """Build fixed-shape token observations from Grid2Op observations."""

    GROUP_TOKEN_TYPES = [
        ("global", TOKEN_GLOBAL),
        ("generation", TOKEN_GENERATION),
        ("load", TOKEN_LOAD),
        ("line_loading", TOKEN_LINE_LOADING),
        ("line_status", TOKEN_LINE_STATUS),
        ("topology", TOKEN_TOPOLOGY),
        ("substation_cooldown", TOKEN_SUBSTATION_COOLDOWN),
        ("line_cooldown", TOKEN_LINE_COOLDOWN),
        ("maintenance", TOKEN_MAINTENANCE),
        ("redispatch", TOKEN_REDISPATCH),
        ("curtailment", TOKEN_CURTAILMENT),
        ("local_domain", TOKEN_LOCAL_DOMAIN),
    ]

    def __init__(
        self,
        g2op_env,
        observation_domains: Dict[str, List[int]],
        tokenizer_type: str = "group",
        include_neighbors: bool = True,
        include_maintenance: bool = False,
        max_token_feature_dim: int = 128,
        include_busbar_tokens: bool = False,
    ) -> None:
        tokenizer_type = tokenizer_type.lower()
        if tokenizer_type not in {"group", "entity", "hybrid"}:
            raise ValueError(
                "tokenizer_type must be 'group', 'entity', or 'hybrid', "
                f"got {tokenizer_type!r}."
            )
        if max_token_feature_dim < 8:
            raise ValueError("max_token_feature_dim must be at least 8.")

        self.g2op_env = g2op_env
        self.observation_domains = {
            agent: np.asarray(nodes, dtype=np.int64)
            for agent, nodes in observation_domains.items()
        }
        self.agent_index = {
            agent: idx for idx, agent in enumerate(sorted(self.observation_domains))
        }
        self.tokenizer_type = tokenizer_type
        self.include_neighbors = bool(include_neighbors)
        self.include_maintenance = bool(include_maintenance)
        self.include_busbar_tokens = bool(include_busbar_tokens)
        self.token_feature_dim = int(max_token_feature_dim)

        self.n_sub = int(g2op_env.n_sub)
        self.n_line = int(g2op_env.n_line)
        self.n_gen = int(getattr(g2op_env, "n_gen", 0))
        self.n_load = int(getattr(g2op_env, "n_load", 0))
        self.n_busbar = self._get_n_busbar(g2op_env)
        self.n_bus_nodes = self.n_sub * self.n_busbar

        self.line_or = self._env_array("line_or_to_subid", self.n_line, dtype=np.int64)
        self.line_ex = self._env_array("line_ex_to_subid", self.n_line, dtype=np.int64)
        self.gen_to_sub = self._env_array("gen_to_subid", self.n_gen, dtype=np.int64)
        self.load_to_sub = self._env_array("load_to_subid", self.n_load, dtype=np.int64)
        self.line_or_pos = self._topo_pos_array("line_or_pos_topo_vect", self.n_line)
        self.line_ex_pos = self._topo_pos_array("line_ex_pos_topo_vect", self.n_line)
        self.gen_pos = self._topo_pos_array("gen_pos_topo_vect", self.n_gen)
        self.load_pos = self._topo_pos_array("load_pos_topo_vect", self.n_load)

        self.specs = {
            "state": self._make_spec(
                sub_ids=np.arange(self.n_sub),
                line_ids=np.arange(self.n_line),
                controlled_nodes=np.arange(self.n_sub),
                agent_id=None,
            )
        }
        for agent, domain_nodes in self.observation_domains.items():
            sub_ids, line_ids = self._local_ids(domain_nodes)
            self.specs[agent] = self._make_spec(
                sub_ids=sub_ids,
                line_ids=line_ids,
                controlled_nodes=domain_nodes,
                agent_id=agent,
            )

    def build(self, obs) -> Dict[str, Dict[str, np.ndarray]]:
        cache = self._make_obs_cache(obs)
        tokens = {
            "state": self.build_for_spec(obs, self.specs["state"], cache=cache)
        }
        for agent, spec in self.specs.items():
            if agent == "state":
                continue
            tokens[agent] = self.build_for_spec(obs, spec, cache=cache)
        return tokens

    def build_for_spec(
        self,
        obs,
        spec: Dict[str, Any],
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        if cache is None:
            cache = self._make_obs_cache(obs)
        if self.tokenizer_type == "group":
            return self._build_group_for_spec(spec, cache)
        if self.tokenizer_type == "entity":
            return self._build_entity_for_spec(spec, cache)
        return self._build_hybrid_for_spec(spec, cache)

    def _make_spec(
        self,
        sub_ids,
        line_ids,
        controlled_nodes,
        agent_id: Optional[str],
    ) -> Dict[str, Any]:
        sub_ids = np.asarray(sub_ids, dtype=np.int64)
        line_ids = np.asarray(line_ids, dtype=np.int64)
        controlled_nodes = np.asarray(controlled_nodes, dtype=np.int64)
        gen_ids = np.nonzero(np.isin(self.gen_to_sub, sub_ids))[0].astype(np.int64)
        load_ids = np.nonzero(np.isin(self.load_to_sub, sub_ids))[0].astype(np.int64)

        group_names = [
            name
            for name, _ in self.GROUP_TOKEN_TYPES
            if self.include_maintenance or name != "maintenance"
        ]
        entity_count = 1 + len(gen_ids) + len(load_ids) + len(line_ids) + len(sub_ids)
        if self.include_busbar_tokens:
            entity_count += len(sub_ids) * self.n_busbar

        if self.tokenizer_type == "group":
            n_tokens = len(group_names)
        elif self.tokenizer_type == "entity":
            n_tokens = entity_count
        else:
            n_tokens = len(group_names) + entity_count

        return {
            "tokenizer_type": self.tokenizer_type,
            "n_tokens": int(n_tokens),
            "token_feature_dim": self.token_feature_dim,
            "n_token_types": TOKEN_BUSBAR_ENTITY + 1,
            "n_entity_types": ENTITY_BUSBAR + 1,
            "max_entity_id": int(
                max(self.n_line, self.n_gen, self.n_load, self.n_sub, self.n_bus_nodes, 1)
            ),
            "n_sub": self.n_sub,
            "n_busbar": self.n_busbar,
            "n_agents": max(len(self.observation_domains), 1),
            "group_token_names": group_names,
            "sub_ids": sub_ids,
            "line_ids": line_ids,
            "gen_ids": gen_ids,
            "load_ids": load_ids,
            "controlled_nodes": controlled_nodes,
            "agent_index": -1 if agent_id is None else self.agent_index[agent_id],
        }

    def _empty_tokens(self, spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
        n_tokens = int(spec["n_tokens"])
        feature_dim = int(spec["token_feature_dim"])
        return {
            "features": np.zeros((n_tokens, feature_dim), dtype=np.float32),
            "feature_mask": np.zeros((n_tokens, feature_dim), dtype=np.float32),
            "token_mask": np.zeros((n_tokens,), dtype=np.float32),
            "type_ids": np.full((n_tokens,), TOKEN_PAD, dtype=np.int64),
            "entity_type_ids": np.full((n_tokens,), ENTITY_PAD, dtype=np.int64),
            "entity_ids": np.full((n_tokens,), -1, dtype=np.int64),
            "substation_ids": np.full((n_tokens,), -1, dtype=np.int64),
            "bus_ids": np.full((n_tokens,), -1, dtype=np.int64),
            "line_or_sub_ids": np.full((n_tokens,), -1, dtype=np.int64),
            "line_ex_sub_ids": np.full((n_tokens,), -1, dtype=np.int64),
            "line_or_bus_ids": np.full((n_tokens,), -1, dtype=np.int64),
            "line_ex_bus_ids": np.full((n_tokens,), -1, dtype=np.int64),
            "agent_ids": np.full((n_tokens,), int(spec["agent_index"]), dtype=np.int64),
        }

    def _set_token(
        self,
        tokens: Dict[str, np.ndarray],
        row: int,
        token_type: int,
        entity_type: int,
        entity_id: int,
        chunks: List[np.ndarray],
        *,
        substation_id: int = -1,
        bus_id: int = -1,
        line_or_sub_id: int = -1,
        line_ex_sub_id: int = -1,
        line_or_bus_id: int = -1,
        line_ex_bus_id: int = -1,
    ) -> None:
        features, mask = self._feature_vector(chunks)
        tokens["features"][row] = features
        tokens["feature_mask"][row] = mask
        tokens["token_mask"][row] = 1.0
        tokens["type_ids"][row] = int(token_type)
        tokens["entity_type_ids"][row] = int(entity_type)
        tokens["entity_ids"][row] = int(entity_id)
        tokens["substation_ids"][row] = int(substation_id)
        tokens["bus_ids"][row] = int(bus_id)
        tokens["line_or_sub_ids"][row] = int(line_or_sub_id)
        tokens["line_ex_sub_ids"][row] = int(line_ex_sub_id)
        tokens["line_or_bus_ids"][row] = int(line_or_bus_id)
        tokens["line_ex_bus_ids"][row] = int(line_ex_bus_id)

    def _build_group_for_spec(
        self,
        spec: Dict[str, Any],
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        tokens = self._empty_tokens(spec)
        token_type_by_name = dict(self.GROUP_TOKEN_TYPES)
        for row, name in enumerate(spec["group_token_names"]):
            self._set_token(
                tokens,
                row,
                token_type_by_name[name],
                ENTITY_GLOBAL,
                row,
                self._group_chunks(name, spec, cache),
            )
        return tokens

    def _build_hybrid_for_spec(
        self,
        spec: Dict[str, Any],
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        tokens = self._empty_tokens(spec)
        group_spec = dict(spec)
        group_spec["n_tokens"] = len(spec["group_token_names"])
        group_tokens = self._build_group_for_spec(group_spec, cache)
        for key, value in group_tokens.items():
            tokens[key][: len(value)] = value

        entity_spec = dict(spec)
        entity_spec["n_tokens"] = int(spec["n_tokens"]) - len(spec["group_token_names"])
        entity_tokens = self._build_entity_for_spec(entity_spec, cache)
        offset = len(spec["group_token_names"])
        for key, value in entity_tokens.items():
            tokens[key][offset : offset + len(value)] = value
        return tokens

    def _build_entity_for_spec(
        self,
        spec: Dict[str, Any],
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        tokens = self._empty_tokens(spec)
        row = 0
        self._set_token(
            tokens,
            row,
            TOKEN_CLS,
            ENTITY_GLOBAL,
            0,
            self._global_chunks(spec, cache),
        )
        row += 1

        controlled = set(int(x) for x in spec["controlled_nodes"])
        observed = set(int(x) for x in spec["sub_ids"])

        for gen_id in spec["gen_ids"]:
            if row >= spec["n_tokens"]:
                break
            sub_id = int(self.gen_to_sub[gen_id])
            bus_id = self._asset_bus_id(cache["topo_vect"], self.gen_pos, gen_id)
            domain = 1.0 if sub_id in controlled else 0.0
            self._set_token(
                tokens,
                row,
                TOKEN_GENERATOR_ENTITY,
                ENTITY_GENERATOR,
                int(gen_id),
                [
                    self._values(cache, "gen_p", [gen_id]),
                    self._values(cache, "gen_theta", [gen_id]),
                    self._values(cache, "target_dispatch", [gen_id]),
                    self._values(cache, "actual_dispatch", [gen_id]),
                    self._values(cache, "gen_margin_up", [gen_id]),
                    self._values(cache, "gen_margin_down", [gen_id]),
                    self._values(cache, "gen_p_before_curtail", [gen_id]),
                    self._values(cache, "curtailment", [gen_id]),
                    self._values(cache, "curtailment_limit", [gen_id]),
                    np.asarray([self._scale_bus(bus_id), domain], dtype=np.float32),
                ],
                substation_id=sub_id,
                bus_id=bus_id,
            )
            row += 1

        for load_id in spec["load_ids"]:
            if row >= spec["n_tokens"]:
                break
            sub_id = int(self.load_to_sub[load_id])
            bus_id = self._asset_bus_id(cache["topo_vect"], self.load_pos, load_id)
            domain = 1.0 if sub_id in controlled else 0.0
            self._set_token(
                tokens,
                row,
                TOKEN_LOAD_ENTITY,
                ENTITY_LOAD,
                int(load_id),
                [
                    self._values(cache, "load_p", [load_id]),
                    self._values(cache, "load_theta", [load_id]),
                    np.asarray([self._scale_bus(bus_id), domain], dtype=np.float32),
                ],
                substation_id=sub_id,
                bus_id=bus_id,
            )
            row += 1

        for line_id in spec["line_ids"]:
            if row >= spec["n_tokens"]:
                break
            line_id = int(line_id)
            or_sub = int(self.line_or[line_id])
            ex_sub = int(self.line_ex[line_id])
            or_bus = self._asset_bus_id(cache["topo_vect"], self.line_or_pos, line_id)
            ex_bus = self._asset_bus_id(cache["topo_vect"], self.line_ex_pos, line_id)
            touches_controlled = or_sub in controlled or ex_sub in controlled
            boundary = float((or_sub in controlled) ^ (ex_sub in controlled))
            self._set_token(
                tokens,
                row,
                TOKEN_LINE_ENTITY,
                ENTITY_LINE,
                line_id,
                [
                    self._values(cache, "line_status", [line_id]),
                    self._values(cache, "rho", [line_id]),
                    self._values(cache, "timestep_overflow", [line_id]),
                    self._values(cache, "time_before_cooldown_line", [line_id]),
                    self._values(cache, "time_next_maintenance", [line_id]),
                    self._values(cache, "duration_next_maintenance", [line_id]),
                    np.asarray(
                        [
                            self._scale_bus(or_bus),
                            self._scale_bus(ex_bus),
                            boundary,
                            float(touches_controlled),
                        ],
                        dtype=np.float32,
                    ),
                ],
                line_or_sub_id=or_sub,
                line_ex_sub_id=ex_sub,
                line_or_bus_id=or_bus,
                line_ex_bus_id=ex_bus,
            )
            row += 1

        for sub_id in spec["sub_ids"]:
            if row >= spec["n_tokens"]:
                break
            sub_id = int(sub_id)
            line_mask = (self.line_or == sub_id) | (self.line_ex == sub_id)
            line_ids = np.nonzero(line_mask)[0]
            topo_positions = self._topo_positions_for_subs(
                [sub_id],
                line_ids,
                np.nonzero(self.gen_to_sub == sub_id)[0],
                np.nonzero(self.load_to_sub == sub_id)[0],
            )
            bus_values = self._topo_values(cache["topo_vect"], topo_positions)
            bus_counts = np.asarray(
                [
                    np.sum(bus_values == bus)
                    for bus in range(self.n_busbar)
                ],
                dtype=np.float32,
            )
            rho = self._values(cache, "rho", line_ids)
            self._set_token(
                tokens,
                row,
                TOKEN_SUBSTATION_ENTITY,
                ENTITY_SUBSTATION,
                sub_id,
                [
                    self._values(cache, "time_before_cooldown_sub", [sub_id]),
                    np.asarray(
                        [
                            np.sum(self.gen_to_sub == sub_id),
                            np.sum(self.load_to_sub == sub_id),
                            np.sum(line_mask),
                            float(sub_id in controlled),
                            float(sub_id in observed and sub_id not in controlled),
                            self._safe_max(rho),
                            self._safe_mean(rho),
                        ],
                        dtype=np.float32,
                    ),
                    bus_counts,
                ],
                substation_id=sub_id,
            )
            row += 1

        if self.include_busbar_tokens:
            for sub_id in spec["sub_ids"]:
                for bus_id in range(self.n_busbar):
                    if row >= spec["n_tokens"]:
                        break
                    sub_id = int(sub_id)
                    self._set_token(
                        tokens,
                        row,
                        TOKEN_BUSBAR_ENTITY,
                        ENTITY_BUSBAR,
                        sub_id * self.n_busbar + bus_id,
                        self._busbar_chunks(sub_id, bus_id, cache),
                        substation_id=sub_id,
                        bus_id=bus_id,
                    )
                    row += 1

        return tokens

    def _group_chunks(
        self,
        name: str,
        spec: Dict[str, Any],
        cache: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        if name == "global":
            return self._global_chunks(spec, cache)
        if name == "generation":
            gen_ids = spec["gen_ids"]
            return [
                self._values(cache, attr, gen_ids)
                for attr in [
                    "gen_p",
                    "gen_theta",
                    "target_dispatch",
                    "actual_dispatch",
                    "gen_margin_up",
                    "gen_margin_down",
                    "gen_p_before_curtail",
                    "curtailment",
                    "curtailment_limit",
                ]
            ] + [self._summary(self._values(cache, "gen_p", gen_ids))]
        if name == "load":
            load_ids = spec["load_ids"]
            return [
                self._values(cache, attr, load_ids)
                for attr in ["load_p", "load_theta"]
            ] + [self._summary(self._values(cache, "load_p", load_ids))]
        if name == "line_loading":
            line_ids = spec["line_ids"]
            rho = self._values(cache, "rho", line_ids)
            return [
                rho,
                self._values(cache, "timestep_overflow", line_ids),
                self._summary(rho),
            ]
        if name == "line_status":
            line_ids = spec["line_ids"]
            status = self._values(cache, "line_status", line_ids)
            return [status, self._summary(status)]
        if name == "topology":
            positions = self._topo_positions_for_subs(
                spec["sub_ids"],
                spec["line_ids"],
                spec["gen_ids"],
                spec["load_ids"],
            )
            topo = self._topo_values(cache["topo_vect"], positions)
            scaled = np.asarray(
                [self._scale_bus(int(bus)) for bus in topo],
                dtype=np.float32,
            )
            return [scaled, self._summary(scaled)]
        if name == "substation_cooldown":
            values = self._values(cache, "time_before_cooldown_sub", spec["sub_ids"])
            return [self._scale_cooldown(values), self._summary(values)]
        if name == "line_cooldown":
            values = self._values(cache, "time_before_cooldown_line", spec["line_ids"])
            return [self._scale_cooldown(values), self._summary(values)]
        if name == "maintenance":
            line_ids = spec["line_ids"]
            next_maintenance = self._values(cache, "time_next_maintenance", line_ids)
            duration = self._values(cache, "duration_next_maintenance", line_ids)
            return [
                self._scale_maintenance(next_maintenance),
                self._scale_maintenance(duration),
                self._summary(next_maintenance),
            ]
        if name == "redispatch":
            gen_ids = spec["gen_ids"]
            return [
                self._values(cache, attr, gen_ids)
                for attr in [
                    "target_dispatch",
                    "actual_dispatch",
                    "gen_margin_up",
                    "gen_margin_down",
                ]
            ]
        if name == "curtailment":
            gen_ids = spec["gen_ids"]
            return [
                self._values(cache, attr, gen_ids)
                for attr in [
                    "gen_p_before_curtail",
                    "curtailment",
                    "curtailment_limit",
                ]
            ]
        if name == "local_domain":
            controlled = set(int(x) for x in spec["controlled_nodes"])
            domain = np.asarray(
                [1.0 if int(sub) in controlled else 0.0 for sub in spec["sub_ids"]],
                dtype=np.float32,
            )
            counts = np.asarray(
                [
                    len(spec["controlled_nodes"]),
                    len(spec["sub_ids"]),
                    len(spec["line_ids"]),
                    len(spec["gen_ids"]),
                    len(spec["load_ids"]),
                    spec["agent_index"],
                ],
                dtype=np.float32,
            )
            return [domain, counts]
        return [np.zeros((0,), dtype=np.float32)]

    def _global_chunks(
        self,
        spec: Dict[str, Any],
        cache: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        line_ids = spec["line_ids"]
        sub_ids = spec["sub_ids"]
        rho = self._values(cache, "rho", line_ids)
        line_status = self._values(cache, "line_status", line_ids)
        line_cd = self._values(cache, "time_before_cooldown_line", line_ids)
        sub_cd = self._values(cache, "time_before_cooldown_sub", sub_ids)
        maintenance = self._values(cache, "time_next_maintenance", line_ids)
        scalars = np.asarray(
            [
                self._safe_max(rho),
                self._safe_mean(rho),
                self._safe_std(rho),
                float(np.sum(rho >= 0.90)) if rho.size else 0.0,
                float(np.sum(rho >= 1.00)) if rho.size else 0.0,
                float(np.sum(line_status <= 0.0)) if line_status.size else 0.0,
                float(np.sum(line_cd > 0.0)) if line_cd.size else 0.0,
                float(np.sum(sub_cd > 0.0)) if sub_cd.size else 0.0,
                float(np.sum(maintenance >= 0.0)) if maintenance.size else 0.0,
                float(spec["agent_index"]),
            ],
            dtype=np.float32,
        )
        return [scalars, self._summary(rho)]

    def _busbar_chunks(
        self,
        sub_id: int,
        bus_id: int,
        cache: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        gen_ids = np.nonzero(self.gen_to_sub == sub_id)[0]
        load_ids = np.nonzero(self.load_to_sub == sub_id)[0]
        gen_bus = self._asset_bus_ids(cache["topo_vect"], self.gen_pos)
        load_bus = self._asset_bus_ids(cache["topo_vect"], self.load_pos)
        line_or_bus = self._asset_bus_ids(cache["topo_vect"], self.line_or_pos)
        line_ex_bus = self._asset_bus_ids(cache["topo_vect"], self.line_ex_pos)
        line_endpoint_count = np.sum(
            ((self.line_or == sub_id) & (line_or_bus == bus_id))
            | ((self.line_ex == sub_id) & (line_ex_bus == bus_id))
        )
        return [
            np.asarray(
                [
                    np.sum(self._values(cache, "gen_p", gen_ids[gen_bus[gen_ids] == bus_id])),
                    np.sum(self._values(cache, "load_p", load_ids[load_bus[load_ids] == bus_id])),
                    float(line_endpoint_count),
                    self._scale_bus(bus_id),
                ],
                dtype=np.float32,
            )
        ]

    def _feature_vector(self, chunks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        values = np.zeros((self.token_feature_dim,), dtype=np.float32)
        mask = np.zeros((self.token_feature_dim,), dtype=np.float32)
        cursor = 0
        for chunk in chunks:
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.size == 0:
                continue
            take = min(arr.size, self.token_feature_dim - cursor)
            if take <= 0:
                break
            values[cursor : cursor + take] = arr[:take]
            mask[cursor : cursor + take] = 1.0
            cursor += take
        return values, mask

    def _summary(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros((8,), dtype=np.float32)
        return np.asarray(
            [
                float(finite.mean()),
                float(finite.std()),
                float(finite.min()),
                float(finite.max()),
                float(finite.sum()),
                float(finite.size),
                float(np.sum(finite > 0.0)),
                float(np.sum(finite >= 1.0)),
            ],
            dtype=np.float32,
        )

    def _make_obs_cache(self, obs) -> Dict[str, np.ndarray]:
        return {
            "gen_p": self._obs_array(obs, "gen_p", self.n_gen),
            "gen_theta": self._obs_array(obs, "gen_theta", self.n_gen),
            "target_dispatch": self._obs_array(obs, "target_dispatch", self.n_gen),
            "actual_dispatch": self._obs_array(obs, "actual_dispatch", self.n_gen),
            "gen_margin_up": self._obs_array(obs, "gen_margin_up", self.n_gen),
            "gen_margin_down": self._obs_array(obs, "gen_margin_down", self.n_gen),
            "gen_p_before_curtail": self._obs_array(obs, "gen_p_before_curtail", self.n_gen),
            "curtailment": self._obs_array(obs, "curtailment", self.n_gen),
            "curtailment_limit": self._obs_array(obs, "curtailment_limit", self.n_gen),
            "load_p": self._obs_array(obs, "load_p", self.n_load),
            "load_theta": self._obs_array(obs, "load_theta", self.n_load),
            "rho": self._obs_array(obs, "rho", self.n_line),
            "line_status": self._obs_array(obs, "line_status", self.n_line, default=1.0),
            "timestep_overflow": self._obs_array(obs, "timestep_overflow", self.n_line),
            "time_before_cooldown_line": self._obs_array(obs, "time_before_cooldown_line", self.n_line),
            "time_next_maintenance": self._obs_array(obs, "time_next_maintenance", self.n_line, default=-1.0),
            "duration_next_maintenance": self._obs_array(obs, "duration_next_maintenance", self.n_line),
            "time_before_cooldown_sub": self._obs_array(obs, "time_before_cooldown_sub", self.n_sub),
            "topo_vect": np.asarray(getattr(obs, "topo_vect", []), dtype=np.float32),
        }

    def _values(
        self,
        cache: Dict[str, np.ndarray],
        name: str,
        ids,
    ) -> np.ndarray:
        values = cache.get(name)
        ids = np.asarray(ids, dtype=np.int64)
        if values is None or ids.size == 0:
            return np.zeros((0,), dtype=np.float32)
        valid = (ids >= 0) & (ids < len(values))
        if not np.any(valid):
            return np.zeros((0,), dtype=np.float32)
        return np.nan_to_num(values[ids[valid]].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _obs_array(self, obs, name: str, expected: int, default: float = 0.0) -> np.ndarray:
        if expected <= 0:
            return np.zeros((0,), dtype=np.float32)
        if not hasattr(obs, name):
            return np.full((expected,), default, dtype=np.float32)
        values = np.asarray(getattr(obs, name), dtype=np.float32)
        if len(values) != expected:
            return np.full((expected,), default, dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _local_ids(self, domain_nodes) -> Tuple[np.ndarray, np.ndarray]:
        domain_nodes = np.asarray(domain_nodes, dtype=np.int64)
        if self.include_neighbors:
            touches_domain = np.isin(self.line_or, domain_nodes) | np.isin(
                self.line_ex, domain_nodes
            )
            line_ids = np.nonzero(touches_domain)[0]
            node_ids = np.unique(
                np.concatenate([domain_nodes, self.line_or[line_ids], self.line_ex[line_ids]])
            )
        else:
            inside_domain = np.isin(self.line_or, domain_nodes) & np.isin(
                self.line_ex, domain_nodes
            )
            line_ids = np.nonzero(inside_domain)[0]
            node_ids = np.unique(domain_nodes)
        return node_ids.astype(np.int64), line_ids.astype(np.int64)

    def _topo_positions_for_subs(
        self,
        sub_ids,
        line_ids,
        gen_ids,
        load_ids,
    ) -> np.ndarray:
        sub_ids = np.asarray(sub_ids, dtype=np.int64)
        line_ids = np.asarray(line_ids, dtype=np.int64)
        gen_ids = np.asarray(gen_ids, dtype=np.int64)
        load_ids = np.asarray(load_ids, dtype=np.int64)
        positions = []
        if line_ids.size:
            line_mask = np.isin(self.line_or[line_ids], sub_ids)
            positions.extend(self.line_or_pos[line_ids[line_mask]].tolist())
            line_mask = np.isin(self.line_ex[line_ids], sub_ids)
            positions.extend(self.line_ex_pos[line_ids[line_mask]].tolist())
        if gen_ids.size:
            positions.extend(self.gen_pos[gen_ids].tolist())
        if load_ids.size:
            positions.extend(self.load_pos[load_ids].tolist())
        positions = np.asarray(positions, dtype=np.int64)
        return positions[positions >= 0]

    def _topo_values(self, topo_vect: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if topo_vect.size == 0 or positions.size == 0:
            return np.zeros((0,), dtype=np.int64)
        valid = (positions >= 0) & (positions < len(topo_vect))
        values = np.full((len(positions),), -1, dtype=np.int64)
        values[valid] = np.asarray(topo_vect[positions[valid]], dtype=np.int64) - 1
        values[(values < 0) | (values >= self.n_busbar)] = -1
        return values

    def _asset_bus_ids(self, topo_vect: np.ndarray, topo_pos: np.ndarray) -> np.ndarray:
        return self._topo_values(topo_vect, np.asarray(topo_pos, dtype=np.int64))

    def _asset_bus_id(self, topo_vect: np.ndarray, topo_pos: np.ndarray, idx: int) -> int:
        if idx < 0 or idx >= len(topo_pos):
            return -1
        return int(self._topo_values(topo_vect, np.asarray([topo_pos[idx]], dtype=np.int64))[0])

    def _scale_bus(self, bus_id: int) -> float:
        if bus_id < 0:
            return 0.0
        return float(bus_id + 1) / float(max(self.n_busbar, 1))

    def _scale_cooldown(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 10.0) / 10.0

    def _scale_maintenance(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        values = np.where(values < 0.0, 0.0, values)
        return np.clip(values, 0.0, 2016.0) / 2016.0

    def _safe_max(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float32)
        finite = values[np.isfinite(values)]
        return 0.0 if finite.size == 0 else float(finite.max())

    def _safe_mean(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float32)
        finite = values[np.isfinite(values)]
        return 0.0 if finite.size == 0 else float(finite.mean())

    def _safe_std(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float32)
        finite = values[np.isfinite(values)]
        return 0.0 if finite.size == 0 else float(finite.std())

    def _env_array(self, name: str, expected: int, dtype=np.float32) -> np.ndarray:
        if expected <= 0:
            return np.zeros((0,), dtype=dtype)
        values = getattr(self.g2op_env, name, None)
        if values is None:
            return np.zeros((expected,), dtype=dtype)
        values = np.asarray(values, dtype=dtype)
        if len(values) != expected:
            return np.zeros((expected,), dtype=dtype)
        return values

    def _topo_pos_array(self, name: str, expected: int) -> np.ndarray:
        if expected <= 0:
            return np.zeros((0,), dtype=np.int64)
        values = getattr(self.g2op_env, name, None)
        if values is None or len(values) != expected:
            raise ValueError(
                f"Tokenizer requires Grid2Op metadata '{name}' with length {expected}."
            )
        return np.asarray(values, dtype=np.int64)

    def _get_n_busbar(self, g2op_env) -> int:
        raw = getattr(g2op_env, "n_busbar_per_sub", 2)
        if raw is None:
            return 2
        values = np.asarray(raw)
        if values.size == 0:
            return 2
        return max(2, int(values.max()))

