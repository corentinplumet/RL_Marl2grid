from common.imports import *


ASSET_EDGE_DIRECTIONS = {
    "bidirectional",
    "asset_to_busbar",
    "busbar_to_asset",
}
LINE_NODE_EDGE_DIRECTIONS = {
    "bidirectional",
    "line_to_busbar",
    "busbar_to_line",
}

# How voltage angles enter the graph. "node" keeps the absolute per-asset angle
# on the nodes; "edge_diff" replaces it with the angle difference across each
# transmission line.
ANGLE_REPRESENTATIONS = {"node", "edge_diff"}
NODE_ANGLE_FEATURES = ("gen_theta", "load_theta", "theta")
EDGE_ANGLE_FEATURE = "theta_diff"


def _validate_angle_representation(value: str) -> str:
    representation = str(value).strip().lower()
    if representation not in ANGLE_REPRESENTATIONS:
        choices = ", ".join(sorted(ANGLE_REPRESENTATIONS))
        raise ValueError(
            f"Unsupported angle representation '{value}'. Use one of: {choices}."
        )
    return representation


def _validate_edge_direction(value: str, allowed, option_name: str) -> str:
    direction = str(value).strip().lower()
    if direction not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported {option_name} '{value}'. Use one of: {choices}."
        )
    return direction


class GridGraphBuilder:
    """Build fixed-shape busbar graph observations from the current Grid2Op state.

    Nodes are busbars. Edges are fixed potential line connections, with edge
    masks activating only the current bus-to-bus connections from topo_vect.
    """

    BASE_NODE_FEATURES = [
        "gen_p",
        "gen_theta",
        "load_p",
        "load_theta",
        "time_before_cooldown_sub",
        "domain_mask",
    ]

    BASE_EDGE_FEATURES = [
        "line_status",
        "rho",
        "timestep_overflow",
        "time_before_cooldown_line",
    ]

    MAINTENANCE_EDGE_FEATURES = [
        "time_next_maintenance",
        "duration_next_maintenance",
    ]

    EDGE_TYPE_SELF = 0
    EDGE_TYPE_PHYSICAL_LINE = 1
    EDGE_TYPE_SAME_SUBSTATION = 2
    EDGE_TYPE_NAMES = {
        "self": EDGE_TYPE_SELF,
        "physical_line": EDGE_TYPE_PHYSICAL_LINE,
        "same_substation_busbar": EDGE_TYPE_SAME_SUBSTATION,
    }

    def __init__(
        self,
        g2op_env,
        observation_domains: Dict[str, List[int]],
        include_neighbors: bool = True,
        include_maintenance: bool = False,
        add_self_edges: bool = False,
        add_substation_edges: bool = False,
        angle_representation: str = "node",
        context_requires_connection: bool = True,
    ) -> None:
        self.g2op_env = g2op_env
        self.observation_domains = {
            agent: np.asarray(nodes, dtype=np.int64)
            for agent, nodes in observation_domains.items()
        }
        self.include_neighbors = include_neighbors
        self.add_self_edges = bool(add_self_edges)
        self.add_substation_edges = bool(add_substation_edges)
        # A contextual node is only observable when an active physical relation
        # ties it to the region the agent controls. Set to False to restore the
        # older behaviour, where every node of the local graph was always valid.
        self.context_requires_connection = bool(context_requires_connection)
        self.angle_representation = _validate_angle_representation(
            angle_representation
        )
        self.node_features = list(self.BASE_NODE_FEATURES)
        self.edge_features = list(self.BASE_EDGE_FEATURES)
        if self.angle_representation == "edge_diff":
            # An absolute voltage angle is measured against whichever bus is
            # slack, so its level is grid-specific while only differences carry
            # physical meaning. Moving the angle onto the lines makes it
            # gauge-invariant, and dense: every line has two ends, whereas a
            # busbar only has an angle when an asset is attached to it.
            self.node_features = [
                name
                for name in self.node_features
                if name not in NODE_ANGLE_FEATURES
            ]
            self.edge_features = self.edge_features + [EDGE_ANGLE_FEATURE]
        if include_maintenance:
            self.edge_features += self.MAINTENANCE_EDGE_FEATURES

        self.n_sub = int(g2op_env.n_sub)
        self.n_line = int(g2op_env.n_line)
        self.n_busbar = self._get_n_busbar(g2op_env)
        self.n_bus_nodes = self.n_sub * self.n_busbar
        self.line_or = self._env_array("line_or_to_subid", self.n_line, dtype=np.int64)
        self.line_ex = self._env_array("line_ex_to_subid", self.n_line, dtype=np.int64)
        self.gen_to_sub = self._env_array("gen_to_subid", getattr(g2op_env, "n_gen", 0), dtype=np.int64)
        self.load_to_sub = self._env_array("load_to_subid", getattr(g2op_env, "n_load", 0), dtype=np.int64)
        self.line_or_pos = self._topo_pos_array("line_or_pos_topo_vect", self.n_line)
        self.line_ex_pos = self._topo_pos_array("line_ex_pos_topo_vect", self.n_line)
        self.gen_pos = self._topo_pos_array("gen_pos_topo_vect", getattr(g2op_env, "n_gen", 0))
        self.load_pos = self._topo_pos_array("load_pos_topo_vect", getattr(g2op_env, "n_load", 0))

        self.specs = {"state": self._make_spec(np.arange(self.n_sub), np.arange(self.n_line))}
        for agent, domain_nodes in self.observation_domains.items():
            sub_ids, line_ids = self._local_ids(domain_nodes)
            self.specs[agent] = self._make_spec(sub_ids, line_ids, controlled_nodes=domain_nodes)

    @property
    def node_dim(self) -> int:
        return len(self.node_features)

    @property
    def edge_dim(self) -> int:
        return len(self.edge_features)

    def build(self, obs) -> Dict[str, Dict[str, np.ndarray]]:
        cache = self._make_obs_cache(obs)
        state_graph = self.build_for_spec(obs, self.specs["state"], cache=cache)
        graphs = {"state": state_graph}
        for agent, spec in self.specs.items():
            if agent == "state":
                continue
            graphs[agent] = self.build_for_spec(obs, spec, cache=cache)
        return graphs

    def build_for_spec(
        self,
        obs,
        spec: Dict[str, Any],
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        return self._build_bus_for_spec(obs, spec, cache=cache)

    def _make_spec(self, node_ids, line_ids, controlled_nodes=None) -> Dict[str, Any]:
        return self._make_bus_spec(node_ids, line_ids, controlled_nodes=controlled_nodes)

    def _make_bus_spec(self, sub_ids, line_ids, controlled_nodes=None) -> Dict[str, Any]:
        sub_ids = np.asarray(sub_ids, dtype=np.int64)
        line_ids = np.asarray(line_ids, dtype=np.int64)
        controlled_nodes = sub_ids if controlled_nodes is None else np.asarray(controlled_nodes, dtype=np.int64)
        node_ids = self._bus_node_ids(sub_ids)
        local_index = {int(node_id): idx for idx, node_id in enumerate(node_ids)}

        directed_edges = []
        edge_line_ids = []
        edge_or_bus_ids = []
        edge_ex_bus_ids = []
        edge_types = []

        if self.add_self_edges:
            for local_node_idx in range(len(node_ids)):
                directed_edges.append((local_node_idx, local_node_idx))
                edge_line_ids.append(-1)
                edge_or_bus_ids.append(-1)
                edge_ex_bus_ids.append(-1)
                edge_types.append(self.EDGE_TYPE_SELF)

        for line_id in line_ids:
            or_sub, ex_sub = int(self.line_or[line_id]), int(self.line_ex[line_id])
            for or_bus in range(self.n_busbar):
                or_node = self._bus_node_id(or_sub, or_bus)
                if or_node not in local_index:
                    continue
                for ex_bus in range(self.n_busbar):
                    ex_node = self._bus_node_id(ex_sub, ex_bus)
                    if ex_node not in local_index:
                        continue

                    directed_edges.append((local_index[or_node], local_index[ex_node]))
                    directed_edges.append((local_index[ex_node], local_index[or_node]))
                    edge_line_ids.extend([line_id, line_id])
                    edge_or_bus_ids.extend([or_bus, or_bus])
                    edge_ex_bus_ids.extend([ex_bus, ex_bus])
                    edge_types.extend(
                        [
                            self.EDGE_TYPE_PHYSICAL_LINE,
                            self.EDGE_TYPE_PHYSICAL_LINE,
                        ]
                    )

        if self.add_substation_edges:
            for sub_id in sub_ids:
                sub_node_ids = self._bus_node_ids([sub_id])
                local_sub_nodes = [
                    local_index[int(node_id)]
                    for node_id in sub_node_ids
                    if int(node_id) in local_index
                ]
                for src_idx in local_sub_nodes:
                    for dst_idx in local_sub_nodes:
                        if src_idx == dst_idx:
                            continue
                        directed_edges.append((src_idx, dst_idx))
                        edge_line_ids.append(-1)
                        edge_or_bus_ids.append(-1)
                        edge_ex_bus_ids.append(-1)
                        edge_types.append(self.EDGE_TYPE_SAME_SUBSTATION)

        edge_index = np.asarray(directed_edges, dtype=np.int64).T
        if edge_index.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_line_ids = np.asarray(edge_line_ids, dtype=np.int64)
        edge_or_bus_ids = np.asarray(edge_or_bus_ids, dtype=np.int64)
        edge_ex_bus_ids = np.asarray(edge_ex_bus_ids, dtype=np.int64)
        edge_types = np.asarray(edge_types, dtype=np.int64)

        controlled_node_mask = np.isin(
            node_ids // self.n_busbar,
            controlled_nodes,
        ).astype(np.float32)

        return {
            "graph_type": "bus",
            "node_ids": node_ids,
            "line_ids": line_ids,
            "controlled_nodes": controlled_nodes,
            "controlled_node_mask": controlled_node_mask,
            "edge_index": edge_index,
            "edge_line_ids": edge_line_ids,
            "edge_or_bus_ids": edge_or_bus_ids,
            "edge_ex_bus_ids": edge_ex_bus_ids,
            "edge_type": edge_types,
            "edge_type_names": dict(self.EDGE_TYPE_NAMES),
            "n_edge_types": len(self.EDGE_TYPE_NAMES),
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "node_feature_names": list(self.node_features),
            "edge_feature_names": list(self.edge_features),
            "n_sub": self.n_sub,
            "n_busbar": self.n_busbar,
            "n_bus_nodes": self.n_bus_nodes,
        }

    def _local_ids(self, domain_nodes):
        domain_nodes = np.asarray(domain_nodes, dtype=np.int64)
        if self.include_neighbors:
            touches_domain = np.isin(self.line_or, domain_nodes) | np.isin(self.line_ex, domain_nodes)
            line_ids = np.nonzero(touches_domain)[0]
            node_ids = np.unique(np.concatenate([domain_nodes, self.line_or[line_ids], self.line_ex[line_ids]]))
        else:
            inside_domain = np.isin(self.line_or, domain_nodes) & np.isin(self.line_ex, domain_nodes)
            line_ids = np.nonzero(inside_domain)[0]
            node_ids = np.unique(domain_nodes)
        return node_ids, line_ids

    def _make_obs_cache(self, obs) -> Dict[str, np.ndarray]:
        line_status = self._obs_array(obs, "line_status", expected=self.n_line)
        if line_status is None:
            line_status = np.ones(self.n_line, dtype=np.float32)
        return {
            "base_node_features": self._base_bus_node_features(obs),
            "line_features": self._line_feature_matrix(obs),
            "line_status": line_status,
            "line_or_bus": self._topo_bus_ids(obs, self.line_or_pos) - 1,
            "line_ex_bus": self._topo_bus_ids(obs, self.line_ex_pos) - 1,
        }

    def _build_bus_for_spec(
        self,
        obs,
        spec: Dict[str, Any],
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        if cache is None:
            cache = self._make_obs_cache(obs)
        node_features = self._bus_node_features_from_cache(cache, spec)
        edge_features, edge_mask = self._bus_edge_features_from_cache(spec, cache)
        node_mask = self._visible_node_mask(spec, edge_mask)
        node_features = self._hide_invisible_node_features(
            node_features, node_mask
        )
        edge_features, edge_mask = self._hide_invisible_node_edges(
            spec, edge_features, edge_mask, node_mask
        )
        return {
            "node_features": node_features.astype(np.float32, copy=False),
            "edge_features": edge_features.astype(np.float32, copy=False),
            "node_mask": node_mask,
            "edge_mask": edge_mask.astype(np.float32, copy=False),
            "edge_type": spec["edge_type"].astype(np.int64, copy=False),
            "controlled_node_mask": spec["controlled_node_mask"].astype(
                np.float32, copy=False
            ),
        }

    def _base_bus_node_features(self, obs):
        features = np.zeros((self.n_bus_nodes, self.node_dim), dtype=np.float32)
        col = {name: idx for idx, name in enumerate(self.node_features)}

        self._put_bus_asset_feature(features, col["gen_p"], obs, self.gen_to_sub, self.gen_pos, self._obs_array(obs, "gen_p"), mode="sum")
        if "gen_theta" in col:
            self._put_bus_asset_feature(features, col["gen_theta"], obs, self.gen_to_sub, self.gen_pos, self._obs_array(obs, "gen_theta"), mode="mean")

        self._put_bus_asset_feature(features, col["load_p"], obs, self.load_to_sub, self.load_pos, self._obs_array(obs, "load_p"), mode="sum")
        if "load_theta" in col:
            self._put_bus_asset_feature(features, col["load_theta"], obs, self.load_to_sub, self.load_pos, self._obs_array(obs, "load_theta"), mode="mean")

        sub_cooldown = self._obs_array(obs, "time_before_cooldown_sub", expected=self.n_sub)
        if sub_cooldown is not None:
            for sub_id in range(self.n_sub):
                features[self._bus_node_ids([sub_id]), col["time_before_cooldown_sub"]] = sub_cooldown[sub_id]

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _bus_node_features_from_cache(self, cache, spec):
        node_features = cache["base_node_features"][spec["node_ids"]].copy()
        col = {name: idx for idx, name in enumerate(self.node_features)}
        node_sub_ids = spec["node_ids"] // self.n_busbar
        node_features[:, col["domain_mask"]] = np.isin(
            node_sub_ids, spec["controlled_nodes"]
        ).astype(np.float32)
        return np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

    def _bus_node_features(self, obs, controlled_nodes):
        cache = {"base_node_features": self._base_bus_node_features(obs)}
        spec = {
            "node_ids": np.arange(self.n_bus_nodes, dtype=np.int64),
            "controlled_nodes": np.asarray(controlled_nodes, dtype=np.int64),
        }
        return self._bus_node_features_from_cache(cache, spec)

    def _bus_edge_features_from_cache(self, spec, cache):
        edge_line_ids = spec["edge_line_ids"]
        if len(edge_line_ids) == 0:
            return (
                np.zeros((0, self.edge_dim), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        edge_features = np.zeros((len(edge_line_ids), self.edge_dim), dtype=np.float32)
        physical = edge_line_ids >= 0
        if np.any(physical):
            physical_line_ids = edge_line_ids[physical]
            edge_features[physical] = cache["line_features"][physical_line_ids]

        line_status = cache["line_status"]
        active = np.ones((len(edge_line_ids),), dtype=bool)
        if np.any(physical):
            physical_line_ids = edge_line_ids[physical]
            or_bus = cache["line_or_bus"][physical_line_ids]
            ex_bus = cache["line_ex_bus"][physical_line_ids]
            active[physical] = (
                (line_status[physical_line_ids] > 0)
                & (or_bus == spec["edge_or_bus_ids"][physical])
                & (ex_bus == spec["edge_ex_bus_ids"][physical])
            )
        if "rho" in self.edge_features:
            rho_idx = self.edge_features.index("rho")
            edge_features[~physical, rho_idx] = 1.0
        edge_features[~active] = 0.0
        return np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0), active.astype(np.float32)

    def _bus_edge_features(self, obs, spec):
        return self._bus_edge_features_from_cache(spec, self._make_obs_cache(obs))

    def _line_feature_matrix(self, obs):
        features = np.zeros((self.n_line, self.edge_dim), dtype=np.float32)
        for idx, name in enumerate(self.edge_features):
            if name == EDGE_ANGLE_FEATURE:
                values = self._line_theta_difference(obs)
            else:
                values = self._obs_array(obs, name, expected=self.n_line)
            if values is not None:
                features[:, idx] = values
        return features

    def _line_theta_difference(self, obs):
        """Voltage angle drop across each line, origin minus extremity.

        Unlike the per-asset angles this is defined on every line and does not
        depend on the slack bus the angles are referenced to.
        """
        theta_or = self._obs_array(obs, "theta_or", expected=self.n_line)
        theta_ex = self._obs_array(obs, "theta_ex", expected=self.n_line)
        if theta_or is None or theta_ex is None:
            return None
        return np.asarray(theta_or, dtype=np.float32) - np.asarray(
            theta_ex, dtype=np.float32
        )

    def _put_bus_asset_feature(self, features, col, obs, mapping, topo_pos, values, mode="sum"):
        if values is None or len(mapping) == 0 or len(values) != len(mapping):
            return
        mapping = np.asarray(mapping, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        bus_ids = self._topo_bus_ids(obs, topo_pos)
        node_ids = self._asset_bus_node_ids(mapping, bus_ids)
        valid = (node_ids >= 0) & np.isfinite(values)
        if not np.any(valid):
            return
        sums = np.zeros(self.n_bus_nodes, dtype=np.float32)
        np.add.at(sums, node_ids[valid], values[valid])
        if mode == "mean":
            counts = np.zeros(self.n_bus_nodes, dtype=np.float32)
            np.add.at(counts, node_ids[valid], 1.0)
            sums = np.divide(sums, np.maximum(counts, 1.0))
        features[:, col] = sums

    def _obs_array(self, obs, name, expected=None):
        if not hasattr(obs, name):
            return None
        values = np.asarray(getattr(obs, name), dtype=np.float32)
        if expected is not None and len(values) != expected:
            return None
        return values

    def _topo_bus_ids(self, obs, topo_pos):
        topo_vect = self._obs_array(obs, "topo_vect")
        return self._topo_bus_ids_from_pos(topo_vect, topo_pos)

    def _topo_bus_ids_from_pos(self, topo_vect, topo_pos):
        buses = np.zeros((len(topo_pos),), dtype=np.int64)
        if topo_vect is None:
            return buses
        topo_pos = np.asarray(topo_pos, dtype=np.int64)
        valid = (topo_pos >= 0) & (topo_pos < len(topo_vect))
        buses[valid] = np.asarray(topo_vect[topo_pos[valid]], dtype=np.int64)
        buses[(buses < 1) | (buses > self.n_busbar)] = 0
        return buses

    def _asset_bus_node_ids(self, mapping, bus_ids):
        node_ids = np.full((len(mapping),), -1, dtype=np.int64)
        valid = (
            (mapping >= 0)
            & (mapping < self.n_sub)
            & (bus_ids >= 1)
            & (bus_ids <= self.n_busbar)
        )
        node_ids[valid] = mapping[valid] * self.n_busbar + (bus_ids[valid] - 1)
        return node_ids

    def _bus_node_id(self, sub_id, bus_id):
        return int(sub_id) * self.n_busbar + int(bus_id)

    #: Relations that are computational rather than electrical. They never
    #: make a contextual node observable, because two busbars of one substation
    #: are not electrically joined and a self edge connects nothing.
    STRUCTURAL_EDGE_TYPE_NAMES = ("self", "same_substation_busbar")

    def _visible_node_mask(self, spec, edge_mask) -> np.ndarray:
        """Valid-node mask for one agent at one step.

        Controlled nodes are always valid. A contextual node is valid only when
        an active, non-structural relation connects it to the controlled
        region, directly or through other contextual nodes. Without this, an
        agent pools busbars of a neighbouring substation that no live line
        attaches to its own region, which is state it has no way of observing.
        """
        controlled = np.asarray(spec["controlled_node_mask"], dtype=bool)
        if not self.context_requires_connection or controlled.all():
            return np.ones(controlled.shape, dtype=np.float32)

        edge_index = np.asarray(spec["edge_index"], dtype=np.int64)
        if edge_index.size == 0:
            return controlled.astype(np.float32)

        type_names = spec.get("edge_type_names", {})
        structural = [
            type_names[name]
            for name in self.STRUCTURAL_EDGE_TYPE_NAMES
            if name in type_names
        ]
        carries_state = np.asarray(edge_mask, dtype=bool)
        if structural:
            carries_state &= ~np.isin(
                np.asarray(spec["edge_type"], dtype=np.int64), structural
            )

        src = edge_index[0][carries_state]
        dst = edge_index[1][carries_state]
        visible = controlled.copy()
        # Propagate to a fixed point. The relation is symmetric for visibility
        # even when the edge itself is one-way: an attachment that exists at
        # all means the two endpoints are electrically joined.
        for _ in range(len(visible)):
            grown = visible.copy()
            # ``np.logical_or.at`` is the unbuffered form: with plain fancy
            # indexing a repeated destination keeps only the last write, which
            # silently drops visibility when a node has several active edges.
            np.logical_or.at(grown, dst, visible[src])
            np.logical_or.at(grown, src, visible[dst])
            if np.array_equal(grown, visible):
                break
            visible = grown
        return visible.astype(np.float32)

    @staticmethod
    def _hide_invisible_node_features(node_features, node_mask) -> np.ndarray:
        """Remove state from fixed-shape candidate rows that are not visible.

        Local graph specs enumerate candidate busbar assignments so tensor
        shapes remain constant while topology changes. An inactive contextual
        candidate must not retain raw neighboring state in the observation,
        even though its edges and readout entry are masked. Zeroing the entire
        row makes the information boundary a property of graph construction
        rather than something that depends on a particular encoder or pooling
        implementation.
        """
        visible = np.asarray(node_mask, dtype=bool)
        if bool(np.all(visible)):
            return node_features
        sanitized = np.asarray(node_features).copy()
        sanitized[~visible] = 0.0
        return sanitized

    @staticmethod
    def _hide_invisible_node_edges(
        spec, edge_features, edge_mask, node_mask
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deactivate every edge incident to an invisible candidate row.

        Physical candidate edges are already topology-masked, but optional
        self and same-substation relations are structurally active. Removing
        all incident edges prevents an invisible placeholder from entering
        message passing through encoder biases, node-id embeddings, or type
        embeddings after its dynamic feature vector has been zeroed.
        """
        edge_index = np.asarray(spec["edge_index"], dtype=np.int64)
        active = np.asarray(edge_mask, dtype=bool).copy()
        if edge_index.size > 0:
            visible = np.asarray(node_mask, dtype=bool)
            active &= visible[edge_index[0]] & visible[edge_index[1]]
        sanitized = np.asarray(edge_features).copy()
        sanitized[~active] = 0.0
        return sanitized, active.astype(np.float32)

    def _bus_node_ids(self, sub_ids):
        sub_ids = np.asarray(sub_ids, dtype=np.int64)
        return np.asarray(
            [self._bus_node_id(sub_id, bus) for sub_id in sub_ids for bus in range(self.n_busbar)],
            dtype=np.int64,
        )

    def _env_array(self, name, expected, dtype=np.float32):
        values = getattr(self.g2op_env, name, None)
        if values is None:
            return np.zeros((expected,), dtype=dtype)
        values = np.asarray(values, dtype=dtype)
        if len(values) != expected:
            return np.zeros((expected,), dtype=dtype)
        return values

    def _topo_pos_array(self, name, expected):
        if expected == 0:
            return np.zeros((0,), dtype=np.int64)
        values = getattr(self.g2op_env, name, None)
        if values is None or len(values) != expected:
            raise ValueError(
                f"Bus-node graph requires Grid2Op metadata '{name}' with length {expected}."
            )
        return np.asarray(values, dtype=np.int64)

    def _get_n_busbar(self, g2op_env):
        raw = getattr(g2op_env, "n_busbar_per_sub", 2)
        if raw is None:
            return 2
        values = np.asarray(raw)
        if values.size == 0:
            return 2
        return max(2, int(values.max()))


class HeterogeneousGridGraphBuilder(GridGraphBuilder):
    """Build a fixed-shape typed graph of busbars, generators, and loads.

    The representation is stored in homogeneous tensors for compatibility with
    the existing rollout and encoder code, while ``node_type`` and ``edge_type``
    retain the heterogeneous schema. Physical-line and asset-attachment edges
    enumerate every possible busbar assignment and use masks to activate the
    assignment selected by the current Grid2Op ``topo_vect``. Generator and
    load relations may independently be one-way or bidirectional.

    In a local agent graph, one-hop neighboring substations contribute only the
    busbar attached through a live shared line. Generators and loads are added
    only for substations controlled by the agent; assets at neighboring
    substations must never enter its graph or message-passing computation.
    """

    NODE_TYPE_BUSBAR = 0
    NODE_TYPE_GENERATOR = 1
    NODE_TYPE_LOAD = 2
    NODE_TYPE_NAMES = {
        "busbar": NODE_TYPE_BUSBAR,
        "generator": NODE_TYPE_GENERATOR,
        "load": NODE_TYPE_LOAD,
    }

    EDGE_TYPE_SELF = 0
    EDGE_TYPE_PHYSICAL_LINE = 1
    EDGE_TYPE_SAME_SUBSTATION = 2
    EDGE_TYPE_GENERATOR_TO_BUSBAR = 3
    EDGE_TYPE_BUSBAR_TO_GENERATOR = 4
    EDGE_TYPE_LOAD_TO_BUSBAR = 5
    EDGE_TYPE_BUSBAR_TO_LOAD = 6
    EDGE_TYPE_NAMES = {
        "self": EDGE_TYPE_SELF,
        "physical_line": EDGE_TYPE_PHYSICAL_LINE,
        "same_substation_busbar": EDGE_TYPE_SAME_SUBSTATION,
        "generator_to_busbar": EDGE_TYPE_GENERATOR_TO_BUSBAR,
        "busbar_to_generator": EDGE_TYPE_BUSBAR_TO_GENERATOR,
        "load_to_busbar": EDGE_TYPE_LOAD_TO_BUSBAR,
        "busbar_to_load": EDGE_TYPE_BUSBAR_TO_LOAD,
    }

    ASSET_NONE = 0
    ASSET_GENERATOR = 1
    ASSET_LOAD = 2

    HETERO_NODE_FEATURES = [
        "is_busbar",
        "is_generator",
        "is_load",
        "p",
        "theta",
        "time_before_cooldown_sub",
        "connected",
        "domain_mask",
    ]

    def __init__(
        self,
        g2op_env,
        observation_domains: Dict[str, List[int]],
        include_neighbors: bool = True,
        include_maintenance: bool = False,
        add_self_edges: bool = False,
        add_substation_edges: bool = False,
        generator_edge_direction: str = "bidirectional",
        load_edge_direction: str = "bidirectional",
        angle_representation: str = "node",
        include_legacy_self_relation_feature: bool = False,
        context_requires_connection: bool = True,
    ) -> None:
        # Initialize the shared Grid2Op metadata and helper methods first. The
        # bus-only specs produced by the parent are immediately replaced below.
        super().__init__(
            g2op_env,
            observation_domains,
            include_neighbors=include_neighbors,
            include_maintenance=include_maintenance,
            add_self_edges=add_self_edges,
            add_substation_edges=add_substation_edges,
            angle_representation=angle_representation,
            context_requires_connection=context_requires_connection,
        )
        self.n_gen = int(getattr(g2op_env, "n_gen", len(self.gen_to_sub)))
        self.n_load = int(getattr(g2op_env, "n_load", len(self.load_to_sub)))
        self.generator_edge_direction = _validate_edge_direction(
            generator_edge_direction,
            ASSET_EDGE_DIRECTIONS,
            "generator edge direction",
        )
        self.load_edge_direction = _validate_edge_direction(
            load_edge_direction,
            ASSET_EDGE_DIRECTIONS,
            "load edge direction",
        )
        # Explicit self edges are not part of the GINE graph schemas used by
        # the experiments. Older checkpoints nevertheless allocated an
        # always-zero ``relation_self`` input column. Keep that obsolete width
        # only when reconstructing one of those checkpoints.
        self.include_legacy_self_relation_feature = bool(
            include_legacy_self_relation_feature
        )
        self.node_features = list(self.HETERO_NODE_FEATURES)
        if self.angle_representation == "edge_diff":
            self.node_features = [
                name
                for name in self.node_features
                if name not in NODE_ANGLE_FEATURES
            ]
        # theta_diff was appended to the parent's edge features, so it rides
        # along with the rest of the physical line channels.
        self.physical_edge_features = list(self.edge_features)
        self.relation_edge_types = {
            edge_type: name
            for name, edge_type in self.EDGE_TYPE_NAMES.items()
            if name != "self" or self.include_legacy_self_relation_feature
        }
        self.relation_edge_features = [
            f"relation_{name}" for name in self.relation_edge_types.values()
        ]
        self.relation_edge_columns = {
            edge_type: column
            for column, edge_type in enumerate(self.relation_edge_types)
        }
        self.edge_features = (
            self.physical_edge_features + self.relation_edge_features
        )

        # Busbar slots occupy [0, B); generator and load slots are static role
        # identifiers used only by the optional node-id embeddings.
        self.node_id_stride = self.n_busbar + 2
        self.specs = {
            "state": self._make_heterogeneous_spec(
                np.arange(self.n_sub), np.arange(self.n_line)
            )
        }
        for agent, domain_nodes in self.observation_domains.items():
            sub_ids, line_ids = self._local_ids(domain_nodes)
            self.specs[agent] = self._make_heterogeneous_spec(
                sub_ids,
                line_ids,
                controlled_nodes=domain_nodes,
            )

    def build_for_spec(
        self,
        obs,
        spec: Dict[str, Any],
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        return self._build_heterogeneous_for_spec(obs, spec, cache=cache)

    def _make_heterogeneous_spec(
        self,
        sub_ids,
        line_ids,
        controlled_nodes=None,
    ) -> Dict[str, Any]:
        sub_ids = np.unique(np.asarray(sub_ids, dtype=np.int64))
        line_ids = np.asarray(line_ids, dtype=np.int64)
        controlled_nodes = (
            sub_ids
            if controlled_nodes is None
            else np.asarray(controlled_nodes, dtype=np.int64)
        )
        # ``sub_ids`` contains the controlled substations plus the far ends of
        # boundary lines. The far-end busbar is needed to carry the physical
        # line relation, but its generators and loads are private to the other
        # agent. Selecting assets from ``controlled_nodes`` enforces that
        # information boundary at construction time, before message passing or
        # pooling can expose them. For the global state graph,
        # ``controlled_nodes == sub_ids``, so all assets remain present.
        gen_ids = np.nonzero(np.isin(self.gen_to_sub, controlled_nodes))[0].astype(
            np.int64
        )
        load_ids = np.nonzero(np.isin(self.load_to_sub, controlled_nodes))[0].astype(
            np.int64
        )

        bus_entity_ids = self._bus_node_ids(sub_ids)
        gen_entity_ids = self.n_bus_nodes + gen_ids
        load_entity_ids = self.n_bus_nodes + self.n_gen + load_ids
        entity_ids = np.concatenate(
            [bus_entity_ids, gen_entity_ids, load_entity_ids]
        ).astype(np.int64, copy=False)
        local_index = {
            int(entity_id): idx for idx, entity_id in enumerate(entity_ids)
        }

        bus_sub_ids = bus_entity_ids // self.n_busbar
        node_substation_ids = np.concatenate(
            [bus_sub_ids, self.gen_to_sub[gen_ids], self.load_to_sub[load_ids]]
        ).astype(np.int64, copy=False)
        node_slots = np.concatenate(
            [
                bus_entity_ids % self.n_busbar,
                np.full(len(gen_ids), self.n_busbar, dtype=np.int64),
                np.full(len(load_ids), self.n_busbar + 1, dtype=np.int64),
            ]
        )
        node_ids = node_substation_ids * self.node_id_stride + node_slots
        node_type = np.concatenate(
            [
                np.full(
                    len(bus_entity_ids), self.NODE_TYPE_BUSBAR, dtype=np.int64
                ),
                np.full(
                    len(gen_ids), self.NODE_TYPE_GENERATOR, dtype=np.int64
                ),
                np.full(len(load_ids), self.NODE_TYPE_LOAD, dtype=np.int64),
            ]
        )
        controlled_node_mask = np.isin(
            node_substation_ids, controlled_nodes
        ).astype(np.float32)

        directed_edges = []
        edge_line_ids = []
        edge_or_bus_ids = []
        edge_ex_bus_ids = []
        edge_asset_kind = []
        edge_asset_ids = []
        edge_asset_bus_ids = []
        edge_types = []

        def append_edge(
            src_entity,
            dst_entity,
            edge_type,
            line_id=-1,
            or_bus=-1,
            ex_bus=-1,
            asset_kind=0,
            asset_id=-1,
            asset_bus=-1,
        ):
            directed_edges.append(
                (local_index[int(src_entity)], local_index[int(dst_entity)])
            )
            edge_line_ids.append(line_id)
            edge_or_bus_ids.append(or_bus)
            edge_ex_bus_ids.append(ex_bus)
            edge_asset_kind.append(asset_kind)
            edge_asset_ids.append(asset_id)
            edge_asset_bus_ids.append(asset_bus)
            edge_types.append(edge_type)

        if self.add_self_edges:
            for entity_id in entity_ids:
                append_edge(entity_id, entity_id, self.EDGE_TYPE_SELF)

        for line_id in line_ids:
            or_sub = int(self.line_or[line_id])
            ex_sub = int(self.line_ex[line_id])
            for or_bus in range(self.n_busbar):
                or_entity = self._bus_node_id(or_sub, or_bus)
                if or_entity not in local_index:
                    continue
                for ex_bus in range(self.n_busbar):
                    ex_entity = self._bus_node_id(ex_sub, ex_bus)
                    if ex_entity not in local_index:
                        continue
                    append_edge(
                        or_entity,
                        ex_entity,
                        self.EDGE_TYPE_PHYSICAL_LINE,
                        line_id=line_id,
                        or_bus=or_bus,
                        ex_bus=ex_bus,
                    )
                    append_edge(
                        ex_entity,
                        or_entity,
                        self.EDGE_TYPE_PHYSICAL_LINE,
                        line_id=line_id,
                        or_bus=or_bus,
                        ex_bus=ex_bus,
                    )

        if self.add_substation_edges:
            for sub_id in sub_ids:
                bus_entities = self._bus_node_ids([sub_id])
                for src_entity in bus_entities:
                    for dst_entity in bus_entities:
                        if src_entity == dst_entity:
                            continue
                        append_edge(
                            src_entity,
                            dst_entity,
                            self.EDGE_TYPE_SAME_SUBSTATION,
                        )

        for gen_id in gen_ids:
            sub_id = int(self.gen_to_sub[gen_id])
            gen_entity = self.n_bus_nodes + int(gen_id)
            for bus_id in range(self.n_busbar):
                bus_entity = self._bus_node_id(sub_id, bus_id)
                if self.generator_edge_direction in {
                    "bidirectional",
                    "asset_to_busbar",
                }:
                    append_edge(
                        gen_entity,
                        bus_entity,
                        self.EDGE_TYPE_GENERATOR_TO_BUSBAR,
                        asset_kind=self.ASSET_GENERATOR,
                        asset_id=int(gen_id),
                        asset_bus=bus_id,
                    )
                if self.generator_edge_direction in {
                    "bidirectional",
                    "busbar_to_asset",
                }:
                    append_edge(
                        bus_entity,
                        gen_entity,
                        self.EDGE_TYPE_BUSBAR_TO_GENERATOR,
                        asset_kind=self.ASSET_GENERATOR,
                        asset_id=int(gen_id),
                        asset_bus=bus_id,
                    )

        for load_id in load_ids:
            sub_id = int(self.load_to_sub[load_id])
            load_entity = self.n_bus_nodes + self.n_gen + int(load_id)
            for bus_id in range(self.n_busbar):
                bus_entity = self._bus_node_id(sub_id, bus_id)
                if self.load_edge_direction in {
                    "bidirectional",
                    "asset_to_busbar",
                }:
                    append_edge(
                        load_entity,
                        bus_entity,
                        self.EDGE_TYPE_LOAD_TO_BUSBAR,
                        asset_kind=self.ASSET_LOAD,
                        asset_id=int(load_id),
                        asset_bus=bus_id,
                    )
                if self.load_edge_direction in {
                    "bidirectional",
                    "busbar_to_asset",
                }:
                    append_edge(
                        bus_entity,
                        load_entity,
                        self.EDGE_TYPE_BUSBAR_TO_LOAD,
                        asset_kind=self.ASSET_LOAD,
                        asset_id=int(load_id),
                        asset_bus=bus_id,
                    )

        edge_index = np.asarray(directed_edges, dtype=np.int64).T
        if edge_index.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return {
            "graph_type": "heterogeneous",
            "node_ids": node_ids.astype(np.int64, copy=False),
            "entity_ids": entity_ids,
            "node_type": node_type,
            "node_type_names": dict(self.NODE_TYPE_NAMES),
            "n_node_types": len(self.NODE_TYPE_NAMES),
            "node_substation_ids": node_substation_ids,
            "node_id_stride": self.node_id_stride,
            "n_bus_id_embeddings": self.node_id_stride,
            "line_ids": line_ids,
            "gen_ids": gen_ids,
            "load_ids": load_ids,
            "controlled_nodes": controlled_nodes,
            "controlled_node_mask": controlled_node_mask,
            "edge_index": edge_index,
            "edge_line_ids": np.asarray(edge_line_ids, dtype=np.int64),
            "edge_or_bus_ids": np.asarray(edge_or_bus_ids, dtype=np.int64),
            "edge_ex_bus_ids": np.asarray(edge_ex_bus_ids, dtype=np.int64),
            "edge_asset_kind": np.asarray(edge_asset_kind, dtype=np.int64),
            "edge_asset_ids": np.asarray(edge_asset_ids, dtype=np.int64),
            "edge_asset_bus_ids": np.asarray(
                edge_asset_bus_ids, dtype=np.int64
            ),
            "edge_type": np.asarray(edge_types, dtype=np.int64),
            "edge_type_names": dict(self.EDGE_TYPE_NAMES),
            "n_edge_types": len(self.EDGE_TYPE_NAMES),
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "node_feature_names": list(self.node_features),
            "edge_feature_names": list(self.edge_features),
            "n_sub": self.n_sub,
            "n_busbar": self.n_busbar,
            "n_bus_nodes": self.n_bus_nodes,
            "n_gen": self.n_gen,
            "n_load": self.n_load,
            "generator_edge_direction": self.generator_edge_direction,
            "load_edge_direction": self.load_edge_direction,
        }

    def _make_obs_cache(self, obs) -> Dict[str, np.ndarray]:
        line_status = self._obs_array(obs, "line_status", expected=self.n_line)
        if line_status is None:
            line_status = np.ones(self.n_line, dtype=np.float32)
        return {
            "line_features": self._line_feature_matrix(obs),
            "line_status": line_status,
            "line_or_bus": self._topo_bus_ids(obs, self.line_or_pos) - 1,
            "line_ex_bus": self._topo_bus_ids(obs, self.line_ex_pos) - 1,
            "gen_bus": self._topo_bus_ids(obs, self.gen_pos) - 1,
            "load_bus": self._topo_bus_ids(obs, self.load_pos) - 1,
        }

    def _build_heterogeneous_for_spec(
        self,
        obs,
        spec: Dict[str, Any],
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        if cache is None:
            cache = self._make_obs_cache(obs)
        node_features = self._heterogeneous_node_features(obs, spec, cache)
        edge_features, edge_mask = self._heterogeneous_edge_features(spec, cache)
        node_mask = self._visible_node_mask(spec, edge_mask)
        node_features = self._hide_invisible_node_features(
            node_features, node_mask
        )
        edge_features, edge_mask = self._hide_invisible_node_edges(
            spec, edge_features, edge_mask, node_mask
        )
        return {
            "node_features": node_features.astype(np.float32, copy=False),
            "edge_features": edge_features.astype(np.float32, copy=False),
            "node_mask": node_mask,
            "edge_mask": edge_mask.astype(np.float32, copy=False),
            "node_type": spec["node_type"].astype(np.int64, copy=False),
            "edge_type": spec["edge_type"].astype(np.int64, copy=False),
            "controlled_node_mask": spec["controlled_node_mask"].astype(
                np.float32, copy=False
            ),
        }

    def _heterogeneous_node_features(self, obs, spec, cache):
        features = np.zeros(
            (len(spec["node_ids"]), self.node_dim), dtype=np.float32
        )
        col = {name: idx for idx, name in enumerate(self.node_features)}
        node_type = spec["node_type"]
        bus_rows = np.nonzero(node_type == self.NODE_TYPE_BUSBAR)[0]
        gen_rows = np.nonzero(node_type == self.NODE_TYPE_GENERATOR)[0]
        load_rows = np.nonzero(node_type == self.NODE_TYPE_LOAD)[0]

        features[bus_rows, col["is_busbar"]] = 1.0
        features[gen_rows, col["is_generator"]] = 1.0
        features[load_rows, col["is_load"]] = 1.0
        features[bus_rows, col["connected"]] = 1.0
        features[:, col["domain_mask"]] = spec["controlled_node_mask"]

        sub_cooldown = self._obs_array(
            obs, "time_before_cooldown_sub", expected=self.n_sub
        )
        if sub_cooldown is not None and len(bus_rows) > 0:
            features[bus_rows, col["time_before_cooldown_sub"]] = sub_cooldown[
                spec["node_substation_ids"][bus_rows]
            ]

        self._put_entity_node_features(
            features,
            gen_rows,
            spec["gen_ids"],
            cache["gen_bus"],
            self._obs_array(obs, "gen_p", expected=self.n_gen),
            self._obs_array(obs, "gen_theta", expected=self.n_gen),
            col,
        )
        self._put_entity_node_features(
            features,
            load_rows,
            spec["load_ids"],
            cache["load_bus"],
            self._obs_array(obs, "load_p", expected=self.n_load),
            self._obs_array(obs, "load_theta", expected=self.n_load),
            col,
        )
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _put_entity_node_features(
        self,
        features,
        rows,
        asset_ids,
        asset_bus,
        p_values,
        theta_values,
        col,
    ):
        if len(rows) == 0:
            return
        connected = (
            (asset_bus[asset_ids] >= 0)
            & (asset_bus[asset_ids] < self.n_busbar)
        )
        features[rows, col["connected"]] = connected.astype(np.float32)
        if p_values is not None:
            features[rows[connected], col["p"]] = p_values[asset_ids[connected]]
        if theta_values is not None and "theta" in col:
            features[rows[connected], col["theta"]] = theta_values[
                asset_ids[connected]
            ]

    def _heterogeneous_edge_features(self, spec, cache):
        n_edges = len(spec["edge_type"])
        edge_features = np.zeros((n_edges, self.edge_dim), dtype=np.float32)
        active = np.ones((n_edges,), dtype=bool)
        edge_type = spec["edge_type"]

        relation_offset = len(self.physical_edge_features)
        for edge_type_id, relation_column in self.relation_edge_columns.items():
            rows = np.nonzero(edge_type == edge_type_id)[0]
            edge_features[rows, relation_offset + relation_column] = 1.0

        physical = spec["edge_line_ids"] >= 0
        if np.any(physical):
            line_ids = spec["edge_line_ids"][physical]
            edge_features[physical, : len(self.physical_edge_features)] = cache[
                "line_features"
            ][line_ids, : len(self.physical_edge_features)]
            active[physical] = (
                (cache["line_status"][line_ids] > 0)
                & (
                    cache["line_or_bus"][line_ids]
                    == spec["edge_or_bus_ids"][physical]
                )
                & (
                    cache["line_ex_bus"][line_ids]
                    == spec["edge_ex_bus_ids"][physical]
                )
            )

        generator_edges = spec["edge_asset_kind"] == self.ASSET_GENERATOR
        if np.any(generator_edges):
            gen_ids = spec["edge_asset_ids"][generator_edges]
            active[generator_edges] = (
                cache["gen_bus"][gen_ids]
                == spec["edge_asset_bus_ids"][generator_edges]
            )

        load_edges = spec["edge_asset_kind"] == self.ASSET_LOAD
        if np.any(load_edges):
            load_ids = spec["edge_asset_ids"][load_edges]
            active[load_edges] = (
                cache["load_bus"][load_ids]
                == spec["edge_asset_bus_ids"][load_edges]
            )

        # If rho is selected as a GCN message weight, structural relations need
        # unit weight so generator/load and optional relation messages survive.
        if "rho" in self.physical_edge_features:
            rho_idx = self.physical_edge_features.index("rho")
            edge_features[~physical, rho_idx] = 1.0

        edge_features[~active] = 0.0
        return (
            np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0),
            active.astype(np.float32),
        )


class HeterogeneousLineGraphBuilder(HeterogeneousGridGraphBuilder):
    """Build a typed graph with explicit transmission-line nodes.

    Busbars, generators, loads, and physical lines are nodes. In a local agent
    graph, equipment nodes belong only to controlled substations. Every line
    touching that domain is included as a controlled node, but a boundary line
    is connected only to its endpoint inside the domain: no neighboring
    busbar, generator, or load is instantiated. The line node itself carries
    the shared-line measurements needed by the controlled busbar.

    In the global state graph both endpoint substations are present, so every
    line node has candidate attachment edges at both ends as before. Configured
    message directions may be one-way or bidirectional, and masks select the
    current endpoint busbars.
    """

    NODE_TYPE_BUSBAR = 0
    NODE_TYPE_GENERATOR = 1
    NODE_TYPE_LOAD = 2
    NODE_TYPE_TRANSMISSION_LINE = 3
    NODE_TYPE_NAMES = {
        "busbar": NODE_TYPE_BUSBAR,
        "generator": NODE_TYPE_GENERATOR,
        "load": NODE_TYPE_LOAD,
        "transmission_line": NODE_TYPE_TRANSMISSION_LINE,
    }

    EDGE_TYPE_SELF = 0
    EDGE_TYPE_SAME_SUBSTATION = 1
    EDGE_TYPE_BUSBAR_TO_LINE_ORIGIN = 2
    EDGE_TYPE_LINE_ORIGIN_TO_BUSBAR = 3
    EDGE_TYPE_BUSBAR_TO_LINE_EXTREMITY = 4
    EDGE_TYPE_LINE_EXTREMITY_TO_BUSBAR = 5
    EDGE_TYPE_GENERATOR_TO_BUSBAR = 6
    EDGE_TYPE_BUSBAR_TO_GENERATOR = 7
    EDGE_TYPE_LOAD_TO_BUSBAR = 8
    EDGE_TYPE_BUSBAR_TO_LOAD = 9

    # The parent constructor briefly builds and then discards an element-edge
    # spec, for which this compatibility alias is required.
    EDGE_TYPE_PHYSICAL_LINE = EDGE_TYPE_BUSBAR_TO_LINE_ORIGIN
    EDGE_TYPE_NAMES = {
        "self": EDGE_TYPE_SELF,
        "same_substation_busbar": EDGE_TYPE_SAME_SUBSTATION,
        "busbar_to_line_origin": EDGE_TYPE_BUSBAR_TO_LINE_ORIGIN,
        "line_origin_to_busbar": EDGE_TYPE_LINE_ORIGIN_TO_BUSBAR,
        "busbar_to_line_extremity": EDGE_TYPE_BUSBAR_TO_LINE_EXTREMITY,
        "line_extremity_to_busbar": EDGE_TYPE_LINE_EXTREMITY_TO_BUSBAR,
        "generator_to_busbar": EDGE_TYPE_GENERATOR_TO_BUSBAR,
        "busbar_to_generator": EDGE_TYPE_BUSBAR_TO_GENERATOR,
        "load_to_busbar": EDGE_TYPE_LOAD_TO_BUSBAR,
        "busbar_to_load": EDGE_TYPE_BUSBAR_TO_LOAD,
    }

    BASE_LINE_NODE_FEATURES = [
        "is_busbar",
        "is_generator",
        "is_load",
        "is_transmission_line",
        "p",
        "theta",
        "line_status",
        "rho",
        "timestep_overflow",
        "time_before_cooldown_line",
        "time_before_cooldown_sub",
        "connected",
        "domain_mask",
    ]

    def __init__(
        self,
        g2op_env,
        observation_domains: Dict[str, List[int]],
        include_neighbors: bool = True,
        include_maintenance: bool = False,
        add_self_edges: bool = False,
        add_substation_edges: bool = False,
        generator_edge_direction: str = "bidirectional",
        load_edge_direction: str = "bidirectional",
        line_node_edge_direction: str = "bidirectional",
        angle_representation: str = "node",
        include_legacy_self_relation_feature: bool = False,
        context_requires_connection: bool = True,
    ) -> None:
        if _validate_angle_representation(angle_representation) != "node":
            # This builder puts each line on its own node, so an angle drop is
            # a node attribute here rather than an edge one. Supporting it
            # needs a different placement than the two graph types above.
            raise ValueError(
                "angle_representation='edge_diff' is not implemented for the "
                "heterogeneous line-node graph."
            )
        super().__init__(
            g2op_env,
            observation_domains,
            include_neighbors=include_neighbors,
            include_maintenance=include_maintenance,
            add_self_edges=add_self_edges,
            add_substation_edges=add_substation_edges,
            generator_edge_direction=generator_edge_direction,
            load_edge_direction=load_edge_direction,
            angle_representation=angle_representation,
            include_legacy_self_relation_feature=(
                include_legacy_self_relation_feature
            ),
            context_requires_connection=context_requires_connection,
        )
        self.line_node_edge_direction = _validate_edge_direction(
            line_node_edge_direction,
            LINE_NODE_EDGE_DIRECTIONS,
            "line-node edge direction",
        )
        self.node_features = list(self.BASE_LINE_NODE_FEATURES)
        if include_maintenance:
            insert_at = self.node_features.index("time_before_cooldown_sub")
            self.node_features[insert_at:insert_at] = self.MAINTENANCE_EDGE_FEATURES

        # All physical line state, including rho, is carried by the explicit
        # transmission-line node. Attachment edges encode only their relation.
        self.physical_edge_features = []
        self.relation_edge_features = [
            f"relation_{name}" for name in self.relation_edge_types.values()
        ]
        self.edge_features = (
            self.physical_edge_features + self.relation_edge_features
        )
        self.node_id_stride = self.n_busbar + 3

        self.specs = {
            "state": self._make_line_node_spec(
                np.arange(self.n_sub), np.arange(self.n_line)
            )
        }
        for agent, domain_nodes in self.observation_domains.items():
            sub_ids, line_ids = self._local_ids(domain_nodes)
            self.specs[agent] = self._make_line_node_spec(
                sub_ids,
                line_ids,
                controlled_nodes=domain_nodes,
            )

    def _local_ids(self, domain_nodes):
        """Return controlled substations and every line touching the domain.

        The explicit line node replaces the far-end busbar as boundary context.
        Consequently ``include_neighbors`` does not expand equipment nodes for
        this representation: shared lines are part of the agent's own graph,
        while neighboring substations are never included.
        """
        domain_nodes = np.unique(np.asarray(domain_nodes, dtype=np.int64))
        touches_domain = np.isin(self.line_or, domain_nodes) | np.isin(
            self.line_ex, domain_nodes
        )
        line_ids = np.nonzero(touches_domain)[0].astype(np.int64)
        return domain_nodes, line_ids

    def build_for_spec(
        self,
        obs,
        spec: Dict[str, Any],
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        return self._build_line_node_for_spec(obs, spec, cache=cache)

    def _make_line_node_spec(
        self,
        sub_ids,
        line_ids,
        controlled_nodes=None,
    ) -> Dict[str, Any]:
        sub_ids = np.unique(np.asarray(sub_ids, dtype=np.int64))
        line_ids = np.asarray(line_ids, dtype=np.int64)
        controlled_nodes = (
            sub_ids
            if controlled_nodes is None
            else np.asarray(controlled_nodes, dtype=np.int64)
        )
        gen_ids = np.nonzero(np.isin(self.gen_to_sub, sub_ids))[0].astype(np.int64)
        load_ids = np.nonzero(np.isin(self.load_to_sub, sub_ids))[0].astype(
            np.int64
        )

        bus_entity_ids = self._bus_node_ids(sub_ids)
        gen_entity_ids = self.n_bus_nodes + gen_ids
        load_entity_ids = self.n_bus_nodes + self.n_gen + load_ids
        line_entity_offset = self.n_bus_nodes + self.n_gen + self.n_load
        line_entity_ids = line_entity_offset + line_ids
        entity_ids = np.concatenate(
            [
                bus_entity_ids,
                gen_entity_ids,
                load_entity_ids,
                line_entity_ids,
            ]
        ).astype(np.int64, copy=False)
        local_index = {
            int(entity_id): idx for idx, entity_id in enumerate(entity_ids)
        }

        bus_sub_ids = bus_entity_ids // self.n_busbar
        # A line touches two substations. Its origin is used only for the
        # optional substation embedding; both endpoints remain explicit edges.
        node_substation_ids = np.concatenate(
            [
                bus_sub_ids,
                self.gen_to_sub[gen_ids],
                self.load_to_sub[load_ids],
                self.line_or[line_ids],
            ]
        ).astype(np.int64, copy=False)
        node_slots = np.concatenate(
            [
                bus_entity_ids % self.n_busbar,
                np.full(len(gen_ids), self.n_busbar, dtype=np.int64),
                np.full(len(load_ids), self.n_busbar + 1, dtype=np.int64),
                np.full(len(line_ids), self.n_busbar + 2, dtype=np.int64),
            ]
        )
        node_ids = node_substation_ids * self.node_id_stride + node_slots
        node_type = np.concatenate(
            [
                np.full(
                    len(bus_entity_ids), self.NODE_TYPE_BUSBAR, dtype=np.int64
                ),
                np.full(
                    len(gen_ids), self.NODE_TYPE_GENERATOR, dtype=np.int64
                ),
                np.full(len(load_ids), self.NODE_TYPE_LOAD, dtype=np.int64),
                np.full(
                    len(line_ids),
                    self.NODE_TYPE_TRANSMISSION_LINE,
                    dtype=np.int64,
                ),
            ]
        )
        equipment_controlled = np.isin(
            node_substation_ids[: len(bus_entity_ids) + len(gen_ids) + len(load_ids)],
            controlled_nodes,
        )
        line_controlled = (
            np.isin(self.line_or[line_ids], controlled_nodes)
            | np.isin(self.line_ex[line_ids], controlled_nodes)
        )
        controlled_node_mask = np.concatenate(
            [equipment_controlled, line_controlled]
        ).astype(np.float32)

        directed_edges = []
        edge_line_ids = []
        edge_line_endpoint = []
        edge_line_bus_ids = []
        edge_asset_kind = []
        edge_asset_ids = []
        edge_asset_bus_ids = []
        edge_types = []

        def append_edge(
            src_entity,
            dst_entity,
            edge_type,
            line_id=-1,
            line_endpoint=-1,
            line_bus=-1,
            asset_kind=0,
            asset_id=-1,
            asset_bus=-1,
        ):
            directed_edges.append(
                (local_index[int(src_entity)], local_index[int(dst_entity)])
            )
            edge_line_ids.append(line_id)
            edge_line_endpoint.append(line_endpoint)
            edge_line_bus_ids.append(line_bus)
            edge_asset_kind.append(asset_kind)
            edge_asset_ids.append(asset_id)
            edge_asset_bus_ids.append(asset_bus)
            edge_types.append(edge_type)

        if self.add_self_edges:
            for entity_id in entity_ids:
                append_edge(entity_id, entity_id, self.EDGE_TYPE_SELF)

        for line_id in line_ids:
            line_entity = line_entity_offset + int(line_id)
            or_sub = int(self.line_or[line_id])
            ex_sub = int(self.line_ex[line_id])
            for bus_id in range(self.n_busbar):
                or_bus_entity = self._bus_node_id(or_sub, bus_id)
                ex_bus_entity = self._bus_node_id(ex_sub, bus_id)
                # A local graph deliberately omits the far-end busbars of a
                # boundary line. Only create attachment candidates for endpoint
                # entities that actually belong to this graph. The global state
                # graph still contains both endpoints and therefore retains all
                # four relation directions per busbar candidate.
                if or_bus_entity in local_index:
                    if self.line_node_edge_direction in {
                        "bidirectional",
                        "busbar_to_line",
                    }:
                        append_edge(
                            or_bus_entity,
                            line_entity,
                            self.EDGE_TYPE_BUSBAR_TO_LINE_ORIGIN,
                            line_id=int(line_id),
                            line_endpoint=0,
                            line_bus=bus_id,
                        )
                    if self.line_node_edge_direction in {
                        "bidirectional",
                        "line_to_busbar",
                    }:
                        append_edge(
                            line_entity,
                            or_bus_entity,
                            self.EDGE_TYPE_LINE_ORIGIN_TO_BUSBAR,
                            line_id=int(line_id),
                            line_endpoint=0,
                            line_bus=bus_id,
                        )
                if ex_bus_entity in local_index:
                    if self.line_node_edge_direction in {
                        "bidirectional",
                        "busbar_to_line",
                    }:
                        append_edge(
                            ex_bus_entity,
                            line_entity,
                            self.EDGE_TYPE_BUSBAR_TO_LINE_EXTREMITY,
                            line_id=int(line_id),
                            line_endpoint=1,
                            line_bus=bus_id,
                        )
                    if self.line_node_edge_direction in {
                        "bidirectional",
                        "line_to_busbar",
                    }:
                        append_edge(
                            line_entity,
                            ex_bus_entity,
                            self.EDGE_TYPE_LINE_EXTREMITY_TO_BUSBAR,
                            line_id=int(line_id),
                            line_endpoint=1,
                            line_bus=bus_id,
                        )

        if self.add_substation_edges:
            for sub_id in sub_ids:
                bus_entities = self._bus_node_ids([sub_id])
                for src_entity in bus_entities:
                    for dst_entity in bus_entities:
                        if src_entity == dst_entity:
                            continue
                        append_edge(
                            src_entity,
                            dst_entity,
                            self.EDGE_TYPE_SAME_SUBSTATION,
                        )

        for gen_id in gen_ids:
            sub_id = int(self.gen_to_sub[gen_id])
            gen_entity = self.n_bus_nodes + int(gen_id)
            for bus_id in range(self.n_busbar):
                bus_entity = self._bus_node_id(sub_id, bus_id)
                if self.generator_edge_direction in {
                    "bidirectional",
                    "asset_to_busbar",
                }:
                    append_edge(
                        gen_entity,
                        bus_entity,
                        self.EDGE_TYPE_GENERATOR_TO_BUSBAR,
                        asset_kind=self.ASSET_GENERATOR,
                        asset_id=int(gen_id),
                        asset_bus=bus_id,
                    )
                if self.generator_edge_direction in {
                    "bidirectional",
                    "busbar_to_asset",
                }:
                    append_edge(
                        bus_entity,
                        gen_entity,
                        self.EDGE_TYPE_BUSBAR_TO_GENERATOR,
                        asset_kind=self.ASSET_GENERATOR,
                        asset_id=int(gen_id),
                        asset_bus=bus_id,
                    )

        for load_id in load_ids:
            sub_id = int(self.load_to_sub[load_id])
            load_entity = self.n_bus_nodes + self.n_gen + int(load_id)
            for bus_id in range(self.n_busbar):
                bus_entity = self._bus_node_id(sub_id, bus_id)
                if self.load_edge_direction in {
                    "bidirectional",
                    "asset_to_busbar",
                }:
                    append_edge(
                        load_entity,
                        bus_entity,
                        self.EDGE_TYPE_LOAD_TO_BUSBAR,
                        asset_kind=self.ASSET_LOAD,
                        asset_id=int(load_id),
                        asset_bus=bus_id,
                    )
                if self.load_edge_direction in {
                    "bidirectional",
                    "busbar_to_asset",
                }:
                    append_edge(
                        bus_entity,
                        load_entity,
                        self.EDGE_TYPE_BUSBAR_TO_LOAD,
                        asset_kind=self.ASSET_LOAD,
                        asset_id=int(load_id),
                        asset_bus=bus_id,
                    )

        edge_index = np.asarray(directed_edges, dtype=np.int64).T
        if edge_index.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        busbar_id_to_node_row = np.full(
            self.n_bus_nodes, -1, dtype=np.int64
        )
        line_id_to_node_row = np.full(self.n_line, -1, dtype=np.int64)
        gen_id_to_node_row = np.full(self.n_gen, -1, dtype=np.int64)
        load_id_to_node_row = np.full(self.n_load, -1, dtype=np.int64)

        n_bus_rows = len(bus_entity_ids)
        n_gen_rows = len(gen_ids)
        n_load_rows = len(load_ids)
        busbar_id_to_node_row[bus_entity_ids] = np.arange(
            n_bus_rows, dtype=np.int64
        )
        gen_id_to_node_row[gen_ids] = n_bus_rows + np.arange(
            n_gen_rows, dtype=np.int64
        )
        load_id_to_node_row[load_ids] = (
            n_bus_rows + n_gen_rows + np.arange(n_load_rows, dtype=np.int64)
        )
        line_id_to_node_row[line_ids] = (
            n_bus_rows
            + n_gen_rows
            + n_load_rows
            + np.arange(len(line_ids), dtype=np.int64)
        )
        substation_busbar_node_rows = busbar_id_to_node_row.reshape(
            self.n_sub, self.n_busbar
        )

        return {
            "graph_type": "heterogeneous_line",
            "node_ids": node_ids.astype(np.int64, copy=False),
            "entity_ids": entity_ids,
            "node_type": node_type,
            "node_type_names": dict(self.NODE_TYPE_NAMES),
            "n_node_types": len(self.NODE_TYPE_NAMES),
            "node_substation_ids": node_substation_ids,
            "node_id_stride": self.node_id_stride,
            "n_bus_id_embeddings": self.node_id_stride,
            "line_ids": line_ids,
            "gen_ids": gen_ids,
            "load_ids": load_ids,
            "busbar_id_to_node_row": busbar_id_to_node_row,
            "line_id_to_node_row": line_id_to_node_row,
            "gen_id_to_node_row": gen_id_to_node_row,
            "load_id_to_node_row": load_id_to_node_row,
            "substation_busbar_node_rows": substation_busbar_node_rows,
            "controlled_nodes": controlled_nodes,
            "controlled_node_mask": controlled_node_mask,
            "edge_index": edge_index,
            "edge_line_ids": np.asarray(edge_line_ids, dtype=np.int64),
            "edge_line_endpoint": np.asarray(
                edge_line_endpoint, dtype=np.int64
            ),
            "edge_line_bus_ids": np.asarray(edge_line_bus_ids, dtype=np.int64),
            "edge_asset_kind": np.asarray(edge_asset_kind, dtype=np.int64),
            "edge_asset_ids": np.asarray(edge_asset_ids, dtype=np.int64),
            "edge_asset_bus_ids": np.asarray(
                edge_asset_bus_ids, dtype=np.int64
            ),
            "edge_type": np.asarray(edge_types, dtype=np.int64),
            "edge_type_names": dict(self.EDGE_TYPE_NAMES),
            "n_edge_types": len(self.EDGE_TYPE_NAMES),
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "node_feature_names": list(self.node_features),
            "edge_feature_names": list(self.edge_features),
            "n_sub": self.n_sub,
            "n_busbar": self.n_busbar,
            "n_bus_nodes": self.n_bus_nodes,
            "n_gen": self.n_gen,
            "n_load": self.n_load,
            "n_line": self.n_line,
            "generator_edge_direction": self.generator_edge_direction,
            "load_edge_direction": self.load_edge_direction,
            "line_node_edge_direction": self.line_node_edge_direction,
        }

    def _build_line_node_for_spec(
        self,
        obs,
        spec: Dict[str, Any],
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        if cache is None:
            cache = self._make_obs_cache(obs)
        node_features = self._line_node_features(obs, spec, cache)
        edge_features, edge_mask = self._line_node_edge_features(spec, cache)
        node_mask = self._visible_node_mask(spec, edge_mask)
        node_features = self._hide_invisible_node_features(
            node_features, node_mask
        )
        edge_features, edge_mask = self._hide_invisible_node_edges(
            spec, edge_features, edge_mask, node_mask
        )
        return {
            "node_features": node_features.astype(np.float32, copy=False),
            "edge_features": edge_features.astype(np.float32, copy=False),
            "node_mask": node_mask,
            "edge_mask": edge_mask.astype(np.float32, copy=False),
            "node_type": spec["node_type"].astype(np.int64, copy=False),
            "edge_type": spec["edge_type"].astype(np.int64, copy=False),
            "controlled_node_mask": spec["controlled_node_mask"].astype(
                np.float32, copy=False
            ),
        }

    def _line_node_features(self, obs, spec, cache):
        features = np.zeros(
            (len(spec["node_ids"]), self.node_dim), dtype=np.float32
        )
        col = {name: idx for idx, name in enumerate(self.node_features)}
        node_type = spec["node_type"]
        bus_rows = np.nonzero(node_type == self.NODE_TYPE_BUSBAR)[0]
        gen_rows = np.nonzero(node_type == self.NODE_TYPE_GENERATOR)[0]
        load_rows = np.nonzero(node_type == self.NODE_TYPE_LOAD)[0]
        line_rows = np.nonzero(
            node_type == self.NODE_TYPE_TRANSMISSION_LINE
        )[0]

        features[bus_rows, col["is_busbar"]] = 1.0
        features[gen_rows, col["is_generator"]] = 1.0
        features[load_rows, col["is_load"]] = 1.0
        features[line_rows, col["is_transmission_line"]] = 1.0
        features[bus_rows, col["connected"]] = 1.0
        features[:, col["domain_mask"]] = spec["controlled_node_mask"]

        sub_cooldown = self._obs_array(
            obs, "time_before_cooldown_sub", expected=self.n_sub
        )
        if sub_cooldown is not None and len(bus_rows) > 0:
            features[bus_rows, col["time_before_cooldown_sub"]] = sub_cooldown[
                spec["node_substation_ids"][bus_rows]
            ]

        self._put_entity_node_features(
            features,
            gen_rows,
            spec["gen_ids"],
            cache["gen_bus"],
            self._obs_array(obs, "gen_p", expected=self.n_gen),
            self._obs_array(obs, "gen_theta", expected=self.n_gen),
            col,
        )
        self._put_entity_node_features(
            features,
            load_rows,
            spec["load_ids"],
            cache["load_bus"],
            self._obs_array(obs, "load_p", expected=self.n_load),
            self._obs_array(obs, "load_theta", expected=self.n_load),
            col,
        )

        line_ids = spec["line_ids"]
        if len(line_rows) > 0:
            features[line_rows, col["connected"]] = (
                cache["line_status"][line_ids] > 0
            ).astype(np.float32)
            for name in [
                "line_status",
                "rho",
                "timestep_overflow",
                "time_before_cooldown_line",
                "time_next_maintenance",
                "duration_next_maintenance",
            ]:
                if name not in col:
                    continue
                values = self._obs_array(obs, name, expected=self.n_line)
                if values is not None:
                    features[line_rows, col[name]] = values[line_ids]

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _line_node_edge_features(self, spec, cache):
        n_edges = len(spec["edge_type"])
        edge_features = np.zeros((n_edges, self.edge_dim), dtype=np.float32)
        active = np.ones((n_edges,), dtype=bool)
        edge_type = spec["edge_type"]

        relation_offset = len(self.physical_edge_features)
        for edge_type_id, relation_column in self.relation_edge_columns.items():
            rows = np.nonzero(edge_type == edge_type_id)[0]
            edge_features[rows, relation_offset + relation_column] = 1.0

        line_edges = spec["edge_line_ids"] >= 0
        if np.any(line_edges):
            line_ids = spec["edge_line_ids"][line_edges]
            endpoint = spec["edge_line_endpoint"][line_edges]
            endpoint_bus = np.where(
                endpoint == 0,
                cache["line_or_bus"][line_ids],
                cache["line_ex_bus"][line_ids],
            )
            active[line_edges] = (
                (cache["line_status"][line_ids] > 0)
                & (endpoint_bus == spec["edge_line_bus_ids"][line_edges])
            )

        generator_edges = spec["edge_asset_kind"] == self.ASSET_GENERATOR
        if np.any(generator_edges):
            gen_ids = spec["edge_asset_ids"][generator_edges]
            active[generator_edges] = (
                cache["gen_bus"][gen_ids]
                == spec["edge_asset_bus_ids"][generator_edges]
            )

        load_edges = spec["edge_asset_kind"] == self.ASSET_LOAD
        if np.any(load_edges):
            load_ids = spec["edge_asset_ids"][load_edges]
            active[load_edges] = (
                cache["load_bus"][load_ids]
                == spec["edge_asset_bus_ids"][load_edges]
            )

        edge_features[~active] = 0.0
        return (
            np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0),
            active.astype(np.float32),
        )


def make_grid_graph_builder(
    graph_type: str,
    g2op_env,
    observation_domains: Dict[str, List[int]],
    **kwargs,
):
    """Create the requested graph representation without changing env callers."""
    graph_type = str(graph_type).lower()
    if graph_type == "bus":
        builder_cls = GridGraphBuilder
        kwargs.pop("generator_edge_direction", None)
        kwargs.pop("load_edge_direction", None)
        kwargs.pop("line_node_edge_direction", None)
        kwargs.pop("include_legacy_self_relation_feature", None)
    elif graph_type in {"heterogeneous", "hetero"}:
        builder_cls = HeterogeneousGridGraphBuilder
        kwargs.pop("line_node_edge_direction", None)
    elif graph_type in {
        "heterogeneous_line",
        "heterogeneous_line_nodes",
        "hetero_line",
    }:
        builder_cls = HeterogeneousLineGraphBuilder
    else:
        raise ValueError(
            f"Unsupported graph type '{graph_type}'. Use 'bus', "
            "'heterogeneous', or 'heterogeneous_line'."
        )
    return builder_cls(g2op_env, observation_domains, **kwargs)
