from common.imports import *


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

    def __init__(
        self,
        g2op_env,
        observation_domains: Dict[str, List[int]],
        include_neighbors: bool = True,
        include_maintenance: bool = False,
    ) -> None:
        self.g2op_env = g2op_env
        self.observation_domains = {
            agent: np.asarray(nodes, dtype=np.int64)
            for agent, nodes in observation_domains.items()
        }
        self.include_neighbors = include_neighbors
        self.node_features = list(self.BASE_NODE_FEATURES)
        self.edge_features = list(self.BASE_EDGE_FEATURES)
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
        state_graph = self.build_for_spec(obs, self.specs["state"])
        graphs = {"state": state_graph}
        for agent, spec in self.specs.items():
            if agent == "state":
                continue
            graphs[agent] = self.build_for_spec(obs, spec)
        return graphs

    def build_for_spec(self, obs, spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return self._build_bus_for_spec(obs, spec)

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

        edge_index = np.asarray(directed_edges, dtype=np.int64).T
        if edge_index.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return {
            "node_ids": node_ids,
            "line_ids": line_ids,
            "controlled_nodes": controlled_nodes,
            "edge_index": edge_index,
            "edge_line_ids": np.asarray(edge_line_ids, dtype=np.int64),
            "edge_or_bus_ids": np.asarray(edge_or_bus_ids, dtype=np.int64),
            "edge_ex_bus_ids": np.asarray(edge_ex_bus_ids, dtype=np.int64),
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
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

    def _build_bus_for_spec(self, obs, spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
        node_features = self._bus_node_features(obs, spec["controlled_nodes"])[spec["node_ids"]]
        edge_features, edge_mask = self._bus_edge_features(obs, spec)
        return {
            "node_features": node_features.astype(np.float32, copy=False),
            "edge_features": edge_features.astype(np.float32, copy=False),
            "node_mask": np.ones((len(spec["node_ids"]),), dtype=np.float32),
            "edge_mask": edge_mask.astype(np.float32, copy=False),
        }

    def _bus_node_features(self, obs, controlled_nodes):
        features = np.zeros((self.n_bus_nodes, self.node_dim), dtype=np.float32)
        col = {name: idx for idx, name in enumerate(self.node_features)}

        self._put_bus_asset_feature(features, col["gen_p"], obs, self.gen_to_sub, self.gen_pos, self._obs_array(obs, "gen_p"), mode="sum")
        self._put_bus_asset_feature(features, col["gen_theta"], obs, self.gen_to_sub, self.gen_pos, self._obs_array(obs, "gen_theta"), mode="mean")

        self._put_bus_asset_feature(features, col["load_p"], obs, self.load_to_sub, self.load_pos, self._obs_array(obs, "load_p"), mode="sum")
        self._put_bus_asset_feature(features, col["load_theta"], obs, self.load_to_sub, self.load_pos, self._obs_array(obs, "load_theta"), mode="mean")

        sub_cooldown = self._obs_array(obs, "time_before_cooldown_sub", expected=self.n_sub)
        if sub_cooldown is not None:
            for sub_id in range(self.n_sub):
                features[self._bus_node_ids([sub_id]), col["time_before_cooldown_sub"]] = sub_cooldown[sub_id]

        controlled_mask = np.isin(np.arange(self.n_sub), controlled_nodes).astype(np.float32)
        for sub_id, is_controlled in enumerate(controlled_mask):
            features[self._bus_node_ids([sub_id]), col["domain_mask"]] = is_controlled

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _bus_edge_features(self, obs, spec):
        edge_line_ids = spec["edge_line_ids"]
        if len(edge_line_ids) == 0:
            return (
                np.zeros((0, self.edge_dim), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        edge_features = self._line_feature_matrix(obs)[edge_line_ids]
        line_status = self._obs_array(obs, "line_status", expected=self.n_line)
        line_status = np.ones(self.n_line, dtype=np.float32) if line_status is None else line_status
        or_bus = self._topo_bus_ids(obs, self.line_or_pos)[edge_line_ids] - 1
        ex_bus = self._topo_bus_ids(obs, self.line_ex_pos)[edge_line_ids] - 1
        active = (
            (line_status[edge_line_ids] > 0)
            & (or_bus == spec["edge_or_bus_ids"])
            & (ex_bus == spec["edge_ex_bus_ids"])
        )
        edge_features = edge_features.copy()
        edge_features[~active] = 0.0
        return np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0), active.astype(np.float32)

    def _line_feature_matrix(self, obs):
        features = np.zeros((self.n_line, self.edge_dim), dtype=np.float32)
        for idx, name in enumerate(self.edge_features):
            values = self._obs_array(obs, name, expected=self.n_line)
            if values is not None:
                features[:, idx] = values
        return features

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
