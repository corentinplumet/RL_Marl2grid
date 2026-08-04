from common.imports import *


class GraphFeatureProcessor:
    """Apply deterministic physical scaling and shared running normalization.

    Running statistics are updated from the full state graph once per
    environment observation. The same statistics are then used for every local
    actor graph and for the centralized state graph. Node statistics are kept
    separately per node type so that sparse heterogeneous feature blocks do not
    contaminate one another. Edge statistics only use active physical-line
    relations; masked candidates and structural relations are excluded.
    """

    STATS_PREFIX = "__graph_"
    NODE_STATS_TEMPLATE = "__graph_node_type_{}__"
    EDGE_STATS_KEY = "__graph_physical_edge__"

    # Binary indicators, masks, type one-hot features, and rho are intentionally
    # excluded. In particular, rho can be used directly as a GCN edge weight
    # and must retain its physical, non-negative meaning.
    RUNNING_FEATURES = {
        "gen_p",
        "load_p",
        "p",
        "gen_theta",
        "load_theta",
        "theta",
        "theta_diff",
        "time_before_cooldown_sub",
        "timestep_overflow",
        "time_before_cooldown_line",
        "time_next_maintenance",
        "duration_next_maintenance",
    }

    PHYSICAL_EDGE_RELATIONS = {
        "physical_line",
        "busbar_to_line_origin",
        "line_origin_to_busbar",
        "busbar_to_line_extremity",
        "line_extremity_to_busbar",
    }

    def __init__(
        self,
        g2op_env,
        graph_spec: Dict[str, Any],
        physical_scaling: bool = False,
        running_normalization: bool = False,
        power_scale_mw: float = 0.0,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        self.physical_scaling = bool(physical_scaling)
        self.running_normalization = bool(running_normalization)
        self.clip = float(clip)
        self.epsilon = float(epsilon)

        self.node_feature_names = list(graph_spec.get("node_feature_names", []))
        self.edge_feature_names = list(graph_spec.get("edge_feature_names", []))
        self.node_running_mask = np.asarray(
            [name in self.RUNNING_FEATURES for name in self.node_feature_names],
            dtype=bool,
        )
        self.edge_running_mask = np.asarray(
            [name in self.RUNNING_FEATURES for name in self.edge_feature_names],
            dtype=bool,
        )

        self.scale_factors = self._resolve_physical_scales(
            g2op_env,
            power_scale_mw=power_scale_mw,
        )
        self.node_scale = self._feature_scale_vector(self.node_feature_names)
        self.edge_scale = self._feature_scale_vector(self.edge_feature_names)

        n_node_types = int(graph_spec.get("n_node_types", 1))
        self.node_type_ids = tuple(range(max(1, n_node_types)))
        edge_type_names = graph_spec.get("edge_type_names", {}) or {}
        self.physical_edge_types = {
            int(edge_type)
            for name, edge_type in edge_type_names.items()
            if str(name) in self.PHYSICAL_EDGE_RELATIONS
        }
        self._stats: Dict[str, Dict[str, Any]] = {}

    def process(
        self,
        graphs: Dict[str, Dict[str, np.ndarray]],
        update: bool,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Scale and optionally standardize a state/local graph collection."""
        if self.physical_scaling:
            for graph in graphs.values():
                self._apply_physical_scaling(graph)

        if self.running_normalization:
            state_graph = graphs.get("state")
            if update and state_graph is not None:
                self._update_from_state_graph(state_graph)
            for graph in graphs.values():
                self._apply_running_normalization(graph)
        return graphs

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            key: {
                "count": float(value["count"]),
                "mean": None
                if value["mean"] is None
                else value["mean"].copy(),
                "var": None if value["var"] is None else value["var"].copy(),
            }
            for key, value in self._stats.items()
        }

    def set_stats(self, stats: Dict[str, Dict[str, Any]]) -> None:
        self._stats.clear()
        if not stats:
            return
        for key, value in stats.items():
            if not str(key).startswith(self.STATS_PREFIX):
                continue
            mean = value.get("mean")
            var = value.get("var")
            self._stats[str(key)] = {
                "count": float(value.get("count", 0.0)),
                "mean": None
                if mean is None
                else np.asarray(mean, dtype=np.float64).copy(),
                "var": None
                if var is None
                else np.asarray(var, dtype=np.float64).copy(),
            }

    def _resolve_physical_scales(
        self,
        g2op_env,
        power_scale_mw: float,
    ) -> Dict[str, float]:
        configured_power_scale = float(power_scale_mw)
        if configured_power_scale > 0.0:
            power_scale = configured_power_scale
        else:
            gen_pmax = np.asarray(
                getattr(g2op_env, "gen_pmax", []), dtype=np.float64
            ).reshape(-1)
            valid = np.abs(gen_pmax[np.isfinite(gen_pmax)])
            valid = valid[valid > self.epsilon]
            power_scale = float(valid.max()) if valid.size else 100.0

        parameters = getattr(g2op_env, "parameters", None)

        def parameter_scale(name: str, fallback: float) -> float:
            value = getattr(parameters, name, fallback) if parameters is not None else fallback
            try:
                value = abs(float(value))
            except (TypeError, ValueError):
                value = fallback
            return max(value, 1.0)

        episode_horizon = 1.0
        for owner in (g2op_env, getattr(g2op_env, "chronics_handler", None)):
            if owner is None:
                continue
            candidate = getattr(owner, "max_episode_duration", None)
            try:
                candidate = candidate() if callable(candidate) else candidate
                candidate = float(candidate)
            except (TypeError, ValueError):
                continue
            if np.isfinite(candidate) and candidate > 0.0:
                episode_horizon = candidate
                break

        return {
            "power_mw": max(power_scale, self.epsilon),
            "angle_degree": 180.0,
            "substation_cooldown": parameter_scale(
                "NB_TIMESTEP_COOLDOWN_SUB", 1.0
            ),
            "line_cooldown": parameter_scale(
                "NB_TIMESTEP_COOLDOWN_LINE", 1.0
            ),
            "overflow_duration": parameter_scale(
                "NB_TIMESTEP_OVERFLOW_ALLOWED", 1.0
            ),
            "maintenance_horizon": max(float(episode_horizon), 1.0),
        }

    def _feature_scale_vector(self, feature_names: List[str]) -> np.ndarray:
        feature_to_scale = {
            "gen_p": self.scale_factors["power_mw"],
            "load_p": self.scale_factors["power_mw"],
            "p": self.scale_factors["power_mw"],
            "gen_theta": self.scale_factors["angle_degree"],
            "load_theta": self.scale_factors["angle_degree"],
            "theta": self.scale_factors["angle_degree"],
            "theta_diff": self.scale_factors["angle_degree"],
            "time_before_cooldown_sub": self.scale_factors[
                "substation_cooldown"
            ],
            "timestep_overflow": self.scale_factors["overflow_duration"],
            "time_before_cooldown_line": self.scale_factors["line_cooldown"],
            "time_next_maintenance": self.scale_factors[
                "maintenance_horizon"
            ],
            "duration_next_maintenance": self.scale_factors[
                "maintenance_horizon"
            ],
        }
        return np.asarray(
            [max(float(feature_to_scale.get(name, 1.0)), self.epsilon) for name in feature_names],
            dtype=np.float32,
        )

    def _apply_physical_scaling(self, graph: Dict[str, np.ndarray]) -> None:
        node_features = np.asarray(graph["node_features"], dtype=np.float32).copy()
        edge_features = np.asarray(graph["edge_features"], dtype=np.float32).copy()
        if node_features.size:
            node_features /= self.node_scale
        if edge_features.size:
            edge_features /= self.edge_scale
        graph["node_features"] = np.nan_to_num(
            node_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        graph["edge_features"] = np.nan_to_num(
            edge_features, nan=0.0, posinf=0.0, neginf=0.0
        )

    def _update_from_state_graph(self, graph: Dict[str, np.ndarray]) -> None:
        node_features = np.asarray(graph["node_features"], dtype=np.float64)
        node_types = self._node_types(graph, len(node_features))
        node_mask = np.asarray(
            graph.get("node_mask", np.ones(len(node_features))), dtype=np.float32
        )
        for node_type in self.node_type_ids:
            rows = (node_types == node_type) & (node_mask > 0.0)
            if np.any(rows):
                self._update_stats(
                    self.NODE_STATS_TEMPLATE.format(node_type),
                    node_features[rows],
                )

        edge_features = np.asarray(graph["edge_features"], dtype=np.float64)
        edge_rows = self._active_physical_edge_rows(graph, len(edge_features))
        if np.any(edge_rows):
            self._update_stats(self.EDGE_STATS_KEY, edge_features[edge_rows])

    def _apply_running_normalization(self, graph: Dict[str, np.ndarray]) -> None:
        node_features = np.asarray(graph["node_features"], dtype=np.float32).copy()
        node_types = self._node_types(graph, len(node_features))
        node_mask = np.asarray(
            graph.get("node_mask", np.ones(len(node_features))), dtype=np.float32
        )
        for node_type in self.node_type_ids:
            rows = (node_types == node_type) & (node_mask > 0.0)
            self._standardize_rows(
                node_features,
                rows,
                self.node_running_mask,
                self._stats.get(self.NODE_STATS_TEMPLATE.format(node_type)),
            )

        edge_features = np.asarray(graph["edge_features"], dtype=np.float32).copy()
        edge_rows = self._active_physical_edge_rows(graph, len(edge_features))
        self._standardize_rows(
            edge_features,
            edge_rows,
            self.edge_running_mask,
            self._stats.get(self.EDGE_STATS_KEY),
        )
        graph["node_features"] = np.nan_to_num(
            node_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        graph["edge_features"] = np.nan_to_num(
            edge_features, nan=0.0, posinf=0.0, neginf=0.0
        )

    def _node_types(self, graph: Dict[str, np.ndarray], n_nodes: int) -> np.ndarray:
        node_types = graph.get("node_type")
        if node_types is None:
            return np.zeros(n_nodes, dtype=np.int64)
        return np.asarray(node_types, dtype=np.int64)

    def _active_physical_edge_rows(
        self,
        graph: Dict[str, np.ndarray],
        n_edges: int,
    ) -> np.ndarray:
        active = np.asarray(
            graph.get("edge_mask", np.ones(n_edges)), dtype=np.float32
        ) > 0.0
        edge_types = graph.get("edge_type")
        if edge_types is None or not self.physical_edge_types:
            return active
        return active & np.isin(
            np.asarray(edge_types, dtype=np.int64),
            np.asarray(sorted(self.physical_edge_types), dtype=np.int64),
        )

    def _update_stats(self, key: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            return
        batch_count = float(values.shape[0])
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_var = np.sum(centered * centered, axis=0)

        current = self._stats.get(key)
        if current is None or current["mean"] is None or current["count"] <= 0.0:
            self._stats[key] = {
                "count": batch_count,
                "mean": batch_mean,
                "var": batch_var,
            }
            return

        old_count = float(current["count"])
        new_count = old_count + batch_count
        delta = batch_mean - current["mean"]
        current["mean"] = current["mean"] + delta * (batch_count / new_count)
        current["var"] = (
            current["var"]
            + batch_var
            + delta * delta * (old_count * batch_count / new_count)
        )
        current["count"] = new_count

    def _standardize_rows(
        self,
        features: np.ndarray,
        row_mask: np.ndarray,
        feature_mask: np.ndarray,
        stats: Optional[Dict[str, Any]],
    ) -> None:
        if (
            stats is None
            or stats.get("mean") is None
            or stats.get("var") is None
            or float(stats.get("count", 0.0)) <= 0.0
            or not np.any(row_mask)
            or not np.any(feature_mask)
        ):
            return
        mean = np.asarray(stats["mean"], dtype=np.float64)
        variance = np.asarray(stats["var"], dtype=np.float64) / float(
            stats["count"]
        )
        std = np.sqrt(np.maximum(variance, self.epsilon))
        selected = features[np.ix_(row_mask, feature_mask)].astype(np.float64)
        normalized = (selected - mean[feature_mask]) / std[feature_mask]
        if self.clip > 0.0:
            normalized = np.clip(normalized, -self.clip, self.clip)
        features[np.ix_(row_mask, feature_mask)] = normalized.astype(np.float32)
