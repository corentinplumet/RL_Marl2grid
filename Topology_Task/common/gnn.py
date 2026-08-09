from common.imports import *
from common.utils import get_flat_obs

try:
    from torch_geometric.nn import GATConv, GCNConv, GINEConv, SAGEConv, global_add_pool, global_max_pool
    from torch_geometric.utils import softmax as pyg_softmax
except ModuleNotFoundError:
    GATConv = GCNConv = GINEConv = SAGEConv = global_add_pool = global_max_pool = None
    pyg_softmax = None


SUMMARY_EDGE_DIRECTIONS = {"bidirectional", "toward_summary"}
VIRTUAL_EDGE_DIRECTIONS = {"inherit", "bidirectional", "toward_virtual"}


def _validate_summary_edge_direction(value: str) -> str:
    direction = str(value).strip().lower()
    if direction not in SUMMARY_EDGE_DIRECTIONS:
        choices = ", ".join(sorted(SUMMARY_EDGE_DIRECTIONS))
        raise ValueError(
            f"Unsupported summary edge direction '{value}'. Use one of: {choices}."
        )
    return direction


def _resolve_virtual_edge_direction(value: str, summary_direction: str) -> str:
    direction = str(value).strip().lower()
    if direction not in VIRTUAL_EDGE_DIRECTIONS:
        choices = ", ".join(sorted(VIRTUAL_EDGE_DIRECTIONS))
        raise ValueError(
            f"Unsupported virtual edge direction '{value}'. Use one of: {choices}."
        )
    if direction == "inherit":
        return (
            "bidirectional"
            if summary_direction == "bidirectional"
            else "toward_virtual"
        )
    return direction


# Readout names live in a dependency-free module so launchers can validate
# a config without importing torch. Re-exported here for existing callers.
from common.readouts import (  # noqa: E402
    POOLING_AGGREGATIONS,
    SPARSE_TRANSFORMER_ONLY_AGGREGATIONS,
)


class _VirtualNodeEncoderBase(nn.Module):
    """Shared runtime construction for virtual-node graph readout."""

    def _init_hierarchy_nodes(
        self,
        graph_spec: Dict[str, Any],
        feature_dim: int,
        use_virtual_node: bool,
        add_substation_nodes: bool,
        summary_edge_direction: str,
        virtual_edge_direction: str,
    ) -> None:
        self.use_virtual_node = bool(use_virtual_node)
        self.add_substation_nodes = bool(add_substation_nodes)
        # Summary and virtual nodes relate every busbar they gather. Outside the
        # controlled domain that relation is not one the agent can act on, and
        # it would let two contextual busbars exchange through the summary.
        self.structural_relations_controlled_only = bool(
            graph_spec.get("structural_relations_controlled_only", False)
        )
        self.summary_edge_direction = _validate_summary_edge_direction(
            summary_edge_direction
        )
        self.virtual_edge_direction = _resolve_virtual_edge_direction(
            virtual_edge_direction,
            self.summary_edge_direction,
        )
        node_ids = np.asarray(graph_spec["node_ids"], dtype=np.int64)
        busbar_mask = (node_ids % self.node_id_stride) < self.n_busbar
        # Which substations may host a summary node. This is a property of the
        # partition, fixed for the episode, so it is resolved once here rather
        # than read from the per-step observation.
        controlled = np.asarray(
            graph_spec.get(
                "controlled_node_mask", np.ones(len(node_ids), dtype=np.float32)
            ),
            dtype=np.float32,
        ) > 0
        self.register_buffer(
            "controlled_substation_ids",
            th.tensor(
                np.unique(node_ids[busbar_mask & controlled] // self.node_id_stride),
                dtype=th.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "virtual_busbar_mask",
            th.tensor(busbar_mask, dtype=th.bool),
            persistent=False,
        )

        edge_feature_names = list(graph_spec.get("edge_feature_names", []))
        if self.add_substation_nodes:
            # A summary node is initialised from what its substation is made of,
            # not from which substation it is. An index embedding would be
            # shaped by the number of substations and could not cross grids;
            # this projection is shaped by the descriptor, so it can. Its bias
            # is the vector every substation shares, and its weight is the
            # structural modulation on top of it.
            substation_features = np.asarray(
                graph_spec.get(
                    "substation_features",
                    np.zeros((max(1, int(graph_spec.get("n_sub", 1))), 1)),
                ),
                dtype=np.float32,
            )
            self.register_buffer(
                "substation_features",
                th.tensor(substation_features, dtype=th.float32),
                persistent=False,
            )
            self.substation_node_encoder = nn.Linear(
                substation_features.shape[1], int(feature_dim)
            )
            nn.init.normal_(
                self.substation_node_encoder.weight, mean=0.0, std=0.02
            )
            nn.init.normal_(self.substation_node_encoder.bias, mean=0.0, std=0.02)
        else:
            self.substation_node_encoder = None
        if self.use_virtual_node:
            self.virtual_node_embedding = nn.Parameter(
                th.empty(1, int(feature_dim))
            )
            nn.init.normal_(self.virtual_node_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter("virtual_node_embedding", None)

    def _append_hierarchy_nodes(
        self,
        x: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
        batch: th.Tensor,
        flat_node_ids: th.Tensor,
        node_mask: Optional[th.Tensor] = None,
        edge_type: Optional[th.Tensor] = None,
        substation_edge_type: Optional[int] = None,
        virtual_edge_type: Optional[int] = None,
        controlled_node_mask: Optional[th.Tensor] = None,
    ):
        """Append optional substation and graph-level virtual nodes."""
        if not self.use_virtual_node and not self.add_substation_nodes:
            return (
                x,
                edge_index,
                edge_attr,
                batch,
                node_mask,
                edge_type,
                controlled_node_mask,
                None,
            )

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        n_real_nodes = x.shape[0]
        if flat_node_ids.numel() != n_real_nodes:
            raise ValueError(
                "Hierarchy nodes require one static node id per physical node."
            )
        graph_ids = th.arange(batch_size, device=x.device)
        busbar_ids = th.nonzero(
            th.remainder(flat_node_ids, self.node_id_stride) < self.n_busbar,
            as_tuple=False,
        ).flatten()
        if self.structural_relations_controlled_only:
            busbar_substations = th.div(
                flat_node_ids[busbar_ids],
                self.node_id_stride,
                rounding_mode="floor",
            ).long()
            busbar_ids = busbar_ids[
                th.isin(busbar_substations, self.controlled_substation_ids)
            ]
        active_busbar_ids = busbar_ids
        if node_mask is not None:
            active_busbar_ids = busbar_ids[node_mask[busbar_ids] > 0]

        def append_structural_edges(
            source_nodes: th.Tensor,
            summary_nodes: th.Tensor,
            relation_type: Optional[int],
            direction: str,
        ) -> None:
            nonlocal edge_index, edge_attr, edge_type
            toward_summary = th.stack([source_nodes, summary_nodes], dim=0)
            if direction == "bidirectional":
                structural_edges = th.cat(
                    [
                        toward_summary,
                        th.stack([summary_nodes, source_nodes], dim=0),
                    ],
                    dim=1,
                )
            else:
                structural_edges = toward_summary
            edge_index = th.cat([edge_index, structural_edges], dim=1)
            structural_edge_attr = th.zeros(
                (structural_edges.shape[1], self.edge_dim),
                dtype=edge_attr.dtype,
                device=edge_attr.device,
            )
            edge_attr = th.cat([edge_attr, structural_edge_attr], dim=0)
            if edge_type is not None:
                if relation_type is None:
                    raise ValueError(
                        "A relation type is required for hierarchy-node edges."
                    )
                edge_type = th.cat(
                    [
                        edge_type,
                        th.full(
                            (structural_edges.shape[1],),
                            int(relation_type),
                            dtype=edge_type.dtype,
                            device=edge_type.device,
                        ),
                    ],
                    dim=0,
                )

        substation_indices = None
        if self.add_substation_nodes:
            first_graph_busbars = busbar_ids[batch[busbar_ids] == 0]
            included_substation_ids = th.unique(
                th.div(
                    flat_node_ids[first_graph_busbars],
                    self.node_id_stride,
                    rounding_mode="floor",
                ).long(),
                sorted=True,
            )
            n_included_substations = int(included_substation_ids.numel())
            if n_included_substations < 1:
                raise ValueError("No busbars are available for substation nodes.")

            substation_graph_ids = graph_ids.repeat_interleave(
                n_included_substations
            )
            repeated_substation_ids = included_substation_ids.repeat(batch_size)
            substation_indices = n_real_nodes + th.arange(
                batch_size * n_included_substations,
                device=x.device,
            )
            substation_x = self.substation_node_encoder(
                self.substation_features[repeated_substation_ids]
            ).to(dtype=x.dtype)
            x = th.cat([x, substation_x], dim=0)
            batch = th.cat([batch, substation_graph_ids], dim=0)

            busbar_substation_ids = th.div(
                flat_node_ids[active_busbar_ids],
                self.node_id_stride,
                rounding_mode="floor",
            ).long()
            local_substation_ids = th.searchsorted(
                included_substation_ids, busbar_substation_ids
            )
            substation_for_busbar = (
                n_real_nodes
                + batch[active_busbar_ids] * n_included_substations
                + local_substation_ids
            )
            append_structural_edges(
                active_busbar_ids,
                substation_for_busbar,
                substation_edge_type,
                self.summary_edge_direction,
            )

            if node_mask is not None:
                node_mask = th.cat(
                    [
                        node_mask,
                        th.ones(
                            len(substation_indices),
                            dtype=node_mask.dtype,
                            device=x.device,
                        ),
                    ]
                )
            if controlled_node_mask is not None:
                all_busbar_substation_ids = th.div(
                    flat_node_ids[busbar_ids],
                    self.node_id_stride,
                    rounding_mode="floor",
                ).long()
                all_local_substation_ids = th.searchsorted(
                    included_substation_ids, all_busbar_substation_ids
                )
                substation_control_index = (
                    batch[busbar_ids] * n_included_substations
                    + all_local_substation_ids
                )
                substation_controlled = th.zeros(
                    batch_size * n_included_substations,
                    dtype=controlled_node_mask.dtype,
                    device=x.device,
                )
                substation_controlled.index_add_(
                    0,
                    substation_control_index,
                    controlled_node_mask[busbar_ids],
                )
                substation_controlled.clamp_(0, 1)
                controlled_node_mask = th.cat(
                    [controlled_node_mask, substation_controlled], dim=0
                )

        virtual_indices = None
        if self.use_virtual_node:
            summary_nodes = (
                substation_indices
                if substation_indices is not None
                else active_busbar_ids
            )
            n_nodes_before_virtual = x.shape[0]
            virtual_indices = n_nodes_before_virtual + graph_ids
            virtual_for_edge = virtual_indices[batch[summary_nodes].long()]
            append_structural_edges(
                summary_nodes,
                virtual_for_edge,
                virtual_edge_type,
                self.virtual_edge_direction,
            )

            virtual_x = self.virtual_node_embedding.to(dtype=x.dtype).expand(
                batch_size, -1
            )
            x = th.cat([x, virtual_x], dim=0)
            batch = th.cat([batch, graph_ids], dim=0)
            if node_mask is not None:
                node_mask = th.cat(
                    [
                        node_mask,
                        th.ones(
                            batch_size,
                            dtype=node_mask.dtype,
                            device=x.device,
                        ),
                    ]
                )
            if controlled_node_mask is not None:
                controlled_node_mask = th.cat(
                    [
                        controlled_node_mask,
                        th.ones(
                            batch_size,
                            dtype=controlled_node_mask.dtype,
                            device=x.device,
                        ),
                    ]
                )
        return (
            x,
            edge_index,
            edge_attr,
            batch,
            node_mask,
            edge_type,
            controlled_node_mask,
            virtual_indices,
        )


class GraphEncoder(_VirtualNodeEncoderBase):
    """Thin PyTorch Geometric encoder for fixed Grid2Op graph observations."""

    def __init__(
        self,
        graph_spec: Dict[str, Any],
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        conv_type: str = "gat",
        graphsage_aggr: str = "mean",
        readout_aggr: str = "mean",
        layer_norm: bool = True,
        heads: int = 1,
        node_pre_encoder: bool = False,
        edge_pre_encoder: bool = False,
        node_id_embeddings: bool = False,
        node_id_emb_dim: int = 8,
        gcn_edge_weight_feature: str = "none",
        add_substation_nodes: bool = False,
        summary_edge_direction: str = "bidirectional",
        virtual_edge_direction: str = "inherit",
    ) -> None:
        super().__init__()
        if GCNConv is None:
            raise ImportError(
                "GNN encoders now use PyTorch Geometric. Install torch-geometric "
                "before running with an encoder set to 'gnn'."
            )
        if n_layers < 1:
            raise ValueError("A GNN encoder needs at least one layer.")

        self.conv_type = conv_type.lower()
        self.node_out_dim = int(hidden_dim)
        self.uses_edge_attr = self.conv_type in {"gat", "gine"}
        self.readout_aggr = readout_aggr.lower()
        if self.readout_aggr not in POOLING_AGGREGATIONS:
            raise ValueError(f"Unsupported GNN readout aggregation: {readout_aggr}")
        self.edge_dim = int(graph_spec["edge_dim"])
        self.edge_feature_names = list(graph_spec.get("edge_feature_names", []))
        self.gcn_edge_weight_feature = str(gcn_edge_weight_feature).lower()
        self.gcn_edge_weight_idx = self._resolve_gcn_edge_weight_idx(graph_spec)
        edge_names = list(graph_spec.get("edge_feature_names", []))
        self.physical_edge_indicator_idx = (
            edge_names.index("relation_physical_line")
            if "relation_physical_line" in edge_names
            else (
                edge_names.index("line_status")
                if "line_status" in edge_names
                else None
            )
        )
        self.register_buffer("edge_index", th.tensor(graph_spec["edge_index"], dtype=th.long))
        self.register_buffer("node_ids", th.tensor(graph_spec["node_ids"], dtype=th.long))

        node_dim = int(graph_spec["node_dim"])
        self.node_id_embeddings = bool(node_id_embeddings)
        self.n_busbar = int(graph_spec.get("n_busbar", 2))
        self.node_id_stride = int(
            graph_spec.get("node_id_stride", self.n_busbar)
        )
        self.n_bus_id_embeddings = int(
            graph_spec.get("n_bus_id_embeddings", self.n_busbar)
        )
        if self.node_id_embeddings:
            if node_id_emb_dim < 1:
                raise ValueError("node_id_emb_dim must be at least 1.")
            n_sub = int(
                graph_spec.get(
                    "n_sub",
                    int(self.node_ids.max().item() // self.node_id_stride) + 1,
                )
            )
            # Same reasoning as the summary node: an index embedding is shaped
            # by the number of substations and cannot cross grids. Project the
            # structural descriptor instead, so the learned tensor is shaped by
            # the descriptor and a checkpoint stays loadable on any network.
            substation_features = np.asarray(
                graph_spec.get(
                    "substation_features", np.zeros((max(1, n_sub), 1))
                ),
                dtype=np.float32,
            )
            self.register_buffer(
                "node_id_substation_features",
                th.tensor(substation_features, dtype=th.float32),
                persistent=False,
            )
            self.sub_id_embedding = nn.Linear(
                substation_features.shape[1], node_id_emb_dim
            )
            self.bus_id_embedding = nn.Embedding(
                self.n_bus_id_embeddings, node_id_emb_dim
            )
            node_dim = node_dim + 2 * node_id_emb_dim

        if node_pre_encoder:
            node_encoder_layers = [nn.Linear(node_dim, hidden_dim), nn.ReLU()]
            if layer_norm:
                node_encoder_layers.append(nn.LayerNorm(hidden_dim))
            self.node_pre_encoder = nn.Sequential(*node_encoder_layers)
            conv_input_dim = hidden_dim
        else:
            self.node_pre_encoder = nn.Identity()
            conv_input_dim = node_dim

        self._init_hierarchy_nodes(
            graph_spec,
            feature_dim=conv_input_dim,
            use_virtual_node=self.readout_aggr == "virtual_node",
            add_substation_nodes=add_substation_nodes,
            summary_edge_direction=summary_edge_direction,
            virtual_edge_direction=virtual_edge_direction,
        )

        if edge_pre_encoder and self.uses_edge_attr:
            edge_encoder_layers = [nn.Linear(self.edge_dim, hidden_dim), nn.ReLU()]
            if layer_norm:
                edge_encoder_layers.append(nn.LayerNorm(hidden_dim))
            self.edge_pre_encoder = nn.Sequential(*edge_encoder_layers)
            conv_edge_dim = hidden_dim
        else:
            self.edge_pre_encoder = nn.Identity()
            conv_edge_dim = self.edge_dim

        dims = [conv_input_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList(
            [
                self._make_conv(
                    in_dim=dims[idx],
                    hidden_dim=hidden_dim,
                    edge_dim=conv_edge_dim,
                    conv_type=self.conv_type,
                    graphsage_aggr=graphsage_aggr,
                    heads=heads,
                )
                for idx in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity() for _ in range(n_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        graph_obs: Dict[str, th.Tensor],
        edge_index: Optional[th.Tensor] = None,
        node_ids: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        embedding, _ = self.forward_with_nodes(
            graph_obs,
            edge_index=edge_index,
            node_ids=node_ids,
        )
        return embedding

    def forward_with_nodes(
        self,
        graph_obs: Dict[str, th.Tensor],
        edge_index: Optional[th.Tensor] = None,
        node_ids: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Return the graph readout and physical post-message-passing nodes."""
        unbatched = graph_obs["node_features"].dim() == 2
        n_nodes = int(graph_obs["node_features"].shape[-2])
        (
            x,
            edge_index,
            edge_attr,
            batch,
            node_mask,
            flat_node_ids,
            controlled_node_mask,
            energized_node_mask,
        ) = self._to_pyg_batch(graph_obs, edge_index=edge_index, node_ids=node_ids)
        x = self._append_node_id_embeddings(x, flat_node_ids)
        x = self.node_pre_encoder(x)
        n_physical_nodes = int(x.shape[0])
        (
            x,
            edge_index,
            edge_attr,
            batch,
            node_mask,
            _,
            controlled_node_mask,
            virtual_indices,
        ) = self._append_hierarchy_nodes(
            x,
            edge_index,
            edge_attr,
            batch,
            flat_node_ids,
            node_mask=node_mask,
            controlled_node_mask=controlled_node_mask,
        )
        if (
            energized_node_mask is not None
            and energized_node_mask.numel() < x.shape[0]
        ):
            energized_node_mask = th.cat(
                [
                    energized_node_mask,
                    th.zeros(
                        x.shape[0] - energized_node_mask.numel(),
                        dtype=energized_node_mask.dtype,
                        device=energized_node_mask.device,
                    ),
                ]
            )
        edge_attr = self.edge_pre_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            if self.uses_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            elif self.conv_type == "gcn" and self.gcn_edge_weight_idx is not None:
                x = conv(
                    x,
                    edge_index,
                    edge_weight=self._gcn_edge_weight(edge_attr),
                )
            else:
                x = conv(x, edge_index)
            x = norm(F.relu(x))

        batch_size = max(1, n_physical_nodes // n_nodes)
        node_embeddings = x[:n_physical_nodes].reshape(
            batch_size, n_nodes, self.node_out_dim
        )

        pooled = (
            x[virtual_indices]
            if virtual_indices is not None
            else self._pool_nodes(
                x,
                batch,
                node_mask,
                controlled_node_mask,
                energized_node_mask,
            )
        )
        embedding = self.readout(pooled)
        if unbatched:
            return embedding.squeeze(0), node_embeddings.squeeze(0)
        return embedding, node_embeddings

    def _pool_nodes(
        self,
        x: th.Tensor,
        batch: th.Tensor,
        node_mask: Optional[th.Tensor],
        controlled_node_mask: Optional[th.Tensor] = None,
        energized_node_mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Aggregate node states into one vector per graph.

        A ``controlled_*`` readout restricts the aggregation to the nodes the
        agent can act on. Contextual neighbours still take part in message
        passing, so their state reaches the readout through the controlled
        nodes, but they do not enter the average themselves. This keeps the
        size of the pooled set fixed for an agent, whereas pooling every valid
        node makes it move with the topology.
        """
        controlled_only = self.readout_aggr.startswith("controlled_")
        readout = self.readout_aggr.removeprefix("controlled_")
        energized_only = readout.startswith("energized_")
        base_aggr = readout.removeprefix("energized_")

        active_mask = th.ones((x.shape[0],), dtype=x.dtype, device=x.device)
        if node_mask is not None:
            active_mask = active_mask * node_mask.to(dtype=x.dtype)
        if controlled_only:
            if controlled_node_mask is None:
                raise ValueError(
                    f"readout_aggr={self.readout_aggr!r} needs a "
                    "controlled_node_mask in the graph observation."
                )
            active_mask = active_mask * controlled_node_mask.to(dtype=x.dtype)
        if energized_only:
            if energized_node_mask is None:
                raise ValueError(
                    f"readout_aggr={self.readout_aggr!r} needs an "
                    "energized_node_mask in the graph observation."
                )
            active_mask = active_mask * energized_node_mask.to(dtype=x.dtype)

        if base_aggr == "max":
            masked_x = x.masked_fill(active_mask.unsqueeze(-1) <= 0, -th.inf)
            pooled = global_max_pool(masked_x, batch)
            return th.nan_to_num(pooled, nan=0.0, neginf=0.0, posinf=0.0)

        pooled = global_add_pool(x * active_mask.unsqueeze(-1), batch)
        if base_aggr == "mean":
            denom = global_add_pool(
                active_mask.unsqueeze(-1), batch
            ).clamp_min(1.0)
            pooled = pooled / denom
        return pooled

    def _to_pyg_batch(
        self,
        graph_obs: Dict[str, th.Tensor],
        edge_index: Optional[th.Tensor] = None,
        node_ids: Optional[th.Tensor] = None,
    ):
        nodes = graph_obs["node_features"]
        edge_features = graph_obs["edge_features"]
        node_mask = graph_obs.get("node_mask")
        edge_mask = graph_obs.get("edge_mask")
        controlled_node_mask = graph_obs.get("controlled_node_mask")
        energized_node_mask = graph_obs.get("energized_node_mask")

        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            node_mask = None if node_mask is None else node_mask.unsqueeze(0)
            edge_mask = None if edge_mask is None else edge_mask.unsqueeze(0)
            controlled_node_mask = (
                None
                if controlled_node_mask is None
                else controlled_node_mask.unsqueeze(0)
            )
            energized_node_mask = (
                None
                if energized_node_mask is None
                else energized_node_mask.unsqueeze(0)
            )

        base_edge_index = self.edge_index if edge_index is None else edge_index
        base_node_ids = self.node_ids if node_ids is None else node_ids
        batch_size, n_nodes, node_dim = nodes.shape
        n_edges = base_edge_index.shape[1]

        x = nodes.reshape(batch_size * n_nodes, node_dim)
        batch = th.arange(batch_size, device=nodes.device).repeat_interleave(n_nodes)
        flat_node_ids = base_node_ids.to(nodes.device).repeat(batch_size)

        base_edge_index = base_edge_index.to(nodes.device)
        edge_index = base_edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
        offsets = (th.arange(batch_size, device=nodes.device) * n_nodes).view(batch_size, 1, 1)
        edge_index = (edge_index + offsets).permute(1, 0, 2).reshape(2, batch_size * n_edges)

        edge_attr = edge_features.reshape(batch_size * n_edges, self.edge_dim)
        if edge_mask is not None:
            keep_edges = edge_mask.reshape(batch_size * n_edges).bool()
            edge_index = edge_index[:, keep_edges]
            edge_attr = edge_attr[keep_edges]

        flat_node_mask = None if node_mask is None else node_mask.reshape(batch_size * n_nodes)
        flat_controlled_mask = (
            None
            if controlled_node_mask is None
            else controlled_node_mask.reshape(batch_size * n_nodes)
        )
        flat_energized_mask = (
            None
            if energized_node_mask is None
            else energized_node_mask.reshape(batch_size * n_nodes)
        )
        return (
            x,
            edge_index,
            edge_attr,
            batch,
            flat_node_mask,
            flat_node_ids,
            flat_controlled_mask,
            flat_energized_mask,
        )

    def _append_node_id_embeddings(
        self,
        x: th.Tensor,
        node_ids: th.Tensor,
    ) -> th.Tensor:
        if not self.node_id_embeddings:
            return x
        sub_ids = th.div(
            node_ids, self.node_id_stride, rounding_mode="floor"
        ).long()
        bus_ids = th.remainder(node_ids, self.node_id_stride).long()
        return th.cat(
            [
                x,
                self.sub_id_embedding(
                    self.node_id_substation_features[sub_ids]
                ),
                self.bus_id_embedding(bus_ids),
            ],
            dim=-1,
        )

    def _make_conv(
        self,
        in_dim: int,
        hidden_dim: int,
        edge_dim: int,
        conv_type: str,
        graphsage_aggr: str,
        heads: int,
    ) -> nn.Module:
        if conv_type == "gcn":
            return GCNConv(in_dim, hidden_dim)
        if conv_type == "gat":
            return GATConv(
                in_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                edge_dim=edge_dim,
            )
        if conv_type == "gine":
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            return GINEConv(mlp, edge_dim=edge_dim)
        if conv_type == "graphsage":
            sage_aggr = "add" if graphsage_aggr == "sum" else graphsage_aggr
            return SAGEConv(in_dim, hidden_dim, aggr=sage_aggr)
        raise ValueError(f"Unsupported GNN type: {conv_type}")

    def _gcn_edge_weight(self, edge_attr: th.Tensor) -> th.Tensor:
        """Loading as a message weight, with unit weight on non-physical edges.

        Zero would delete a message, so a structural or attachment relation
        cannot simply take the loading of a line it does not represent. The
        substitution is made here rather than written into the edge attribute,
        which keeps the attribute an honest measurement for the encoders that
        read it as one.
        """
        weight = edge_attr[:, self.gcn_edge_weight_idx]
        if self.physical_edge_indicator_idx is None:
            return weight
        physical = edge_attr[:, self.physical_edge_indicator_idx] > 0
        return th.where(physical, weight, th.ones_like(weight))

    def _resolve_gcn_edge_weight_idx(self, graph_spec: Dict[str, Any]) -> Optional[int]:
        if self.gcn_edge_weight_feature in {"", "none", "false"}:
            return None
        if self.conv_type != "gcn":
            raise ValueError("--gcn-edge-weight-feature can only be used with --gnn-type gcn.")

        edge_feature_names = list(graph_spec.get("edge_feature_names", []))
        if not edge_feature_names:
            if self.gcn_edge_weight_feature == "rho" and self.edge_dim >= 2:
                return 1
            raise ValueError(
                "GCN edge weighting requires graph specs with edge_feature_names."
            )
        if self.gcn_edge_weight_feature not in edge_feature_names:
            raise ValueError(
                f"Unknown GCN edge weight feature '{self.gcn_edge_weight_feature}'. "
                f"Available edge features: {edge_feature_names}"
            )
        return edge_feature_names.index(self.gcn_edge_weight_feature)


class SparseGraphTransformerLayer(nn.Module):
    """Relation-aware sparse transformer block over a fixed graph edge list."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        edge_dim: int,
        n_edge_types: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        ffn_multiplier: int = 4,
        use_edge_attr: bool = True,
        use_edge_type_embeddings: bool = True,
        relation_bias: bool = True,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("sparse graph transformer needs at least one head.")
        if hidden_dim % heads != 0:
            raise ValueError(
                "gnn_hidden_dim must be divisible by gnn_heads for "
                "gnn_type=sparse_transformer."
            )

        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.n_edge_types = int(n_edge_types)
        self.use_edge_attr = bool(use_edge_attr and edge_dim > 0)
        self.use_edge_type_embeddings = bool(use_edge_type_embeddings)
        self.use_relation_bias = bool(relation_bias)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        if self.use_edge_attr:
            self.edge_k_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
            self.edge_v_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        else:
            self.edge_k_proj = None
            self.edge_v_proj = None

        if self.use_edge_type_embeddings:
            self.edge_type_k = nn.Embedding(n_edge_types, hidden_dim)
            self.edge_type_v = nn.Embedding(n_edge_types, hidden_dim)
        else:
            self.edge_type_k = None
            self.edge_type_v = None

        if self.use_relation_bias:
            self.relation_bias = nn.Embedding(n_edge_types, heads)
        else:
            self.relation_bias = None

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        ffn_hidden_dim = max(hidden_dim, int(ffn_multiplier) * hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
        edge_type: th.Tensor,
    ) -> th.Tensor:
        if edge_index.numel() == 0:
            attn_out = th.zeros_like(x)
        else:
            src, dst = edge_index[0].long(), edge_index[1].long()
            edge_type = edge_type.long().clamp(0, self.n_edge_types - 1)

            q = self.q_proj(x).view(-1, self.heads, self.head_dim)
            k = self.k_proj(x).view(-1, self.heads, self.head_dim)
            v = self.v_proj(x).view(-1, self.heads, self.head_dim)

            edge_k = k[src]
            edge_v = v[src]

            if self.edge_k_proj is not None and edge_attr.numel() > 0:
                edge_k = edge_k + self.edge_k_proj(edge_attr).view(
                    -1, self.heads, self.head_dim
                )
                edge_v = edge_v + self.edge_v_proj(edge_attr).view(
                    -1, self.heads, self.head_dim
                )

            if self.edge_type_k is not None:
                edge_k = edge_k + self.edge_type_k(edge_type).view(
                    -1, self.heads, self.head_dim
                )
                edge_v = edge_v + self.edge_type_v(edge_type).view(
                    -1, self.heads, self.head_dim
                )

            scores = (q[dst] * edge_k).sum(dim=-1) / np.sqrt(self.head_dim)
            if self.relation_bias is not None:
                scores = scores + self.relation_bias(edge_type)

            alpha = pyg_softmax(scores, dst, num_nodes=x.shape[0])
            alpha = self.attention_dropout(alpha)
            messages = alpha.unsqueeze(-1) * edge_v
            attn_out = th.zeros(
                x.shape[0],
                self.heads,
                self.head_dim,
                dtype=x.dtype,
                device=x.device,
            )
            attn_out.index_add_(0, dst, messages)
            attn_out = attn_out.reshape(x.shape[0], self.hidden_dim)

        x = self.norm1(x + self.dropout(self.out_proj(attn_out)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class SparseGraphTransformerEncoder(_VirtualNodeEncoderBase):
    """Sparse graph transformer encoder for fixed-shape Grid2Op graphs."""

    def __init__(
        self,
        graph_spec: Dict[str, Any],
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        heads: int = 4,
        readout_aggr: str = "mean",
        layer_norm: bool = True,
        node_id_embeddings: bool = False,
        node_id_emb_dim: int = 8,
        edge_pre_encoder: bool = False,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        ffn_multiplier: int = 4,
        use_edge_attr: bool = True,
        use_edge_type_embeddings: bool = True,
        relation_bias: bool = True,
        add_substation_nodes: bool = False,
        summary_edge_direction: str = "bidirectional",
        virtual_edge_direction: str = "inherit",
    ) -> None:
        super().__init__()
        if pyg_softmax is None:
            raise ImportError(
                "gnn_type=sparse_transformer requires torch-geometric."
            )
        if n_layers < 1:
            raise ValueError("Sparse graph transformer needs at least one layer.")

        self.readout_aggr = str(readout_aggr).lower()
        self.node_out_dim = int(hidden_dim)
        if self.readout_aggr not in (
            POOLING_AGGREGATIONS + SPARSE_TRANSFORMER_ONLY_AGGREGATIONS
        ):
            raise ValueError(
                f"Unsupported sparse graph transformer readout: {readout_aggr}"
            )

        self.edge_dim = int(graph_spec["edge_dim"])
        self.edge_feature_names = list(graph_spec.get("edge_feature_names", []))
        self.base_n_edge_types = int(graph_spec.get("n_edge_types", 3))
        self.substation_edge_type = (
            self.base_n_edge_types if add_substation_nodes else None
        )
        self.virtual_edge_type = (
            self.base_n_edge_types + int(add_substation_nodes)
            if self.readout_aggr == "virtual_node"
            else None
        )
        self.n_edge_types = (
            self.base_n_edge_types
            + int(self.substation_edge_type is not None)
            + int(self.virtual_edge_type is not None)
        )
        self.register_buffer(
            "edge_index",
            th.tensor(graph_spec["edge_index"], dtype=th.long),
        )
        edge_type = graph_spec.get("edge_type")
        if edge_type is None:
            edge_type = np.ones((self.edge_index.shape[1],), dtype=np.int64)
        self.register_buffer("edge_type", th.tensor(edge_type, dtype=th.long))
        self.register_buffer(
            "node_ids",
            th.tensor(graph_spec["node_ids"], dtype=th.long),
        )
        controlled_node_mask = graph_spec.get("controlled_node_mask")
        if controlled_node_mask is None:
            controlled_node_mask = np.ones(
                (len(graph_spec["node_ids"]),),
                dtype=np.float32,
            )
        self.register_buffer(
            "controlled_node_mask",
            th.tensor(controlled_node_mask, dtype=th.float32),
        )

        node_dim = int(graph_spec["node_dim"])
        self.node_id_embeddings = bool(node_id_embeddings)
        self.n_busbar = int(graph_spec.get("n_busbar", 2))
        self.node_id_stride = int(
            graph_spec.get("node_id_stride", self.n_busbar)
        )
        self.n_bus_id_embeddings = int(
            graph_spec.get("n_bus_id_embeddings", self.n_busbar)
        )
        if self.node_id_embeddings:
            if node_id_emb_dim < 1:
                raise ValueError("node_id_emb_dim must be at least 1.")
            n_sub = int(
                graph_spec.get(
                    "n_sub",
                    int(self.node_ids.max().item() // self.node_id_stride) + 1,
                )
            )
            # Same reasoning as the summary node: an index embedding is shaped
            # by the number of substations and cannot cross grids. Project the
            # structural descriptor instead, so the learned tensor is shaped by
            # the descriptor and a checkpoint stays loadable on any network.
            substation_features = np.asarray(
                graph_spec.get(
                    "substation_features", np.zeros((max(1, n_sub), 1))
                ),
                dtype=np.float32,
            )
            self.register_buffer(
                "node_id_substation_features",
                th.tensor(substation_features, dtype=th.float32),
                persistent=False,
            )
            self.sub_id_embedding = nn.Linear(
                substation_features.shape[1], node_id_emb_dim
            )
            self.bus_id_embedding = nn.Embedding(
                self.n_bus_id_embeddings, node_id_emb_dim
            )
            node_dim = node_dim + 2 * node_id_emb_dim

        node_encoder_layers = [nn.Linear(node_dim, hidden_dim), nn.ReLU()]
        if layer_norm:
            node_encoder_layers.append(nn.LayerNorm(hidden_dim))
        self.node_pre_encoder = nn.Sequential(*node_encoder_layers)
        self._init_hierarchy_nodes(
            graph_spec,
            feature_dim=hidden_dim,
            use_virtual_node=self.readout_aggr == "virtual_node",
            add_substation_nodes=add_substation_nodes,
            summary_edge_direction=summary_edge_direction,
            virtual_edge_direction=virtual_edge_direction,
        )

        if edge_pre_encoder and self.edge_dim > 0 and use_edge_attr:
            edge_encoder_layers = [nn.Linear(self.edge_dim, hidden_dim), nn.ReLU()]
            if layer_norm:
                edge_encoder_layers.append(nn.LayerNorm(hidden_dim))
            self.edge_pre_encoder = nn.Sequential(*edge_encoder_layers)
            layer_edge_dim = hidden_dim
        else:
            self.edge_pre_encoder = nn.Identity()
            layer_edge_dim = self.edge_dim

        self.layers = nn.ModuleList(
            [
                SparseGraphTransformerLayer(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    edge_dim=layer_edge_dim,
                    n_edge_types=self.n_edge_types,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    ffn_multiplier=ffn_multiplier,
                    use_edge_attr=use_edge_attr,
                    use_edge_type_embeddings=use_edge_type_embeddings,
                    relation_bias=relation_bias,
                    layer_norm=layer_norm,
                )
                for _ in range(n_layers)
            ]
        )

        self.pool_score = (
            nn.Linear(hidden_dim, 1)
            if self.readout_aggr in {"attention", "controlled_attention"}
            else None
        )
        self.readout = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU())

    def forward(
        self,
        graph_obs: Dict[str, th.Tensor],
        edge_index: Optional[th.Tensor] = None,
        node_ids: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        embedding, _ = self.forward_with_nodes(
            graph_obs,
            edge_index=edge_index,
            node_ids=node_ids,
        )
        return embedding

    def forward_with_nodes(
        self,
        graph_obs: Dict[str, th.Tensor],
        edge_index: Optional[th.Tensor] = None,
        node_ids: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Return the graph readout and physical post-message-passing nodes."""
        unbatched = graph_obs["node_features"].dim() == 2
        n_nodes = int(graph_obs["node_features"].shape[-2])
        (
            x,
            edge_index,
            edge_attr,
            edge_type,
            batch,
            node_mask,
            controlled_node_mask,
            energized_node_mask,
            flat_node_ids,
        ) = self._to_pyg_batch(graph_obs, edge_index=edge_index, node_ids=node_ids)

        x = self._append_node_id_embeddings(x, flat_node_ids)
        x = self.node_pre_encoder(x)
        n_physical_nodes = int(x.shape[0])
        (
            x,
            edge_index,
            edge_attr,
            batch,
            node_mask,
            edge_type,
            controlled_node_mask,
            virtual_indices,
        ) = self._append_hierarchy_nodes(
            x,
            edge_index,
            edge_attr,
            batch,
            flat_node_ids,
            node_mask=node_mask,
            edge_type=edge_type,
            substation_edge_type=self.substation_edge_type,
            virtual_edge_type=self.virtual_edge_type,
            controlled_node_mask=controlled_node_mask,
        )
        if (
            energized_node_mask is not None
            and energized_node_mask.numel() < x.shape[0]
        ):
            energized_node_mask = th.cat(
                [
                    energized_node_mask,
                    th.zeros(
                        x.shape[0] - energized_node_mask.numel(),
                        dtype=energized_node_mask.dtype,
                        device=energized_node_mask.device,
                    ),
                ]
            )
        edge_attr = self.edge_pre_encoder(edge_attr)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, edge_type)

        batch_size = max(1, n_physical_nodes // n_nodes)
        node_embeddings = x[:n_physical_nodes].reshape(
            batch_size, n_nodes, self.node_out_dim
        )

        pooled = (
            x[virtual_indices]
            if virtual_indices is not None
            else self._pool_nodes(
                x,
                batch,
                node_mask,
                controlled_node_mask,
                energized_node_mask,
            )
        )
        embedding = self.readout(pooled)
        if unbatched:
            return embedding.squeeze(0), node_embeddings.squeeze(0)
        return embedding, node_embeddings

    def _to_pyg_batch(
        self,
        graph_obs: Dict[str, th.Tensor],
        edge_index: Optional[th.Tensor] = None,
        node_ids: Optional[th.Tensor] = None,
    ):
        nodes = graph_obs["node_features"]
        edge_features = graph_obs["edge_features"]
        node_mask = graph_obs.get("node_mask")
        edge_mask = graph_obs.get("edge_mask")
        edge_type_obs = graph_obs.get("edge_type")
        controlled_node_mask = graph_obs.get("controlled_node_mask")
        energized_node_mask = graph_obs.get("energized_node_mask")

        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            node_mask = None if node_mask is None else node_mask.unsqueeze(0)
            edge_mask = None if edge_mask is None else edge_mask.unsqueeze(0)
            edge_type_obs = (
                None if edge_type_obs is None else edge_type_obs.unsqueeze(0)
            )
            controlled_node_mask = (
                None
                if controlled_node_mask is None
                else controlled_node_mask.unsqueeze(0)
            )
            energized_node_mask = (
                None
                if energized_node_mask is None
                else energized_node_mask.unsqueeze(0)
            )

        base_edge_index = self.edge_index if edge_index is None else edge_index
        base_node_ids = self.node_ids if node_ids is None else node_ids
        batch_size, n_nodes, node_dim = nodes.shape
        n_edges = base_edge_index.shape[1]

        x = nodes.reshape(batch_size * n_nodes, node_dim)
        batch = th.arange(batch_size, device=nodes.device).repeat_interleave(n_nodes)
        flat_node_ids = base_node_ids.to(nodes.device).repeat(batch_size)

        base_edge_index = base_edge_index.to(nodes.device)
        edge_index = base_edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
        offsets = (th.arange(batch_size, device=nodes.device) * n_nodes).view(
            batch_size, 1, 1
        )
        edge_index = (edge_index + offsets).permute(1, 0, 2).reshape(
            2, batch_size * n_edges
        )

        edge_attr = edge_features.reshape(batch_size * n_edges, self.edge_dim)
        if edge_type_obs is None:
            edge_type = self.edge_type.to(nodes.device).repeat(batch_size)
        else:
            edge_type = edge_type_obs.reshape(batch_size * n_edges).to(
                device=nodes.device,
                dtype=th.long,
            )
        if edge_mask is not None:
            keep_edges = edge_mask.reshape(batch_size * n_edges).bool()
            edge_index = edge_index[:, keep_edges]
            edge_attr = edge_attr[keep_edges]
            edge_type = edge_type[keep_edges]

        flat_node_mask = (
            None if node_mask is None else node_mask.reshape(batch_size * n_nodes)
        )
        if controlled_node_mask is None:
            controlled_node_mask = self.controlled_node_mask.to(nodes.device).repeat(
                batch_size
            )
        else:
            controlled_node_mask = controlled_node_mask.reshape(batch_size * n_nodes)
        flat_energized_mask = (
            None
            if energized_node_mask is None
            else energized_node_mask.reshape(batch_size * n_nodes)
        )
        return (
            x,
            edge_index,
            edge_attr,
            edge_type,
            batch,
            flat_node_mask,
            controlled_node_mask,
            flat_energized_mask,
            flat_node_ids,
        )

    def _append_node_id_embeddings(
        self,
        x: th.Tensor,
        node_ids: th.Tensor,
    ) -> th.Tensor:
        if not self.node_id_embeddings:
            return x
        sub_ids = th.div(
            node_ids, self.node_id_stride, rounding_mode="floor"
        ).long()
        bus_ids = th.remainder(node_ids, self.node_id_stride).long()
        return th.cat(
            [
                x,
                self.sub_id_embedding(
                    self.node_id_substation_features[sub_ids]
                ),
                self.bus_id_embedding(bus_ids),
            ],
            dim=-1,
        )

    def _pool_nodes(
        self,
        x: th.Tensor,
        batch: th.Tensor,
        node_mask: Optional[th.Tensor],
        controlled_node_mask: Optional[th.Tensor],
        energized_node_mask: Optional[th.Tensor],
    ) -> th.Tensor:
        controlled_only = self.readout_aggr.startswith("controlled_")
        readout = self.readout_aggr.removeprefix("controlled_")
        energized_only = readout.startswith("energized_")
        base_aggr = readout.removeprefix("energized_")

        active_mask = th.ones((x.shape[0],), dtype=x.dtype, device=x.device)
        if node_mask is not None:
            active_mask = active_mask * node_mask.to(dtype=x.dtype)

        if controlled_only:
            if controlled_node_mask is None:
                raise ValueError(
                    f"readout_aggr={self.readout_aggr!r} needs a "
                    "controlled_node_mask in the graph observation."
                )
            active_mask = active_mask * controlled_node_mask.to(dtype=x.dtype)
        if energized_only:
            if energized_node_mask is None:
                raise ValueError(
                    f"readout_aggr={self.readout_aggr!r} needs an "
                    "energized_node_mask in the graph observation."
                )
            active_mask = active_mask * energized_node_mask.to(dtype=x.dtype)

        if base_aggr == "attention":
            return self._attention_pool(x, batch, active_mask)

        if base_aggr == "max":
            masked_x = x.masked_fill(active_mask.unsqueeze(-1) <= 0, -th.inf)
            pooled = global_max_pool(masked_x, batch)
            return th.nan_to_num(pooled, nan=0.0, neginf=0.0, posinf=0.0)

        masked_x = x * active_mask.unsqueeze(-1)
        pooled = global_add_pool(masked_x, batch)
        if base_aggr == "mean":
            denom = global_add_pool(active_mask.unsqueeze(-1), batch).clamp_min(1.0)
            pooled = pooled / denom
        return pooled

    def _attention_pool(
        self,
        x: th.Tensor,
        batch: th.Tensor,
        active_mask: th.Tensor,
    ) -> th.Tensor:
        scores = self.pool_score(x).squeeze(-1)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled = []
        for graph_idx in range(num_graphs):
            graph_nodes = batch == graph_idx
            selected = graph_nodes & (active_mask > 0)
            if not th.any(selected):
                selected = graph_nodes
            weights = th.softmax(scores[selected], dim=0)
            pooled.append((weights.unsqueeze(-1) * x[selected]).sum(dim=0))
        return th.stack(pooled, dim=0)


def build_graph_encoder(graph_spec: Dict[str, Any], args: Dict[str, Any]) -> GraphEncoder:
    if str(args.gnn_type).lower() == "sparse_transformer":
        readout_aggr = getattr(args, "sparse_gt_pooling", "")
        if not readout_aggr:
            readout_aggr = getattr(args, "gnn_readout_aggr", "mean")
        return SparseGraphTransformerEncoder(
            graph_spec=graph_spec,
            hidden_dim=args.gnn_hidden_dim,
            out_dim=args.gnn_out_dim,
            n_layers=args.gnn_layers,
            heads=args.gnn_heads,
            readout_aggr=readout_aggr,
            layer_norm=args.gnn_layer_norm,
            node_id_embeddings=getattr(args, "gnn_node_id_embeddings", False),
            node_id_emb_dim=getattr(args, "gnn_node_id_emb_dim", 8),
            edge_pre_encoder=getattr(args, "gnn_edge_pre_encoder", False),
            dropout=getattr(args, "sparse_gt_dropout", 0.0),
            attention_dropout=getattr(args, "sparse_gt_attention_dropout", 0.0),
            ffn_multiplier=getattr(args, "sparse_gt_ffn_multiplier", 4),
            use_edge_attr=getattr(args, "sparse_gt_use_edge_attr", True),
            use_edge_type_embeddings=getattr(
                args, "sparse_gt_use_edge_type_embeddings", True
            ),
            relation_bias=getattr(args, "sparse_gt_relation_bias", True),
            add_substation_nodes=getattr(args, "gnn_add_substation_nodes", False),
            summary_edge_direction=getattr(
                args, "gnn_summary_edge_direction", "bidirectional"
            ),
            virtual_edge_direction=getattr(
                args, "gnn_virtual_edge_direction", "inherit"
            ),
        )
    return GraphEncoder(
        graph_spec=graph_spec,
        hidden_dim=args.gnn_hidden_dim,
        out_dim=args.gnn_out_dim,
        n_layers=args.gnn_layers,
        conv_type=args.gnn_type,
        graphsage_aggr=getattr(args, "graphsage_aggr", getattr(args, "gnn_aggr", "mean")),
        readout_aggr=getattr(args, "gnn_readout_aggr", "mean"),
        layer_norm=args.gnn_layer_norm,
        heads=args.gnn_heads,
        node_pre_encoder=getattr(args, "gnn_node_pre_encoder", False),
        edge_pre_encoder=getattr(args, "gnn_edge_pre_encoder", False),
        node_id_embeddings=getattr(args, "gnn_node_id_embeddings", False),
        node_id_emb_dim=getattr(args, "gnn_node_id_emb_dim", 8),
        gcn_edge_weight_feature=getattr(args, "gcn_edge_weight_feature", "none"),
        add_substation_nodes=getattr(args, "gnn_add_substation_nodes", False),
        summary_edge_direction=getattr(
            args, "gnn_summary_edge_direction", "bidirectional"
        ),
        virtual_edge_direction=getattr(
            args, "gnn_virtual_edge_direction", "inherit"
        ),
    )


class GraphAndFlatEncoder(nn.Module):
    def __init__(
        self,
        graph_spec: Dict[str, Any],
        flat_dim: int,
        args: Dict[str, Any],
        use_flat: bool = False,
        graph_encoder: Optional[GraphEncoder] = None,
    ) -> None:
        super().__init__()
        self.use_flat = use_flat
        self.register_buffer(
            "edge_index",
            th.tensor(graph_spec["edge_index"], dtype=th.long),
            persistent=False,
        )
        self.register_buffer(
            "node_ids",
            th.tensor(graph_spec["node_ids"], dtype=th.long),
            persistent=False,
        )
        self.graph_encoder = graph_encoder or build_graph_encoder(graph_spec, args)
        self.out_dim = args.gnn_out_dim + (flat_dim if use_flat else 0)
        self.node_out_dim = int(self.graph_encoder.node_out_dim)

    def forward(self, obs: Dict[str, th.Tensor], graph_key: str = "graph") -> th.Tensor:
        graph_embedding = self.graph_encoder(
            obs[graph_key],
            edge_index=self.edge_index,
            node_ids=self.node_ids,
        )
        return self._append_flat(obs, graph_key, graph_embedding)

    def forward_with_nodes(
        self,
        obs: Dict[str, th.Tensor],
        graph_key: str = "graph",
    ) -> Tuple[th.Tensor, th.Tensor]:
        graph_embedding, node_embeddings = self.graph_encoder.forward_with_nodes(
            obs[graph_key],
            edge_index=self.edge_index,
            node_ids=self.node_ids,
        )
        return (
            self._append_flat(obs, graph_key, graph_embedding),
            node_embeddings,
        )

    def _append_flat(
        self,
        obs: Dict[str, th.Tensor],
        graph_key: str,
        graph_embedding: th.Tensor,
    ) -> th.Tensor:
        if not self.use_flat:
            return graph_embedding

        flat = get_flat_obs(obs)
        if flat.dim() == 1 and graph_embedding.dim() == 1:
            return th.cat([graph_embedding, flat], dim=-1)
        if flat.dim() == 1:
            flat = flat.unsqueeze(0)
        if graph_embedding.dim() == 1:
            graph_embedding = graph_embedding.unsqueeze(0)
        encoded = th.cat([graph_embedding, flat], dim=-1)
        return encoded.squeeze(0) if obs[graph_key]["node_features"].dim() == 2 else encoded
