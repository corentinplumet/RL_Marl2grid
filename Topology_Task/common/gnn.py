from common.imports import *
from common.utils import get_flat_obs

try:
    from torch_geometric.nn import GATConv, GCNConv, GINEConv, SAGEConv, global_add_pool, global_max_pool
except ModuleNotFoundError:
    GATConv = GCNConv = GINEConv = SAGEConv = global_add_pool = global_max_pool = None


class GraphEncoder(nn.Module):
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
        self.readout_aggr = readout_aggr.lower()
        if self.readout_aggr not in {"mean", "sum", "max"}:
            raise ValueError(f"Unsupported GNN readout aggregation: {readout_aggr}")
        self.edge_dim = int(graph_spec["edge_dim"])
        self.register_buffer("edge_index", th.tensor(graph_spec["edge_index"], dtype=th.long))

        node_dim = int(graph_spec["node_dim"])
        dims = [node_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList(
            [
                self._make_conv(
                    in_dim=dims[idx],
                    hidden_dim=hidden_dim,
                    edge_dim=self.edge_dim,
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
    ) -> th.Tensor:
        unbatched = graph_obs["node_features"].dim() == 2
        x, edge_index, edge_attr, batch, node_mask = self._to_pyg_batch(
            graph_obs, edge_index=edge_index
        )

        for conv, norm in zip(self.convs, self.norms):
            if self.conv_type in {"gat", "gine"}:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = norm(F.relu(x))

        pooled = self._pool_nodes(x, batch, node_mask)
        embedding = self.readout(pooled)
        return embedding.squeeze(0) if unbatched else embedding

    def _pool_nodes(
        self,
        x: th.Tensor,
        batch: th.Tensor,
        node_mask: Optional[th.Tensor],
    ) -> th.Tensor:
        if node_mask is not None:
            if self.readout_aggr == "max":
                masked_x = x.masked_fill(node_mask.unsqueeze(-1) <= 0, -th.inf)
                pooled = global_max_pool(masked_x, batch)
                return th.nan_to_num(pooled, nan=0.0, neginf=0.0, posinf=0.0)

            x = x * node_mask.unsqueeze(-1)
            pooled = global_add_pool(x, batch)
            if self.readout_aggr == "mean":
                denom = global_add_pool(node_mask.unsqueeze(-1), batch).clamp_min(1.0)
                pooled = pooled / denom
            return pooled

        if self.readout_aggr == "max":
            return global_max_pool(x, batch)

        pooled = global_add_pool(x, batch)
        if self.readout_aggr == "mean":
            denom = th.bincount(batch, minlength=int(batch.max().item()) + 1).to(x).unsqueeze(-1).clamp_min(1.0)
            pooled = pooled / denom
        return pooled

    def _to_pyg_batch(
        self,
        graph_obs: Dict[str, th.Tensor],
        edge_index: Optional[th.Tensor] = None,
    ):
        nodes = graph_obs["node_features"]
        edge_features = graph_obs["edge_features"]
        node_mask = graph_obs.get("node_mask")
        edge_mask = graph_obs.get("edge_mask")

        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            node_mask = None if node_mask is None else node_mask.unsqueeze(0)
            edge_mask = None if edge_mask is None else edge_mask.unsqueeze(0)

        base_edge_index = self.edge_index if edge_index is None else edge_index
        batch_size, n_nodes, node_dim = nodes.shape
        n_edges = base_edge_index.shape[1]

        x = nodes.reshape(batch_size * n_nodes, node_dim)
        batch = th.arange(batch_size, device=nodes.device).repeat_interleave(n_nodes)

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
        return x, edge_index, edge_attr, batch, flat_node_mask

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


def build_graph_encoder(graph_spec: Dict[str, Any], args: Dict[str, Any]) -> GraphEncoder:
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
        self.graph_encoder = graph_encoder or build_graph_encoder(graph_spec, args)
        self.out_dim = args.gnn_out_dim + (flat_dim if use_flat else 0)

    def forward(self, obs: Dict[str, th.Tensor], graph_key: str = "graph") -> th.Tensor:
        graph_embedding = self.graph_encoder(obs[graph_key], edge_index=self.edge_index)
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
