"""Runtime utilities for the learned action-risk surrogate."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch as th
import torch.nn as nn

from common.gnn import build_graph_encoder


LABEL_UNILATERAL = 0
LABEL_JOINT = 1


class RiskSurrogate(nn.Module):
    """Action-conditional graph model trained in Phase 2."""

    def __init__(
        self,
        graph_spec: dict[str, Any],
        n_agents: int,
        action_vocab_size: int,
        args: Namespace,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.action_vocab_size = int(action_vocab_size)
        self.include_pre_risk = bool(getattr(args, "include_pre_risk", True))
        self.include_n_non_idle = bool(getattr(args, "include_n_non_idle", False))

        self.graph_encoder = build_graph_encoder(graph_spec, args)
        self.agent_embedding = nn.Embedding(self.n_agents, args.cond_emb_dim)
        self.action_embedding = nn.Embedding(self.action_vocab_size, args.cond_emb_dim)

        cond_dim = 2 * args.cond_emb_dim
        scalar_dim = int(self.include_pre_risk) + int(self.include_n_non_idle)
        self.head = build_mlp(
            input_dim=args.gnn_out_dim + cond_dim + scalar_dim,
            hidden_dim=args.mlp_hidden_dim,
            n_layers=args.mlp_layers,
            dropout=args.dropout,
        )

    def forward(self, batch: dict[str, th.Tensor]) -> th.Tensor:
        graph_obs = {
            "node_features": batch["node_features"],
            "edge_features": batch["edge_features"],
            "node_mask": batch["node_mask"],
            "edge_mask": batch["edge_mask"],
        }
        graph_embedding = self.graph_encoder(graph_obs)
        cond_embedding = self._condition_embedding(batch)

        pieces = [graph_embedding, cond_embedding]
        if self.include_pre_risk:
            pieces.append(batch["pre_risk"].unsqueeze(-1))
        if self.include_n_non_idle:
            pieces.append(batch["n_non_idle_agents"].unsqueeze(-1))
        return self.head(th.cat(pieces, dim=-1)).squeeze(-1)

    @th.no_grad()
    def predict_local_action_risks(
        self,
        state_graph: dict[str, th.Tensor],
        agent_index: int,
        n_actions: int,
        pre_risk: th.Tensor | None = None,
    ) -> th.Tensor:
        """Score all local actions while running the GNN only once per state."""
        if n_actions > self.action_vocab_size:
            raise ValueError(
                f"Requested {n_actions} actions, but surrogate action vocab is "
                f"{self.action_vocab_size}."
            )
        if agent_index >= self.n_agents:
            raise ValueError(
                f"Requested agent {agent_index}, but surrogate has {self.n_agents} agents."
            )

        graph_embedding = self.graph_encoder(state_graph)
        unbatched = graph_embedding.dim() == 1
        if unbatched:
            graph_embedding = graph_embedding.unsqueeze(0)

        batch_size = graph_embedding.shape[0]
        device = graph_embedding.device
        action_ids = th.arange(n_actions, device=device, dtype=th.long)
        action_ids = action_ids.unsqueeze(0).expand(batch_size, -1)
        agent_ids = th.full(
            (batch_size, n_actions),
            int(agent_index),
            device=device,
            dtype=th.long,
        )

        cond_embedding = th.cat(
            [
                self.agent_embedding(agent_ids),
                self.action_embedding(action_ids),
            ],
            dim=-1,
        )
        graph_embedding = graph_embedding.unsqueeze(1).expand(-1, n_actions, -1)

        pieces = [graph_embedding, cond_embedding]
        if self.include_pre_risk:
            if pre_risk is None:
                raise ValueError("pre_risk is required by this risk surrogate.")
            if pre_risk.dim() == 0:
                pre_risk = pre_risk.unsqueeze(0)
            pieces.append(pre_risk.view(batch_size, 1, 1).expand(-1, n_actions, -1))
        if self.include_n_non_idle:
            pieces.append((action_ids != 0).to(dtype=th.float32).unsqueeze(-1))

        x = th.cat(pieces, dim=-1).reshape(batch_size * n_actions, -1)
        risks = self.head(x).view(batch_size, n_actions)
        return risks.squeeze(0) if unbatched else risks

    def _condition_embedding(self, batch: dict[str, th.Tensor]) -> th.Tensor:
        agent_index = batch["agent_index"].clamp(min=0, max=self.n_agents - 1)
        action_id = batch["action_id"].clamp(min=0, max=self.action_vocab_size - 1)

        local_cond = th.cat(
            [
                self.agent_embedding(agent_index),
                self.action_embedding(action_id),
            ],
            dim=-1,
        )

        joint_action_ids = batch["joint_action_ids"].clamp(
            min=0,
            max=self.action_vocab_size - 1,
        )
        batch_size = joint_action_ids.shape[0]
        joint_agent_ids = th.arange(
            self.n_agents,
            device=joint_action_ids.device,
            dtype=th.long,
        ).unsqueeze(0).expand(batch_size, -1)
        joint_cond = th.cat(
            [
                self.agent_embedding(joint_agent_ids),
                self.action_embedding(joint_action_ids),
            ],
            dim=-1,
        ).mean(dim=1)

        is_joint = batch["label_type"].eq(LABEL_JOINT).unsqueeze(-1)
        return th.where(is_joint, joint_cond, local_cond)


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(max(n_layers, 1)):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, 1))
    return nn.Sequential(*layers)


def load_risk_surrogate(
    checkpoint_path: str | Path,
    map_location: str | th.device = "cpu",
) -> tuple[RiskSurrogate, dict[str, Any]]:
    """Load a frozen Phase 2 surrogate checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Risk-prior checkpoint not found: {path}")

    try:
        payload = th.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = th.load(path, map_location=map_location)

    args = Namespace(**payload["args"])
    metadata = payload["metadata"]
    graph_spec = payload.get("graph_spec", metadata["state_graph_spec"])
    model = RiskSurrogate(
        graph_spec=graph_spec,
        n_agents=int(payload.get("n_agents", len(metadata["agent_ids"]))),
        action_vocab_size=int(
            payload.get("action_vocab_size", max(metadata["n_actions_by_agent"].values()))
        ),
        args=args,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, payload


def graph_pre_risk(
    state_graph: dict[str, th.Tensor],
    graph_spec: dict[str, Any],
) -> th.Tensor:
    """Estimate max rho from the graph features used by the surrogate."""
    edge_features = state_graph["edge_features"]
    edge_feature_names = list(graph_spec.get("edge_feature_names", []))
    if "rho" in edge_feature_names:
        rho_idx = edge_feature_names.index("rho")
    else:
        rho_idx = 1 if int(graph_spec.get("edge_dim", edge_features.shape[-1])) > 1 else 0

    rho = edge_features[..., rho_idx]
    edge_mask = state_graph.get("edge_mask")
    if edge_mask is not None:
        rho = rho.masked_fill(edge_mask <= 0, 0.0)
    return rho.max(dim=-1).values
