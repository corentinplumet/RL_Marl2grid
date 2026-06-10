#!/usr/bin/env python3
"""Train the Phase 2 action-conditional risk surrogate.

The model learns:

    state_graph, agent_id, local_action_id -> predicted one-step risk

It can optionally consume sampled joint-action labels too, but the default
training target is the unilateral label set because that is the signal used by
the per-agent Gibbs logit prior in Phase 3.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TASK_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = TASK_DIR.parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


LABEL_UNILATERAL = 0
LABEL_JOINT = 1


@dataclass
class LoadedRiskFile:
    path: Path
    metadata: dict[str, Any]
    node_features: np.ndarray
    edge_features: np.ndarray
    node_mask: np.ndarray
    edge_mask: np.ndarray
    agent_index: np.ndarray
    action_id: np.ndarray
    joint_action_ids: np.ndarray
    label_type: np.ndarray
    n_non_idle_agents: np.ndarray
    target_risk: np.ndarray
    pre_risk: np.ndarray
    simulation_error: np.ndarray
    hazard_index: np.ndarray


class RiskPriorDataset(Dataset):
    def __init__(self, files: list[LoadedRiskFile], row_refs: np.ndarray) -> None:
        self.files = files
        self.row_refs = np.asarray(row_refs, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.row_refs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, th.Tensor]:
        file_idx, row_idx = self.row_refs[idx]
        data = self.files[int(file_idx)]
        row = int(row_idx)
        return {
            "node_features": th.from_numpy(data.node_features[row]).float(),
            "edge_features": th.from_numpy(data.edge_features[row]).float(),
            "node_mask": th.from_numpy(data.node_mask[row]).float(),
            "edge_mask": th.from_numpy(data.edge_mask[row]).float(),
            "agent_index": th.tensor(data.agent_index[row], dtype=th.long),
            "action_id": th.tensor(data.action_id[row], dtype=th.long),
            "joint_action_ids": th.from_numpy(data.joint_action_ids[row]).long(),
            "label_type": th.tensor(data.label_type[row], dtype=th.long),
            "n_non_idle_agents": th.tensor(data.n_non_idle_agents[row], dtype=th.float32),
            "target_risk": th.tensor(data.target_risk[row], dtype=th.float32),
            "pre_risk": th.tensor(data.pre_risk[row], dtype=th.float32),
            "hazard_index": th.tensor(data.hazard_index[row], dtype=th.long),
            "dataset_index": th.tensor(file_idx, dtype=th.long),
        }


class RiskSurrogate(nn.Module):
    def __init__(
        self,
        graph_spec: dict[str, Any],
        n_agents: int,
        action_vocab_size: int,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.action_vocab_size = int(action_vocab_size)
        self.include_pre_risk = bool(args.include_pre_risk)
        self.include_n_non_idle = bool(args.include_n_non_idle)

        from common.gnn import build_graph_encoder

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
        x = th.cat(pieces, dim=-1)
        return self.head(x).squeeze(-1)

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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Phase 1 .npz datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "risk_prior_models",
        help="Directory where checkpoints and metrics are written.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Name used for the output subdirectory. Defaults to dataset stem plus seed.",
    )
    parser.add_argument(
        "--label-types",
        choices=["unilateral", "joint", "all"],
        default="unilateral",
        help="Which Phase 1 labels to train on. Phase 2 starts with unilateral.",
    )
    parser.add_argument(
        "--exclude-simulation-errors",
        type=parse_bool,
        default=True,
        help="Drop exception-based simulator failures before training.",
    )
    parser.add_argument("--limit-examples", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["huber", "mse", "mae"], default="huber")
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--penalty-threshold", type=float, default=1.999)
    parser.add_argument("--ranking-k", type=int, default=5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, cuda:0, or auto. auto uses CUDA when available, else CPU.",
    )

    parser.add_argument("--gnn-type", default="gine", choices=["gat", "gcn", "gine", "graphsage"])
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-out-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--gnn-heads", type=int, default=1)
    parser.add_argument("--gnn-layer-norm", type=parse_bool, default=True)
    parser.add_argument("--gnn-readout-aggr", default="mean", choices=["mean", "sum", "max"])
    parser.add_argument("--gnn-aggr", default="mean")
    parser.add_argument("--graphsage-aggr", default="mean")
    parser.add_argument("--gnn-node-pre-encoder", type=parse_bool, default=True)
    parser.add_argument("--gnn-edge-pre-encoder", type=parse_bool, default=True)
    parser.add_argument("--gnn-node-id-embeddings", type=parse_bool, default=True)
    parser.add_argument("--gnn-node-id-emb-dim", type=int, default=8)
    parser.add_argument("--gcn-edge-weight-feature", default="none")

    parser.add_argument("--cond-emb-dim", type=int, default=32)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--include-pre-risk",
        type=parse_bool,
        default=True,
        help="Concatenate current max rho. It is observable and helps calibration.",
    )
    parser.add_argument(
        "--include-n-non-idle",
        type=parse_bool,
        default=False,
        help="Optional scalar mainly useful when training with joint labels.",
    )

    parser.add_argument("--wandb", type=parse_bool, default=False)
    parser.add_argument("--wandb-project", default="risk_prior")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-mode", default="")
    parser.add_argument("--wandb-log-artifact", type=parse_bool, default=True)
    return parser.parse_args()


def metadata_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(".meta.json")


def load_risk_file(path: Path) -> LoadedRiskFile:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    meta_path = metadata_path(path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    metadata = json.loads(meta_path.read_text())
    with np.load(path) as data:
        return LoadedRiskFile(
            path=path,
            metadata=metadata,
            node_features=data["node_features"].astype(np.float32, copy=False),
            edge_features=data["edge_features"].astype(np.float32, copy=False),
            node_mask=data["node_mask"].astype(np.float32, copy=False),
            edge_mask=data["edge_mask"].astype(np.float32, copy=False),
            agent_index=data["agent_index"].astype(np.int64, copy=False),
            action_id=data["action_id"].astype(np.int64, copy=False),
            joint_action_ids=data["joint_action_ids"].astype(np.int64, copy=False),
            label_type=data["label_type"].astype(np.int64, copy=False),
            n_non_idle_agents=data["n_non_idle_agents"].astype(np.float32, copy=False),
            target_risk=data["target_risk"].astype(np.float32, copy=False),
            pre_risk=data["pre_risk"].astype(np.float32, copy=False),
            simulation_error=data["simulation_error"].astype(np.bool_, copy=False),
            hazard_index=data["hazard_index"].astype(np.int64, copy=False),
        )


def validate_metadata(files: list[LoadedRiskFile]) -> None:
    first = files[0].metadata
    first_agents = first["agent_ids"]
    first_actions = first["n_actions_by_agent"]
    first_graph = first["state_graph_spec"]
    for loaded in files[1:]:
        meta = loaded.metadata
        if meta["agent_ids"] != first_agents:
            raise ValueError("All datasets must use the same agent ordering.")
        if meta["n_actions_by_agent"] != first_actions:
            raise ValueError("All datasets must use the same action spaces.")
        graph = meta["state_graph_spec"]
        for key in ["node_dim", "edge_dim", "edge_index", "node_ids"]:
            if graph[key] != first_graph[key]:
                raise ValueError(f"All datasets must use the same graph spec key: {key}")


def filtered_row_refs(
    files: list[LoadedRiskFile],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    label_map = {
        "unilateral": {LABEL_UNILATERAL},
        "joint": {LABEL_JOINT},
        "all": {LABEL_UNILATERAL, LABEL_JOINT},
    }
    selected_labels = label_map[args.label_types]
    row_refs: list[np.ndarray] = []
    group_pairs: list[np.ndarray] = []

    for file_idx, loaded in enumerate(files):
        mask = np.isin(loaded.label_type, list(selected_labels))
        if args.exclude_simulation_errors:
            mask &= ~loaded.simulation_error
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            continue
        row_refs.append(
            np.column_stack(
                [
                    np.full(rows.size, file_idx, dtype=np.int64),
                    rows.astype(np.int64, copy=False),
                ]
            )
        )
        group_pairs.append(
            np.column_stack(
                [
                    np.full(rows.size, file_idx, dtype=np.int64),
                    loaded.hazard_index[rows].astype(np.int64, copy=False),
                ]
            )
        )

    if not row_refs:
        raise ValueError("No rows remain after applying label/error filters.")

    refs = np.concatenate(row_refs, axis=0)
    groups = np.concatenate(group_pairs, axis=0)

    if args.limit_examples and args.limit_examples > 0 and refs.shape[0] > args.limit_examples:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(refs.shape[0], size=args.limit_examples, replace=False)
        keep.sort()
        refs = refs[keep]
        groups = groups[keep]

    return refs, groups


def split_by_hazard_group(
    row_refs: np.ndarray,
    group_pairs: np.ndarray,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1.")
    unique_groups, inverse = np.unique(group_pairs, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    group_ids = np.arange(unique_groups.shape[0])
    rng.shuffle(group_ids)
    n_val = max(1, int(round(unique_groups.shape[0] * val_frac)))
    n_val = min(n_val, unique_groups.shape[0] - 1) if unique_groups.shape[0] > 1 else 1
    val_group = np.zeros(unique_groups.shape[0], dtype=bool)
    val_group[group_ids[:n_val]] = True
    val_mask = val_group[inverse]
    train_refs = row_refs[~val_mask]
    val_refs = row_refs[val_mask]
    if train_refs.size == 0 or val_refs.size == 0:
        raise ValueError("Train/validation split produced an empty side.")
    split_info = {
        "n_groups": int(unique_groups.shape[0]),
        "n_train_groups": int((~val_group).sum()),
        "n_val_groups": int(val_group.sum()),
        "n_train_examples": int(train_refs.shape[0]),
        "n_val_examples": int(val_refs.shape[0]),
    }
    return train_refs, val_refs, split_info


def move_batch(batch: dict[str, th.Tensor], device: th.device) -> dict[str, th.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_loss(args: argparse.Namespace):
    if args.loss == "huber":
        return nn.SmoothL1Loss(beta=args.huber_beta, reduction="none")
    if args.loss == "mse":
        return nn.MSELoss(reduction="none")
    if args.loss == "mae":
        return nn.L1Loss(reduction="none")
    raise ValueError(f"Unknown loss: {args.loss}")


def weighted_loss(
    per_example_loss: th.Tensor,
    target: th.Tensor,
    penalty_threshold: float,
    penalty_weight: float,
) -> th.Tensor:
    if penalty_weight <= 1.0:
        return per_example_loss.mean()
    weights = th.ones_like(target)
    weights = th.where(target >= penalty_threshold, weights * penalty_weight, weights)
    return (per_example_loss * weights).mean()


def train_one_epoch(
    model: RiskSurrogate,
    loader: DataLoader,
    optimizer: th.optim.Optimizer,
    criterion: nn.Module,
    device: th.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_n = 0

    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["target_risk"]
        pred = model(batch)
        per_example = criterion(pred, target)
        loss = weighted_loss(
            per_example,
            target=target,
            penalty_threshold=args.penalty_threshold,
            penalty_weight=args.penalty_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        n = int(target.numel())
        total_n += n
        total_loss += float(loss.item()) * n
        err = pred.detach() - target
        total_abs += float(err.abs().sum().item())
        total_sq += float((err * err).sum().item())

    return {
        "loss": total_loss / max(total_n, 1),
        "mae": total_abs / max(total_n, 1),
        "mse": total_sq / max(total_n, 1),
        "rmse": math.sqrt(total_sq / max(total_n, 1)),
    }


@th.no_grad()
def evaluate(
    model: RiskSurrogate,
    loader: DataLoader,
    criterion: nn.Module,
    device: th.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    label_types: list[np.ndarray] = []
    agent_indices: list[np.ndarray] = []
    hazard_indices: list[np.ndarray] = []
    dataset_indices: list[np.ndarray] = []

    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["target_risk"]
        pred = model(batch)
        loss = criterion(pred, target)

        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())
        label_types.append(batch["label_type"].detach().cpu().numpy())
        agent_indices.append(batch["agent_index"].detach().cpu().numpy())
        hazard_indices.append(batch["hazard_index"].detach().cpu().numpy())
        dataset_indices.append(batch["dataset_index"].detach().cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    per_loss = np.concatenate(losses)
    labels = np.concatenate(label_types)
    agents = np.concatenate(agent_indices)
    hazards = np.concatenate(hazard_indices)
    dataset_ids = np.concatenate(dataset_indices)
    err = y_pred - y_true

    penalty_true = y_true >= args.penalty_threshold
    penalty_pred = y_pred >= args.penalty_threshold
    metrics: dict[str, Any] = {
        "loss": float(per_loss.mean()),
        "mae": float(np.abs(err).mean()),
        "mse": float(np.square(err).mean()),
        "rmse": float(np.sqrt(np.square(err).mean())),
        "pred_mean": float(y_pred.mean()),
        "target_mean": float(y_true.mean()),
        "pred_min": float(y_pred.min()),
        "pred_max": float(y_pred.max()),
        "target_min": float(y_true.min()),
        "target_max": float(y_true.max()),
        "penalty_frac": float(penalty_true.mean()),
        "penalty_accuracy": float((penalty_true == penalty_pred).mean()),
        "penalty_auroc": binary_auc(y_true=penalty_true, scores=y_pred),
    }
    metrics.update(
        ranking_metrics(
            y_true=y_true,
            y_pred=y_pred,
            label_type=labels,
            agent_index=agents,
            hazard_index=hazards,
            dataset_index=dataset_ids,
            k=args.ranking_k,
        )
    )
    return metrics


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = y_true.astype(bool, copy=False)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    pos_rank_sum = float(ranks[y_true].sum())
    auc = (pos_rank_sum - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_type: np.ndarray,
    agent_index: np.ndarray,
    hazard_index: np.ndarray,
    dataset_index: np.ndarray,
    k: int,
) -> dict[str, Any]:
    mask = label_type == LABEL_UNILATERAL
    if int(mask.sum()) == 0:
        return {
            "unilateral_rank_groups": 0,
            "spearman_mean": None,
            "risky_topk_recall": None,
            "safe_bottomk_recall": None,
        }

    group_keys = np.column_stack(
        [
            dataset_index[mask].astype(np.int64, copy=False),
            hazard_index[mask].astype(np.int64, copy=False),
            agent_index[mask].astype(np.int64, copy=False),
        ]
    )
    true_u = y_true[mask]
    pred_u = y_pred[mask]
    _, inverse = np.unique(group_keys, axis=0, return_inverse=True)

    spearman_values: list[float] = []
    risky_recalls: list[float] = []
    safe_recalls: list[float] = []
    n_groups = int(inverse.max()) + 1 if inverse.size else 0
    for group_id in range(n_groups):
        rows = np.flatnonzero(inverse == group_id)
        if rows.size < 2:
            continue
        target = true_u[rows]
        pred = pred_u[rows]
        true_rank = average_ranks(target)
        pred_rank = average_ranks(pred)
        if np.std(true_rank) > 0.0 and np.std(pred_rank) > 0.0:
            spearman_values.append(float(np.corrcoef(true_rank, pred_rank)[0, 1]))

        k_eff = min(max(k, 1), rows.size)
        true_risky = set(np.argsort(target)[-k_eff:])
        pred_risky = set(np.argsort(pred)[-k_eff:])
        true_safe = set(np.argsort(target)[:k_eff])
        pred_safe = set(np.argsort(pred)[:k_eff])
        risky_recalls.append(len(true_risky & pred_risky) / k_eff)
        safe_recalls.append(len(true_safe & pred_safe) / k_eff)

    return {
        "unilateral_rank_groups": int(n_groups),
        "spearman_mean": mean_or_none(spearman_values),
        "risky_topk_recall": mean_or_none(risky_recalls),
        "safe_bottomk_recall": mean_or_none(safe_recalls),
    }


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def choose_device(raw: str) -> th.device:
    if raw == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(raw)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    result["dataset"] = [str(path) for path in args.dataset]
    result["output_dir"] = str(args.output_dir)
    return result


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    if len(args.dataset) == 1:
        stem = args.dataset[0].stem
    else:
        stem = f"{len(args.dataset)}datasets"
    return f"{stem}_{args.label_types}_seed{args.seed}"


def init_wandb(args: argparse.Namespace, name: str, metadata: dict[str, Any], split: dict[str, int]):
    if not args.wandb:
        return None
    if args.wandb_mode:
        import os

        os.environ["WANDB_MODE"] = args.wandb_mode

    import wandb

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": name,
        "config": {
            **serializable_args(args),
            "agent_ids": metadata["agent_ids"],
            "n_actions_by_agent": metadata["n_actions_by_agent"],
            **{f"split/{key}": value for key, value in split.items()},
        },
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_group:
        init_kwargs["group"] = args.wandb_group
    return wandb.init(**init_kwargs)


def save_checkpoint(
    path: Path,
    model: RiskSurrogate,
    optimizer: th.optim.Optimizer,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    split: dict[str, int],
    epoch: int,
    best_metric: float,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": serializable_args(args),
        "metadata": metadata,
        "split": split,
        "epoch": int(epoch),
        "best_val_mae": float(best_metric),
        "metrics": metrics,
        "n_agents": int(len(metadata["agent_ids"])),
        "action_vocab_size": int(max(metadata["n_actions_by_agent"].values())),
        "graph_spec": metadata["state_graph_spec"],
    }
    th.save(payload, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    files = [load_risk_file(path) for path in args.dataset]
    validate_metadata(files)
    metadata = files[0].metadata
    row_refs, group_pairs = filtered_row_refs(files, args)
    train_refs, val_refs, split_info = split_by_hazard_group(
        row_refs=row_refs,
        group_pairs=group_pairs,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    name = run_name(args)
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RiskPriorDataset(files, train_refs)
    val_dataset = RiskPriorDataset(files, val_refs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=th.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=th.cuda.is_available(),
    )

    device = choose_device(args.device)
    n_agents = len(metadata["agent_ids"])
    action_vocab_size = int(max(metadata["n_actions_by_agent"].values()))
    model = RiskSurrogate(
        graph_spec=metadata["state_graph_spec"],
        n_agents=n_agents,
        action_vocab_size=action_vocab_size,
        args=args,
    ).to(device)
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = make_loss(args)

    summary = {
        "run_name": name,
        "device": str(device),
        "datasets": [str(path) for path in args.dataset],
        "n_loaded_examples": int(sum(len(item.target_risk) for item in files)),
        "n_selected_examples": int(row_refs.shape[0]),
        "label_types": args.label_types,
        **split_info,
    }
    write_json(output_dir / "train_config.json", {"args": serializable_args(args), **summary})
    print(json.dumps(summary, indent=2, sort_keys=True))

    wb_run = init_wandb(args, name=name, metadata=metadata, split=split_info)
    best_val_mae = float("inf")
    best_metrics: dict[str, Any] = {}
    best_checkpoint = output_dir / "best_surrogate.pt"
    last_checkpoint = output_dir / "last_surrogate.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            args=args,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
        )

        epoch_metrics = {
            "epoch": epoch,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
        }
        print(json.dumps(epoch_metrics, sort_keys=True))
        if wb_run is not None:
            wb_run.log(epoch_metrics, step=epoch)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = float(val_metrics["mae"])
            best_metrics = epoch_metrics
            save_checkpoint(
                path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                args=args,
                metadata=metadata,
                split=split_info,
                epoch=epoch,
                best_metric=best_val_mae,
                metrics=epoch_metrics,
            )

    final_val_metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        args=args,
    )
    final_metrics = {
        "best": best_metrics,
        "final": {f"val/{key}": value for key, value in final_val_metrics.items()},
        "summary": summary,
    }
    write_json(output_dir / "metrics.json", final_metrics)
    save_checkpoint(
        path=last_checkpoint,
        model=model,
        optimizer=optimizer,
        args=args,
        metadata=metadata,
        split=split_info,
        epoch=args.epochs,
        best_metric=best_val_mae,
        metrics=final_metrics,
    )

    if wb_run is not None:
        wb_run.summary["best_val_mae"] = best_val_mae
        wb_run.summary["best_checkpoint"] = str(best_checkpoint)
        if args.wandb_log_artifact:
            import wandb

            artifact = wandb.Artifact(name=name, type="risk_prior_surrogate")
            artifact.add_file(str(best_checkpoint))
            artifact.add_file(str(output_dir / "metrics.json"))
            artifact.add_file(str(output_dir / "train_config.json"))
            wb_run.log_artifact(artifact)
        wb_run.finish()

    print(f"Saved best checkpoint to {best_checkpoint}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
