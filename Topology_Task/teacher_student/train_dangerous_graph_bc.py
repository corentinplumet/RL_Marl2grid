#!/usr/bin/env python3
"""Fine-tune a graph candidate actor on dangerous-state action outcomes."""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch as th
import torch.nn.functional as F

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from common.utils import set_random_seed, str2bool
from env.eval import Evaluator
from full_test_eval.evaluate_checkpoint import (
    _as_namespace,
    _build_actors,
    _build_zero_shot_transfer_actors,
    _configure_legacy_connected_feature,
    _extract_obs_stats,
    _load_checkpoint,
    _merge_missing_defaults,
    _repo_relative,
    _resolve_checkpoint_path,
    _resolve_device,
)
from teacher_student.dangerous_graph_bc import (
    chronic_row_split,
    load_agent_batch,
    load_candidate_outcome_batch,
)
from teacher_student.dataset import (
    list_shards,
    load_metadata,
    make_joint_agent_minibatches,
    make_minibatches,
    resolve_dataset_dir,
)
from teacher_student.losses import (
    classification_counts,
    empty_classification_counts,
    empty_utility_ranking_counts,
    finalize_classification_counts,
    finalize_utility_ranking_counts,
    intervention_bce_loss,
    masked_centered_utility_huber_loss,
    merge_classification_counts,
    merge_utility_ranking_counts,
    observed_unsafe_action_loss,
    utility_ranking_counts,
    weighted_action_cross_entropy,
)
from teacher_student.scratch_initialization import (
    apply_scratch_architecture_overrides,
    build_scratch_actors,
    scratch_checkpoint_base,
)


def _unique_parameters(modules: Iterable[th.nn.Module]) -> List[th.nn.Parameter]:
    parameters: List[th.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def _to_tensor_obs(
    obs: Dict[str, Any], indices: np.ndarray, device: th.device
) -> Dict[str, Any]:
    return {
        "flat": th.as_tensor(obs["flat"][indices], dtype=th.float32, device=device),
        "graph": {
            key: th.as_tensor(value[indices], dtype=th.float32, device=device)
            for key, value in obs["graph"].items()
        },
    }


def _distillation_kl(
    student_logits: th.Tensor,
    reference_logits: th.Tensor,
    temperature: float,
) -> th.Tensor:
    temperature = max(float(temperature), 1e-6)
    reference_probs = F.softmax(reference_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, reference_probs, reduction="batchmean") * (
        temperature * temperature
    )


def _batch_losses(
    *,
    actor: Any,
    obs_np: Dict[str, Any],
    target_np: np.ndarray,
    reference_np: np.ndarray,
    candidate_np: Dict[str, np.ndarray] | None,
    indices: np.ndarray,
    device: th.device,
    cli: Namespace,
) -> tuple[
    th.Tensor,
    th.Tensor,
    th.Tensor,
    th.Tensor,
    th.Tensor,
    th.Tensor,
    int,
]:
    obs = _to_tensor_obs(obs_np, indices, device)
    target = th.as_tensor(target_np[indices], dtype=th.long, device=device)
    reference = th.as_tensor(reference_np[indices], dtype=th.float32, device=device)
    logits = actor._actor_logits(obs)
    zero = logits.sum() * 0.0
    if cli.objective == "hard_ce":
        action_loss = weighted_action_cross_entropy(
            logits,
            target,
            action0_weight=cli.action0_weight,
            nonidle_weight=cli.nonidle_weight,
        )
        utility_loss = zero
        unsafe_loss = zero
    else:
        if candidate_np is None:
            raise ValueError("utility_regression requires candidate outcomes.")
        utility = th.as_tensor(
            candidate_np["utility_vs_noop"][indices],
            dtype=th.float32,
            device=device,
        )
        trainable_mask = th.as_tensor(
            candidate_np["trainable_mask"][indices],
            dtype=th.bool,
            device=device,
        )
        observed_mask = th.as_tensor(
            candidate_np["observed_mask"][indices],
            dtype=th.bool,
            device=device,
        )
        action_loss = zero
        utility_loss = masked_centered_utility_huber_loss(
            logits,
            utility,
            trainable_mask,
            beta=cli.utility_huber_beta,
            min_candidates=cli.min_trainable_candidates,
        )
        unsafe_loss = observed_unsafe_action_loss(
            logits,
            utility,
            trainable_mask,
            observed_mask,
            margin=cli.unsafe_margin,
        )
    aux_loss = (
        intervention_bce_loss(logits, target, pos_weight=cli.aux_pos_weight)
        if cli.aux_intervention_loss
        else zero
    )
    distill_loss = (
        _distillation_kl(logits, reference, cli.distill_temperature)
        if cli.distill_weight > 0.0
        else zero
    )
    loss = (
        action_loss
        + cli.utility_weight * utility_loss
        + cli.unsafe_weight * unsafe_loss
        + cli.aux_weight * aux_loss
        + cli.distill_weight * distill_loss
    )
    return (
        loss,
        action_loss,
        utility_loss,
        unsafe_loss,
        aux_loss,
        distill_loss,
        int(target.numel()),
    )


def _accumulate_train_losses(
    values: Dict[str, float],
    *,
    loss: th.Tensor,
    action_loss: th.Tensor,
    utility_loss: th.Tensor,
    unsafe_loss: th.Tensor,
    aux_loss: th.Tensor,
    distill_loss: th.Tensor,
    n: int,
) -> None:
    weight = float(n)
    values["n"] += weight
    values["loss"] += float(loss.detach().cpu()) * weight
    values["action_loss"] += float(action_loss.detach().cpu()) * weight
    values["utility_loss"] += float(utility_loss.detach().cpu()) * weight
    values["unsafe_loss"] += float(unsafe_loss.detach().cpu()) * weight
    values["aux_loss"] += float(aux_loss.detach().cpu()) * weight
    values["distill_loss"] += float(distill_loss.detach().cpu()) * weight


def _resolve_output(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()
    if path.suffix != ".tar":
        path = path.with_suffix(".tar")
    return path


def _default_best_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_best{output_path.suffix}")


def _task_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()
    return path


def _atomic_save(record: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    try:
        th.save(record, temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metric_accumulator() -> Dict[str, float]:
    return {
        "n": 0.0,
        "loss": 0.0,
        "action_loss": 0.0,
        "utility_loss": 0.0,
        "unsafe_loss": 0.0,
        "aux_loss": 0.0,
        "distill_loss": 0.0,
    }


def _average_metrics(values: Dict[str, float]) -> Dict[str, float]:
    denominator = max(values["n"], 1.0)
    return {
        key: (value if key == "n" else value / denominator)
        for key, value in values.items()
    }


def _evaluate(
    actors: Dict[str, Any],
    shards: List[Path],
    agent_ids: List[str],
    device: th.device,
    batch_size: int,
    max_batches: int,
    cli: Namespace,
    row_indices_by_shard: Dict[Path, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    classification_sums = {agent: empty_classification_counts() for agent in agent_ids}
    ranking_sums = {agent: empty_utility_ranking_counts() for agent in agent_ids}
    seen = {agent: 0 for agent in agent_ids}
    unlimited = max_batches <= 0
    for actor in actors.values():
        actor.eval()
    with th.no_grad():
        for shard in shards:
            for agent in agent_ids:
                if not unlimited and seen[agent] >= max_batches:
                    continue
                obs_np, target_np, _ = load_agent_batch(shard, agent)
                indices_all = row_indices_by_shard[shard]
                candidate_np = None
                if cli.objective == "utility_regression":
                    candidate_np = load_candidate_outcome_batch(shard, agent)
                    candidate_mask = candidate_np["trainable_mask"][
                        indices_all
                    ] & np.isfinite(candidate_np["utility_vs_noop"][indices_all])
                    indices_all = indices_all[
                        candidate_mask.sum(axis=1) >= cli.min_trainable_candidates
                    ]
                for start in range(0, len(indices_all), batch_size):
                    if not unlimited and seen[agent] >= max_batches:
                        break
                    indices = indices_all[start : start + batch_size]
                    obs = _to_tensor_obs(obs_np, indices, device)
                    target = th.as_tensor(
                        target_np[indices], dtype=th.long, device=device
                    )
                    logits = actors[agent]._actor_logits(obs)
                    merge_classification_counts(
                        classification_sums[agent],
                        classification_counts(logits, target),
                    )
                    if candidate_np is not None:
                        utility = th.as_tensor(
                            candidate_np["utility_vs_noop"][indices],
                            dtype=th.float32,
                            device=device,
                        )
                        trainable_mask = th.as_tensor(
                            candidate_np["trainable_mask"][indices],
                            dtype=th.bool,
                            device=device,
                        )
                        observed_mask = th.as_tensor(
                            candidate_np["observed_mask"][indices],
                            dtype=th.bool,
                            device=device,
                        )
                        merge_utility_ranking_counts(
                            ranking_sums[agent],
                            utility_ranking_counts(
                                logits,
                                utility,
                                trainable_mask,
                                observed_mask,
                                min_candidates=cli.min_trainable_candidates,
                            ),
                        )
                    seen[agent] += 1
    for actor in actors.values():
        actor.train()
    metrics = {}
    for agent in agent_ids:
        metrics[agent] = finalize_classification_counts(classification_sums[agent])
        if cli.objective == "utility_regression":
            metrics[agent].update(finalize_utility_ranking_counts(ranking_sums[agent]))
    return metrics


def _print_eval_metrics(
    eval_metrics: Dict[str, Dict[str, float]],
    agent_ids: List[str],
    train_metrics: Dict[str, Dict[str, float]] | None = None,
) -> None:
    for agent in agent_ids:
        metrics = eval_metrics[agent]
        loss_prefix = (
            f"loss={train_metrics[agent]['loss']:.5f} "
            f"util={train_metrics[agent]['utility_loss']:.5f} "
            f"unsafe={train_metrics[agent]['unsafe_loss']:.5f} "
            if train_metrics is not None
            else ""
        )
        if "ranking_rows" in metrics:
            print(
                f"{agent}: {loss_prefix}"
                f"rank_top1={metrics.get('masked_top1_accuracy', float('nan')):.3f} "
                f"rank_top3={metrics.get('masked_top3_accuracy', float('nan')):.3f} "
                f"regret={metrics.get('masked_mean_regret', float('nan')):.5f} "
                f"deploy_safe={metrics.get('deployment_safe_rate', float('nan')):.3f} "
                f"deploy_best="
                f"{metrics.get('deployment_best_accuracy', float('nan')):.3f} "
                f"unsafe_outrank="
                f"{metrics.get('unsafe_outrank_rate', float('nan')):.3f} "
                f"hard_acc={metrics.get('accuracy', float('nan')):.3f} "
                f"rows={int(metrics.get('ranking_rows', 0))}",
                flush=True,
            )
        else:
            print(
                f"{agent}: {loss_prefix}"
                f"acc={metrics.get('accuracy', float('nan')):.3f} "
                f"a0_acc={metrics.get('action0_accuracy', float('nan')):.3f} "
                f"nonidle_acc={metrics.get('nonidle_accuracy', float('nan')):.3f} "
                f"nonidle_top3="
                f"{metrics.get('nonidle_top3_accuracy', float('nan')):.3f} "
                f"nonidle_top5="
                f"{metrics.get('nonidle_top5_accuracy', float('nan')):.3f} "
                f"nonidle_rank_top5="
                f"{metrics.get('nonidle_rank_top5_accuracy', float('nan')):.3f} "
                f"false_noop="
                f"{metrics.get('false_noop_rate', float('nan')):.3f} "
                f"false_intervention="
                f"{metrics.get('false_intervention_rate', float('nan')):.3f} "
                f"n/nonidle={int(metrics.get('n', 0))}/"
                f"{int(metrics.get('nonidle_n', 0))}",
                flush=True,
            )


def _eval_selection_score(
    eval_metrics: Dict[str, Dict[str, float]],
    agent_ids: List[str],
    objective: str,
) -> float:
    """Return the objective-specific score used to retain the best epoch."""
    if objective == "utility_regression":
        weighted_score = 0.0
        total_rows = 0.0
        for agent in agent_ids:
            metrics = eval_metrics[agent]
            rows = float(metrics.get("ranking_rows", 0.0))
            safe_rate = float(metrics.get("deployment_safe_rate", float("nan")))
            best_rate = float(metrics.get("deployment_best_accuracy", float("nan")))
            if rows > 0 and math.isfinite(safe_rate) and math.isfinite(best_rate):
                weighted_score += rows * (0.7 * best_rate + 0.3 * safe_rate)
                total_rows += rows
        return weighted_score / total_rows if total_rows > 0 else float("-inf")

    scores = []
    for agent in agent_ids:
        metrics = eval_metrics[agent]
        idle_accuracy = float(metrics.get("action0_accuracy", float("nan")))
        nonidle_accuracy = float(metrics.get("nonidle_accuracy", float("nan")))
        if math.isfinite(idle_accuracy) and math.isfinite(nonidle_accuracy):
            scores.append(0.5 * (idle_accuracy + nonidle_accuracy))
    return float(np.mean(scores)) if scores else float("-inf")


def _unshare_candidate_scorers(actors: Dict[str, Any]) -> int:
    """Clone learned scorer modules while keeping the graph encoder shared."""
    cloned = 0
    first = True
    for actor in actors.values():
        candidate_scorer = getattr(actor, "actor", None)
        if candidate_scorer is None or not hasattr(candidate_scorer, "scorer"):
            raise ValueError(
                "--unshare-candidate-scorer requires candidate-pool actors."
            )
        if first:
            first = False
            continue
        candidate_scorer.scorer = copy.deepcopy(candidate_scorer.scorer)
        if candidate_scorer.do_nothing_actor is not None:
            candidate_scorer.do_nothing_actor = copy.deepcopy(
                candidate_scorer.do_nothing_actor
            )
        candidate_scorer.do_nothing_logit_bias = th.nn.Parameter(
            candidate_scorer.do_nothing_logit_bias.detach().clone(),
            requires_grad=candidate_scorer.do_nothing_logit_bias.requires_grad,
        )
        cloned += 1
    return cloned


def _save_checkpoint(
    *,
    output_path: Path,
    source_record: Dict[str, Any],
    args: Namespace,
    actors: Dict[str, Any],
    optimizer: th.optim.Optimizer,
    summary: Dict[str, Any],
) -> None:
    record = dict(source_record)
    record["args"] = args
    for agent, actor in actors.items():
        record[agent] = actor.state_dict()
    record["actor_optim"] = optimizer.state_dict()
    record["wb_run_name"] = ""
    training_state = dict(record.get("training_state", {}) or {})
    training_state["dangerous_graph_bc"] = summary
    record["training_state"] = training_state
    record["dangerous_graph_bc"] = summary
    record["resume_state"] = None
    record["checkpoint_boundary"] = {
        "phase": "supervised_finetune",
        "exact": False,
        "completed_rollout": int(record.get("last_rollout", 0)),
        "next_rollout": int(record.get("last_rollout", 0)) + 1,
    }
    _atomic_save(record, output_path)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Warm-start checkpoint, or architecture-template checkpoint when "
            "--initialization scratch. Scratch mode never loads its learned "
            "actor, critic, optimizer, or normalization state."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=TASK_DIR / "checkpoint")
    parser.add_argument(
        "--initialization",
        choices=["warm_start", "scratch"],
        default="warm_start",
        help=(
            "warm_start transfers the checkpoint actor weights; scratch uses "
            "the checkpoint only as an architecture template and initializes "
            "fresh actors directly on the dataset environment/action space."
        ),
    )
    parser.add_argument(
        "--scratch-gnn-layers",
        type=int,
        default=None,
        help=(
            "Override gnn_layers while constructing a fresh actor. Valid only "
            "with --initialization scratch; all other architecture settings "
            "remain those of the template checkpoint."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--best-output",
        type=Path,
        default=None,
        help=(
            "Checkpoint receiving the best macro balanced-accuracy epoch. "
            "Defaults to <output_stem>_best.tar."
        ),
    )
    parser.add_argument("--exp-tag", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--objective",
        choices=["hard_ce", "utility_regression"],
        default="hard_ce",
        help=(
            "hard_ce reproduces one-hot behavior cloning; utility_regression "
            "fits all safe candidate utilities and penalizes observed unsafe actions."
        ),
    )
    parser.add_argument("--utility-weight", type=float, default=1.0)
    parser.add_argument("--utility-huber-beta", type=float, default=0.5)
    parser.add_argument("--unsafe-weight", type=float, default=0.5)
    parser.add_argument("--unsafe-margin", type=float, default=1.0)
    parser.add_argument("--min-trainable-candidates", type=int, default=2)
    parser.add_argument("--balanced-nonidle-frac", type=float, default=0.20)
    parser.add_argument("--action0-weight", type=float, default=1.0)
    parser.add_argument("--nonidle-weight", type=float, default=3.0)
    parser.add_argument("--aux-intervention-loss", type=str2bool, default=True)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--aux-pos-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=0.10)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--freeze-encoder", type=str2bool, default=False)
    parser.add_argument(
        "--unshare-candidate-scorer",
        type=str2bool,
        default=False,
        help=(
            "Clone the transferred scorer for each target agent while keeping "
            "the graph encoder shared. This is a diagnostic ablation."
        ),
    )
    parser.add_argument(
        "--agent-update-mode",
        choices=["mixed", "sequential"],
        default="mixed",
        help=(
            "mixed accumulates one minibatch loss per agent before each shared "
            "optimizer step; sequential reproduces the legacy ordering."
        ),
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--validation-chronic-frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of unique chronic fingerprints held out for model selection. "
            "Zero evaluates on the training rows for an intentional capacity test."
        ),
    )
    parser.add_argument("--validation-seed", type=int, default=1701)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="Evaluation batches per agent; use 0 to evaluate every selected shard.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--transfer-action-head-source-agent", default="agent_0")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.epochs <= 0 or cli.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")
    if not 0.0 <= cli.balanced_nonidle_frac <= 1.0:
        raise ValueError("--balanced-nonidle-frac must be in [0, 1].")
    if not 0.0 <= cli.validation_chronic_frac < 1.0:
        raise ValueError("--validation-chronic-frac must be in [0, 1).")
    if cli.min_trainable_candidates < 2:
        raise ValueError("--min-trainable-candidates must be at least 2.")
    if (
        cli.utility_weight < 0.0
        or cli.utility_huber_beta <= 0.0
        or cli.unsafe_weight < 0.0
        or cli.unsafe_margin < 0.0
    ):
        raise ValueError("Invalid utility-regression loss weight or scale.")
    if cli.distill_weight < 0.0 or cli.distill_temperature <= 0.0:
        raise ValueError("Invalid distillation weight or temperature.")
    if cli.initialization == "scratch" and cli.distill_weight > 0.0:
        raise ValueError(
            "Scratch initialization cannot use reference-policy distillation: "
            "set --distill-weight 0. A nonzero KL weight would reintroduce the "
            "source policy into the scratch control."
        )

    dataset_dir = resolve_dataset_dir(cli.dataset)
    metadata = load_metadata(dataset_dir)
    if metadata.get("dataset_mode") != "dangerous_graph_bc":
        raise ValueError("Dataset is not a dangerous_graph_bc dataset.")
    if cli.objective == "utility_regression" and not bool(
        metadata.get("has_candidate_outcomes", False)
    ):
        raise ValueError(
            "utility_regression requires a dataset collected with "
            "--store-candidate-outcomes true."
        )
    shards = list_shards(dataset_dir, cli.max_shards)
    agent_ids = list(metadata["agent_ids"])
    (
        train_rows_by_shard,
        validation_rows_by_shard,
        n_train_chronics,
        n_validation_chronics,
    ) = chronic_row_split(shards, cli.validation_chronic_frac, cli.validation_seed)

    checkpoint_dir = cli.checkpoint_dir.expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (TASK_DIR / checkpoint_dir).resolve()
    source = cli.checkpoint or metadata.get("checkpoint")
    if not source:
        raise ValueError(
            "Provide --checkpoint to select the warm-start weights or, with "
            "--initialization scratch, the architecture template. Checkpoint-free "
            "collection datasets intentionally do not choose an architecture."
        )
    checkpoint_path = _resolve_checkpoint_path(str(source), checkpoint_dir)
    output_path = _resolve_output(cli.output)
    best_output_path = (
        _resolve_output(cli.best_output)
        if cli.best_output is not None
        else _default_best_output(output_path)
    )
    if best_output_path == output_path:
        raise ValueError("--best-output must differ from --output.")

    dataset_checkpoint_raw = metadata.get("checkpoint")
    dataset_checkpoint_value = (
        str(dataset_checkpoint_raw) if dataset_checkpoint_raw else ""
    )
    if cli.distill_weight > 0.0:
        if not dataset_checkpoint_value or not bool(
            metadata.get("has_policy_logits", True)
        ):
            raise ValueError(
                "The dataset has no genuine source-policy logits, so "
                "reference-policy KL cannot be used. Set --distill-weight 0."
            )
        dataset_checkpoint = _resolve_checkpoint_path(
            dataset_checkpoint_value, checkpoint_dir
        )
        if dataset_checkpoint.resolve() != checkpoint_path.resolve():
            raise ValueError(
                "The dataset policy logits were generated by "
                f"{dataset_checkpoint}, but --checkpoint selects {checkpoint_path}. "
                "Hard action labels can be shared across compatible models; "
                "reference-policy KL cannot. Set --distill-weight 0 or collect a "
                "dataset with this checkpoint."
            )

    cpu_record = _load_checkpoint(checkpoint_path, th.device("cpu"))
    args = _merge_missing_defaults(_as_namespace(cpu_record["args"]))
    args = _configure_legacy_connected_feature(args, cpu_record)
    scratch_architecture_overrides = apply_scratch_architecture_overrides(
        args,
        initialization=cli.initialization,
        gnn_layers=cli.scratch_gnn_layers,
    )
    source_env_id = str(getattr(args, "env_id", ""))
    dataset_env_id = str(metadata.get("env_id", "") or "")
    if not dataset_env_id:
        raise ValueError("Dataset metadata has no env_id.")
    cross_grid_transfer = source_env_id != dataset_env_id
    checkpoint_action_space = str(getattr(args, "reduced_action_space", "") or "")
    dataset_action_space = str(metadata.get("reduced_action_space", "") or "")
    if not dataset_action_space:
        raise ValueError("Dataset metadata has no reduced_action_space path.")
    if not _task_path(dataset_action_space).exists():
        raise FileNotFoundError(
            f"Dataset action-space file does not exist: "
            f"{_task_path(dataset_action_space)}"
        )
    args.env_id = dataset_env_id
    args.reduced_action_space = dataset_action_space
    if cross_grid_transfer:
        # Bus14 flat-observation statistics are not meaningful on WCCI. The
        # transferable candidate actors use graph-only inputs
        # (gnn_concat_flat=false) with deterministic physical scaling.
        args.norm_obs = False
    args.track = False
    args.eval_action_heuristic = "none"
    args.deterministic_eval = True
    args.n_threads = int(cli.n_threads)
    args.exp_tag = cli.exp_tag or output_path.stem
    args.seed = int(cli.seed)
    args.resume_run_name = None
    args.resume_wandb_id = None
    args.resume_wandb_name = None
    if (
        str(getattr(args, "actor_encoder", "")) != "gnn"
        or str(getattr(args, "actor_action_head", "")) != "candidate_pool"
    ):
        raise ValueError("Dangerous graph BC requires a GNN candidate-pool checkpoint.")

    set_random_seed(cli.seed)
    device = _resolve_device(args, cli.device)
    if cli.initialization == "scratch":
        source_record = scratch_checkpoint_base(args)
    else:
        source_record = _load_checkpoint(checkpoint_path, device)
        if cross_grid_transfer:
            source_record = dict(source_record)
            training_state = dict(source_record.get("training_state", {}) or {})
            training_state.pop("obs_stats", None)
            source_record["training_state"] = training_state
    evaluator = Evaluator(args, logger=None, device=device, chronic_split="train")
    obs_stats = (
        {}
        if cli.initialization == "scratch" or cross_grid_transfer
        else _extract_obs_stats(source_record)
    )
    if obs_stats:
        evaluator.env.env.set_obs_stats(obs_stats)
    same_action_space = bool(
        checkpoint_action_space
        and _task_path(checkpoint_action_space) == _task_path(dataset_action_space)
    )
    if cli.initialization == "scratch":
        actors = build_scratch_actors(args, evaluator, device)
    elif same_action_space:
        actors = _build_actors(source_record, args, evaluator, device)
    else:
        transferable = {
            "share_actor_gnn": bool(getattr(args, "share_actor_gnn", False)),
            "share_candidate_scorer": bool(
                getattr(args, "share_candidate_scorer", False)
            ),
            "gnn_concat_flat=false": not bool(getattr(args, "gnn_concat_flat", False)),
        }
        missing_transfer = [name for name, ok in transferable.items() if not ok]
        if missing_transfer:
            raise ValueError(
                "Training on another action-space size requires a transferable "
                "shared actor; missing: " + ", ".join(missing_transfer)
            )
        actors, _ = _build_zero_shot_transfer_actors(
            checkpoint_path,
            args,
            evaluator,
            device,
            cli.transfer_action_head_source_agent,
        )
        for actor in actors.values():
            for parameter in actor.parameters():
                parameter.requires_grad_(True)
    unshared_scorers = 0
    if cli.unshare_candidate_scorer:
        unshared_scorers = _unshare_candidate_scorers(actors)
        # The saved checkpoint must rebuild independent scorers; otherwise its
        # per-agent states would all be loaded into one shared module at eval.
        args.share_candidate_scorer = False
    if sorted(actors) != sorted(agent_ids):
        raise ValueError(
            f"Dataset agents {agent_ids} do not match checkpoint actors "
            f"{sorted(actors)}"
        )
    action_sizes = {
        agent: int(evaluator.env.env.action_space[agent].n) for agent in agent_ids
    }
    dataset_action_sizes = {
        key: int(value) for key, value in metadata["action_sizes"].items()
    }
    if action_sizes != dataset_action_sizes:
        raise ValueError(
            f"Dataset action sizes {metadata['action_sizes']} do not match "
            f"checkpoint {action_sizes}"
        )
    schema_fields = (
        "actor_encoder",
        "actor_action_head",
        "gnn_graph_type",
        "gnn_physical_scaling",
        "gnn_running_norm",
    )
    schema_mismatches = {
        field: (metadata.get(field), getattr(args, field, None))
        for field in schema_fields
        if metadata.get(field) != getattr(args, field, None)
    }
    if schema_mismatches:
        raise ValueError(
            "Dataset and checkpoint graph schemas differ: "
            + ", ".join(
                f"{field}={dataset_value!r}/{checkpoint_value!r}"
                for field, (
                    dataset_value,
                    checkpoint_value,
                ) in schema_mismatches.items()
            )
        )

    if cli.freeze_encoder:
        seen_encoders: set[int] = set()
        for actor in actors.values():
            encoder = actor.encoder.graph_encoder
            if id(encoder) in seen_encoders:
                continue
            seen_encoders.add(id(encoder))
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
    for actor in actors.values():
        actor.train()
    parameters = _unique_parameters(actors.values())
    optimizer = th.optim.Adam(
        parameters, lr=cli.lr, eps=1e-5, weight_decay=cli.weight_decay
    )
    rng = np.random.default_rng(cli.seed)

    print("========== Dangerous-state graph BC training ==========")
    print(f"Dataset: {dataset_dir}")
    print(f"Initialization: {cli.initialization}")
    if cli.initialization == "scratch":
        print(f"Architecture template: {_repo_relative(checkpoint_path)}")
        print("Learned template weights loaded: False")
        if scratch_architecture_overrides:
            layer_override = scratch_architecture_overrides["gnn_layers"]
            print(
                "GNN message-passing layers: "
                f"{layer_override['template']} -> {layer_override['target']}"
            )
    else:
        print(f"Checkpoint: {_repo_relative(checkpoint_path)}")
        print("Learned template weights loaded: True")
    print(f"Architecture environment: {source_env_id} -> {dataset_env_id}")
    print(f"Output: {output_path}")
    print(f"Best output: {best_output_path}")
    print(f"Shards / epochs: {len(shards)} / {cli.epochs}")
    print(f"Objective: {cli.objective}")
    print(f"Batch / LR: {cli.batch_size} / {cli.lr}")
    print(
        "Chronic split train/validation: "
        f"{n_train_chronics}/{n_validation_chronics} "
        f"(validation_fraction={cli.validation_chronic_frac})"
    )
    if cli.objective == "hard_ce":
        print(f"Balanced nonidle: {cli.balanced_nonidle_frac}")
        print(
            "CE weights action0/nonidle: " f"{cli.action0_weight}/{cli.nonidle_weight}"
        )
    else:
        print(
            "Utility/unsafe weights: "
            f"{cli.utility_weight}/{cli.unsafe_weight} "
            f"huber_beta={cli.utility_huber_beta} "
            f"unsafe_margin={cli.unsafe_margin}"
        )
        print(f"Minimum trainable candidates: {cli.min_trainable_candidates}")
    print(f"Aux intervention weight: {cli.aux_weight}")
    print(f"Reference-policy KL weight: {cli.distill_weight}")
    print(f"Freeze encoder: {cli.freeze_encoder}")
    print(f"Agent update mode: {cli.agent_update_mode}")
    print(
        "Candidate scorer sharing: "
        f"{'unshared diagnostic' if cli.unshare_candidate_scorer else 'shared'}"
    )
    if cli.unshare_candidate_scorer:
        print(f"Cloned target scorers: {unshared_scorers}")
    print(f"Unique trainable parameters: {sum(p.numel() for p in parameters)}")
    print("========================================================")

    optimizer_steps = 0
    baseline_label = (
        "random initialization baseline"
        if cli.initialization == "scratch"
        else "source baseline"
    )
    print(f"========== epoch 0 {baseline_label} ==========", flush=True)
    baseline_metrics = _evaluate(
        actors,
        shards,
        agent_ids,
        device,
        cli.batch_size,
        cli.eval_batches,
        cli,
        validation_rows_by_shard,
    )
    _print_eval_metrics(baseline_metrics, agent_ids)
    history: List[Dict[str, Any]] = [
        {
            "epoch": 0,
            "optimizer_steps": 0,
            "train": None,
            "eval": baseline_metrics,
        }
    ]
    best_score = float("-inf")
    best_epoch: int | None = None
    try:
        for epoch in range(1, cli.epochs + 1):
            sums = {agent: _metric_accumulator() for agent in agent_ids}
            epoch_shards = list(shards)
            rng.shuffle(epoch_shards)
            for shard_index, shard in enumerate(epoch_shards, start=1):
                arrays = {}
                eligible_indices = {}
                base_indices = train_rows_by_shard[shard]
                for agent in agent_ids:
                    obs_np, target_np, reference_np = load_agent_batch(shard, agent)
                    candidate_np = (
                        load_candidate_outcome_batch(shard, agent)
                        if cli.objective == "utility_regression"
                        else None
                    )
                    indices = base_indices
                    if candidate_np is not None:
                        candidate_mask = candidate_np["trainable_mask"][
                            indices
                        ] & np.isfinite(candidate_np["utility_vs_noop"][indices])
                        indices = indices[
                            candidate_mask.sum(axis=1) >= cli.min_trainable_candidates
                        ]
                    arrays[agent] = (
                        obs_np,
                        target_np,
                        reference_np,
                        candidate_np,
                    )
                    eligible_indices[agent] = indices
                if cli.agent_update_mode == "mixed":
                    sampling_targets = {
                        agent: arrays[agent][1][eligible_indices[agent]]
                        for agent in agent_ids
                        if len(eligible_indices[agent]) > 0
                    }
                    for joint_local_indices in make_joint_agent_minibatches(
                        sampling_targets,
                        cli.batch_size,
                        rng,
                        balanced_nonidle_frac=(
                            cli.balanced_nonidle_frac
                            if cli.objective == "hard_ce"
                            else 0.0
                        ),
                    ):
                        optimizer.zero_grad(set_to_none=True)
                        active_agents = len(joint_local_indices)
                        for agent, local_indices in joint_local_indices.items():
                            indices = eligible_indices[agent][local_indices]
                            (
                                obs_np,
                                target_np,
                                reference_np,
                                candidate_np,
                            ) = arrays[agent]
                            (
                                loss,
                                action_loss,
                                utility_loss,
                                unsafe_loss,
                                aux_loss,
                                distill_loss,
                                n,
                            ) = _batch_losses(
                                actor=actors[agent],
                                obs_np=obs_np,
                                target_np=target_np,
                                reference_np=reference_np,
                                candidate_np=candidate_np,
                                indices=indices,
                                device=device,
                                cli=cli,
                            )
                            (loss / max(active_agents, 1)).backward()
                            _accumulate_train_losses(
                                sums[agent],
                                loss=loss,
                                action_loss=action_loss,
                                utility_loss=utility_loss,
                                unsafe_loss=unsafe_loss,
                                aux_loss=aux_loss,
                                distill_loss=distill_loss,
                                n=n,
                            )
                        if cli.max_grad_norm > 0:
                            th.nn.utils.clip_grad_norm_(parameters, cli.max_grad_norm)
                        optimizer.step()
                        optimizer_steps += 1
                else:
                    for agent in agent_ids:
                        (
                            obs_np,
                            target_np,
                            reference_np,
                            candidate_np,
                        ) = arrays[agent]
                        available = eligible_indices[agent]
                        for local_indices in make_minibatches(
                            target_np[available],
                            cli.batch_size,
                            rng,
                            balanced_nonidle_frac=(
                                cli.balanced_nonidle_frac
                                if cli.objective == "hard_ce"
                                else 0.0
                            ),
                        ):
                            indices = available[local_indices]
                            (
                                loss,
                                action_loss,
                                utility_loss,
                                unsafe_loss,
                                aux_loss,
                                distill_loss,
                                n,
                            ) = _batch_losses(
                                actor=actors[agent],
                                obs_np=obs_np,
                                target_np=target_np,
                                reference_np=reference_np,
                                candidate_np=candidate_np,
                                indices=indices,
                                device=device,
                                cli=cli,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            if cli.max_grad_norm > 0:
                                th.nn.utils.clip_grad_norm_(
                                    parameters, cli.max_grad_norm
                                )
                            optimizer.step()
                            optimizer_steps += 1
                            _accumulate_train_losses(
                                sums[agent],
                                loss=loss,
                                action_loss=action_loss,
                                utility_loss=utility_loss,
                                unsafe_loss=unsafe_loss,
                                aux_loss=aux_loss,
                                distill_loss=distill_loss,
                                n=n,
                            )

                if shard_index % max(cli.progress_every, 1) == 0 or shard_index == len(
                    epoch_shards
                ):
                    print(
                        f"epoch={epoch}/{cli.epochs} shard={shard_index}/{len(epoch_shards)} "
                        f"optimizer_steps={optimizer_steps}",
                        flush=True,
                    )

            train_metrics = {
                agent: _average_metrics(sums[agent]) for agent in agent_ids
            }
            eval_metrics = _evaluate(
                actors,
                shards,
                agent_ids,
                device,
                cli.batch_size,
                cli.eval_batches,
                cli,
                validation_rows_by_shard,
            )
            epoch_record = {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "train": train_metrics,
                "eval": eval_metrics,
                "selection_score": _eval_selection_score(
                    eval_metrics, agent_ids, cli.objective
                ),
            }
            history.append(epoch_record)
            _print_eval_metrics(eval_metrics, agent_ids, train_metrics)
            selection_score = float(epoch_record["selection_score"])
            is_best = selection_score > best_score
            if is_best:
                best_score = selection_score
                best_epoch = epoch

            summary = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "trainer": "teacher_student.train_dangerous_graph_bc",
                "dataset": str(dataset_dir),
                "initialization": cli.initialization,
                "learned_source_weights_loaded": cli.initialization == "warm_start",
                "architecture_template_checkpoint": str(checkpoint_path),
                "architecture_template_global_step": int(
                    cpu_record.get("global_step", 0)
                ),
                "scratch_architecture_overrides": scratch_architecture_overrides,
                "source_checkpoint": (
                    str(checkpoint_path) if cli.initialization == "warm_start" else None
                ),
                "source_checkpoint_global_step": (
                    int(cpu_record.get("global_step", 0))
                    if cli.initialization == "warm_start"
                    else None
                ),
                "source_env_id": source_env_id,
                "target_env_id": dataset_env_id,
                "cross_grid_actor_transfer": bool(
                    cross_grid_transfer and cli.initialization == "warm_start"
                ),
                "source_obs_stats_discarded": bool(
                    cross_grid_transfer or cli.initialization == "scratch"
                ),
                "agent_update_mode": cli.agent_update_mode,
                "candidate_scorer_unshared": bool(cli.unshare_candidate_scorer),
                "selection_metric": (
                    "row_weighted_0.7_deployment_best_plus_0.3_deployment_safe"
                    if cli.objective == "utility_regression"
                    else "macro_mean_0.5_action0_plus_0.5_nonidle_accuracy"
                ),
                "selection_score": selection_score,
                "best_selection_score": best_score,
                "best_epoch": best_epoch,
                "objective": cli.objective,
                "objective_components": (
                    "centered_standardized_utility_huber_plus_observed_unsafe_"
                    "margin_plus_intervention_bce"
                    if cli.objective == "utility_regression"
                    else "balanced_weighted_ce_plus_intervention_bce"
                ),
                "n_train_chronics": n_train_chronics,
                "n_validation_chronics": n_validation_chronics,
                "args": vars(cli),
                "history": history,
                "optimizer_steps": optimizer_steps,
            }
            _save_checkpoint(
                output_path=output_path,
                source_record=source_record,
                args=args,
                actors=actors,
                optimizer=optimizer,
                summary=summary,
            )
            print(f"saved {output_path}", flush=True)
            if is_best:
                _save_checkpoint(
                    output_path=best_output_path,
                    source_record=source_record,
                    args=args,
                    actors=actors,
                    optimizer=optimizer,
                    summary=summary,
                )
                print(
                    f"saved best {best_output_path} "
                    f"epoch={best_epoch} score={best_score:.6f}",
                    flush=True,
                )
    finally:
        evaluator.env.close()

    print("========== Graph BC complete ==========")
    print(f"Output: {output_path}")
    print(f"Best output: {best_output_path}")
    print(f"Best epoch / score: {best_epoch} / {best_score:.6f}")
    print(f"Optimizer steps: {optimizer_steps}")


if __name__ == "__main__":
    main()
