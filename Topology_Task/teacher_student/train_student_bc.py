#!/usr/bin/env python3
"""Train a behavior-cloned student actor from a teacher-student dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th
from gymnasium.spaces import Box, Discrete

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from alg.mappo.agent import Actor
from common.utils import set_random_seed, set_torch, str2bool
from full_test_eval.evaluate_checkpoint import (
    _as_namespace,
    _extract_obs_stats,
    _load_checkpoint,
    _merge_missing_defaults,
    _repo_relative,
    _resolve_checkpoint_path,
)
from teacher_student.dataset import (
    list_shards,
    load_agent_arrays,
    load_agent_policy_logits,
    load_agent_was_overwritten,
    load_metadata,
    make_minibatches,
    resolve_dataset_dir,
    shard_has_policy_logits,
    task_relative,
)
from teacher_student.losses import (
    classification_counts,
    empty_classification_counts,
    finalize_classification_counts,
    intervention_bce_loss,
    merge_classification_counts,
    soft_label_kl_loss,
    weighted_action_cross_entropy,
)


class DummyActorEnv:
    """Minimal env facade needed by Actor for flat MLP policies."""

    def __init__(
        self,
        agent_ids: List[str],
        obs_shapes: Dict[str, List[int]],
        action_sizes: Dict[str, int],
    ) -> None:
        self.observation_space = {
            agent: Box(
                low=-np.inf,
                high=np.inf,
                shape=tuple(obs_shapes[agent]),
                dtype=np.float32,
            )
            for agent in agent_ids
        }
        self.action_space = {
            agent: Discrete(int(action_sizes[agent])) for agent in agent_ids
        }
        self.graph_specs = None


def _actor_logits(actor: Actor, obs: th.Tensor) -> th.Tensor:
    if getattr(actor, "intervention_gate", False):
        raise NotImplementedError(
            "Phase 3 BC training currently supports flat non-gated actors only."
        )
    if getattr(actor, "encoder_type", "mlp") != "mlp":
        raise NotImplementedError(
            "Phase 3 BC training currently supports actor_encoder='mlp' only."
        )
    return actor.actor(actor._encode(obs))


def _resolve_output(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()
    if path.suffix != ".tar":
        path = path.with_suffix(".tar")
    return path


def _resolve_teacher_checkpoint(
    teacher_checkpoint: Optional[str],
    metadata: Dict[str, Any],
    checkpoint_dir: Path,
) -> Path:
    candidate = teacher_checkpoint or str(metadata.get("checkpoint", ""))
    if not candidate:
        raise ValueError(
            "No --teacher-checkpoint was provided and metadata.json does not "
            "contain a checkpoint path."
        )
    return _resolve_checkpoint_path(candidate, checkpoint_dir)


def _build_students(
    args: Namespace,
    metadata: Dict[str, Any],
    record: Dict[str, Any],
    device: th.device,
    init_from_teacher: bool,
) -> Dict[str, Actor]:
    agent_ids = list(metadata["agent_ids"])
    if getattr(args, "actor_encoder", "mlp") != "mlp":
        raise NotImplementedError(
            "Phase 3 BC training currently supports only actor_encoder='mlp'."
        )
    if bool(getattr(args, "intervention_gate", False)):
        raise NotImplementedError(
            "Phase 3 BC training currently supports only non-gated actors."
        )
    env = DummyActorEnv(agent_ids, metadata["obs_shapes"], metadata["action_sizes"])
    continuous_actions = getattr(args, "action_type", "topology") == "redispatch"
    actors: Dict[str, Actor] = {}
    for idx, agent in enumerate(agent_ids):
        actor = Actor(idx, env, args, continuous_actions).to(device)
        if init_from_teacher:
            actor.load_state_dict(record[agent])
        actor.train()
        actors[agent] = actor
    return actors


def _merge_metric_sum(dst: Dict[str, float], src: Dict[str, float]) -> None:
    n = float(src.get("n", 0.0))
    dst["n"] = dst.get("n", 0.0) + n
    for key, value in src.items():
        if key == "n":
            continue
        dst[key] = dst.get(key, 0.0) + float(value) * n


def _finalize_metric_sum(src: Dict[str, float]) -> Dict[str, float]:
    n = max(float(src.get("n", 0.0)), 1.0)
    return {
        key: (float(value) / n if key != "n" else float(value))
        for key, value in src.items()
    }


def evaluate_students(
    actors: Dict[str, Actor],
    shards: List[Path],
    agent_ids: List[str],
    device: th.device,
    batch_size: int,
    max_batches_per_agent: int,
) -> Dict[str, Dict[str, float]]:
    metric_sums = {agent: empty_classification_counts() for agent in agent_ids}
    batches_seen = {agent: 0 for agent in agent_ids}
    for actor in actors.values():
        actor.eval()
    with th.no_grad():
        for shard in shards:
            if all(batches_seen[agent] >= max_batches_per_agent for agent in agent_ids):
                break
            for agent in agent_ids:
                if batches_seen[agent] >= max_batches_per_agent:
                    continue
                obs_np, target_np = load_agent_arrays(shard, agent)
                for start in range(0, obs_np.shape[0], batch_size):
                    if batches_seen[agent] >= max_batches_per_agent:
                        break
                    obs = th.as_tensor(
                        obs_np[start : start + batch_size],
                        dtype=th.float32,
                        device=device,
                    )
                    target = th.as_tensor(
                        target_np[start : start + batch_size],
                        dtype=th.long,
                        device=device,
                    )
                    logits = _actor_logits(actors[agent], obs)
                    merge_classification_counts(
                        metric_sums[agent],
                        classification_counts(logits, target),
                    )
                    batches_seen[agent] += 1
    for actor in actors.values():
        actor.train()
    return {
        agent: finalize_classification_counts(metric_sums[agent]) for agent in agent_ids
    }


def save_student_checkpoint(
    output_path: Path,
    *,
    teacher_record: Dict[str, Any],
    args: Namespace,
    actors: Dict[str, Actor],
    obs_stats: Dict[str, Any],
    train_summary: Dict[str, Any],
    optimizer_steps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {
        "args": args,
        "global_step": int(optimizer_steps),
        "wb_run_name": "",
        "last_rollout": 0,
        "teacher_student": train_summary,
        "training_state": {
            "obs_stats": obs_stats,
            "teacher_student": train_summary,
        },
    }
    for agent, actor in actors.items():
        record[agent] = actor.state_dict()
    for key in ("critic", "critic_optim"):
        if key in teacher_record:
            record[key] = teacher_record[key]
    th.save(record, output_path)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=TASK_DIR / "checkpoint",
        help="Directory used to resolve relative teacher checkpoint stems.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exp-tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--action0-weight", type=float, default=1.0)
    parser.add_argument("--nonidle-weight", type=float, default=5.0)
    parser.add_argument("--balanced-nonidle-frac", type=float, default=0.5)
    parser.add_argument("--aux-intervention-loss", type=str2bool, default=True)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--aux-pos-weight", type=float, default=1.0)
    parser.add_argument(
        "--soft-distillation-loss",
        type=str2bool,
        default=False,
        help=(
            "Add KL distillation from saved teacher policy logits. Requires "
            "datasets collected with --save-policy-logits true."
        ),
    )
    parser.add_argument("--soft-weight", type=float, default=1.0)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--init-from-teacher", type=str2bool, default=True)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 <= cli.balanced_nonidle_frac <= 1.0:
        raise ValueError("--balanced-nonidle-frac must be in [0, 1].")
    if cli.soft_temperature <= 0.0:
        raise ValueError("--soft-temperature must be positive.")
    if cli.soft_distillation_loss and cli.soft_weight <= 0.0:
        raise ValueError(
            "--soft-weight must be positive when soft distillation is enabled."
        )

    dataset_dir = resolve_dataset_dir(cli.dataset)
    metadata = load_metadata(dataset_dir)
    shards = list_shards(dataset_dir, cli.max_shards)
    agent_ids = list(metadata["agent_ids"])
    if cli.soft_distillation_loss:
        missing = [
            f"{task_relative(shard)}:{agent}"
            for shard in shards
            for agent in agent_ids
            if not shard_has_policy_logits(shard, agent)
        ]
        if missing:
            sample = "\n".join(missing[:10])
            raise FileNotFoundError(
                "Soft-label distillation requires policy logits in every shard. "
                "Recollect with --save-policy-logits true. Missing examples:\n"
                f"{sample}"
            )
    checkpoint_dir = cli.checkpoint_dir.expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (TASK_DIR / checkpoint_dir).resolve()
    teacher_checkpoint = _resolve_teacher_checkpoint(
        cli.teacher_checkpoint, metadata, checkpoint_dir
    )
    output_path = _resolve_output(cli.output)

    teacher_record = _load_checkpoint(teacher_checkpoint, th.device("cpu"))
    teacher_args = _merge_missing_defaults(_as_namespace(teacher_record["args"]))
    teacher_args.track = False
    teacher_args.eval_action_heuristic = "none"
    teacher_args.deterministic_eval = True
    teacher_args.exp_tag = cli.exp_tag or output_path.stem
    teacher_args.resume_run_name = None
    teacher_args.resume_wandb_id = None
    teacher_args.resume_wandb_name = None
    obs_stats = _extract_obs_stats(teacher_record)

    if cli.device == "cpu":
        device = set_torch(
            cli.n_threads, getattr(teacher_args, "th_deterministic", False), cuda=False
        )
    elif cli.device == "cuda":
        device = set_torch(
            cli.n_threads, getattr(teacher_args, "th_deterministic", False), cuda=True
        )
        if device.type != "cuda":
            raise RuntimeError(
                "--device cuda was requested, but CUDA is not available."
            )
    elif cli.device == "mps":
        if not th.backends.mps.is_available():
            raise RuntimeError("--device mps was requested, but MPS is not available.")
        th.set_num_threads(cli.n_threads)
        device = th.device("mps")
    else:
        device = set_torch(
            cli.n_threads,
            getattr(teacher_args, "th_deterministic", False),
            cuda=bool(getattr(teacher_args, "cuda", False)),
        )
    set_random_seed(cli.seed)
    rng = np.random.default_rng(cli.seed)

    actors = _build_students(
        teacher_args,
        metadata,
        teacher_record,
        device,
        init_from_teacher=cli.init_from_teacher,
    )
    optimizers = {
        agent: th.optim.Adam(
            actors[agent].parameters(),
            lr=cli.lr,
            eps=1e-5,
            weight_decay=cli.weight_decay,
        )
        for agent in agent_ids
    }

    print("========== Teacher-student BC training ==========")
    print(f"Dataset: {task_relative(dataset_dir)}")
    print(f"Teacher checkpoint: {_repo_relative(teacher_checkpoint)}")
    print(f"Output: {task_relative(output_path)}")
    print(f"Agents: {', '.join(agent_ids)}")
    print(f"Shards: {len(shards)}")
    print(f"Epochs: {cli.epochs}")
    print(f"Batch size: {cli.batch_size}")
    print(f"Balanced nonidle frac: {cli.balanced_nonidle_frac}")
    print(f"CE weights: action0={cli.action0_weight} nonidle={cli.nonidle_weight}")
    print(f"Aux intervention loss: {cli.aux_intervention_loss} weight={cli.aux_weight}")
    print(
        "Soft distillation loss: "
        f"{cli.soft_distillation_loss} weight={cli.soft_weight} "
        f"temperature={cli.soft_temperature}"
    )
    print(f"Init from teacher: {cli.init_from_teacher}")
    print(f"Device: {device}")
    print("=================================================")

    optimizer_steps = 0
    history = []
    for epoch in range(1, cli.epochs + 1):
        epoch_metric_sums = {
            agent: {
                "n": 0.0,
                "loss": 0.0,
                "action_loss": 0.0,
                "aux_loss": 0.0,
                "soft_loss": 0.0,
            }
            for agent in agent_ids
        }
        epoch_shards = list(shards)
        rng.shuffle(epoch_shards)
        for shard_idx, shard in enumerate(epoch_shards, start=1):
            for agent in agent_ids:
                obs_np, target_np = load_agent_arrays(shard, agent)
                policy_logits_np = (
                    load_agent_policy_logits(shard, agent)
                    if cli.soft_distillation_loss
                    else None
                )
                was_overwritten_np = (
                    load_agent_was_overwritten(shard, agent)
                    if cli.soft_distillation_loss
                    else None
                )
                actor = actors[agent]
                optimizer = optimizers[agent]
                for batch_idx in make_minibatches(
                    target_np,
                    cli.batch_size,
                    rng,
                    balanced_nonidle_frac=cli.balanced_nonidle_frac,
                ):
                    obs = th.as_tensor(
                        obs_np[batch_idx],
                        dtype=th.float32,
                        device=device,
                    )
                    target = th.as_tensor(
                        target_np[batch_idx],
                        dtype=th.long,
                        device=device,
                    )
                    logits = _actor_logits(actor, obs)
                    action_loss = weighted_action_cross_entropy(
                        logits,
                        target,
                        action0_weight=cli.action0_weight,
                        nonidle_weight=cli.nonidle_weight,
                    )
                    aux_loss = (
                        intervention_bce_loss(
                            logits,
                            target,
                            pos_weight=cli.aux_pos_weight,
                        )
                        if cli.aux_intervention_loss
                        else logits.sum() * 0.0
                    )
                    if cli.soft_distillation_loss:
                        assert policy_logits_np is not None
                        assert was_overwritten_np is not None
                        teacher_policy_logits = th.as_tensor(
                            policy_logits_np[batch_idx],
                            dtype=th.float32,
                            device=device,
                        )
                        was_overwritten = th.as_tensor(
                            was_overwritten_np[batch_idx],
                            dtype=th.bool,
                            device=device,
                        )
                        soft_loss = soft_label_kl_loss(
                            logits,
                            teacher_policy_logits,
                            target,
                            was_overwritten,
                            temperature=cli.soft_temperature,
                        )
                    else:
                        soft_loss = logits.sum() * 0.0
                    loss = (
                        action_loss
                        + float(cli.aux_weight) * aux_loss
                        + float(cli.soft_weight) * soft_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if cli.max_grad_norm > 0.0:
                        th.nn.utils.clip_grad_norm_(
                            actor.parameters(), cli.max_grad_norm
                        )
                    optimizer.step()
                    optimizer_steps += 1

                    n = float(target.numel())
                    sums = epoch_metric_sums[agent]
                    sums["n"] += n
                    sums["loss"] += float(loss.detach().cpu().item()) * n
                    sums["action_loss"] += float(action_loss.detach().cpu().item()) * n
                    sums["aux_loss"] += float(aux_loss.detach().cpu().item()) * n
                    sums["soft_loss"] += float(soft_loss.detach().cpu().item()) * n

            if shard_idx % max(cli.progress_every, 1) == 0 or shard_idx == len(
                epoch_shards
            ):
                print(
                    f"epoch {epoch}/{cli.epochs} shard {shard_idx}/{len(epoch_shards)} "
                    f"optimizer_steps={optimizer_steps}",
                    flush=True,
                )

        train_metrics = {
            agent: _finalize_metric_sum(epoch_metric_sums[agent]) for agent in agent_ids
        }
        eval_metrics = evaluate_students(
            actors,
            shards,
            agent_ids,
            device,
            batch_size=cli.batch_size,
            max_batches_per_agent=cli.eval_batches,
        )
        epoch_record = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        history.append(epoch_record)

        print(f"========== epoch {epoch} summary ==========")
        for agent in agent_ids:
            t = train_metrics[agent]
            e = eval_metrics[agent]
            print(
                f"{agent}: train_loss={t['loss']:.6f} "
                f"eval_acc={e['accuracy']:.4f} "
                f"eval_a0_acc={e['action0_accuracy']:.4f} "
                f"eval_nonidle_acc={e['nonidle_accuracy']:.4f} "
                f"eval_pred_nonidle={e['pred_nonidle_frac']:.4f} "
                f"eval_false_noop={e['false_noop_rate']:.4f} "
                f"eval_false_intervention={e['false_intervention_rate']:.4f}"
            )

        train_summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trainer": "teacher_student.train_student_bc",
            "phase": 3,
            "dataset": str(dataset_dir),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_global_step": int(teacher_record.get("global_step", 0)),
            "dataset_metadata": {
                "split": metadata.get("split"),
                "eval_action_heuristic": metadata.get("eval_action_heuristic"),
                "eval_action_rho_threshold": metadata.get("eval_action_rho_threshold"),
                "n_env_steps": metadata.get("n_env_steps"),
                "n_agent_examples": metadata.get("n_agent_examples"),
                "n_completed_episodes": metadata.get("n_completed_episodes"),
                "n_unique_chronic_fingerprints": metadata.get(
                    "n_unique_chronic_fingerprints"
                ),
            },
            "student_training_objective": (
                "weighted_ce_plus_intervention_bce_plus_soft_kl"
                if cli.soft_distillation_loss and cli.aux_intervention_loss
                else (
                    "weighted_ce_plus_soft_kl"
                    if cli.soft_distillation_loss
                    else (
                        "weighted_ce_plus_intervention_bce"
                        if cli.aux_intervention_loss
                        else "weighted_ce"
                    )
                )
            ),
            "args": vars(cli),
            "history": history,
        }
        save_student_checkpoint(
            output_path,
            teacher_record=teacher_record,
            args=teacher_args,
            actors=actors,
            obs_stats=obs_stats,
            train_summary=train_summary,
            optimizer_steps=optimizer_steps,
        )
        print(f"saved checkpoint: {task_relative(output_path)}", flush=True)

    print("========== BC training complete ==========")
    print(f"Output checkpoint: {task_relative(output_path)}")
    print(f"Optimizer steps: {optimizer_steps}")


if __name__ == "__main__":
    main()
