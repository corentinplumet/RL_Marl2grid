#!/usr/bin/env python3
"""Evaluate a saved MAPPO checkpoint on a full chronic split."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import torch as th


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(TASK_DIR))
    except ValueError:
        return str(path.resolve())


def _as_namespace(value: Any) -> Namespace:
    if isinstance(value, Namespace):
        return value
    if isinstance(value, dict):
        return Namespace(**value)
    raise TypeError(f"Checkpoint args must be a Namespace or dict, got {type(value)!r}.")


def str2bool(s: str) -> bool:
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _merge_missing_defaults(args: Namespace) -> Namespace:
    """Add new code defaults to older checkpoints without changing saved values."""

    from alg.mappo.config import get_alg_args
    from env.config import get_env_args

    defaults: Dict[str, Any] = {
        "alg": "MAPPO",
        "seed": 0,
        "track": False,
        "verbose": True,
        "th_deterministic": False,
        "cuda": True,
        "n_threads": 4,
    }
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        for namespace in (get_env_args(), get_alg_args()):
            defaults.update(vars(namespace))
    finally:
        sys.argv = original_argv
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _resolve_checkpoint_path(path_or_stem: str, checkpoint_dir: Path) -> Path:
    candidate = Path(path_or_stem).expanduser()
    if candidate.suffix != ".tar":
        candidate = candidate.with_suffix(".tar")

    if candidate.is_absolute() or candidate.parent != Path("."):
        if candidate.exists():
            return candidate.resolve()
        relative_to_task = (TASK_DIR / candidate).resolve()
        if relative_to_task.exists():
            return relative_to_task
        raise FileNotFoundError(f"Could not find checkpoint {path_or_stem!r}.")

    resolved = (checkpoint_dir / candidate.name).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint {candidate.name!r} in {_repo_relative(checkpoint_dir)}."
        )
    return resolved


def _checkpoint_global_step(path: Path, device: str = "cpu") -> Optional[int]:
    try:
        record = th.load(path, map_location=device, weights_only=False)
    except Exception:
        return None
    step = record.get("global_step")
    return None if step is None else int(step)


def _summary_matches(summary: Dict[str, Any], model: str) -> bool:
    haystack = " ".join(
        str(summary.get(key, ""))
        for key in ("file", "exp_tag", "wb_run_name")
    )
    return model in haystack


def _candidate_score(
    path: Path,
    summary: Dict[str, Any],
    model: str,
    step: Optional[int],
) -> Tuple[int, float]:
    name = path.stem
    exp_tag = str(summary.get("exp_tag", ""))
    wb_run_name = str(summary.get("wb_run_name", ""))
    score = 0
    if name == model:
        score += 80
    if exp_tag == model:
        score += 80
    if name.endswith(model):
        score += 50
    if exp_tag.endswith(model):
        score += 50
    if model in name:
        score += 25
    if model in exp_tag:
        score += 25
    if model in wb_run_name:
        score += 10
    if name.startswith("best_test_"):
        score += 10
    if name.startswith("final_"):
        score += 5
    if step is not None and str(step) in name:
        score += 20
    return score, path.stat().st_mtime


def _format_matches(paths: Iterable[Path], max_items: int = 20) -> str:
    lines = []
    for path in list(paths)[:max_items]:
        lines.append(f"  - {_repo_relative(path)}")
    return "\n".join(lines)


def _find_checkpoint_by_model_step(
    model: str,
    step: Optional[int],
    checkpoint_dir: Path,
) -> Path:
    candidates = []
    failures = []
    for path in sorted(checkpoint_dir.glob("*.tar")):
        try:
            summary = _checkpoint_summary(path)
        except Exception as exc:
            failures.append((path, exc))
            continue
        if _summary_matches(summary, model):
            candidates.append((path, summary))
    if not candidates:
        failure_msg = ""
        if failures:
            failure_msg = f" ({len(failures)} unreadable checkpoints skipped)"
        raise FileNotFoundError(
            f"No .tar checkpoint in {_repo_relative(checkpoint_dir)} contains "
            f"model {model!r} in filename, exp_tag, or wb_run_name{failure_msg}."
        )

    if step is not None:
        exact_step_matches = []
        for path, summary in candidates:
            if summary["global_step"] == step:
                exact_step_matches.append(path)
        if len(exact_step_matches) == 1:
            return exact_step_matches[0].resolve()
        if len(exact_step_matches) > 1:
            raise RuntimeError(
                "Multiple checkpoints matched the requested global step. "
                "Use --checkpoint to disambiguate:\n"
                + _format_matches(exact_step_matches)
            )

        filename_matches = [path for path, _ in candidates if str(step) in path.stem]
        if len(filename_matches) == 1:
            return filename_matches[0].resolve()
        if len(filename_matches) > 1:
            raise RuntimeError(
                "Multiple checkpoint filenames matched the requested step. "
                "Use --checkpoint to disambiguate:\n"
                + _format_matches(filename_matches)
            )

        raise FileNotFoundError(
            f"No checkpoint for model {model!r} matched global_step={step}. "
            "Use --checkpoint with the exact file if this is a best/final checkpoint."
        )

    ranked = sorted(
        candidates,
        key=lambda item: _candidate_score(item[0], item[1], model, step=None),
        reverse=True,
    )
    if len(ranked) > 1:
        best_score = _candidate_score(ranked[0][0], ranked[0][1], model, None)[0]
        tied = [
            path
            for path, summary in ranked
            if _candidate_score(path, summary, model, None)[0] == best_score
        ]
        if len(tied) > 1:
            raise RuntimeError(
                "Multiple checkpoints matched this model. Use --step or --checkpoint:\n"
                + _format_matches(tied)
            )
    return ranked[0][0].resolve()


def _checkpoint_summary(path: Path) -> Dict[str, Any]:
    record = th.load(path, map_location="cpu", weights_only=False)
    args = _as_namespace(record.get("args", {}))
    return {
        "file": _repo_relative(path),
        "global_step": record.get("global_step", None),
        "wb_run_name": record.get("wb_run_name", ""),
        "exp_tag": getattr(args, "exp_tag", ""),
        "total_timesteps": getattr(args, "total_timesteps", ""),
        "env_id": getattr(args, "env_id", ""),
        "seed": getattr(args, "seed", ""),
        "actor_encoder": getattr(args, "actor_encoder", ""),
        "gnn_type": getattr(args, "gnn_type", ""),
        "share_actor_gnn": getattr(args, "share_actor_gnn", ""),
        "intervention_gate": getattr(args, "intervention_gate", ""),
        "intervention_gate_eval_mode": getattr(args, "intervention_gate_eval_mode", ""),
        "eval_action_heuristic": getattr(args, "eval_action_heuristic", ""),
        "eval_action_rho_threshold": getattr(args, "eval_action_rho_threshold", ""),
    }


def _print_checkpoint_table(checkpoint_dir: Path, model_filter: Optional[str]) -> None:
    paths = sorted(checkpoint_dir.glob("*.tar"))
    if not paths:
        filter_msg = f" matching {model_filter!r}" if model_filter else ""
        print(f"No checkpoints{filter_msg} found in {_repo_relative(checkpoint_dir)}.")
        return

    rows = []
    failures = []
    for path in paths:
        try:
            summary = _checkpoint_summary(path)
            if model_filter and not _summary_matches(summary, model_filter):
                continue
            rows.append(summary)
        except Exception as exc:
            failures.append((path, exc))

    if not rows:
        filter_msg = f" matching {model_filter!r}" if model_filter else ""
        print(
            f"No readable checkpoints{filter_msg} found in "
            f"{_repo_relative(checkpoint_dir)}."
        )
        if failures:
            print(f"{len(failures)} checkpoint(s) could not be read.")
        return

    rows.sort(
        key=lambda row: (
            -1 if row["global_step"] is None else int(row["global_step"]),
            row["file"],
        ),
        reverse=True,
    )
    headers = [
        "global_step",
        "total",
        "tag",
        "file",
        "env_id",
        "seed",
        "actor",
        "gate",
        "heuristic",
        "rho",
    ]
    table = []
    for row in rows:
        actor = row["actor_encoder"]
        if row["actor_encoder"] == "gnn":
            actor = f"gnn/{row['gnn_type']}"
            if row["share_actor_gnn"]:
                actor += "/shared"
        gate = "yes" if row["intervention_gate"] else "no"
        if row["intervention_gate_eval_mode"]:
            gate += f":{row['intervention_gate_eval_mode']}"
        table.append(
            [
                str(row["global_step"]),
                str(row["total_timesteps"]),
                str(row["exp_tag"]),
                row["file"],
                str(row["env_id"]),
                str(row["seed"]),
                str(actor),
                gate,
                str(row["eval_action_heuristic"] or "none"),
                str(row["eval_action_rho_threshold"]),
            ]
        )

    widths = [
        max(len(headers[idx]), *(len(line[idx]) for line in table))
        for idx in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for line in table:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(line)))

    if failures:
        print("\nUnreadable checkpoints:")
        for path, exc in failures:
            print(f"  - {_repo_relative(path)}: {exc}")


def _load_checkpoint(path: Path, device: th.device) -> Dict[str, Any]:
    record = th.load(path, map_location=device, weights_only=False)
    required = {"args", "global_step"}
    missing = sorted(required.difference(record.keys()))
    if missing:
        raise KeyError(f"Checkpoint {_repo_relative(path)} is missing keys: {missing}")
    return record


def _resolve_device(args: Namespace, requested: str) -> th.device:
    from common.utils import set_torch

    if requested == "cpu":
        args.cuda = False
        return set_torch(args.n_threads, args.th_deterministic, cuda=False)
    if requested == "cuda":
        if not th.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        args.cuda = True
        return set_torch(args.n_threads, args.th_deterministic, cuda=True)
    if requested == "mps":
        if not th.backends.mps.is_available():
            raise RuntimeError("--device mps was requested, but MPS is not available.")
        th.set_num_threads(args.n_threads)
        th.backends.cudnn.deterministic = bool(args.th_deterministic)
        return th.device("mps")
    return set_torch(args.n_threads, args.th_deterministic, cuda=bool(args.cuda))


def _apply_eval_overrides(args: Namespace, cli: Namespace) -> Namespace:
    args.track = False
    args.split_chronics = bool(cli.split_chronics)
    args.eval_all_split_chronics = bool(cli.eval_all_split_chronics)
    args.deterministic_eval = bool(cli.deterministic_eval)
    args.trace_rollout_actions = False

    if cli.eval_episodes is not None:
        args.eval_episodes = int(cli.eval_episodes)

    if cli.eval_action_heuristic != "checkpoint":
        args.eval_action_heuristic = cli.eval_action_heuristic
    if cli.eval_action_rho_threshold is not None:
        args.eval_action_rho_threshold = float(cli.eval_action_rho_threshold)

    if cli.intervention_gate_eval_mode != "checkpoint":
        args.intervention_gate_eval_mode = cli.intervention_gate_eval_mode

    if cli.n_threads is not None:
        args.n_threads = int(cli.n_threads)
    args.n_threads = max(int(getattr(args, "n_threads", 1)), 1)
    return args


def _build_actors(record: Dict[str, Any], args: Namespace, evaluator: Any, device: th.device) -> Dict[str, Any]:
    from alg.mappo.agent import Actor
    from alg.mappo.core import _build_shared_actor_graph_encoder

    actor_env = evaluator.env.env
    agent_ids = [f"agent_{idx}" for idx in range(len(actor_env.observation_space.keys()))]
    continuous_actions = getattr(args, "action_type", "topology") == "redispatch"
    shared_encoder = _build_shared_actor_graph_encoder(actor_env, args, agent_ids)

    actors: Dict[str, Actor] = {}
    for idx, agent_id in enumerate(agent_ids):
        if agent_id not in record:
            raise KeyError(f"Checkpoint is missing actor state for {agent_id}.")
        actor = Actor(
            idx,
            actor_env,
            args,
            continuous_actions=continuous_actions,
            shared_graph_encoder=shared_encoder,
        ).to(device)
        actor.load_state_dict(record[agent_id])
        actor.eval()
        actors[agent_id] = actor
    return actors


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "checkpoint"


def _default_output_json(checkpoint_path: Path, model: Optional[str], step: int) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    label = _safe_name(model or checkpoint_path.stem)
    output_dir = TASK_DIR / "outputs" / "full_test_eval"
    return output_dir / f"{label}_step{step}_job{job_id}.json"


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MAPPO checkpoint on the full test split."
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List readable checkpoints in --checkpoint-dir and exit.",
    )
    selector = parser.add_mutually_exclusive_group(required=False)
    selector.add_argument(
        "--checkpoint",
        type=str,
        help="Exact checkpoint path, or checkpoint stem inside --checkpoint-dir.",
    )
    selector.add_argument(
        "--model",
        type=str,
        help="Model/run-name substring used to find a checkpoint in --checkpoint-dir.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Requested checkpoint global_step when using --model.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=TASK_DIR / "checkpoint",
        help="Directory containing .tar checkpoints.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Chronic split to evaluate.",
    )
    parser.add_argument(
        "--split-chronics",
        type=str2bool,
        default=True,
        help="Enable train/test chronic splitting before evaluation.",
    )
    parser.add_argument(
        "--eval-all-split-chronics",
        type=str2bool,
        default=True,
        help="Evaluate every chronic in the selected split when splitting is enabled.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override number of evaluated episodes; by default the full split is used.",
    )
    parser.add_argument(
        "--deterministic-eval",
        type=str2bool,
        default=True,
        help="Use greedy actions during evaluation.",
    )
    parser.add_argument(
        "--eval-action-heuristic",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "none", "rho_threshold", "local_rho_threshold"],
        help="Keep or override the checkpoint's evaluation action heuristic.",
    )
    parser.add_argument(
        "--eval-action-rho-threshold",
        type=float,
        default=None,
        help="Override the rho threshold used by evaluation heuristics.",
    )
    parser.add_argument(
        "--intervention-gate-eval-mode",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "final_action_map", "hierarchical_greedy"],
        help="Keep or override deterministic eval mode for gated actors.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Override Torch thread count.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the JSON result summary.",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.eval_episodes is not None and cli.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive when provided.")
    if cli.n_threads is not None and cli.n_threads <= 0:
        raise ValueError("--n-threads must be positive when provided.")

    checkpoint_dir = cli.checkpoint_dir.expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (TASK_DIR / checkpoint_dir).resolve()

    if cli.list_checkpoints:
        _print_checkpoint_table(checkpoint_dir, cli.model)
        return
    if not cli.checkpoint and not cli.model:
        raise ValueError("Pass --checkpoint, --model, or --list-checkpoints.")

    from common.utils import set_random_seed
    from env.eval import Evaluator

    if cli.checkpoint:
        checkpoint_path = _resolve_checkpoint_path(cli.checkpoint, checkpoint_dir)
    else:
        checkpoint_path = _find_checkpoint_by_model_step(
            cli.model, cli.step, checkpoint_dir
        )

    first_record = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = _merge_missing_defaults(_as_namespace(first_record["args"]))
    args = _apply_eval_overrides(args, cli)
    set_random_seed(getattr(args, "seed", 0))
    device = _resolve_device(args, cli.device)
    record = _load_checkpoint(checkpoint_path, device)
    checkpoint_step = int(record.get("global_step", 0))
    if cli.checkpoint and cli.step is not None and checkpoint_step != cli.step:
        raise ValueError(
            f"--step {cli.step} was requested, but checkpoint "
            f"{_repo_relative(checkpoint_path)} has global_step={checkpoint_step}."
        )

    evaluator = Evaluator(
        args,
        logger=None,
        device=device,
        chronic_split=cli.split,
        metric_prefix=cli.split,
    )
    actors = _build_actors(record, args, evaluator, device)

    print(f"Checkpoint: {_repo_relative(checkpoint_path)}")
    print(f"Checkpoint global_step: {checkpoint_step}")
    print(f"Evaluating split: {cli.split}")
    print(f"Evaluation episodes: {cli.eval_episodes or evaluator.eval_episodes}")
    print(f"Device: {device}")
    print(f"Eval heuristic: {getattr(args, 'eval_action_heuristic', 'none')}")
    if getattr(args, "eval_action_heuristic", "none") != "none":
        print(f"Eval rho threshold: {getattr(args, 'eval_action_rho_threshold', 0.90)}")

    survival = evaluator.evaluate(checkpoint_step, actors, eval_ep=cli.eval_episodes)

    output_json = cli.output_json or _default_output_json(
        checkpoint_path, cli.model, checkpoint_step
    )
    if not output_json.is_absolute():
        output_json = (TASK_DIR / output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": checkpoint_step,
        "requested_model": cli.model,
        "requested_step": cli.step,
        "split": cli.split,
        "split_chronics": bool(args.split_chronics),
        "eval_all_split_chronics": bool(args.eval_all_split_chronics),
        "eval_episodes": int(cli.eval_episodes or evaluator.eval_episodes),
        "deterministic_eval": bool(args.deterministic_eval),
        "eval_action_heuristic": getattr(args, "eval_action_heuristic", "none"),
        "eval_action_rho_threshold": float(
            getattr(args, "eval_action_rho_threshold", 0.90)
        ),
        "intervention_gate": bool(getattr(args, "intervention_gate", False)),
        "intervention_gate_eval_mode": getattr(
            args, "intervention_gate_eval_mode", "final_action_map"
        ),
        "survival_frac": float(survival),
        "survival_percent": float(100.0 * survival),
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Survival fraction: {survival:.6f}")
    print(f"Survival percent: {100.0 * survival:.3f}%")
    print(f"Saved summary: {_repo_relative(output_json)}")


if __name__ == "__main__":
    main()
