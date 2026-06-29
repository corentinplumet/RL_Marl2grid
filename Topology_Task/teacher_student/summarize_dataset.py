#!/usr/bin/env python3
"""Summarize and sanity-check a teacher-student dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from teacher_student.dataset import (
    list_shards,
    load_metadata,
    metadata_dir,
    summary_path,
    task_relative,
)


def _resolve_dataset(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {path}")
    return path


def _safe_path(path: Path) -> str:
    return task_relative(path)


def _load_metadata(dataset_dir: Path) -> Dict[str, Any]:
    return load_metadata(dataset_dir)


def _metric_template() -> Dict[str, int]:
    return {
        "n": 0,
        "policy_action_0": 0,
        "policy_nonidle": 0,
        "teacher_action_0": 0,
        "teacher_nonidle": 0,
        "force_noop": 0,
        "was_overwritten": 0,
        "overwrite_to_action0": 0,
        "obs_nan_or_inf": 0,
    }


def _add_action_metrics(
    metrics: Dict[str, int],
    policy_action: np.ndarray,
    teacher_action: np.ndarray,
    force_noop: np.ndarray,
    obs: np.ndarray,
) -> None:
    n = int(teacher_action.size)
    overwritten = policy_action != teacher_action
    metrics["n"] += n
    metrics["policy_action_0"] += int(np.sum(policy_action == 0))
    metrics["policy_nonidle"] += int(np.sum(policy_action != 0))
    metrics["teacher_action_0"] += int(np.sum(teacher_action == 0))
    metrics["teacher_nonidle"] += int(np.sum(teacher_action != 0))
    metrics["force_noop"] += int(np.sum(force_noop))
    metrics["was_overwritten"] += int(np.sum(overwritten))
    metrics["overwrite_to_action0"] += int(np.sum(overwritten & (teacher_action == 0)))
    metrics["obs_nan_or_inf"] += int(np.sum(~np.isfinite(obs)))


def _frac(num: int, den: int) -> float:
    return float(num) / float(max(den, 1))


def summarize(dataset_dir: Path) -> Dict[str, Any]:
    metadata = _load_metadata(dataset_dir)
    agent_ids: List[str] = list(metadata["agent_ids"])
    shards = list_shards(dataset_dir)

    agent_metrics = {agent: _metric_template() for agent in agent_ids}
    action_counts = {
        agent: defaultdict(int)
        for agent in agent_ids
    }
    unique_fingerprints = set()
    n_env_steps = 0
    n_done = 0
    episode_lengths = []
    bad_shards = []

    for shard in shards:
        try:
            with np.load(shard) as data:
                n = int(data["dataset_step"].size)
                n_env_steps += n
                n_done += int(np.sum(data["done"]))
                unique_fingerprints.update(map(str, data["chronic_fingerprint"]))
                episode_lengths.extend(
                    int(x) for x in data["done_episode_length"] if int(x) > 0
                )
                for agent in agent_ids:
                    policy = data[f"policy_action_{agent}"]
                    teacher = data[f"teacher_action_{agent}"]
                    force = data[f"force_noop_{agent}"]
                    obs = data[f"obs_{agent}"]
                    _add_action_metrics(agent_metrics[agent], policy, teacher, force, obs)
                    for action_id, count in zip(*np.unique(teacher, return_counts=True)):
                        action_counts[agent][int(action_id)] += int(count)
        except Exception as exc:
            bad_shards.append({"path": _safe_path(shard), "error": repr(exc)})

    agent_summary = {}
    for agent, metrics in agent_metrics.items():
        n = metrics["n"]
        top_actions = sorted(
            action_counts[agent].items(), key=lambda item: item[1], reverse=True
        )[:10]
        agent_summary[agent] = {
            **metrics,
            "policy_action_0_frac": _frac(metrics["policy_action_0"], n),
            "policy_nonidle_frac": _frac(metrics["policy_nonidle"], n),
            "teacher_action_0_frac": _frac(metrics["teacher_action_0"], n),
            "teacher_nonidle_frac": _frac(metrics["teacher_nonidle"], n),
            "force_noop_frac": _frac(metrics["force_noop"], n),
            "was_overwritten_frac": _frac(metrics["was_overwritten"], n),
            "overwrite_to_action0_frac": _frac(metrics["overwrite_to_action0"], n),
            "top_teacher_actions": [
                {"action": action, "count": count, "frac": _frac(count, n)}
                for action, count in top_actions
            ],
        }

    return {
        "dataset": str(dataset_dir),
        "metadata_status": metadata.get("status"),
        "checkpoint": metadata.get("checkpoint"),
        "checkpoint_global_step": metadata.get("checkpoint_global_step"),
        "split": metadata.get("split"),
        "eval_action_heuristic": metadata.get("eval_action_heuristic"),
        "eval_action_rho_threshold": metadata.get("eval_action_rho_threshold"),
        "n_shards": len(shards),
        "n_bad_shards": len(bad_shards),
        "bad_shards": bad_shards,
        "n_env_steps": n_env_steps,
        "n_agent_examples": n_env_steps * len(agent_ids),
        "n_completed_episodes": n_done,
        "n_unique_chronic_fingerprints": len(unique_fingerprints),
        "episode_length_mean": (
            float(np.mean(episode_lengths)) if episode_lengths else None
        ),
        "episode_length_min": int(min(episode_lengths)) if episode_lengths else None,
        "episode_length_max": int(max(episode_lengths)) if episode_lengths else None,
        "agent_summary": agent_summary,
    }


def _print_table(summary: Dict[str, Any]) -> None:
    print("========== Teacher dataset summary ==========")
    print(f"Dataset: {_safe_path(Path(summary['dataset']))}")
    print(f"Status: {summary['metadata_status']}")
    print(f"Checkpoint step: {summary['checkpoint_global_step']}")
    print(f"Split: {summary['split']}")
    print(
        "Heuristic: "
        f"{summary['eval_action_heuristic']} "
        f"rho={summary['eval_action_rho_threshold']}"
    )
    print(f"Shards: {summary['n_shards']} bad={summary['n_bad_shards']}")
    print(f"Env steps: {summary['n_env_steps']}")
    print(f"Agent examples: {summary['n_agent_examples']}")
    print(f"Completed episodes: {summary['n_completed_episodes']}")
    print(f"Unique fingerprints: {summary['n_unique_chronic_fingerprints']}")
    print(
        "Episode length: "
        f"mean={summary['episode_length_mean']} "
        f"min={summary['episode_length_min']} "
        f"max={summary['episode_length_max']}"
    )
    print()
    headers = [
        "agent",
        "n",
        "teacher_a0",
        "teacher_nonidle",
        "policy_nonidle",
        "overwrite",
        "force_noop",
        "obs_bad",
    ]
    rows = []
    for agent, metrics in summary["agent_summary"].items():
        rows.append(
            [
                agent,
                str(metrics["n"]),
                f"{metrics['teacher_action_0_frac']:.4f}",
                f"{metrics['teacher_nonidle_frac']:.4f}",
                f"{metrics['policy_nonidle_frac']:.4f}",
                f"{metrics['was_overwritten_frac']:.4f}",
                f"{metrics['force_noop_frac']:.4f}",
                str(metrics["obs_nan_or_inf"]),
            ]
        )
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    print("  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("=============================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional summary JSON path. Defaults to "
            "DATASET/metadata/summary.json for split-layout datasets."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve_dataset(args.dataset)
    summary = summarize(dataset_dir)
    _print_table(summary)
    metadata_dir(dataset_dir).mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or summary_path(dataset_dir)
    output_json = output_json.expanduser()
    if not output_json.is_absolute():
        output_json = (TASK_DIR / output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved summary: {_safe_path(output_json)}")


if __name__ == "__main__":
    main()
