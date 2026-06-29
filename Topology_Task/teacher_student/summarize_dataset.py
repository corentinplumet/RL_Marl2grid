#!/usr/bin/env python3
"""Summarize and sanity-check a teacher-student dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from teacher_student.dataset import (
    export_metadata_file,
    list_shards,
    load_metadata,
    metadata_dir,
    metadata_export_dir,
    summary_path,
    task_relative,
)


RHO_HIST_BINS = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]


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
        "obs_bad_examples": 0,
        "obs_shape_mismatch_shards": 0,
        "invalid_policy_actions": 0,
        "invalid_teacher_actions": 0,
        "policy_logits_examples": 0,
        "policy_logits_nan_or_inf": 0,
        "policy_logits_missing_shards": 0,
        "policy_logits_shape_mismatch_shards": 0,
    }


def _add_action_metrics(
    metrics: Dict[str, int],
    policy_action: np.ndarray,
    teacher_action: np.ndarray,
    force_noop: np.ndarray,
    obs: np.ndarray,
    *,
    action_size: Optional[int],
    expected_obs_shape: Optional[List[int]],
    policy_logits: Optional[np.ndarray],
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
    if obs.ndim <= 1:
        metrics["obs_bad_examples"] += int(np.sum(~np.isfinite(obs)))
    else:
        metrics["obs_bad_examples"] += int(
            np.sum(np.any(~np.isfinite(obs), axis=tuple(range(1, obs.ndim))))
        )
    if expected_obs_shape is not None and tuple(obs.shape[1:]) != tuple(expected_obs_shape):
        metrics["obs_shape_mismatch_shards"] += 1
    if action_size is not None:
        metrics["invalid_policy_actions"] += int(
            np.sum((policy_action < 0) | (policy_action >= int(action_size)))
        )
        metrics["invalid_teacher_actions"] += int(
            np.sum((teacher_action < 0) | (teacher_action >= int(action_size)))
        )
    if policy_logits is None:
        metrics["policy_logits_missing_shards"] += 1
    else:
        metrics["policy_logits_examples"] += int(policy_logits.shape[0])
        metrics["policy_logits_nan_or_inf"] += int(np.sum(~np.isfinite(policy_logits)))
        if action_size is not None and (
            policy_logits.ndim != 2
            or policy_logits.shape[0] != n
            or policy_logits.shape[1] != int(action_size)
        ):
            metrics["policy_logits_shape_mismatch_shards"] += 1


def _frac(num: int, den: int) -> float:
    return float(num) / float(max(den, 1))


def _empty_histogram() -> Dict[str, Any]:
    return {
        "bin_edges": list(RHO_HIST_BINS),
        "counts": [0 for _ in range(len(RHO_HIST_BINS) - 1)],
        "underflow": 0,
        "overflow": 0,
        "nan_or_inf": 0,
    }


def _add_histogram(histogram: Dict[str, Any], values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return
    finite = values[np.isfinite(values)]
    histogram["nan_or_inf"] += int(values.size - finite.size)
    if finite.size == 0:
        return
    edges = np.asarray(histogram["bin_edges"], dtype=np.float32)
    histogram["underflow"] += int(np.sum(finite < edges[0]))
    histogram["overflow"] += int(np.sum(finite >= edges[-1]))
    in_range = finite[(finite >= edges[0]) & (finite < edges[-1])]
    counts, _ = np.histogram(in_range, bins=edges)
    histogram["counts"] = [
        int(old + new) for old, new in zip(histogram["counts"], counts.tolist())
    ]


def _top_actions(counts: Dict[int, int], n: int, *, limit: int = 10) -> List[Dict[str, Any]]:
    return [
        {"action": int(action), "count": int(count), "frac": _frac(int(count), n)}
        for action, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[
            :limit
        ]
    ]


def summarize(dataset_dir: Path) -> Dict[str, Any]:
    metadata = _load_metadata(dataset_dir)
    agent_ids: List[str] = list(metadata["agent_ids"])
    shards = list_shards(dataset_dir)

    agent_metrics = {agent: _metric_template() for agent in agent_ids}
    action_counts = {
        agent: defaultdict(int)
        for agent in agent_ids
    }
    nonidle_action_counts = {
        agent: defaultdict(int)
        for agent in agent_ids
    }
    rho_histograms = {
        agent: {
            "local_max_rho_when_teacher_action0": _empty_histogram(),
            "local_max_rho_when_teacher_nonidle": _empty_histogram(),
        }
        for agent in agent_ids
    }
    unique_fingerprints = set()
    n_env_steps = 0
    n_done = 0
    episode_lengths = []
    episode_records = []
    chronic_lengths: Dict[str, List[int]] = defaultdict(list)
    chronic_names: Dict[str, str] = {}
    bad_shards = []
    action_sizes = metadata.get("action_sizes", {}) or {}
    obs_shapes = metadata.get("obs_shapes", {}) or {}

    for shard in shards:
        try:
            with np.load(shard) as data:
                n = int(data["dataset_step"].size)
                n_env_steps += n
                n_done += int(np.sum(data["done"]))
                unique_fingerprints.update(map(str, data["chronic_fingerprint"]))
                done_mask = np.asarray(data["done"], dtype=bool)
                done_indices = np.flatnonzero(done_mask)
                for idx in done_indices:
                    episode_length = int(data["done_episode_length"][idx])
                    if episode_length <= 0:
                        continue
                    fingerprint = str(data["chronic_fingerprint"][idx])
                    chronic_name = str(data["chronic_name"][idx])
                    episode_id = int(data["episode_id"][idx])
                    chronic_names[fingerprint] = chronic_name
                    chronic_lengths[fingerprint].append(episode_length)
                    episode_records.append(
                        {
                            "episode_id": episode_id,
                            "chronic_name": chronic_name,
                            "chronic_fingerprint": fingerprint,
                            "episode_length": episode_length,
                        }
                    )
                episode_lengths.extend(
                    int(x) for x in data["done_episode_length"] if int(x) > 0
                )
                for agent in agent_ids:
                    policy = data[f"policy_action_{agent}"]
                    teacher = data[f"teacher_action_{agent}"]
                    force = data[f"force_noop_{agent}"]
                    obs = data[f"obs_{agent}"]
                    logits_key = f"policy_logits_{agent}"
                    policy_logits = data[logits_key] if logits_key in data.files else None
                    _add_action_metrics(
                        agent_metrics[agent],
                        policy,
                        teacher,
                        force,
                        obs,
                        action_size=action_sizes.get(agent),
                        expected_obs_shape=obs_shapes.get(agent),
                        policy_logits=policy_logits,
                    )
                    for action_id, count in zip(*np.unique(teacher, return_counts=True)):
                        action_counts[agent][int(action_id)] += int(count)
                    teacher_nonidle = teacher[teacher != 0]
                    if teacher_nonidle.size:
                        for action_id, count in zip(
                            *np.unique(teacher_nonidle, return_counts=True)
                        ):
                            nonidle_action_counts[agent][int(action_id)] += int(count)
                    rho_key = f"local_max_rho_{agent}"
                    if rho_key in data.files:
                        local_rho = data[rho_key]
                        _add_histogram(
                            rho_histograms[agent][
                                "local_max_rho_when_teacher_action0"
                            ],
                            local_rho[teacher == 0],
                        )
                        _add_histogram(
                            rho_histograms[agent][
                                "local_max_rho_when_teacher_nonidle"
                            ],
                            local_rho[teacher != 0],
                        )
        except Exception as exc:
            bad_shards.append({"path": _safe_path(shard), "error": repr(exc)})

    agent_summary = {}
    for agent, metrics in agent_metrics.items():
        n = metrics["n"]
        agent_summary[agent] = {
            **metrics,
            "policy_action_0_frac": _frac(metrics["policy_action_0"], n),
            "policy_nonidle_frac": _frac(metrics["policy_nonidle"], n),
            "teacher_action_0_frac": _frac(metrics["teacher_action_0"], n),
            "teacher_nonidle_frac": _frac(metrics["teacher_nonidle"], n),
            "force_noop_frac": _frac(metrics["force_noop"], n),
            "was_overwritten_frac": _frac(metrics["was_overwritten"], n),
            "overwrite_to_action0_frac": _frac(metrics["overwrite_to_action0"], n),
            "invalid_policy_action_frac": _frac(metrics["invalid_policy_actions"], n),
            "invalid_teacher_action_frac": _frac(metrics["invalid_teacher_actions"], n),
            "obs_bad_example_frac": _frac(metrics["obs_bad_examples"], n),
            "policy_logits_available_frac": _frac(
                metrics["policy_logits_examples"], n
            ),
            "top_teacher_actions": _top_actions(action_counts[agent], n),
            "top_teacher_nonidle_actions": _top_actions(
                nonidle_action_counts[agent], max(metrics["teacher_nonidle"], 1)
            ),
            "rho_histograms": rho_histograms[agent],
        }

    survival_denominator = metadata.get("episode_length_max")
    if survival_denominator is None and episode_lengths:
        survival_denominator = max(episode_lengths)
    survival_denominator = int(survival_denominator or 0)
    survival_by_chronic = []
    for fingerprint, lengths in sorted(chronic_lengths.items()):
        mean_length = float(np.mean(lengths))
        row = {
            "chronic_fingerprint": fingerprint,
            "chronic_name": chronic_names.get(fingerprint, "unknown"),
            "n_episodes": int(len(lengths)),
            "survival_step_mean": mean_length,
            "survival_step_min": int(min(lengths)),
            "survival_step_max": int(max(lengths)),
        }
        if survival_denominator > 0:
            row["survival_rate_mean"] = mean_length / float(survival_denominator)
        survival_by_chronic.append(row)
    teacher_survival_step_mean = (
        float(np.mean(episode_lengths)) if episode_lengths else None
    )
    teacher_survival_rate_mean = (
        teacher_survival_step_mean / float(survival_denominator)
        if teacher_survival_step_mean is not None and survival_denominator > 0
        else None
    )

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
        "teacher_survival_step_mean": teacher_survival_step_mean,
        "teacher_survival_rate_mean": teacher_survival_rate_mean,
        "teacher_survival_mean": teacher_survival_rate_mean,
        "teacher_survival_denominator_steps": survival_denominator or None,
        "teacher_survival_by_chronic": survival_by_chronic,
        "teacher_survival_episodes": episode_records,
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
    print(
        "Teacher survival: "
        f"mean_step={summary['teacher_survival_step_mean']} "
        f"mean_rate={summary['teacher_survival_rate_mean']}"
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
        "bad_actions",
        "logits_frac",
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
                str(
                    metrics["invalid_policy_actions"]
                    + metrics["invalid_teacher_actions"]
                ),
                f"{metrics['policy_logits_available_frac']:.4f}",
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
    export_path = export_metadata_file(dataset_dir, output_json, "summary")
    if export_path is not None:
        print(f"Saved summary export: {_safe_path(export_path)}")
        print(f"Metadata export dir: {_safe_path(metadata_export_dir(dataset_dir))}")


if __name__ == "__main__":
    main()
