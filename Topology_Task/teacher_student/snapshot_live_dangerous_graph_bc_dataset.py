#!/usr/bin/env python3
"""Snapshot completed shards from a running dangerous graph-BC collection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from teacher_student.calendar_balancing import calendar_month_key
from teacher_student.dataset import metadata_dir, shards_dir


def _task_path(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = TASK_DIR / path
    return path.resolve()


def _stable_prefix(source: Path, min_age_seconds: float) -> List[Path]:
    candidates = sorted(shards_dir(source).glob("shard_*.npz"))
    if not candidates:
        candidates = sorted(source.glob("shard_*.npz"))
    now = time.time()
    stable: List[Path] = []
    for expected_index, path in enumerate(candidates):
        expected_name = f"shard_{expected_index:05d}.npz"
        if path.name != expected_name:
            raise ValueError(
                f"Live shards are not a contiguous prefix: expected {expected_name}, "
                f"found {path.name}."
            )
        before = path.stat()
        if now - before.st_mtime < float(min_age_seconds):
            break
        try:
            with np.load(path, allow_pickle=False) as data:
                state_ids = np.asarray(data["state_id"], dtype=np.int64)
                if not len(state_ids):
                    raise ValueError("empty state_id array")
        except Exception as exc:
            print(f"Stopping before unstable shard {path.name}: {exc}", flush=True)
            break
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            print(f"Stopping before changing shard {path.name}.", flush=True)
            break
        stable.append(path)
    if not stable:
        raise RuntimeError(
            "No completed age-stable shards are available. Wait for the first "
            "shard to be written, then retry."
        )
    return stable


def _inspect_shards(paths: List[Path]) -> Dict[str, Any]:
    total_rows = 0
    expected_state_id = 0
    agents: List[str] | None = None
    action_sizes: Dict[str, int] = {}
    has_candidate_outcomes = True
    month_counts: Counter[str] = Counter()
    fingerprints: set[str] = set()
    fingerprints_by_month: Dict[str, set[str]] = defaultdict(set)
    episodes: set[int] = set()
    agent_summary: Dict[str, Dict[str, int]] = {}

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            state_ids = np.asarray(data["state_id"], dtype=np.int64)
            if not np.array_equal(
                state_ids,
                np.arange(expected_state_id, expected_state_id + len(state_ids)),
            ):
                raise ValueError(f"Non-contiguous state IDs in {path}.")
            expected_state_id += len(state_ids)
            total_rows += len(state_ids)

            shard_agents = sorted(
                key[len("policy_logits_") :]
                for key in data.files
                if key.startswith("policy_logits_")
            )
            if agents is None:
                agents = shard_agents
                if not agents:
                    raise ValueError(f"No policy-logit agent arrays in {path}.")
                agent_summary = {
                    agent: {
                        "dangerous_examples": 0,
                        "concerned": 0,
                        "nonidle_targets": 0,
                    }
                    for agent in agents
                }
            elif shard_agents != agents:
                raise ValueError(f"Agent arrays changed in {path}.")

            for agent in agents:
                logits = np.asarray(data[f"policy_logits_{agent}"])
                size = int(logits.shape[1])
                previous = action_sizes.setdefault(agent, size)
                if previous != size:
                    raise ValueError(f"Action size changed for {agent} in {path}.")
                agent_summary[agent]["dangerous_examples"] += len(state_ids)
                agent_summary[agent]["concerned"] += int(
                    np.asarray(data[f"concerned_{agent}"], dtype=bool).sum()
                )
                agent_summary[agent]["nonidle_targets"] += int(
                    np.asarray(data[f"target_is_nonidle_{agent}"], dtype=bool).sum()
                )
                has_candidate_outcomes = has_candidate_outcomes and (
                    f"candidate_{agent}_rho_after" in data.files
                )

            names = np.asarray(data["chronic_name"], dtype=str)
            datetimes = np.asarray(data["chronic_datetime"], dtype=str)
            shard_fingerprints = np.asarray(data["chronic_fingerprint"], dtype=str)
            if "calendar_month" in data.files:
                months = np.asarray(data["calendar_month"], dtype=str)
            else:
                months = np.asarray(
                    [
                        calendar_month_key(datetime_value, name)
                        for datetime_value, name in zip(datetimes, names)
                    ],
                    dtype=str,
                )
            for month, fingerprint in zip(months, shard_fingerprints):
                month_counts[str(month)] += 1
                fingerprints.add(str(fingerprint))
                fingerprints_by_month[str(month)].add(str(fingerprint))
            episodes.update(np.asarray(data["episode_id"], dtype=np.int64).tolist())

    return {
        "total_rows": total_rows,
        "agent_ids": agents or [],
        "action_sizes": action_sizes,
        "has_candidate_outcomes": has_candidate_outcomes,
        "n_unique_chronic_fingerprints": len(fingerprints),
        "n_episodes_with_rows": len(episodes),
        "dangerous_states_by_month": dict(sorted(month_counts.items())),
        "unique_chronic_fingerprints_by_month": {
            month: len(values)
            for month, values in sorted(fingerprints_by_month.items())
        },
        "agent_summary": agent_summary,
    }


def _copy_or_link(source: Path, destination: Path, mode: str) -> str:
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def _load_config_args(config_path: Path) -> Tuple[Dict[str, Any], Path]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    args = dict(config.get("args", {}))
    if not args:
        raise ValueError(f"Config has no [args] table: {config_path}")
    return args, config_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reduced-action-space", type=Path, required=True)
    parser.add_argument("--action-space-label", default="mk32")
    parser.add_argument("--min-age-seconds", type=float, default=30.0)
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.min_age_seconds < 0.0:
        raise ValueError("--min-age-seconds must be non-negative.")
    source = _task_path(cli.source)
    output = _task_path(cli.output)
    config_path = _task_path(cli.config)
    reduced_action_space = _task_path(cli.reduced_action_space)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not reduced_action_space.is_file():
        raise FileNotFoundError(reduced_action_space)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Snapshot output is not empty: {output}")

    stable_shards = _stable_prefix(source, cli.min_age_seconds)
    inspection = _inspect_shards(stable_shards)
    config_args, config_path = _load_config_args(config_path)
    output_shards = shards_dir(output)
    output_metadata = metadata_dir(output)
    output_shards.mkdir(parents=True, exist_ok=True)
    output_metadata.mkdir(parents=True, exist_ok=True)

    copy_modes = Counter()
    snapshot_paths = []
    for source_path in stable_shards:
        destination = output_shards / source_path.name
        copy_modes[_copy_or_link(source_path, destination, cli.mode)] += 1
        snapshot_paths.append(destination)

    payload = {
        "status": "snapshot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector": "teacher_student.snapshot_live_dangerous_graph_bc_dataset",
        "dataset_mode": "dangerous_graph_bc",
        "snapshot_source": str(source),
        "snapshot_min_age_seconds": float(cli.min_age_seconds),
        "snapshot_copy_modes": dict(copy_modes),
        "source_config": str(config_path),
        "checkpoint": None,
        "has_policy_logits": False,
        "reference_logits_kind": "unavailable_zero_placeholder",
        "env_id": str(config_args.get("env_id", "")),
        "action_space_label": str(cli.action_space_label),
        "reduced_action_space": str(reduced_action_space),
        "actor_encoder": str(config_args.get("actor_encoder", "")),
        "actor_action_head": str(config_args.get("actor_action_head", "")),
        "gnn_graph_type": str(config_args.get("gnn_graph_type", "")),
        "gnn_physical_scaling": bool(
            config_args.get("gnn_physical_scaling", False)
        ),
        "gnn_running_norm": bool(config_args.get("gnn_running_norm", False)),
        "agent_ids": inspection["agent_ids"],
        "action_sizes": inspection["action_sizes"],
        "has_candidate_outcomes": inspection["has_candidate_outcomes"],
        "n_dangerous_states": inspection["total_rows"],
        "n_unique_chronic_fingerprints": inspection[
            "n_unique_chronic_fingerprints"
        ],
        "n_episodes_with_rows": inspection["n_episodes_with_rows"],
        "calendar_balance": {
            "dangerous_states_by_month": inspection["dangerous_states_by_month"],
            "unique_chronic_fingerprints_by_month": inspection[
                "unique_chronic_fingerprints_by_month"
            ],
        },
        "agent_summary": inspection["agent_summary"],
        "n_shards": len(snapshot_paths),
        "shards": [str(path) for path in snapshot_paths],
        "layout": {
            "version": 3 if inspection["has_candidate_outcomes"] else 2,
            "shards_dir": "shards",
            "metadata_file": "metadata/metadata.json",
        },
    }
    metadata_path = output_metadata / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("========== Live collection snapshot ==========")
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Stable shards: {len(snapshot_paths)}")
    print(f"States: {inspection['total_rows']}")
    print(f"Months: {inspection['dangerous_states_by_month']}")
    print(f"Copy modes: {dict(copy_modes)}")
    print(f"Metadata: {metadata_path}")
    print("The source collection was not modified and can continue running.")


if __name__ == "__main__":
    main()
