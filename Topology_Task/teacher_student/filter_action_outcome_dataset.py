#!/usr/bin/env python3
"""Filter action-outcome datasets by the risky-state max-rho threshold."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from teacher_student.dataset import (
    METADATA_DIR_NAME,
    SHARDS_DIR_NAME,
    export_metadata_file,
    load_metadata,
    metadata_dir,
    metadata_path,
    resolve_dataset_dir,
    shards_dir,
    task_relative,
)


STATE_CONTEXTS_DIR_NAME = "state_contexts"


def state_contexts_dir(dataset_dir: Path) -> Path:
    return dataset_dir / STATE_CONTEXTS_DIR_NAME


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


def _resolve_output_dir(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()
    return path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _list_parts(dataset_root: Path) -> List[Path]:
    part_root = dataset_root / "parts"
    parts = sorted(part_root.glob("part_*")) if part_root.exists() else []
    return [part for part in parts if part.is_dir()] or [dataset_root]


def _list_npz(directory: Path, pattern: str, max_files: Optional[int]) -> List[Path]:
    files = sorted(directory.glob(pattern)) if directory.exists() else []
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if output_root.resolve() == TASK_DIR.resolve():
            raise ValueError("Refusing to use Topology_Task itself as output dir.")
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {task_relative(output_root)}. "
                "Use --overwrite true or choose another --output-dir."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _threshold_mask(values: np.ndarray, threshold: float, inclusive: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if inclusive:
        return finite & (values >= float(threshold))
    return finite & (values > float(threshold))


def _write_npz(path: Path, arrays: Dict[str, np.ndarray], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def _filter_outcome_shard(
    *,
    shard: Path,
    output_shard: Path,
    agent_ids: Iterable[str],
    threshold: float,
    inclusive: bool,
    compress: bool,
) -> Tuple[int, Dict[str, int], Dict[str, set]]:
    arrays: Dict[str, np.ndarray] = {}
    rows_by_agent: Dict[str, int] = {agent: 0 for agent in agent_ids}
    action_ids_by_agent: Dict[str, set] = {agent: set() for agent in agent_ids}
    matched_keys: set = set()

    with np.load(shard, allow_pickle=False) as data:
        data_files = list(data.files)
        for agent in agent_ids:
            prefix = f"outcome_{agent}_"
            rho_key = f"{prefix}global_max_rho_before"
            if rho_key not in data_files:
                continue

            rho = np.asarray(data[rho_key])
            mask = _threshold_mask(rho, threshold, inclusive)
            row_count = int(rho.shape[0])
            kept_rows = int(mask.sum())
            rows_by_agent[agent] = kept_rows

            action_key = f"{prefix}action_id"
            if action_key in data_files and kept_rows > 0:
                action_ids_by_agent[agent].update(
                    int(x) for x in np.asarray(data[action_key])[mask]
                )

            for key in data_files:
                is_agent_outcome = key.startswith(prefix)
                is_agent_obs = key == f"obs_{agent}"
                if not is_agent_outcome and not is_agent_obs:
                    continue

                arr = np.asarray(data[key])
                matched_keys.add(key)
                if arr.ndim > 0 and int(arr.shape[0]) == row_count:
                    arrays[key] = arr[mask].copy()
                else:
                    arrays[key] = arr.copy()

        for key in data_files:
            if key in matched_keys:
                continue
            arr = np.asarray(data[key])
            arrays[key] = arr.copy()

    total_rows = int(sum(rows_by_agent.values()))
    if total_rows > 0:
        _write_npz(output_shard, arrays, compress)
    return total_rows, rows_by_agent, action_ids_by_agent


def _filter_state_context_shard(
    *,
    shard: Path,
    output_shard: Path,
    threshold: float,
    inclusive: bool,
    compress: bool,
) -> int:
    with np.load(shard, allow_pickle=False) as data:
        if "state_global_max_rho_before" not in data.files:
            raise KeyError(
                f"Missing state_global_max_rho_before in {task_relative(shard)}"
            )
        rho = np.asarray(data["state_global_max_rho_before"])
        mask = _threshold_mask(rho, threshold, inclusive)
        row_count = int(rho.shape[0])
        kept_rows = int(mask.sum())
        if kept_rows == 0:
            return 0

        arrays: Dict[str, np.ndarray] = {}
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim > 0 and int(arr.shape[0]) == row_count:
                arrays[key] = arr[mask].copy()
            else:
                arrays[key] = arr.copy()
    _write_npz(output_shard, arrays, compress)
    return kept_rows


def _output_part_path(source_root: Path, source_part: Path, output_root: Path) -> Path:
    if source_part == source_root:
        return output_root
    return output_root / "parts" / source_part.name


def _part_summary_metadata(
    *,
    source_part: Path,
    output_part: Path,
    source_metadata: Dict[str, Any],
    source_root: Path,
    output_root: Path,
    threshold: float,
    inclusive: bool,
    outcome_shards: List[Path],
    state_context_shards: List[Path],
    rows_by_agent: Dict[str, int],
    unique_actions_by_agent: Dict[str, int],
    state_context_rows: int,
) -> Dict[str, Any]:
    action_sizes = {
        agent: int(size)
        for agent, size in source_metadata.get("action_sizes", {}).items()
    }
    action_coverage_by_agent = {
        agent: (
            unique_actions_by_agent.get(agent, 0) / int(action_sizes[agent])
            if int(action_sizes.get(agent, 0)) > 0
            else float("nan")
        )
        for agent in source_metadata.get("agent_ids", [])
    }
    source_candidate_states = int(source_metadata.get("n_candidate_states") or 0)

    metadata = dict(source_metadata)
    metadata.update(
        {
            "status": "complete_filtered",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collector": "teacher_student.filter_action_outcome_dataset",
            "source_dataset": task_relative(source_root),
            "source_part": task_relative(source_part),
            "filtered_from": task_relative(source_part),
            "filter_output_root": task_relative(output_root),
            "source_collection_rho_threshold": source_metadata.get(
                "collection_rho_threshold"
            ),
            "filtered_collection_rho_threshold": float(threshold),
            "filter_global_max_rho_before_threshold": float(threshold),
            "filter_global_max_rho_before_inclusive": bool(inclusive),
            "filter_condition": (
                f"global_max_rho_before {'>=' if inclusive else '>'} {threshold}"
            ),
            "n_shards": int(len(outcome_shards)),
            "n_state_context_shards": int(len(state_context_shards)),
            "n_candidate_states": int(state_context_rows),
            "n_state_contexts": int(state_context_rows),
            "n_outcome_examples": int(sum(rows_by_agent.values())),
            "n_outcome_examples_by_agent": {
                agent: int(rows_by_agent.get(agent, 0))
                for agent in source_metadata.get("agent_ids", [])
            },
            "n_unique_actions_by_agent": unique_actions_by_agent,
            "action_coverage_by_agent": action_coverage_by_agent,
            "n_skipped_states_below_threshold": max(
                source_candidate_states - int(state_context_rows), 0
            ),
            "layout": {
                "version": 2,
                "shards_dir": SHARDS_DIR_NAME,
                "state_contexts_dir": STATE_CONTEXTS_DIR_NAME,
                "metadata_dir": METADATA_DIR_NAME,
                "metadata_export_dir": "../metadata_exports",
            },
            "shard_preview": {
                "first": [task_relative(path) for path in outcome_shards[:3]],
                "last": [task_relative(path) for path in outcome_shards[-3:]],
            },
            "state_context_shard_preview": {
                "first": [task_relative(path) for path in state_context_shards[:3]],
                "last": [task_relative(path) for path in state_context_shards[-3:]],
            },
        }
    )
    return metadata


def _write_part_metadata(output_part: Path, metadata: Dict[str, Any]) -> None:
    path = metadata_path(output_part)
    _write_json(path, metadata)
    export_metadata_file(output_part, path, "metadata")


def _filter_part(
    *,
    source_root: Path,
    source_part: Path,
    output_root: Path,
    threshold: float,
    inclusive: bool,
    overwrite: bool,
    compress: bool,
    max_shards: Optional[int],
    progress_every_shards: int,
) -> Dict[str, Any]:
    output_part = _output_part_path(source_root, source_part, output_root)
    if output_part.exists() and overwrite:
        shutil.rmtree(output_part)
    output_part.mkdir(parents=True, exist_ok=True)
    shards_dir(output_part).mkdir(parents=True, exist_ok=True)
    state_contexts_dir(output_part).mkdir(parents=True, exist_ok=True)
    metadata_dir(output_part).mkdir(parents=True, exist_ok=True)

    source_metadata = load_metadata(source_part)
    agent_ids = list(source_metadata.get("agent_ids", []))
    if not agent_ids:
        raise ValueError(f"Missing agent_ids in {task_relative(source_part)}")

    source_shards = _list_npz(shards_dir(source_part), "shard_*.npz", max_shards)
    if not source_shards:
        source_shards = _list_npz(source_part, "shard_*.npz", max_shards)
    source_state_contexts = _list_npz(
        state_contexts_dir(source_part), "state_context_*.npz", max_shards
    )

    print(
        f"part {source_part.name}: outcome_shards={len(source_shards)} "
        f"state_context_shards={len(source_state_contexts)}",
        flush=True,
    )

    rows_by_agent = {agent: 0 for agent in agent_ids}
    action_ids_by_agent = {agent: set() for agent in agent_ids}
    written_outcome_shards: List[Path] = []
    written_state_context_shards: List[Path] = []
    part_start = time.perf_counter()

    for index, shard in enumerate(source_shards, start=1):
        output_shard = shards_dir(output_part) / shard.name
        total_rows, shard_rows_by_agent, shard_actions_by_agent = _filter_outcome_shard(
            shard=shard,
            output_shard=output_shard,
            agent_ids=agent_ids,
            threshold=threshold,
            inclusive=inclusive,
            compress=compress,
        )
        if total_rows > 0:
            written_outcome_shards.append(output_shard)
        for agent in agent_ids:
            rows_by_agent[agent] += int(shard_rows_by_agent.get(agent, 0))
            action_ids_by_agent[agent].update(shard_actions_by_agent.get(agent, set()))
        if (
            progress_every_shards > 0
            and (index == 1 or index == len(source_shards) or index % progress_every_shards == 0)
        ):
            print(
                "filter progress: "
                f"part={source_part.name} "
                f"outcome_shard={index}/{len(source_shards)} "
                f"kept_rows={sum(rows_by_agent.values())} "
                f"elapsed={_format_duration(time.perf_counter() - part_start)}",
                flush=True,
            )

    state_context_rows = 0
    for index, shard in enumerate(source_state_contexts, start=1):
        output_shard = state_contexts_dir(output_part) / shard.name
        kept_rows = _filter_state_context_shard(
            shard=shard,
            output_shard=output_shard,
            threshold=threshold,
            inclusive=inclusive,
            compress=compress,
        )
        state_context_rows += int(kept_rows)
        if kept_rows > 0:
            written_state_context_shards.append(output_shard)
        if (
            progress_every_shards > 0
            and (index == 1 or index == len(source_state_contexts) or index % progress_every_shards == 0)
        ):
            print(
                "context progress: "
                f"part={source_part.name} "
                f"context_shard={index}/{len(source_state_contexts)} "
                f"kept_states={state_context_rows} "
                f"elapsed={_format_duration(time.perf_counter() - part_start)}",
                flush=True,
            )

    unique_actions_by_agent = {
        agent: int(len(action_ids_by_agent.get(agent, set()))) for agent in agent_ids
    }
    metadata = _part_summary_metadata(
        source_part=source_part,
        output_part=output_part,
        source_metadata=source_metadata,
        source_root=source_root,
        output_root=output_root,
        threshold=threshold,
        inclusive=inclusive,
        outcome_shards=written_outcome_shards,
        state_context_shards=written_state_context_shards,
        rows_by_agent=rows_by_agent,
        unique_actions_by_agent=unique_actions_by_agent,
        state_context_rows=state_context_rows,
    )
    _write_part_metadata(output_part, metadata)

    print(
        f"part complete: {source_part.name} kept_states={state_context_rows} "
        f"kept_outcomes={sum(rows_by_agent.values())} "
        f"outcome_shards={len(written_outcome_shards)} "
        f"elapsed={_format_duration(time.perf_counter() - part_start)}",
        flush=True,
    )
    return {
        "source_part": task_relative(source_part),
        "output_part": task_relative(output_part),
        "n_candidate_states": int(state_context_rows),
        "n_outcome_examples": int(sum(rows_by_agent.values())),
        "n_outcome_examples_by_agent": {
            agent: int(rows_by_agent.get(agent, 0)) for agent in agent_ids
        },
        "n_unique_actions_by_agent": unique_actions_by_agent,
        "n_shards": int(len(written_outcome_shards)),
        "n_state_context_shards": int(len(written_state_context_shards)),
    }


def _write_root_metadata(
    *,
    source_root: Path,
    output_root: Path,
    part_summaries: List[Dict[str, Any]],
    threshold: float,
    inclusive: bool,
) -> None:
    payload = {
        "status": "complete_filtered",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector": "teacher_student.filter_action_outcome_dataset",
        "dataset_mode": "action_outcomes",
        "source_dataset": task_relative(source_root),
        "filter_global_max_rho_before_threshold": float(threshold),
        "filter_global_max_rho_before_inclusive": bool(inclusive),
        "filter_condition": (
            f"global_max_rho_before {'>=' if inclusive else '>'} {threshold}"
        ),
        "parts": part_summaries,
        "n_parts": int(len(part_summaries)),
        "n_candidate_states": int(
            sum(part["n_candidate_states"] for part in part_summaries)
        ),
        "n_outcome_examples": int(
            sum(part["n_outcome_examples"] for part in part_summaries)
        ),
        "n_shards": int(sum(part["n_shards"] for part in part_summaries)),
        "n_state_context_shards": int(
            sum(part["n_state_context_shards"] for part in part_summaries)
        ),
    }
    metadata_dir(output_root).mkdir(parents=True, exist_ok=True)
    path = metadata_path(output_root)
    _write_json(path, payload)
    export_metadata_file(output_root, path, "metadata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a filtered copy of an action-outcome dataset, keeping only "
            "rows collected from states whose global_max_rho_before crosses a "
            "new threshold."
        )
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rho-threshold", type=float, required=True)
    parser.add_argument(
        "--inclusive",
        type=str,
        default="false",
        help="Use >= threshold instead of > threshold.",
    )
    parser.add_argument("--overwrite", type=str, default="false")
    parser.add_argument("--compress", type=str, default="true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-parts", type=int, default=None)
    parser.add_argument("--progress-every-shards", type=int, default=25)
    return parser.parse_args()


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    cli = parse_args()
    if cli.rho_threshold < 0.0:
        raise ValueError("--rho-threshold must be non-negative.")
    if cli.progress_every_shards < 0:
        raise ValueError("--progress-every-shards must be non-negative.")

    source_root = resolve_dataset_dir(cli.source_dataset)
    output_root = _resolve_output_dir(cli.output_dir)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("--output-dir must be different from --source-dataset.")

    inclusive = _str_to_bool(cli.inclusive)
    overwrite = _str_to_bool(cli.overwrite)
    compress = _str_to_bool(cli.compress)

    _prepare_output_root(output_root, overwrite=overwrite)
    source_parts = _list_parts(source_root)
    if cli.max_parts is not None:
        source_parts = source_parts[: int(cli.max_parts)]
    if not source_parts:
        raise FileNotFoundError(f"No dataset parts found in {task_relative(source_root)}")

    print("========== Filter action-outcome dataset ==========", flush=True)
    print(f"Source: {task_relative(source_root)}", flush=True)
    print(f"Output: {task_relative(output_root)}", flush=True)
    print(
        f"Condition: global_max_rho_before {'>=' if inclusive else '>'} "
        f"{cli.rho_threshold}",
        flush=True,
    )
    print(f"Parts: {len(source_parts)}", flush=True)
    print(f"Max shards per part: {cli.max_shards or 'all'}", flush=True)
    print(f"Compress: {compress}", flush=True)
    print("===================================================", flush=True)

    run_start = time.perf_counter()
    part_summaries: List[Dict[str, Any]] = []
    for part_index, source_part in enumerate(source_parts, start=1):
        print(
            f"filtering part {part_index}/{len(source_parts)}: "
            f"{task_relative(source_part)}",
            flush=True,
        )
        part_summaries.append(
            _filter_part(
                source_root=source_root,
                source_part=source_part,
                output_root=output_root,
                threshold=float(cli.rho_threshold),
                inclusive=inclusive,
                overwrite=overwrite,
                compress=compress,
                max_shards=cli.max_shards,
                progress_every_shards=int(cli.progress_every_shards),
            )
        )

    _write_root_metadata(
        source_root=source_root,
        output_root=output_root,
        part_summaries=part_summaries,
        threshold=float(cli.rho_threshold),
        inclusive=inclusive,
    )

    total_states = sum(part["n_candidate_states"] for part in part_summaries)
    total_outcomes = sum(part["n_outcome_examples"] for part in part_summaries)
    print("========== Filter complete ==========", flush=True)
    print(f"Output: {task_relative(output_root)}", flush=True)
    print(f"Kept states: {total_states}", flush=True)
    print(f"Kept outcome rows: {total_outcomes}", flush=True)
    print(f"Elapsed: {_format_duration(time.perf_counter() - run_start)}", flush=True)
    print("=====================================", flush=True)


if __name__ == "__main__":
    main()
