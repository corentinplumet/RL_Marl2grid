#!/usr/bin/env python3
"""Report which checkpoint runs are complete, incomplete, or corrupted."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script needs PyTorch to read checkpoint metadata. "
        "Run it inside the same environment used for training/evaluation."
    ) from exc


TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = TASK_DIR / "checkpoint"
SAVE_PREFIXES = ("best_test_", "final_final_", "final_")
OBS_STAT_KEYS = ("count", "mean", "var")


@dataclass
class FileRecord:
    file: str
    filename: str
    kind: str
    status: str
    exp_tag: str
    global_step: int | None
    total_timesteps: int | None
    n_envs: int | None
    complete_by_name: bool
    complete_by_step: bool
    has_obs_stats: bool
    error: str


@dataclass
class RunRecord:
    exp_tag: str
    status: str
    complete: bool
    has_corruption: bool
    n_files: int
    n_readable: int
    n_corrupted: int
    n_final: int
    n_best_test: int
    n_regular: int
    n_with_obs_stats: int
    max_global_step: int | None
    total_timesteps: int | None
    files: list[str]
    corrupted_files: list[str]


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _namespace_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_kind(filename: str) -> str:
    if filename.startswith("best_test_"):
        return "best_test"
    if filename.startswith("final_final_"):
        return "final_final"
    if filename.startswith("final_"):
        return "final"
    return "regular"


def _strip_prefixes(stem: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in SAVE_PREFIXES:
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                changed = True
    return stem


def _infer_exp_tag(path: Path) -> str:
    return _strip_prefixes(path.stem)


def _has_complete_obs_stats(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    training_state = state.get("training_state")
    if not isinstance(training_state, dict):
        return False
    obs_stats = training_state.get("obs_stats")
    if not isinstance(obs_stats, dict) or not obs_stats:
        return False
    return all(
        isinstance(stats, dict)
        and all(stats.get(key) is not None for key in OBS_STAT_KEYS)
        for stats in obs_stats.values()
    )


def inspect_file(path: Path, root: Path) -> FileRecord:
    rel_path = str(path.relative_to(root))
    kind = _checkpoint_kind(path.name)
    complete_by_name = kind in {"final", "final_final"}

    try:
        state = _load_checkpoint(path)
    except Exception as exc:  # noqa: BLE001 - report all unreadable files
        return FileRecord(
            file=rel_path,
            filename=path.name,
            kind=kind,
            status="corrupted",
            exp_tag=_infer_exp_tag(path),
            global_step=None,
            total_timesteps=None,
            n_envs=None,
            complete_by_name=complete_by_name,
            complete_by_step=False,
            has_obs_stats=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    args = state.get("args") if isinstance(state, dict) else None
    exp_tag = str(_namespace_get(args, "exp_tag", "") or "").strip()
    exp_tag = exp_tag or _infer_exp_tag(path)
    global_step = _int_or_none(state.get("global_step") if isinstance(state, dict) else None)
    total_timesteps = _int_or_none(_namespace_get(args, "total_timesteps"))
    n_envs = _int_or_none(_namespace_get(args, "n_envs"))

    complete_by_step = False
    if global_step is not None and total_timesteps is not None:
        tolerance = n_envs or 1
        complete_by_step = global_step >= total_timesteps - tolerance

    return FileRecord(
        file=rel_path,
        filename=path.name,
        kind=kind,
        status="readable",
        exp_tag=exp_tag,
        global_step=global_step,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        complete_by_name=complete_by_name,
        complete_by_step=complete_by_step,
        has_obs_stats=_has_complete_obs_stats(state),
        error="",
    )


def _run_status(complete: bool, has_corruption: bool, n_readable: int) -> str:
    if complete and has_corruption:
        return "complete_with_corruption"
    if complete:
        return "complete"
    if has_corruption and n_readable == 0:
        return "corrupted"
    if has_corruption:
        return "incomplete_with_corruption"
    return "incomplete"


def summarize_runs(records: list[FileRecord]) -> list[RunRecord]:
    grouped: dict[str, list[FileRecord]] = defaultdict(list)
    for record in records:
        grouped[record.exp_tag].append(record)

    runs: list[RunRecord] = []
    for exp_tag, files in grouped.items():
        readable = [record for record in files if record.status == "readable"]
        corrupted = [record for record in files if record.status == "corrupted"]
        final_files = [
            record for record in readable if record.kind in {"final", "final_final"}
        ]
        best_test_files = [record for record in readable if record.kind == "best_test"]
        regular_files = [record for record in readable if record.kind == "regular"]
        complete = any(
            record.complete_by_name or record.complete_by_step for record in readable
        )
        max_step_values = [
            record.global_step
            for record in readable
            if record.global_step is not None
        ]
        total_values = [
            record.total_timesteps
            for record in readable
            if record.total_timesteps is not None
        ]
        status = _run_status(complete, bool(corrupted), len(readable))
        runs.append(
            RunRecord(
                exp_tag=exp_tag,
                status=status,
                complete=complete,
                has_corruption=bool(corrupted),
                n_files=len(files),
                n_readable=len(readable),
                n_corrupted=len(corrupted),
                n_final=len(final_files),
                n_best_test=len(best_test_files),
                n_regular=len(regular_files),
                n_with_obs_stats=sum(record.has_obs_stats for record in readable),
                max_global_step=max(max_step_values) if max_step_values else None,
                total_timesteps=max(total_values) if total_values else None,
                files=sorted(record.file for record in files),
                corrupted_files=sorted(record.file for record in corrupted),
            )
        )

    status_order = {
        "corrupted": 0,
        "incomplete_with_corruption": 1,
        "complete_with_corruption": 2,
        "incomplete": 3,
        "complete": 4,
    }
    return sorted(runs, key=lambda run: (status_order[run.status], run.exp_tag))


def _format_step(max_step: int | None, total: int | None) -> str:
    if max_step is None and total is None:
        return "-"
    if total is None:
        return f"{max_step:,}"
    if max_step is None:
        return f"-/{total:,}"
    return f"{max_step:,}/{total:,}"


def _print_table(runs: list[RunRecord], show_files: bool) -> None:
    headers = [
        "status",
        "exp_tag",
        "step",
        "files",
        "final",
        "best",
        "obs",
        "bad",
    ]
    rows = [
        [
            run.status,
            run.exp_tag,
            _format_step(run.max_global_step, run.total_timesteps),
            str(run.n_files),
            str(run.n_final),
            str(run.n_best_test),
            str(run.n_with_obs_stats),
            str(run.n_corrupted),
        ]
        for run in runs
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for run, row in zip(runs, rows):
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
        if show_files:
            for file in run.files:
                marker = " ! " if file in run.corrupted_files else "   "
                print(f"{marker}{file}")


def _write_csv(path: Path, runs: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "exp_tag",
                "status",
                "complete",
                "has_corruption",
                "n_files",
                "n_readable",
                "n_corrupted",
                "n_final",
                "n_best_test",
                "n_regular",
                "n_with_obs_stats",
                "max_global_step",
                "total_timesteps",
                "files",
                "corrupted_files",
            ],
        )
        writer.writeheader()
        for run in runs:
            row = asdict(run)
            row["files"] = "|".join(run.files)
            row["corrupted_files"] = "|".join(run.corrupted_files)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint files and report complete/corrupted runs."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint root directory. Default: {DEFAULT_CHECKPOINT_DIR}",
    )
    parser.add_argument(
        "--only-problems",
        action="store_true",
        help="Show only corrupted or incomplete runs.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print checkpoint files under each run.",
    )
    parser.add_argument(
        "--min-age-seconds",
        type=float,
        default=0.0,
        help=(
            "Ignore .tar files modified more recently than this. Useful when "
            "training jobs are still writing checkpoints."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the full file and run report as JSON.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write the run summary as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.checkpoint_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {root}")

    now = time.time()
    paths = sorted(root.rglob("*.tar"))
    if args.min_age_seconds > 0:
        paths = [
            path
            for path in paths
            if now - path.stat().st_mtime >= args.min_age_seconds
        ]

    records = [inspect_file(path, root) for path in paths]
    runs = summarize_runs(records)
    displayed_runs = [
        run
        for run in runs
        if not args.only_problems or run.status != "complete"
    ]

    n_corrupted_files = sum(record.status == "corrupted" for record in records)
    n_complete_runs = sum(run.status == "complete" for run in runs)
    n_problem_runs = len(runs) - n_complete_runs
    print(f"Checkpoint directory: {root}")
    print(f"Files scanned: {len(records)}")
    print(f"Runs found: {len(runs)}")
    print(f"Complete runs: {n_complete_runs}")
    print(f"Problem/non-complete runs: {n_problem_runs}")
    print(f"Corrupted files: {n_corrupted_files}")
    if args.min_age_seconds > 0:
        print(f"Skipped files younger than {args.min_age_seconds:g}s")
    print("")
    _print_table(displayed_runs, show_files=args.show_files)

    if args.json:
        json_path = args.json.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "checkpoint_dir": str(root),
            "files": [asdict(record) for record in records],
            "runs": [asdict(run) for run in runs],
        }
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote JSON report: {json_path}")

    if args.csv:
        csv_path = args.csv.expanduser().resolve()
        _write_csv(csv_path, runs)
        print(f"Wrote CSV report: {csv_path}")


if __name__ == "__main__":
    main()
