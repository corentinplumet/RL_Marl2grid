#!/usr/bin/env python3
"""Reconstruct canonical histories from downloaded base and continuation runs.

The input is a permanent ``analysis/download`` run_data folder. Continuations
are paired through ``continuation_of_run_id`` / ``continuation_of_run_name`` in
their W&B config. At every continuation boundary, the earlier segment is cut
and the continuation wins for overlapping steps. This also handles originals
to which an archive was previously appended accidentally.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


MERGE_PROFILES = {
    "nls_cas_hl": {
        "group_dir": "NLS_cas_hl",
        "base_run_re": re.compile(
            r"^cas_hl_NLS_(?:mean|tmean)_f[01]_a0h[01]_s0$"
        ),
    },
    "nl_s3dw": {
        "group_dir": "gs_s3dw",
        "base_run_re": re.compile(
            r"^gs_s3dw_bus_n0_none_e0n0v0_"
            r"mp[123]_h(?:16|32|64|128)_s0$"
        ),
    },
}
FIXED_COLUMNS = {
    "run_name",
    "run_id",
    "run_group",
    "run_state",
    "exp_tag",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-") or "run"


def _load_history(run_dir: Path) -> pd.DataFrame:
    parquet = run_dir / "history.parquet"
    csv = run_dir / "history.csv.gz"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No history file in {run_dir}")


def _normalize(
    history: pd.DataFrame,
    *,
    declared_end_step: int | float | str | None = None,
) -> pd.DataFrame:
    history = history.copy()
    if "_step" not in history.columns:
        raise ValueError("History has no _step column")

    step_source = "_step"
    for candidate in ("charts/global_step", "global_step"):
        if candidate not in history.columns:
            continue
        candidate_steps = pd.to_numeric(history[candidate], errors="coerce")
        if candidate_steps.notna().any():
            step_source = candidate
            break

    if step_source != "_step":
        internal_steps = pd.to_numeric(history["_step"], errors="coerce")
        if "wandb_internal_step" in history.columns:
            history["wandb_internal_step"] = pd.to_numeric(
                history["wandb_internal_step"], errors="coerce"
            ).fillna(internal_steps)
        else:
            history["wandb_internal_step"] = internal_steps
        history["_step"] = pd.to_numeric(history[step_source], errors="coerce")
    else:
        history["_step"] = pd.to_numeric(history["_step"], errors="coerce")
        declared_end = pd.to_numeric(
            pd.Series([declared_end_step]), errors="coerce"
        ).iloc[0]
        available_steps = history["_step"].dropna()
        if pd.notna(declared_end) and not available_steps.empty:
            offset = int(declared_end) - int(available_steps.max())
            if offset:
                history["wandb_internal_step"] = history["_step"]
                history["_step"] = history["_step"] + offset
    history = history.dropna(subset=["_step"])
    if history.empty:
        return history
    history["_step"] = history["_step"].astype("int64")
    history = history.sort_values("_step")
    # Multiple logging calls can share a step. ``last`` coalesces their non-null
    # metric columns instead of throwing away all but one sparse row.
    history = history.groupby("_step", as_index=False, dropna=False).last()
    return history.sort_values("_step").reset_index(drop=True)


def _identity(run_dir: Path) -> dict[str, Any]:
    metadata = _read_json(run_dir / "metadata.json")
    config = _read_json(run_dir / "config.json")
    name = str(metadata.get("name") or metadata.get("run_name") or "")
    run_id = str(metadata.get("id") or metadata.get("run_id") or "")
    if not name:
        name = run_dir.name.split("__", 1)[0]
    if not run_id:
        run_id = run_dir.name.rsplit("__", 1)[-1]
    continuation = bool(config.get("is_continuation")) or "[continuation " in name
    return {
        "run_dir": run_dir,
        "name": name,
        "id": run_id,
        "group": metadata.get("group") or config.get("group"),
        "config": config,
        "metadata": metadata,
        "is_continuation": continuation,
        "parent_id": str(config.get("continuation_of_run_id") or ""),
        "parent_name": str(config.get("continuation_of_run_name") or ""),
        "declared_start": config.get("continuation_start_step"),
        "declared_end": config.get("continuation_end_step"),
    }


def _catalog(runs_root: Path) -> list[dict[str, Any]]:
    return [
        _identity(path)
        for path in sorted(runs_root.iterdir())
        if path.is_dir() and ((path / "history.parquet").exists() or (path / "history.csv.gz").exists())
    ]


def _pair_continuations(
    bases: list[dict[str, Any]],
    continuations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    base_by_id = {row["id"]: row for row in bases}
    base_by_name = {row["name"]: row for row in bases}
    pairs = {row["id"]: [] for row in bases}
    for continuation in continuations:
        base = base_by_id.get(continuation["parent_id"])
        if base is None:
            base = base_by_name.get(continuation["parent_name"])
        if base is None:
            # Fallback for an older upload lacking explicit provenance.
            prefix = continuation["name"].split(" [continuation ", 1)[0]
            base = base_by_name.get(prefix)
        if base is None:
            raise RuntimeError(
                f"Cannot pair continuation {continuation['name']} ({continuation['id']})"
            )
        pairs[base["id"]].append(continuation)
    return pairs


def _source_columns(history: pd.DataFrame, row: dict[str, Any], segment: str) -> pd.DataFrame:
    history = history.copy()
    history["wandb_source_run_name"] = row["name"]
    history["wandb_source_run_id"] = row["id"]
    history["history_segment"] = segment
    return history


def _canonical_columns(history: pd.DataFrame, base: dict[str, Any]) -> pd.DataFrame:
    history = history.copy()
    values = {
        "run_name": base["name"],
        "run_id": base["id"],
        "run_group": base["group"],
        "exp_tag": base["config"].get("exp_tag") or base["name"],
    }
    for column, value in values.items():
        history[column] = value
    return history


def _merge_one(
    base: dict[str, Any],
    continuations: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    current = _source_columns(_normalize(_load_history(base["run_dir"])), base, "base")
    boundaries: list[dict[str, Any]] = []
    loaded: list[tuple[int, int, dict[str, Any], pd.DataFrame]] = []
    for continuation in continuations:
        history = _normalize(
            _load_history(continuation["run_dir"]),
            declared_end_step=continuation.get("declared_end"),
        )
        if history.empty:
            raise RuntimeError(f"Continuation history is empty: {continuation['run_dir']}")
        actual_start = int(history["_step"].min())
        actual_end = int(history["_step"].max())
        loaded.append((actual_start, actual_end, continuation, history))

    for index, (start, end, continuation, history) in enumerate(sorted(loaded), start=1):
        before = current[current["_step"] < start].copy()
        previous_max = int(before["_step"].max()) if not before.empty else None
        continuation_history = _source_columns(history, continuation, f"continuation_{index}")
        current = pd.concat([before, continuation_history], ignore_index=True, sort=False)
        current = _normalize(current)
        boundaries.append(
            {
                "segment": index,
                "continuation_run_name": continuation["name"],
                "continuation_run_id": continuation["id"],
                "start_step": start,
                "end_step": end,
                "previous_retained_step": previous_max,
                "gap_steps": start - previous_max if previous_max is not None else None,
                "rows": int(len(history)),
            }
        )
    current = _canonical_columns(current, base)
    return current, boundaries


def _last_non_null(history: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in history.columns:
        if column in FIXED_COLUMNS:
            continue
        values = history[column].dropna()
        if values.empty:
            continue
        value = values.iloc[-1]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[column] = value
    return summary


def parse_args() -> argparse.Namespace:
    task_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(MERGE_PROFILES),
        default="nls_cas_hl",
        help="Run family and default local run_data directory to merge.",
    )
    parser.add_argument(
        "--group-dir",
        type=Path,
        default=None,
        help="Override the profile's run_data directory.",
    )
    parser.add_argument(
        "--base-run-regex",
        default=None,
        help="Override the profile's full-match regex for original run names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <group-dir>/merged_runs",
    )
    parser.add_argument("--no-write-csv", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.group_dir is None:
        folder = MERGE_PROFILES[args.profile]["group_dir"]
        args.group_dir = task_dir / "outputs" / "run_data" / folder
    return args


def main() -> int:
    args = parse_args()
    profile = MERGE_PROFILES[args.profile]
    base_run_re = (
        re.compile(args.base_run_regex)
        if args.base_run_regex
        else profile["base_run_re"]
    )
    group_dir = args.group_dir.expanduser().resolve()
    runs_root = group_dir / "runs"
    output_dir = (args.output_dir or group_dir / "merged_runs").expanduser().resolve()
    if not runs_root.exists():
        raise SystemExit(f"Downloaded runs folder does not exist: {runs_root}")

    catalog = _catalog(runs_root)
    bases = [
        row
        for row in catalog
        if not row["is_continuation"] and base_run_re.fullmatch(row["name"])
    ]
    if not bases:
        raise SystemExit(
            f"No {args.profile} base runs found in the downloaded group."
        )
    base_ids = {row["id"] for row in bases}
    base_names = {row["name"] for row in bases}
    continuations = [
        row
        for row in catalog
        if row["is_continuation"]
        and (
            row["parent_id"] in base_ids
            or row["parent_name"] in base_names
            or row["name"].split(" [continuation ", 1)[0] in base_names
        )
    ]
    pairs = _pair_continuations(bases, continuations)

    print("========== Canonical continuation merge ==========")
    print(f"Profile: {args.profile}")
    print(f"Input: {runs_root}")
    print(f"Output: {output_dir}")
    print(f"Base runs: {len(bases)}")
    print(f"Continuation runs: {len(continuations)}")
    print(f"Dry run: {args.dry_run}")
    print("==================================================")

    if output_dir.exists() and not args.dry_run:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = group_dir / "merged_runs_backups" / timestamp
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output_dir), str(backup))
        print(f"Previous merged output moved to {backup}")
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    all_boundaries: list[dict[str, Any]] = []
    failures = 0
    for base in sorted(bases, key=lambda row: row["name"]):
        try:
            history, boundaries = _merge_one(base, pairs.get(base["id"], []))
            max_step = int(history["_step"].max())
            print(
                f"MERGE {base['name']}: segments={1 + len(boundaries)}, "
                f"rows={len(history):,}, max_step={max_step:,}"
            )
            for boundary in boundaries:
                print(
                    f"  continuation {boundary['continuation_run_id']}: "
                    f"{boundary['start_step']:,}..{boundary['end_step']:,}; "
                    f"gap={boundary['gap_steps']}"
                )
                all_boundaries.append({"base_run_name": base["name"], "base_run_id": base["id"], **boundary})

            run_dir = output_dir / f"{_safe_name(base['name'])}__{_safe_name(base['id'])}"
            row = {
                "name": base["name"],
                "id": base["id"],
                "group": base["group"],
                "segments": 1 + len(boundaries),
                "continuations": len(boundaries),
                "rows": int(len(history)),
                "columns": int(len(history.columns)),
                "max_step": max_step,
                "history_parquet": str(run_dir / "history.parquet"),
                "history_csv": None if args.no_write_csv else str(run_dir / "history.csv.gz"),
                "run_dir": str(run_dir),
            }
            manifest_rows.append(row)
            if args.dry_run:
                continue
            run_dir.mkdir(parents=True, exist_ok=False)
            history.to_parquet(run_dir / "history.parquet", index=False)
            if not args.no_write_csv:
                history.to_csv(run_dir / "history.csv.gz", index=False)
            _write_json(run_dir / "config.json", base["config"])
            _write_json(run_dir / "summary.json", _last_non_null(history))
            _write_json(
                run_dir / "metadata.json",
                {
                    **row,
                    "source_base_run_dir": str(base["run_dir"]),
                    "boundaries": boundaries,
                    "merged_at_utc": datetime.now(timezone.utc).isoformat(),
                    "merge_policy": "cut previous segment before continuation start; continuation wins overlaps",
                },
            )
        except Exception as exc:
            print(f"FAILED {base['name']}: {type(exc).__name__}: {exc}")
            failures += 1

    if not args.dry_run:
        pd.DataFrame(manifest_rows).to_csv(group_dir / "merged_manifest.csv", index=False)
        pd.DataFrame(all_boundaries).to_csv(group_dir / "continuation_boundaries.csv", index=False)
        _write_json(group_dir / "merged_manifest.json", manifest_rows)
    print(f"Done: {len(manifest_rows)} canonical runs, {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
