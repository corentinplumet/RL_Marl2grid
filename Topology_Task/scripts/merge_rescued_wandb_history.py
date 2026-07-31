#!/usr/bin/env python3
"""Merge rescued local W&B histories into the permanent local run_data cache.

This script never uploads to W&B. It reads ``run-*.wandb`` files directly,
extracts their history rows, and appends/overwrites those rows in existing
``outputs/run_data/<group>/runs/<run_name>__<run_id>/history.parquet`` files.
The merge key is ``_step``; continuation rows win when the same step exists in
both sources.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
except ImportError as exc:  # pragma: no cover - depends on active environment
    raise SystemExit(
        "Could not import W&B internals. Run this with the environment that has "
        "wandb installed, for example:\n"
        "  /opt/anaconda3/envs/marl2grid/bin/python "
        "Topology_Task/scripts/merge_rescued_wandb_history.py ..."
    ) from exc


FIXED_HISTORY_COLUMNS = ["run_name", "run_id", "run_group", "run_state", "exp_tag"]


def _task_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-") or "run"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if pd.isna(value):
        return None
    return str(value)


def _parse_wandb_json(value_json: str) -> Any:
    try:
        return json.loads(value_json)
    except Exception:
        return value_json


def _set_nested(row: dict[str, Any], nested_key: list[str], value: Any) -> None:
    if not nested_key:
        return
    if len(nested_key) == 1:
        row[nested_key[0]] = value
        return

    base = nested_key[0]
    current = row.get(base)
    if not isinstance(current, dict):
        current = {}
        row[base] = current
    cursor = current
    for part in nested_key[1:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[nested_key[-1]] = value


def _history_item_key(item: Any) -> list[str]:
    nested_key = list(item.nested_key)
    if nested_key:
        return nested_key
    key = getattr(item, "key", "")
    return [key] if key else []


def read_wandb_history(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    store = DataStore()
    store.open_for_scan(str(path))
    try:
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue

            row: dict[str, Any] = {}
            for item in record.history.item:
                _set_nested(row, _history_item_key(item), _parse_wandb_json(item.value_json))
            if record.history.step.num and "_step" not in row:
                row["_step"] = record.history.step.num
            rows.append(row)
    finally:
        store.close()

    history = pd.DataFrame(rows)
    if history.empty:
        return history
    return _normalize_history(history)


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    if "_step" not in history.columns:
        if "step" in history.columns:
            history = history.rename(columns={"step": "_step"})
        else:
            history.insert(0, "_step", range(len(history)))
    history["_step"] = pd.to_numeric(history["_step"], errors="coerce")
    history = history.dropna(subset=["_step"])
    if history.empty:
        return history
    history = history.sort_values("_step")
    history = history.groupby("_step", as_index=False, dropna=False).last()
    return history.sort_values("_step").reset_index(drop=True)


def _run_name_from_rescue_file(path: Path) -> str:
    name = path.name
    if name.startswith("run-"):
        name = name[len("run-") :]
    if name.endswith(".wandb"):
        name = name[: -len(".wandb")]
    return name


def _run_name_from_run_dir(run_dir: Path) -> str:
    meta = _read_json(run_dir / "metadata.json")
    if meta.get("name"):
        return str(meta["name"]).strip()
    if meta.get("run_name"):
        return str(meta["run_name"]).strip()
    return run_dir.name.split("__", 1)[0]


def _target_runs_by_name(runs_root: Path) -> dict[str, list[Path]]:
    targets: dict[str, list[Path]] = {}
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        name = _run_name_from_run_dir(run_dir)
        targets.setdefault(name, []).append(run_dir)
    return targets


def _load_existing_history(run_dir: Path) -> tuple[pd.DataFrame, Path | None, Path | None]:
    parquet_path = run_dir / "history.parquet"
    csv_path = run_dir / "history.csv.gz"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path), parquet_path, csv_path if csv_path.exists() else None
    if csv_path.exists():
        return pd.read_csv(csv_path), None, csv_path
    raise FileNotFoundError(f"No history.parquet or history.csv.gz in {run_dir}")


def _fixed_values(existing: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    meta = _read_json(run_dir / "metadata.json")
    config = _read_json(run_dir / "config.json")
    values = {
        "run_name": meta.get("name") or meta.get("run_name") or _run_name_from_run_dir(run_dir),
        "run_id": meta.get("id") or meta.get("run_id") or run_dir.name.rsplit("__", 1)[-1],
        "run_group": meta.get("group") or config.get("group"),
        "run_state": meta.get("state"),
        "exp_tag": meta.get("exp_tag") or config.get("exp_tag") or meta.get("name"),
    }
    for col in FIXED_HISTORY_COLUMNS:
        if col in existing.columns:
            series = existing[col].dropna()
            if not series.empty:
                values[col] = series.iloc[0]
    return values


def _with_fixed_columns(history: pd.DataFrame, values: dict[str, Any]) -> pd.DataFrame:
    history = history.copy()
    for col, value in reversed(list(values.items())):
        if col in history.columns:
            history[col] = history[col].fillna(value)
        else:
            history.insert(0, col, value)
    return history


def _merge_histories(existing: pd.DataFrame, rescue: pd.DataFrame) -> pd.DataFrame:
    existing = _normalize_history(existing)
    rescue = _normalize_history(rescue)
    existing["__merge_source_order"] = 0
    rescue["__merge_source_order"] = 1
    merged = pd.concat([existing, rescue], ignore_index=True, sort=False)
    merged = merged.sort_values(["_step", "__merge_source_order"])
    merged = merged.groupby("_step", as_index=False, dropna=False).last()
    merged = merged.drop(columns=["__merge_source_order"], errors="ignore")
    return merged.sort_values("_step").reset_index(drop=True)


def _backup_existing(run_dir: Path, timestamp: str) -> Path:
    backup_dir = run_dir / "backups" / f"rescue_merge_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for filename in ["history.parquet", "history.csv.gz", "metadata.json", "summary.json"]:
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, backup_dir / filename)
    return backup_dir


def _write_history(run_dir: Path, history: pd.DataFrame, *, write_csv: bool) -> dict[str, Any]:
    parquet_path = run_dir / "history.parquet"
    csv_path = run_dir / "history.csv.gz"
    history.to_parquet(parquet_path, index=False)
    if write_csv:
        history.to_csv(csv_path, index=False)
    max_step = pd.to_numeric(history["_step"], errors="coerce").max() if "_step" in history else None
    return {
        "history_parquet": str(parquet_path),
        "history_csv": str(csv_path) if write_csv or csv_path.exists() else None,
        "rows": int(len(history)),
        "columns": int(len(history.columns)),
        "max_step": float(max_step) if pd.notna(max_step) else None,
    }


def _last_non_null_summary(history: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    skip = set(FIXED_HISTORY_COLUMNS)
    for col in history.columns:
        if col in skip:
            continue
        values = history[col].dropna()
        if values.empty:
            continue
        summary[col] = _jsonify(values.iloc[-1])
    return summary


def _update_run_metadata(
    run_dir: Path,
    history: pd.DataFrame,
    history_info: dict[str, Any],
    *,
    rescue_file: Path,
    backup_dir: Path | None,
    timestamp: str,
) -> None:
    metadata_path = run_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    rescue_sources = list(metadata.get("local_rescue_sources") or [])
    rescue_sources.append(str(rescue_file.resolve()))
    metadata.update(
        {
            **history_info,
            "local_rescue_merged_at_utc": _utc_now(),
            "local_rescue_last_backup_dir": str(backup_dir) if backup_dir is not None else None,
            "local_rescue_sources": sorted(set(rescue_sources)),
            "local_rescue_merge_timestamp": timestamp,
        }
    )
    _write_json(metadata_path, metadata)

    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    summary.update(_last_non_null_summary(history))
    _write_json(summary_path, summary)


def _update_manifest(group_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    manifest_path = group_dir / "manifest.csv"
    if not manifest_path.exists():
        return
    manifest = pd.read_csv(manifest_path)
    for row in rows:
        mask = pd.Series(False, index=manifest.index)
        if "name" in manifest.columns:
            mask = mask | (manifest["name"].astype(str) == str(row["run_name"]))
        if "id" in manifest.columns:
            mask = mask | (manifest["id"].astype(str) == str(row["run_id"]))
        if not mask.any():
            continue
        for key, value in {
            "history_parquet": row["history_parquet"],
            "history_csv": row["history_csv"],
            "rows": row["rows"],
            "columns": row["columns"],
            "max_step": row["max_step"],
            "local_rescue_merged_at_utc": row["merged_at_utc"],
            "local_rescue_source": row["rescue_file"],
        }.items():
            if key not in manifest.columns:
                manifest[key] = None
            manifest.loc[mask, key] = value
    manifest.to_csv(manifest_path, index=False)


def parse_args() -> argparse.Namespace:
    task_dir = _task_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rescue-root",
        type=Path,
        default=task_dir.parent / "wandb-rescue-20260730",
        help="Folder containing rescued run-*.wandb files.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=task_dir / "outputs" / "run_data" / "gs_hmd" / "runs",
        help="Local run_data runs folder to merge into.",
    )
    parser.add_argument(
        "--run-name-regex",
        default=None,
        help="Optional regex filter for rescued run names.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report planned merges without writing files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not backup overwritten local history files.")
    parser.add_argument("--no-write-csv", action="store_true", help="Only update history.parquet, not history.csv.gz.")
    parser.add_argument("--allow-missing-target", action="store_true", help="Skip rescued runs missing from runs-root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rescue_root = args.rescue_root.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    if not rescue_root.exists():
        raise SystemExit(f"Rescue root does not exist: {rescue_root}")
    if not runs_root.exists():
        raise SystemExit(f"Local runs root does not exist: {runs_root}")

    name_re = re.compile(args.run_name_regex) if args.run_name_regex else None
    rescue_files = sorted(rescue_root.glob("*/run-*.wandb"))
    if name_re is not None:
        rescue_files = [path for path in rescue_files if name_re.search(_run_name_from_rescue_file(path))]
    if not rescue_files:
        raise SystemExit(f"No rescued run-*.wandb files found under {rescue_root}")

    targets = _target_runs_by_name(runs_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    group_dir = runs_root.parent
    merge_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    print(f"Rescue root: {rescue_root}")
    print(f"Local runs root: {runs_root}")
    print(f"Rescued files: {len(rescue_files)}")
    print(f"Dry run: {args.dry_run}")

    for rescue_file in rescue_files:
        run_name = _run_name_from_rescue_file(rescue_file)
        matches = targets.get(run_name, [])
        if not matches:
            message = f"No local target run found for {run_name}"
            if args.allow_missing_target:
                print(f"SKIP {run_name}: {message}")
                skipped_rows.append({"run_name": run_name, "reason": message, "rescue_file": str(rescue_file)})
                continue
            raise SystemExit(message)
        if len(matches) > 1:
            raise SystemExit(f"Multiple local target runs found for {run_name}: {matches}")

        run_dir = matches[0]
        existing, _, _ = _load_existing_history(run_dir)
        rescue = read_wandb_history(rescue_file)
        if rescue.empty:
            print(f"SKIP {run_name}: no history rows in {rescue_file}")
            skipped_rows.append({"run_name": run_name, "reason": "no history rows", "rescue_file": str(rescue_file)})
            continue

        fixed_values = _fixed_values(existing, run_dir)
        rescue = _with_fixed_columns(rescue, fixed_values)
        merged = _merge_histories(existing, rescue)

        old_max = float(pd.to_numeric(existing["_step"], errors="coerce").max())
        rescue_min = float(pd.to_numeric(rescue["_step"], errors="coerce").min())
        rescue_max = float(pd.to_numeric(rescue["_step"], errors="coerce").max())
        new_max = float(pd.to_numeric(merged["_step"], errors="coerce").max())
        added_rows = len(merged) - len(existing)
        print(
            f"MERGE {run_name}: existing rows={len(existing)} max={old_max:.0f}; "
            f"rescue rows={len(rescue)} range={rescue_min:.0f}-{rescue_max:.0f}; "
            f"merged rows={len(merged)} max={new_max:.0f}; added={added_rows}"
        )

        history_info = {
            "history_parquet": str(run_dir / "history.parquet"),
            "history_csv": str(run_dir / "history.csv.gz"),
            "rows": int(len(merged)),
            "columns": int(len(merged.columns)),
            "max_step": new_max,
        }
        row = {
            "run_name": fixed_values["run_name"],
            "run_id": fixed_values["run_id"],
            "run_dir": str(run_dir),
            "rescue_file": str(rescue_file),
            "existing_rows": int(len(existing)),
            "rescue_rows": int(len(rescue)),
            "merged_rows": int(len(merged)),
            "old_max_step": old_max,
            "rescue_min_step": rescue_min,
            "rescue_max_step": rescue_max,
            "max_step": new_max,
            "rows": int(len(merged)),
            "columns": int(len(merged.columns)),
            "history_parquet": history_info["history_parquet"],
            "history_csv": history_info["history_csv"],
            "merged_at_utc": _utc_now(),
        }
        merge_rows.append(row)

        if args.dry_run:
            continue

        backup_dir = None if args.no_backup else _backup_existing(run_dir, timestamp)
        written_info = _write_history(run_dir, merged, write_csv=not args.no_write_csv)
        _update_run_metadata(
            run_dir,
            merged,
            written_info,
            rescue_file=rescue_file,
            backup_dir=backup_dir,
            timestamp=timestamp,
        )
        row.update(written_info)

    if not args.dry_run:
        _update_manifest(group_dir, merge_rows)

    report_dir = group_dir / "local_rescue_merges"
    if merge_rows or skipped_rows:
        report_dir.mkdir(parents=True, exist_ok=True)
    if merge_rows:
        pd.DataFrame(merge_rows).to_csv(report_dir / f"merge_report_{timestamp}.csv", index=False)
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(report_dir / f"skipped_{timestamp}.csv", index=False)

    print(f"Done. Merged {len(merge_rows)} run(s), skipped {len(skipped_rows)}.")
    if merge_rows:
        print(f"Report: {report_dir / f'merge_report_{timestamp}.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
