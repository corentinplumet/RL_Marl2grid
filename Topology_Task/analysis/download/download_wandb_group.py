#!/usr/bin/env python3
"""Download W&B run histories into a permanent run_data folder.

This is intentionally separate from the notebook cache under
Topology_Task/outputs/wandb_cache. The output layout is explicit and stable:

run_data/<download_name>/
  manifest.csv
  skipped.csv
  runs/<run_name>__<run_id>/
    metadata.json
    config.json
    summary.json
    history.parquet
    history.csv.gz
    files_manifest.csv

The script first tries the W&B full-history artifact because it preserves all
logged keys better than sampled history. If unavailable, it falls back to
run.scan_history().
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import wandb
except ImportError:  # pragma: no cover - environment dependent
    wandb = None


DEFAULT_ENTITY = "corentin-plumet-epfl"
DEFAULT_PROJECT = "Grid2Op"
DEFAULT_STATES = ("finished", "running", "crashed", "killed")
HISTORY_ARTIFACT_TYPE = "wandb-history"
HISTORY_ARTIFACT_VERSION_CANDIDATES = ("latest", "v0", "v1", "v2", "v3", "v4", "v5")


def _repo_task_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_root() -> Path:
    return _repo_task_dir() / "outputs" / "run_data"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return text or "unnamed"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _parse_states(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return DEFAULT_STATES
    value = value.strip()
    if not value or value.lower() in {"all", "none", "*"}:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _coalesce(cli_value: Any, env_values: dict[str, str], key: str, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    env_file_value = env_values.get(key)
    if env_file_value not in {None, ""}:
        return env_file_value
    env_value = os.getenv(key)
    if env_value not in {None, ""}:
        return env_value
    return default


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
    try:
        return value.to_json()
    except Exception:
        return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True), encoding="utf-8")


def _artifact_ref(entity: str, project: str, run_id: str, version: str) -> str:
    return f"{entity}/{project}/run-{run_id}-history:{version}"


def _find_history_shards(download_dir: Path) -> list[Path]:
    shards = sorted(download_dir.glob("[0-9][0-9][0-9][0-9].parquet"))
    if not shards:
        shards = sorted(download_dir.rglob("[0-9][0-9][0-9][0-9].parquet"))
    return shards


def _normalize_history(history: pd.DataFrame, run: Any) -> pd.DataFrame:
    history = history.copy()
    if "_step" not in history.columns:
        if "step" in history.columns:
            history = history.rename(columns={"step": "_step"})
        else:
            history.insert(0, "_step", range(len(history)))
    history["_step"] = pd.to_numeric(history["_step"], errors="coerce")
    history = history.dropna(subset=["_step"]).sort_values("_step")
    history = history.groupby("_step", as_index=False, dropna=False).last()
    history = history.sort_values("_step").reset_index(drop=True)

    fixed_cols = {
        "run_name": run.name,
        "run_id": run.id,
        "run_group": getattr(run, "group", None),
        "run_state": getattr(run, "state", None),
        "exp_tag": dict(run.config).get("exp_tag"),
    }
    for col, value in reversed(list(fixed_cols.items())):
        if col in history.columns:
            history[col] = history[col].fillna(value)
        else:
            history.insert(0, col, value)
    return history


def _history_from_artifact(api: Any, run: Any, entity: str, project: str, raw_dir: Path) -> tuple[pd.DataFrame, str]:
    errors = []
    for version in HISTORY_ARTIFACT_VERSION_CANDIDATES:
        ref = _artifact_ref(entity, project, run.id, version)
        try:
            artifact = api.artifact(ref, type=HISTORY_ARTIFACT_TYPE)
            download_dir = Path(artifact.download(root=str(raw_dir)))
            shards = _find_history_shards(download_dir)
            if not shards:
                raise FileNotFoundError(f"No parquet history shards found in {download_dir}")
            frames = [pd.read_parquet(path) for path in shards]
            history = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
            return history, ref
        except Exception as exc:
            errors.append(f"{ref}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load W&B history artifact. Tried " + " | ".join(errors))


def _history_from_scan(run: Any, page_size: int) -> tuple[pd.DataFrame, str]:
    rows = list(run.scan_history(page_size=page_size))
    if rows:
        return pd.DataFrame(rows), "scan_history"
    sampled = run.history(samples=20_000, pandas=True)
    if sampled is None or sampled.empty:
        raise RuntimeError("W&B returned no scan_history or sampled history rows.")
    return sampled, "sampled_history"


def _write_history(run_dir: Path, history: pd.DataFrame, *, write_csv: bool) -> dict[str, Any]:
    parquet_path = run_dir / "history.parquet"
    csv_path = run_dir / "history.csv.gz"
    parquet_written = False
    try:
        history.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:
        print(f"    parquet write failed, CSV will still be written: {type(exc).__name__}: {exc}", flush=True)
    csv_written = False
    if write_csv or not parquet_written:
        history.to_csv(csv_path, index=False)
        csv_written = True
    return {
        "history_parquet": str(parquet_path) if parquet_written else None,
        "history_csv": str(csv_path) if csv_written else None,
        "rows": int(len(history)),
        "columns": int(len(history.columns)),
        "max_step": float(pd.to_numeric(history.get("_step"), errors="coerce").max())
        if "_step" in history.columns
        else None,
    }


def _write_file_manifest(run: Any, run_dir: Path, *, download_files: bool, force: bool) -> dict[str, Any]:
    rows = []
    files_dir = run_dir / "files"
    if download_files:
        files_dir.mkdir(parents=True, exist_ok=True)
    try:
        wb_files = list(run.files())
    except Exception as exc:
        manifest_path = run_dir / "files_manifest.csv"
        pd.DataFrame(
            [
                {
                    "name": None,
                    "size": None,
                    "md5": None,
                    "updated_at": None,
                    "downloaded": False,
                    "local_path": None,
                    "manifest_error": f"{type(exc).__name__}: {exc}",
                }
            ]
        ).to_csv(manifest_path, index=False)
        return {
            "files_manifest": str(manifest_path),
            "files_total": 0,
            "files_downloaded": 0,
            "files_manifest_error": f"{type(exc).__name__}: {exc}",
        }

    for wb_file in wb_files:
        row = {
            "name": None,
            "size": None,
            "md5": None,
            "updated_at": None,
            "downloaded": False,
            "local_path": None,
        }
        for attr in ("name", "size", "md5", "updated_at"):
            try:
                row[attr] = getattr(wb_file, attr, None)
            except Exception as exc:
                row[f"{attr}_error"] = f"{type(exc).__name__}: {exc}"
        if download_files:
            try:
                local = wb_file.download(root=str(files_dir), replace=force)
                row["downloaded"] = True
                row["local_path"] = str(Path(local.name))
            except Exception as exc:
                row["download_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest_path = run_dir / "files_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return {
        "files_manifest": str(manifest_path),
        "files_total": int(len(rows)),
        "files_downloaded": int(sum(bool(row.get("downloaded")) for row in rows)),
    }


def _run_matches(
    run: Any,
    *,
    group: str | None,
    run_name_regex: re.Pattern[str] | None,
    exp_tag_regex: re.Pattern[str] | None,
    states: Iterable[str] | None,
) -> bool:
    if states is not None and run.state not in set(states):
        return False
    if group and str(getattr(run, "group", "") or "") != group:
        return False
    if run_name_regex and not run_name_regex.search(str(run.name)):
        return False
    exp_tag = str(dict(run.config).get("exp_tag", "") or "")
    if exp_tag_regex and not exp_tag_regex.search(exp_tag):
        return False
    return True


def _query_runs(
    api: Any,
    *,
    entity: str,
    project: str,
    group: str | None,
    run_name_regex: str | None,
    exp_tag_regex: str | None,
    states: Iterable[str] | None,
    max_runs: int | None,
) -> list[Any]:
    path = f"{entity}/{project}"
    filters = {"group": group} if group else None
    try:
        runs = list(api.runs(path, filters=filters, order="+created_at"))
    except Exception as exc:
        if group:
            print(
                f"Group-filtered query failed ({type(exc).__name__}: {exc}); falling back to project scan.",
                flush=True,
            )
        runs = list(api.runs(path, order="+created_at"))

    name_re = re.compile(run_name_regex) if run_name_regex else None
    exp_tag_re = re.compile(exp_tag_regex) if exp_tag_regex else None
    selected = [
        run
        for run in runs
        if _run_matches(
            run,
            group=group,
            run_name_regex=name_re,
            exp_tag_regex=exp_tag_re,
            states=states,
        )
    ]
    if max_runs is not None:
        selected = selected[:max_runs]
    return selected


def _download_one_run(
    api: Any,
    run: Any,
    *,
    entity: str,
    project: str,
    output_dir: Path,
    force: bool,
    write_csv: bool,
    keep_artifact_raw: bool,
    prefer_scan_history: bool,
    download_files: bool,
    scan_page_size: int,
) -> dict[str, Any]:
    run_dir = output_dir / "runs" / f"{_safe_name(run.name)}__{run.id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    history_parquet = run_dir / "history.parquet"
    history_csv = run_dir / "history.csv.gz"
    if not force and (history_parquet.exists() or history_csv.exists()) and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "existing"
        return metadata

    if force and run_dir.exists():
        shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "config.json", dict(run.config))
    _write_json(run_dir / "summary.json", dict(run.summary))

    raw_artifact_dir = run_dir / "history_artifact_raw"
    source = None
    try:
        if prefer_scan_history:
            raw_history, source = _history_from_scan(run, scan_page_size)
        else:
            raw_history, source = _history_from_artifact(api, run, entity, project, raw_artifact_dir)
    except Exception as artifact_exc:
        print(
            f"    artifact history unavailable for {run.name}; using scan_history fallback: "
            f"{type(artifact_exc).__name__}: {artifact_exc}",
            flush=True,
        )
        raw_history, source = _history_from_scan(run, scan_page_size)

    history = _normalize_history(raw_history, run)
    history_info = _write_history(run_dir, history, write_csv=write_csv)
    files_info = _write_file_manifest(run, run_dir, download_files=download_files, force=force)

    if raw_artifact_dir.exists() and not keep_artifact_raw:
        shutil.rmtree(raw_artifact_dir)

    metadata = {
        "status": "downloaded",
        "downloaded_at_utc": _utc_now(),
        "entity": entity,
        "project": project,
        "name": run.name,
        "id": run.id,
        "group": getattr(run, "group", None),
        "state": run.state,
        "created_at": getattr(run, "created_at", None),
        "exp_tag": dict(run.config).get("exp_tag"),
        "history_source": source,
        "run_dir": str(run_dir),
        **history_info,
        **files_info,
    }
    _write_json(metadata_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("download_config.env"))
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--group", default=None, help="Exact W&B run group name.")
    parser.add_argument("--download-name", default=None, help="Output folder name under run_data. Defaults to group or regex.")
    parser.add_argument("--run-name-regex", default=None)
    parser.add_argument("--exp-tag-regex", default=None)
    parser.add_argument("--states", default=None, help="Comma-separated states, or all. Default: finished,running,crashed,killed.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true", default=None)
    parser.add_argument("--write-csv", action="store_true", default=None)
    parser.add_argument("--no-write-csv", action="store_false", dest="write_csv")
    parser.add_argument("--download-files", action="store_true", default=None)
    parser.add_argument("--keep-artifact-raw", action="store_true", default=None)
    parser.add_argument("--prefer-scan-history", action="store_true", default=None)
    parser.add_argument("--scan-page-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    env_values = _load_env_file(cli.config)

    if wandb is None:
        raise SystemExit(
            "The `wandb` package is not installed in this Python environment. "
            "Activate your marl2grid conda env or install wandb before downloading."
        )

    entity = str(_coalesce(cli.entity, env_values, "ENTITY", DEFAULT_ENTITY))
    project = str(_coalesce(cli.project, env_values, "PROJECT", DEFAULT_PROJECT))
    group = _coalesce(cli.group, env_values, "GROUP_NAME", None) or None
    run_name_regex = _coalesce(cli.run_name_regex, env_values, "RUN_NAME_REGEX", None) or None
    exp_tag_regex = _coalesce(cli.exp_tag_regex, env_values, "EXP_TAG_REGEX", None) or None
    states = _parse_states(_coalesce(cli.states, env_values, "RUN_STATES", None))
    max_runs_raw = _coalesce(cli.max_runs, env_values, "MAX_RUNS", None)
    max_runs = int(max_runs_raw) if max_runs_raw not in {None, ""} else None
    output_root = Path(_coalesce(cli.output_root, env_values, "OUTPUT_ROOT", _default_output_root())).resolve()
    force = _parse_bool(_coalesce(cli.force, env_values, "FORCE", False))
    write_csv = _parse_bool(_coalesce(cli.write_csv, env_values, "WRITE_CSV", True))
    download_files = _parse_bool(_coalesce(cli.download_files, env_values, "DOWNLOAD_FILES", False))
    keep_artifact_raw = _parse_bool(_coalesce(cli.keep_artifact_raw, env_values, "KEEP_ARTIFACT_RAW", False))
    prefer_scan_history = _parse_bool(_coalesce(cli.prefer_scan_history, env_values, "PREFER_SCAN_HISTORY", False))
    scan_page_size = int(_coalesce(cli.scan_page_size, env_values, "SCAN_PAGE_SIZE", 200_000))

    if not any([group, run_name_regex, exp_tag_regex]):
        raise SystemExit(
            "Set GROUP_NAME in download_config.env, or pass --group / --run-name-regex / --exp-tag-regex."
        )

    download_name = _coalesce(cli.download_name, env_values, "DOWNLOAD_NAME", None)
    if not download_name:
        download_name = group or run_name_regex or exp_tag_regex
    output_dir = output_root / _safe_name(download_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========== Permanent W&B Run Download ==========")
    print(f"Entity/project: {entity}/{project}")
    print(f"Group: {group or '<not used>'}")
    print(f"Run-name regex: {run_name_regex or '<not used>'}")
    print(f"Exp-tag regex: {exp_tag_regex or '<not used>'}")
    print(f"States: {','.join(states) if states is not None else 'all'}")
    print(f"Output: {output_dir}")
    print(f"Force: {force}")
    print(f"Download files: {download_files}")
    print("================================================")

    api = wandb.Api(timeout=int(os.getenv("WANDB_API_TIMEOUT", "300")))
    runs = _query_runs(
        api,
        entity=entity,
        project=project,
        group=group,
        run_name_regex=run_name_regex,
        exp_tag_regex=exp_tag_regex,
        states=states,
        max_runs=max_runs,
    )
    if not runs:
        raise SystemExit("No W&B runs matched the requested filters.")

    print(f"Selected {len(runs)} run(s).")
    manifest_rows = []
    skipped_rows = []
    t0 = time.time()
    for idx, run in enumerate(runs, start=1):
        print(f"[{idx:>3}/{len(runs)}] {run.name} ({run.id}) state={run.state}", flush=True)
        try:
            metadata = _download_one_run(
                api,
                run,
                entity=entity,
                project=project,
                output_dir=output_dir,
                force=force,
                write_csv=write_csv,
                keep_artifact_raw=keep_artifact_raw,
                prefer_scan_history=prefer_scan_history,
                download_files=download_files,
                scan_page_size=scan_page_size,
            )
            manifest_rows.append(metadata)
            print(
                f"    {metadata.get('status')} rows={metadata.get('rows')} "
                f"cols={metadata.get('columns')} source={metadata.get('history_source')}",
                flush=True,
            )
        except Exception as exc:
            skipped = {
                "name": run.name,
                "id": run.id,
                "group": getattr(run, "group", None),
                "state": run.state,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            skipped_rows.append(skipped)
            print(f"    SKIPPED: {type(exc).__name__}: {exc}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    _write_json(output_dir / "manifest.json", manifest_rows)

    skipped = pd.DataFrame(skipped_rows)
    skipped_path = output_dir / "skipped.csv"
    skipped.to_csv(skipped_path, index=False)

    _write_json(
        output_dir / "download_request.json",
        {
            "downloaded_at_utc": _utc_now(),
            "entity": entity,
            "project": project,
            "group": group,
            "run_name_regex": run_name_regex,
            "exp_tag_regex": exp_tag_regex,
            "states": list(states) if states is not None else None,
            "max_runs": max_runs,
            "force": force,
            "write_csv": write_csv,
            "download_files": download_files,
            "keep_artifact_raw": keep_artifact_raw,
            "prefer_scan_history": prefer_scan_history,
            "scan_page_size": scan_page_size,
            "output_dir": str(output_dir),
        },
    )

    print("========== Download Complete ==========")
    print(f"Downloaded/existing: {len(manifest_rows)} / {len(runs)}")
    print(f"Skipped: {len(skipped_rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"Skipped file: {skipped_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print("=======================================")

    if skipped_rows:
        sys.exit(2)


if __name__ == "__main__":
    main()
