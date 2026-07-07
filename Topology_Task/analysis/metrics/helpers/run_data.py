"""Helpers for permanent W&B run_data downloads.

The permanent downloader writes runs under ``Topology_Task/outputs/run_data``.
These helpers make the plotting notebooks treat that folder like the old W&B
history cache, while resolving paths locally after copying data between machines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional
import json
import os
import re

import pandas as pd


def find_task_dir(start: Optional[Path] = None) -> Path:
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "main.py").is_file():
            return candidate
        task_dir = candidate / "Topology_Task"
        if (task_dir / "main.py").is_file():
            return task_dir
    raise RuntimeError("Could not locate Topology_Task/main.py.")


TASK_DIR = find_task_dir()
RUN_DATA_DIR = Path(os.getenv("MARL2GRID_RUN_DATA_DIR", TASK_DIR / "outputs" / "run_data")).resolve()
LEGACY_DOWNLOAD_RUN_DATA_DIR = TASK_DIR / "analysis" / "download" / "run_data"
RUN_DATA_INDEX_PATH = RUN_DATA_DIR / "run_data_index.csv"


def safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-") or "run"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _split_run_folder_name(run_dir: Path) -> tuple[str, str]:
    name = run_dir.name
    if "__" in name:
        run_name, run_id = name.rsplit("__", 1)
        return run_name, run_id
    return name, name


def _existing_path(*candidates: Any) -> Optional[Path]:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _local_history_path(run_dir: Path, meta: dict[str, Any], key: str, filename: str) -> Optional[Path]:
    # First trust the copied/local folder. Metadata can contain stale absolute
    # paths from a cluster after the folder has been downloaded to a laptop.
    return _existing_path(run_dir / filename, meta.get(key))


def _load_config_summary(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _read_json(run_dir / "config.json"), _read_json(run_dir / "summary.json")


def _row_from_run_dir(run_dir: Path, group_dir: Path) -> Optional[dict[str, Any]]:
    meta = _read_json(run_dir / "metadata.json")
    config, summary = _load_config_summary(run_dir)
    fallback_name, fallback_id = _split_run_folder_name(run_dir)

    run_name = str(meta.get("name") or meta.get("run_name") or config.get("exp_tag") or fallback_name).strip()
    run_id = str(meta.get("id") or meta.get("run_id") or fallback_id).strip()
    if not run_name or not run_id:
        return None

    parquet_path = _local_history_path(run_dir, meta, "history_parquet", "history.parquet")
    csv_path = _local_history_path(run_dir, meta, "history_csv", "history.csv.gz")
    if parquet_path is None and csv_path is None:
        return None

    group_name = group_dir.name
    exp_tag = meta.get("exp_tag") or config.get("exp_tag") or run_name

    return {
        "name": run_name,
        "id": run_id,
        "run_name": run_name,
        "run_id": run_id,
        "state": meta.get("state"),
        "created_at": meta.get("created_at"),
        "group": meta.get("group") or config.get("group"),
        "download_group": group_name,
        "source": "run_data",
        "exp_tag": exp_tag,
        "actor_encoder": config.get("actor_encoder") or meta.get("actor_encoder"),
        "critic_encoder": config.get("critic_encoder") or meta.get("critic_encoder"),
        "gnn_type": config.get("gnn_type") or meta.get("gnn_type"),
        "gnn_include_neighbors": config.get("gnn_include_neighbors") or meta.get("gnn_include_neighbors"),
        "deterministic_eval": config.get("deterministic_eval") or meta.get("deterministic_eval"),
        "n_envs": config.get("n_envs") or meta.get("n_envs"),
        "n_steps": config.get("n_steps") or meta.get("n_steps"),
        "seed": config.get("seed") or meta.get("seed"),
        "total_timesteps": config.get("total_timesteps") or meta.get("total_timesteps"),
        "entropy_coef": config.get("entropy_coef") or meta.get("entropy_coef"),
        "entropy_coef_final": config.get("entropy_coef_final") or meta.get("entropy_coef_final"),
        "init_do_nothing_prob": config.get("init_do_nothing_prob") or meta.get("init_do_nothing_prob"),
        "global_step": summary.get("charts/global_step") or meta.get("max_step"),
        "test_survival": summary.get("test/charts/episodic_survival", summary.get("test/episodic_survival")),
        "train_eval_survival": summary.get(
            "train_eval/charts/episodic_survival",
            summary.get("train_eval/episodic_survival"),
        ),
        "legacy_survival": summary.get("charts/episodic_survival"),
        "artifact": meta.get("history_source") or meta.get("artifact_resolved") or meta.get("artifact_requested"),
        "history_parquet": parquet_path,
        "history_csv": csv_path,
        "rows": meta.get("rows"),
        "columns": meta.get("columns"),
        "run_dir": run_dir.resolve(),
        "run_data_dir": group_dir.resolve(),
    }


def _candidate_group_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    if (root / "runs").is_dir() or (root / "manifest.csv").is_file():
        return [root]
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and ((path / "runs").is_dir() or (path / "manifest.csv").is_file())
    ]


def scan_run_data(root: Optional[Path] = None, *, include_legacy: bool = True) -> pd.DataFrame:
    roots = [Path(root).expanduser().resolve()] if root is not None else [RUN_DATA_DIR]
    if include_legacy and root is None and LEGACY_DOWNLOAD_RUN_DATA_DIR != RUN_DATA_DIR:
        roots.append(LEGACY_DOWNLOAD_RUN_DATA_DIR)

    rows: list[dict[str, Any]] = []
    for current_root in roots:
        for group_dir in _candidate_group_dirs(current_root):
            runs_dir = group_dir / "runs"
            if not runs_dir.exists():
                continue
            for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
                row = _row_from_run_dir(run_dir, group_dir)
                if row is not None:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["run_name", "run_id"], keep="last")
    return df.sort_values(["download_group", "run_name", "run_id"]).reset_index(drop=True)


def write_run_data_index(root: Optional[Path] = None) -> Optional[Path]:
    df = scan_run_data(root)
    if df.empty:
        return None
    target_root = Path(root).expanduser().resolve() if root is not None else RUN_DATA_DIR
    target_root.mkdir(parents=True, exist_ok=True)
    path = target_root / "run_data_index.csv"
    df_to_write = df.copy()
    for column in ["history_parquet", "history_csv", "run_dir", "run_data_dir"]:
        if column in df_to_write.columns:
            df_to_write[column] = df_to_write[column].map(lambda value: str(value) if value else "")
    df_to_write.to_csv(path, index=False)
    return path


def active_history_index_path(fallback: Path) -> Path:
    path = write_run_data_index()
    return path if path is not None and path.exists() else fallback
