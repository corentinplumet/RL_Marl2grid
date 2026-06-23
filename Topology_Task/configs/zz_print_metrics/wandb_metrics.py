"""Reusable W&B metric loading and plotting helpers.

This module was extracted from ``../wandb_run_plots.ipynb`` so the notebook can stay
small while the plotting and cache-loading logic remains editable Python code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import shutil
import time

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:
    raise ImportError("Install plotly first, for example: pip install plotly") from exc

try:
    import wandb
except ImportError:
    wandb = None

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)


ENTITY = os.getenv("WANDB_ENTITY", "corentin-plumet-epfl")
PROJECT = os.getenv("WANDB_PROJECT", "Grid2Op")

# None means every W&B run in the project. Use a regex to focus on a subset.
ENTROPY_DECAY_SEED_SWEEP_REGEX = (
    r"^noval20_mlp_(?:a1_entropy_decay|a3_entropy_decay|a3_logit_decay|a4_entropy_decay|p999_decay)"
    r"(?:_s[0-2])?_(?:det|stoch)$"
)
NO_ENTROPY_DECAY_SEED_SWEEP_REGEX = (
    r"^noval20_mlp_(?:a1_no_entropy_decay|a3_no_entropy_decay|a3_logit_decay_no_entropy_decay|a4_no_entropy_decay)_"
    r"s[0-2]_(?:det|stoch)$"
)
RUN_NAME_REGEX = None
EXCLUDE_RUN_NAME_REGEX = None
RUN_STATES = None  # None includes finished, crashed, and killed cached histories.
MAX_RUNS = None    # None means all matching runs

# Full history is downloaded from W&B history artifacts: run-<run_id>-history:<version>.
CACHE_MODE = "full"
FORCE_REFRESH = False
USE_LOCAL_CACHE_ONLY = True  # False downloads missing histories from W&B; True only reads local cache.
REFRESH_SCAN_HISTORY_FALLBACKS = False  # Replace old API fallback caches when downloading from W&B.
ALLOW_SCAN_HISTORY_FALLBACK = True  # Use W&B scalar history API if the history artifact is unavailable.
CACHE_STALE_STEP_TOLERANCE = 0.99  # Refresh cache if max cached step is behind run summary by more than this ratio.
WRITE_FULL_HISTORY_CSV = True
SKIP_FAILED_DOWNLOADS = True  # Continue when active runs do not have history artifacts yet.
WANDB_API_TIMEOUT = 300
HISTORY_ARTIFACT_TYPE = "wandb-history"
HISTORY_ARTIFACT_VERSION = "latest"
HISTORY_ARTIFACT_FALLBACK_VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5"]
SCAN_HISTORY_FALLBACK_PAGE_SIZE = 200_000
SCAN_HISTORY_FALLBACK_SAMPLES = 20_000
SHOW_FIGURES = False

METRICS = [
    "charts/episodic_survival",
    "test/charts/episodic_survival",
    "test/episodic_survival",
    "validation/episodic_survival",
    "train_eval/charts/episodic_survival",
    "train_eval/episodic_survival",
    "train/entropy_coef",
    "train/action0_logit_bonus",
    "train/entropy_agent_0",
    "train/entropy_agent_1",
    "train/entropy_agent_2",
    "train/frac_action_0_agent_0",
    "train/frac_action_0_agent_1",
    "train/frac_action_0_agent_2",
    "train/approx_kl_agent_0",
    "train/approx_kl_agent_1",
    "train/approx_kl_agent_2",
    "train/lr_actor",
    "train/lr_critic",
]


def find_task_dir(start=None):
    """Locate the Topology_Task directory from a notebook, repo root, or subfolder."""
    start = Path(start or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "main.py").is_file():
            return candidate
        task_dir = candidate / "Topology_Task"
        if (task_dir / "main.py").is_file():
            return task_dir
    raise RuntimeError("Could not locate Topology_Task/main.py from the current working directory.")


def configure_task_dir(task_dir=None):
    """Configure cache and figure paths. Call this if the notebook runs from an unusual cwd."""
    global TASK_DIR, CACHE_DIR, FULL_CACHE_DIR, METRIC_CACHE_DIR, FIG_DIR
    global MANIFEST_PATH, CACHE_INDEX_PATH, FAILED_HISTORY_DOWNLOADS_PATH

    TASK_DIR = Path(task_dir).resolve() if task_dir is not None else find_task_dir()
    CACHE_DIR = TASK_DIR / "outputs" / "wandb_cache"
    FULL_CACHE_DIR = CACHE_DIR / "full_history"
    METRIC_CACHE_DIR = CACHE_DIR / "metric_history"
    FIG_DIR = TASK_DIR / "outputs" / "wandb_figures"
    for directory in [CACHE_DIR, FULL_CACHE_DIR, METRIC_CACHE_DIR, FIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    MANIFEST_PATH = CACHE_DIR / "selected_runs_manifest.csv"
    CACHE_INDEX_PATH = CACHE_DIR / "full_history_cache_index.csv"
    FAILED_HISTORY_DOWNLOADS_PATH = CACHE_DIR / "failed_history_downloads.csv"
    return TASK_DIR


configure_task_dir()

name_re = None
exclude_name_re = None
api = None
all_runs = []
selected_runs = []
runs_df = pd.DataFrame()
histories = []
skipped_history_downloads_df = pd.DataFrame()
history_df = pd.DataFrame(columns=["run_name", "run_id", "metric", "step", "value", "step_millions"])

_UNSET = object()


@dataclass
class WandbPlotData:
    runs_df: pd.DataFrame
    history_df: pd.DataFrame
    skipped_history_downloads_df: pd.DataFrame
    histories: list[pd.DataFrame]

    def as_globals(self):
        return {
            "runs_df": self.runs_df,
            "history_df": self.history_df,
            "skipped_history_downloads_df": self.skipped_history_downloads_df,
            "histories": self.histories,
        }


def refresh_run_filters():
    """Recompile regex filters after changing RUN_NAME_REGEX/EXCLUDE_RUN_NAME_REGEX."""
    global name_re, exclude_name_re
    name_re = re.compile(RUN_NAME_REGEX) if RUN_NAME_REGEX else None
    exclude_name_re = re.compile(EXCLUDE_RUN_NAME_REGEX) if EXCLUDE_RUN_NAME_REGEX else None
    return name_re, exclude_name_re


refresh_run_filters()


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _is_all_config_choice(config_folder):
    if config_folder is None:
        return True
    if isinstance(config_folder, str):
        return config_folder.strip().lower() in {"", "all", "none", "null"}
    return False


def _normalize_config_folder_choices(config_folder):
    if isinstance(config_folder, str) and "," in config_folder:
        return [choice.strip() for choice in config_folder.split(",") if choice.strip()]
    if isinstance(config_folder, (list, tuple, set)):
        return list(config_folder)
    return None


def resolve_config_folder(config_folder):
    """Resolve a config folder name/path under TASK_DIR/configs."""
    if _is_all_config_choice(config_folder):
        return None
    path = Path(config_folder).expanduser()
    candidates = [
        path,
        TASK_DIR / "configs" / path,
        TASK_DIR / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find config folder {config_folder!r}. "
        f"Try a folder under {TASK_DIR / 'configs'}, or use None/'all'."
    )


def _with_inserted_token_after_best_number(name, token):
    match = re.match(r"^(best_\d+)(_.+)$", str(name))
    if not match:
        return None
    return f"{match.group(1)}_{token}{match.group(2)}"


def _unique_text(values):
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def config_folder_run_names(config_folder):
    """Return W&B run-name candidates from one or more TOML config folders.

    Use None, "None", or "all" for no filtering. Pass a list/tuple/set or a
    comma-separated string to combine several folders.
    """
    choices = _normalize_config_folder_choices(config_folder)
    if choices is not None:
        names = []
        for choice in choices:
            choice_names = config_folder_run_names(choice)
            if choice_names is None:
                return None
            names.extend(choice_names)
        return _unique_text(names)

    folder = resolve_config_folder(config_folder)
    if folder is None:
        return None

    names = []
    for config_path in sorted(folder.glob("*.toml")):
        cfg = _load_toml(config_path)
        args = cfg.get("args", {})
        run = cfg.get("run", {})
        base_names = _unique_text([run.get("name"), args.get("exp_tag"), config_path.stem])
        names.extend(base_names)

        include_neighbors = bool(args.get("gnn_include_neighbors", False)) or "include_neighbors" in folder.name
        if include_neighbors:
            for name in base_names:
                names.append(_with_inserted_token_after_best_number(name, "neighbors"))

    names = _unique_text(names)
    if not names:
        raise FileNotFoundError(f"No TOML config run names found under {folder}")
    return names


def run_name_regex_from_config_folder(config_folder):
    """Build a RUN_NAME_REGEX from config folder choice(s), or None for all runs."""
    names = config_folder_run_names(config_folder)
    if names is None:
        return None
    escaped = [re.escape(name) for name in names]
    return r"^\s*(?:" + "|".join(escaped) + r")\s*$"


def configure_run_filter_from_config_folder(config_folder, *, verbose=True):
    """Set RUN_NAME_REGEX from config folder choice(s) and return the regex."""
    global RUN_NAME_REGEX

    RUN_NAME_REGEX = run_name_regex_from_config_folder(config_folder)
    refresh_run_filters()
    if verbose:
        if RUN_NAME_REGEX is None:
            print("Config folder filter: all W&B runs")
        else:
            names = config_folder_run_names(config_folder)
            choices = _normalize_config_folder_choices(config_folder)
            if choices is None:
                print(f"Config folder filter: {resolve_config_folder(config_folder)}")
            else:
                folders = [resolve_config_folder(choice) for choice in choices]
                print("Config folder filter:")
                for folder in folders:
                    print(f"  - {folder}")
            print(f"Matched config run-name candidates: {len(names)}")
    return RUN_NAME_REGEX


def print_wandb_setup():
    print(f"Project: {ENTITY}/{PROJECT}")
    print(f"Task dir: {TASK_DIR}")
    print(f"Cache mode: {CACHE_MODE}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Local-only mode: {USE_LOCAL_CACHE_ONLY}")
    print(f"Force refresh: {FORCE_REFRESH}")
    print(f"Refresh scan-history fallbacks: {REFRESH_SCAN_HISTORY_FALLBACKS}")


def _ensure_wandb_api():
    global api
    if wandb is None:
        raise ImportError("Install wandb first, for example: pip install wandb")
    if api is None:
        api = wandb.Api(timeout=WANDB_API_TIMEOUT)
    return api


# Run discovery helpers extracted from the original notebook.
def run_name_matches(name, exp_tag=""):
    if exclude_name_re is not None and (
        exclude_name_re.search(str(name)) or exclude_name_re.search(str(exp_tag))
    ):
        return False
    if name_re is None:
        return True
    return bool(name_re.search(str(name)) or name_re.search(str(exp_tag)))


def _metadata_json_paths():
    return sorted(FULL_CACHE_DIR.glob("*/metadata.json"))


def _cache_file_from_metadata(meta, meta_path, key, filename):
    raw_path = meta.get(key)
    if raw_path:
        path = Path(raw_path)
        if path.exists():
            return path
    fallback = meta_path.parent / filename
    return fallback if fallback.exists() else None


def cached_runs_from_full_history():
    rows = []
    for meta_path in _metadata_json_paths():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid metadata {meta_path}: {exc}")
            continue

        name = meta.get("name") or meta.get("run_name") or meta_path.parent.name.split("__", 1)[0]
        run_id = meta.get("id") or meta.get("run_id") or meta_path.parent.name.split("__")[-1]
        exp_tag = meta.get("exp_tag") or name
        parquet_path = _cache_file_from_metadata(meta, meta_path, "history_parquet", "history.parquet")
        csv_path = _cache_file_from_metadata(meta, meta_path, "history_csv", "history.csv.gz")
        if parquet_path is None and csv_path is None:
            print(f"Skipping {name}: no history.parquet or history.csv.gz found")
            continue

        rows.append({
            "name": name,
            "id": run_id,
            "state": meta.get("state"),
            "created_at": meta.get("created_at"),
            "exp_tag": exp_tag,
            "actor_encoder": meta.get("actor_encoder"),
            "critic_encoder": meta.get("critic_encoder"),
            "gnn_type": meta.get("gnn_type"),
            "deterministic_eval": meta.get("deterministic_eval"),
            "n_envs": meta.get("n_envs"),
            "n_steps": meta.get("n_steps"),
            "entropy_coef": meta.get("entropy_coef"),
            "entropy_coef_final": meta.get("entropy_coef_final"),
            "init_do_nothing_prob": meta.get("init_do_nothing_prob"),
            "artifact": meta.get("artifact_resolved") or meta.get("artifact_requested"),
            "history_parquet": str(parquet_path) if parquet_path else None,
            "history_csv": str(csv_path) if csv_path else None,
            "rows": meta.get("rows"),
            "columns": meta.get("columns"),
        })
    return pd.DataFrame(rows)


def load_runs(
    *,
    run_name_regex=_UNSET,
    exclude_run_name_regex=_UNSET,
    run_states=_UNSET,
    max_runs=_UNSET,
    use_local_cache_only=_UNSET,
    verbose=True,
):
    """Load the W&B run manifest from local cache or the W&B API."""
    global RUN_NAME_REGEX, EXCLUDE_RUN_NAME_REGEX, RUN_STATES, MAX_RUNS, USE_LOCAL_CACHE_ONLY
    global api, all_runs, selected_runs, runs_df

    if run_name_regex is not _UNSET:
        RUN_NAME_REGEX = run_name_regex
    if exclude_run_name_regex is not _UNSET:
        EXCLUDE_RUN_NAME_REGEX = exclude_run_name_regex
    if run_states is not _UNSET:
        RUN_STATES = run_states
    if max_runs is not _UNSET:
        MAX_RUNS = max_runs
    if use_local_cache_only is not _UNSET:
        USE_LOCAL_CACHE_ONLY = use_local_cache_only
    refresh_run_filters()

    if USE_LOCAL_CACHE_ONLY:
        loaded_runs = cached_runs_from_full_history()
        if loaded_runs.empty:
            raise FileNotFoundError(
                f"No cached histories found under {FULL_CACHE_DIR}. Download finished runs first."
            )
        loaded_runs = loaded_runs[
            loaded_runs.apply(lambda row: run_name_matches(row.get("name"), row.get("exp_tag", "")), axis=1)
        ]
        if RUN_STATES is not None and "state" in loaded_runs.columns:
            loaded_runs = loaded_runs[loaded_runs["state"].isin(RUN_STATES)]
        loaded_runs = loaded_runs.sort_values(["name", "id"]).reset_index(drop=True)
        if MAX_RUNS is not None:
            loaded_runs = loaded_runs.head(MAX_RUNS)
        selected_runs = []
        all_runs = []
        runs_df = loaded_runs
        if verbose:
            print(f"Selected {len(runs_df)} cached runs from {FULL_CACHE_DIR}")
            if "state" in runs_df.columns:
                print(runs_df["state"].value_counts(dropna=False).to_string())
        return runs_df

    api = _ensure_wandb_api()
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    def run_matches(run):
        if RUN_STATES is not None and run.state not in RUN_STATES:
            return False
        exp_tag = str(run.config.get("exp_tag", ""))
        return run_name_matches(run.name, exp_tag)

    selected_runs = [run for run in all_runs if run_matches(run)]
    selected_runs = sorted(selected_runs, key=lambda run: getattr(run, "created_at", "") or "")
    if MAX_RUNS is not None:
        selected_runs = selected_runs[:MAX_RUNS]

    summary_rows = []
    for run in selected_runs:
        summary = dict(run.summary)
        config = dict(run.config)
        summary_rows.append({
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "created_at": getattr(run, "created_at", None),
            "exp_tag": config.get("exp_tag"),
            "actor_encoder": config.get("actor_encoder"),
            "critic_encoder": config.get("critic_encoder"),
            "gnn_type": config.get("gnn_type"),
            "gnn_include_neighbors": config.get("gnn_include_neighbors"),
            "deterministic_eval": config.get("deterministic_eval"),
            "n_envs": config.get("n_envs"),
            "n_steps": config.get("n_steps"),
            "seed": config.get("seed"),
            "total_timesteps": config.get("total_timesteps"),
            "entropy_coef": config.get("entropy_coef"),
            "entropy_coef_final": config.get("entropy_coef_final"),
            "init_do_nothing_prob": config.get("init_do_nothing_prob"),
            "global_step": summary.get("charts/global_step"),
            "test_survival": summary.get("test/charts/episodic_survival", summary.get("test/episodic_survival")),
            "train_eval_survival": summary.get("train_eval/charts/episodic_survival", summary.get("train_eval/episodic_survival")),
            "legacy_survival": summary.get("charts/episodic_survival"),
        })

    runs_df = pd.DataFrame(summary_rows)
    runs_df.to_csv(MANIFEST_PATH, index=False)
    if verbose:
        print(f"Selected {len(selected_runs)} / {len(all_runs)} runs")
        print(f"Saved manifest: {MANIFEST_PATH}")
    return runs_df


# History-cache helpers extracted from the original notebook.
def safe_name(text):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return text or "run"


def run_cache_dir(run_name, run_id):
    return FULL_CACHE_DIR / f"{safe_name(run_name)}__{run_id}"


def full_parquet_path(run_name, run_id):
    return run_cache_dir(run_name, run_id) / "history.parquet"


def full_csv_path(run_name, run_id):
    return run_cache_dir(run_name, run_id) / "history.csv.gz"


def metadata_path(run_name, run_id):
    return run_cache_dir(run_name, run_id) / "metadata.json"


def artifact_path_for_run(run_id, version=None):
    version = version or HISTORY_ARTIFACT_VERSION
    return f"{ENTITY}/{PROJECT}/run-{run_id}-history:{version}"


def artifact_versions_to_try():
    versions = [HISTORY_ARTIFACT_VERSION] + list(HISTORY_ARTIFACT_FALLBACK_VERSIONS)
    deduped = []
    for version in versions:
        if version and version not in deduped:
            deduped.append(version)
    return deduped


def _progress(prefix, idx, total, name):
    return f"[{idx:>2}/{total}] {prefix}: {name}"


def _row_get(row, key, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _read_history_table(parquet_path, csv_path=None):
    parquet_path = Path(parquet_path) if parquet_path is not None else None
    csv_path = Path(csv_path) if csv_path is not None else None
    if parquet_path is not None and parquet_path.exists():
        try:
            history = pd.read_parquet(parquet_path)
            history.attrs["source_path"] = str(parquet_path)
            return history
        except ImportError as exc:
            if csv_path is not None and csv_path.exists():
                print(f"    parquet reader unavailable; loading CSV cache instead: {csv_path}", flush=True)
                history = pd.read_csv(csv_path)
                history.attrs["source_path"] = str(csv_path)
                return history
            raise ImportError(
                "Reading W&B history parquet requires pyarrow or fastparquet. "
                "Install one of them, for example: pip install pyarrow"
            ) from exc
    if csv_path is not None and csv_path.exists():
        history = pd.read_csv(csv_path)
        history.attrs["source_path"] = str(csv_path)
        return history
    raise FileNotFoundError(f"Missing cached history file: parquet={parquet_path}, csv={csv_path}")


def _ensure_run_columns(history, run_name, run_id):
    history = history.copy()
    if "run_name" not in history.columns:
        history.insert(0, "run_name", run_name)
    else:
        history["run_name"] = history["run_name"].fillna(run_name)
    if "run_id" not in history.columns:
        history.insert(1, "run_id", run_id)
    else:
        history["run_id"] = history["run_id"].fillna(run_id)
    return history


def _read_cache_metadata(run_name, run_id):
    meta_path = metadata_path(run_name, run_id)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _is_scan_history_fallback_cache(run_name, run_id):
    metadata = _read_cache_metadata(run_name, run_id)
    artifact = str(metadata.get("artifact_resolved") or metadata.get("artifact") or "")
    source = str(metadata.get("source") or "")
    return artifact == "scan_history:fallback" or "scan_history" in source


def _history_max_step(history):
    if history is None or history.empty:
        return np.nan
    step_col = "_step" if "_step" in history.columns else "step" if "step" in history.columns else None
    if step_col is None:
        return np.nan
    return pd.to_numeric(history[step_col], errors="coerce").max()


def _cached_history_max_step(run_name, run_id):
    parquet_path = full_parquet_path(run_name, run_id)
    csv_path = full_csv_path(run_name, run_id)
    if not parquet_path.exists() and not csv_path.exists():
        return np.nan
    try:
        history = _read_history_table(parquet_path, csv_path=csv_path)
    except Exception:
        return np.nan
    return _history_max_step(history)


def _summary_global_step(row):
    value = _row_get(row, "global_step")
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _cache_is_stale_for_summary(run_name, run_id, row):
    summary_step = _summary_global_step(row)
    if not np.isfinite(summary_step) or summary_step <= 0:
        return False, np.nan, summary_step
    cached_step = _cached_history_max_step(run_name, run_id)
    if not np.isfinite(cached_step):
        return False, cached_step, summary_step
    return cached_step < summary_step * float(CACHE_STALE_STEP_TOLERANCE), cached_step, summary_step


def _find_downloaded_history(download_dir):
    download_dir = Path(download_dir)
    matches = sorted(download_dir.glob("[0-9][0-9][0-9][0-9].parquet"))
    if not matches:
        matches = sorted(download_dir.rglob("[0-9][0-9][0-9][0-9].parquet"))
    if not matches:
        raise FileNotFoundError(f"Could not find W&B history parquet shards under {download_dir}")
    return matches


def _read_downloaded_history_artifact(download_dir):
    shard_paths = _find_downloaded_history(download_dir)
    frames = [pd.read_parquet(path) for path in shard_paths]
    if not frames:
        raise FileNotFoundError(f"No W&B history parquet shards found under {download_dir}")
    if len(frames) == 1:
        return frames[0]
    history = pd.concat(frames, ignore_index=True, sort=False)
    step_col = "_step" if "_step" in history.columns else "step" if "step" in history.columns else None
    if step_col is not None:
        history[step_col] = pd.to_numeric(history[step_col], errors="coerce")
        history = history.dropna(subset=[step_col]).sort_values(step_col)
        history = history.groupby(step_col, as_index=False, dropna=False).last()
        history = history.sort_values(step_col).reset_index(drop=True)
    return history


wandb_download_run = None


def _get_history_artifact(run_id):
    wb_api = _ensure_wandb_api()
    errors = []
    for version in artifact_versions_to_try():
        artifact_ref = artifact_path_for_run(run_id, version=version)
        try:
            artifact = wb_api.artifact(artifact_ref, type=HISTORY_ARTIFACT_TYPE)
            if errors:
                print(f"    resolved history artifact with {artifact_ref}", flush=True)
            return artifact, artifact_ref
        except Exception as api_exc:
            errors.append((artifact_ref, api_exc))
            print(
                f"    api.artifact could not load {artifact_ref}: {type(api_exc).__name__}: {api_exc}",
                flush=True,
            )

    # Fallback to the method recommended in W&B examples.
    global wandb_download_run
    if wandb_download_run is None:
        wandb_download_run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            job_type="download-history-artifacts",
            name="download_history_artifacts",
            reinit=True,
        )

    for version in artifact_versions_to_try():
        artifact_ref = artifact_path_for_run(run_id, version=version)
        try:
            artifact = wandb_download_run.use_artifact(artifact_ref, type=HISTORY_ARTIFACT_TYPE)
            print(f"    resolved history artifact with {artifact_ref}", flush=True)
            return artifact, artifact_ref
        except Exception as use_exc:
            errors.append((artifact_ref, use_exc))
            print(
                f"    use_artifact could not load {artifact_ref}: {type(use_exc).__name__}: {use_exc}",
                flush=True,
            )

    tried = ", ".join(ref for ref, _ in errors)
    raise RuntimeError(f"Could not find a history artifact for run {run_id}. Tried: {tried}")


def _scan_history_all_keys_table(run_id, page_size=SCAN_HISTORY_FALLBACK_PAGE_SIZE):
    wb_api = _ensure_wandb_api()
    run = wb_api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    rows = list(run.scan_history(page_size=page_size))
    if not rows:
        return None
    history = pd.DataFrame(rows)
    if history.empty:
        return None
    if "_step" not in history.columns:
        if "step" in history.columns:
            history = history.rename(columns={"step": "_step"})
        else:
            history.insert(0, "_step", np.arange(len(history), dtype=float))
    return history.drop_duplicates(subset=["_step"], keep="last").sort_values("_step").reset_index(drop=True)


def _sampled_full_history_table(run_id, samples=SCAN_HISTORY_FALLBACK_SAMPLES):
    wb_api = _ensure_wandb_api()
    run = wb_api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    history = run.history(samples=samples, pandas=True)
    if history is None or history.empty:
        return None
    history = history.copy()
    if "_step" not in history.columns:
        if "step" in history.columns:
            history = history.rename(columns={"step": "_step"})
        else:
            history.insert(0, "_step", np.arange(len(history), dtype=float))
    return history.drop_duplicates(subset=["_step"], keep="last").sort_values("_step").reset_index(drop=True)


def _choose_richer_history(*candidates):
    valid = [candidate for candidate in candidates if candidate is not None and not candidate.empty]
    if not valid:
        return None
    merged = valid[0].copy()
    for candidate in valid[1:]:
        candidate = candidate.copy()
        duplicate_cols = [col for col in candidate.columns if col != "_step" and col in merged.columns]
        if duplicate_cols:
            candidate = candidate.rename(columns={col: f"{col}__candidate" for col in duplicate_cols})
        merged = merged.merge(candidate, on="_step", how="outer")
        for col in duplicate_cols:
            candidate_col = f"{col}__candidate"
            if candidate_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[candidate_col])
                merged = merged.drop(columns=[candidate_col])
    return merged.sort_values("_step").reset_index(drop=True)


def download_run_full_history_from_api(row, idx=None, total=None, reason=None):
    idx = idx or 1
    total = total or 1
    run_name = _row_get(row, "name")
    run_id = _row_get(row, "id")
    state = _row_get(row, "state")
    cache_dir = run_cache_dir(run_name, run_id)
    parquet_path = full_parquet_path(run_name, run_id)
    csv_path = full_csv_path(run_name, run_id)
    meta_path = metadata_path(run_name, run_id)

    cache_dir.mkdir(parents=True, exist_ok=True)
    label = f"{run_name} ({run_id})"
    if reason:
        label = f"{label}; {reason}"
    print(_progress("downloading scalar history API fallback", idx, total, label), flush=True)
    t0 = time.time()

    scanned = None
    try:
        scanned = _scan_history_all_keys_table(run_id)
        if scanned is not None:
            print(f"    scan_history all keys: {len(scanned):,} rows, {len(scanned.columns):,} columns", flush=True)
    except Exception as exc:
        print(f"    scan_history all keys failed: {type(exc).__name__}: {exc}", flush=True)

    sampled = None
    try:
        sampled = _sampled_full_history_table(run_id)
        if sampled is not None:
            print(f"    sampled history all keys: {len(sampled):,} rows, {len(sampled.columns):,} columns", flush=True)
    except Exception as exc:
        print(f"    sampled full history failed: {type(exc).__name__}: {exc}", flush=True)

    history = _choose_richer_history(scanned, sampled)
    if history is None or history.empty:
        raise RuntimeError(f"W&B API returned no scalar history rows for {run_id}")

    history_with_ids = _ensure_run_columns(history, run_name, run_id)
    parquet_written = False
    try:
        history_with_ids.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as parquet_exc:
        print(
            f"    parquet write failed; keeping CSV cache only: {type(parquet_exc).__name__}: {parquet_exc}",
            flush=True,
        )
    csv_written = False
    if WRITE_FULL_HISTORY_CSV or not parquet_written:
        history_with_ids.to_csv(csv_path, index=False)
        csv_written = True

    metadata = {
        "name": run_name,
        "id": run_id,
        "state": state,
        "entity": ENTITY,
        "project": PROJECT,
        "artifact_requested": artifact_path_for_run(run_id),
        "artifact_resolved": "scan_history:fallback",
        "history_parquet": str(parquet_path) if parquet_written else None,
        "history_csv": str(csv_path) if csv_written else None,
        "rows": int(len(history_with_ids)),
        "columns": int(len(history_with_ids.columns)),
        "downloaded_at_utc": pd.Timestamp.utcnow().isoformat(),
        "source": "wandb.Api.run(...).scan_history() / run.history() all keys",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"    saved {len(history_with_ids):,} rows, {len(history_with_ids.columns):,} columns "
        f"from scalar history API in {time.time() - t0:.1f}s",
        flush=True,
    )
    return history_with_ids


def load_cached_full_history(run_name, run_id, idx=None, total=None, parquet_path=None, csv_path=None):
    idx = idx or 1
    total = total or 1
    parquet_path = Path(parquet_path) if parquet_path else full_parquet_path(run_name, run_id)
    csv_path = Path(csv_path) if csv_path else full_csv_path(run_name, run_id)
    t0 = time.time()
    print(_progress("loading artifact cache", idx, total, run_name), flush=True)
    history = _read_history_table(parquet_path, csv_path=csv_path)
    source_path = history.attrs.get("source_path", str(parquet_path if parquet_path.exists() else csv_path))
    history = _ensure_run_columns(history, run_name, run_id)
    print(
        f"    loaded {len(history):,} rows, {len(history.columns):,} columns "
        f"from {source_path} in {time.time() - t0:.1f}s",
        flush=True,
    )
    return history


def download_run_full_history_from_artifact(row, idx=None, total=None):
    idx = idx or 1
    total = total or 1
    run_name = _row_get(row, "name")
    run_id = _row_get(row, "id")
    state = _row_get(row, "state")
    artifact_ref = artifact_path_for_run(run_id)
    cache_dir = run_cache_dir(run_name, run_id)
    parquet_path = full_parquet_path(run_name, run_id)
    csv_path = full_csv_path(run_name, run_id)
    meta_path = metadata_path(run_name, run_id)

    refresh_existing_fallback = (
        REFRESH_SCAN_HISTORY_FALLBACKS
        and _is_scan_history_fallback_cache(run_name, run_id)
    )
    cache_exists = parquet_path.exists() or csv_path.exists()
    cache_is_stale, cached_step, summary_step = _cache_is_stale_for_summary(run_name, run_id, row)
    if cache_exists and not FORCE_REFRESH and not refresh_existing_fallback and not cache_is_stale:
        return load_cached_full_history(run_name, run_id, idx=idx, total=total)

    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if refresh_existing_fallback:
        print(f"    refreshing existing scan_history fallback cache for {run_name}", flush=True)
    if cache_is_stale:
        print(
            f"    refreshing stale cache for {run_name}: cached max step "
            f"{cached_step / 1_000_000:.2f}M < summary {summary_step / 1_000_000:.2f}M",
            flush=True,
        )
    print(_progress("downloading history artifact", idx, total, f"{run_name} ({artifact_ref})"), flush=True)
    try:
        artifact, resolved_artifact_ref = _get_history_artifact(run_id)
    except Exception as artifact_exc:
        if not ALLOW_SCAN_HISTORY_FALLBACK:
            raise
        print(
            f"    history artifact unavailable; trying scalar history API fallback: "
            f"{type(artifact_exc).__name__}: {artifact_exc}",
            flush=True,
        )
        return download_run_full_history_from_api(
            row,
            idx=idx,
            total=total,
            reason="history artifact unavailable",
        )
    download_dir = Path(artifact.download(root=str(cache_dir)))
    history = _read_downloaded_history_artifact(download_dir)
    artifact_step = _history_max_step(history)
    summary_step = _summary_global_step(row)
    if (
        ALLOW_SCAN_HISTORY_FALLBACK
        and np.isfinite(summary_step)
        and summary_step > 0
        and np.isfinite(artifact_step)
        and artifact_step < summary_step * float(CACHE_STALE_STEP_TOLERANCE)
    ):
        print(
            f"    history artifact still behind run summary "
            f"({artifact_step / 1_000_000:.2f}M < {summary_step / 1_000_000:.2f}M); "
            "using scalar history API fallback",
            flush=True,
        )
        return download_run_full_history_from_api(
            row,
            idx=idx,
            total=total,
            reason="history artifact behind run summary",
        )
    history_with_ids = _ensure_run_columns(history, run_name, run_id)
    history_with_ids.to_parquet(parquet_path, index=False)
    if WRITE_FULL_HISTORY_CSV:
        history_with_ids.to_csv(csv_path, index=False)

    metadata = {
        "name": run_name,
        "id": run_id,
        "state": state,
        "entity": ENTITY,
        "project": PROJECT,
        "artifact_requested": artifact_ref,
        "artifact_resolved": resolved_artifact_ref,
        "history_parquet": str(parquet_path),
        "history_csv": str(csv_path) if WRITE_FULL_HISTORY_CSV else None,
        "rows": int(len(history_with_ids)),
        "columns": int(len(history_with_ids.columns)),
        "downloaded_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"    saved {len(history_with_ids):,} rows, {len(history_with_ids.columns):,} columns "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return history_with_ids


def full_history_to_long(full_history, metrics):
    if full_history.empty:
        return pd.DataFrame(columns=["run_name", "run_id", "metric", "step", "value"])
    step_col = "_step" if "_step" in full_history.columns else "step"
    if step_col not in full_history.columns:
        raise ValueError("History has neither '_step' nor 'step' column.")
    available_metrics = [metric for metric in metrics if metric in full_history.columns]
    if not available_metrics:
        return pd.DataFrame(columns=["run_name", "run_id", "metric", "step", "value"])

    id_cols = [col for col in ["run_name", "run_id", step_col] if col in full_history.columns]
    long_history = full_history[id_cols + available_metrics].melt(
        id_vars=id_cols,
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    )
    long_history = long_history.dropna(subset=["value"])
    if step_col != "step":
        long_history = long_history.rename(columns={step_col: "step"})
    return long_history[["run_name", "run_id", "metric", "step", "value"]]


def _runs_df_count():
    return len(runs_df) if "runs_df" in globals() else 0


def _check_download_inputs():
    runs_count = _runs_df_count()
    print(
        f"History artifact setup: local_only={USE_LOCAL_CACHE_ONLY}, runs_df={runs_count}",
        flush=True,
    )
    if runs_count == 0:
        raise RuntimeError(
            "runs_df is empty. Rerun the 'Fetch Run List' cell after changing "
            "RUN_NAME_REGEX/RUN_STATES, then rerun this cell."
        )


def download_or_load_histories():
    if CACHE_MODE != "full":
        raise ValueError('This notebook now uses artifact full-history caching. Set CACHE_MODE = "full".')

    _check_download_inputs()
    t0 = time.time()
    histories = []
    cache_index_rows = []
    skipped_rows = []
    total = len(runs_df)
    for idx, row in enumerate(runs_df.itertuples(index=False), start=1):
        run_name = _row_get(row, "name")
        run_id = _row_get(row, "id")
        try:
            if USE_LOCAL_CACHE_ONLY:
                full_history = load_cached_full_history(
                    run_name,
                    run_id,
                    idx=idx,
                    total=total,
                    parquet_path=_row_get(row, "history_parquet"),
                    csv_path=_row_get(row, "history_csv"),
                )
            else:
                full_history = download_run_full_history_from_artifact(row, idx=idx, total=total)
            histories.append(full_history_to_long(full_history, METRICS))
            resolved_ref = artifact_path_for_run(run_id)
            if metadata_path(run_name, run_id).exists():
                try:
                    resolved_ref = json.loads(metadata_path(run_name, run_id).read_text(encoding="utf-8")).get("artifact_resolved", resolved_ref)
                except json.JSONDecodeError:
                    pass
            cache_index_rows.append({
                "name": run_name,
                "id": run_id,
                "artifact": resolved_ref,
                "history_parquet": str(full_parquet_path(run_name, run_id)),
                "history_csv": str(full_csv_path(run_name, run_id)) if WRITE_FULL_HISTORY_CSV else None,
                "rows": len(full_history),
                "columns": len(full_history.columns),
            })
        except Exception as exc:
            print(f"    SKIPPED {run_name}: {type(exc).__name__}: {exc}", flush=True)
            skipped_rows.append({
                "name": run_name,
                "id": run_id,
                "state": _row_get(row, "state"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            if not SKIP_FAILED_DOWNLOADS:
                raise
    cache_index = pd.DataFrame(cache_index_rows)
    cache_index.to_csv(CACHE_INDEX_PATH, index=False)
    skipped_df = pd.DataFrame(skipped_rows)
    skipped_df.to_csv(FAILED_HISTORY_DOWNLOADS_PATH, index=False)
    print(f"Finished history artifact loading in {time.time() - t0:.1f}s", flush=True)
    print(f"Loaded histories for {len(histories)} / {total} selected runs", flush=True)
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} run(s); saved details: {FAILED_HISTORY_DOWNLOADS_PATH}", flush=True)
    print(f"Saved cache index: {CACHE_INDEX_PATH}", flush=True)
    return histories, skipped_df


def load_history_df(verbose=True):
    """Load metric histories for the current global runs_df."""
    global histories, skipped_history_downloads_df, history_df

    histories, skipped_history_downloads_df = download_or_load_histories()
    if histories:
        history_df = pd.concat(histories, ignore_index=True)
    else:
        history_df = pd.DataFrame(columns=["run_name", "run_id", "metric", "step", "value"])
    history_df["step"] = pd.to_numeric(history_df["step"], errors="coerce")
    history_df["value"] = pd.to_numeric(history_df["value"], errors="coerce")
    history_df = history_df.dropna(subset=["step", "value"])
    history_df["step_millions"] = history_df["step"] / 1_000_000
    if verbose:
        print(history_df.shape)
    return history_df


def load_wandb_data(*, verbose=True, **run_filter_kwargs):
    """Load runs and metric history, then update module-level runs_df/history_df."""
    if verbose:
        print_wandb_setup()
    loaded_runs = load_runs(verbose=verbose, **run_filter_kwargs)
    loaded_history = load_history_df(verbose=verbose)
    return WandbPlotData(
        runs_df=loaded_runs,
        history_df=loaded_history,
        skipped_history_downloads_df=skipped_history_downloads_df,
        histories=histories,
    )


# General plotting helpers extracted from the original notebook.
SURVIVAL_METRIC_CANDIDATES = {
    "test": [
        "test/charts/episodic_survival",
        "test/episodic_survival",
        "charts/episodic_survival",
    ],
    "train_eval": [
        "train_eval/charts/episodic_survival",
        "train_eval/episodic_survival",
    ],
    "validation": ["validation/episodic_survival"],
    "legacy": ["charts/episodic_survival"],
}

DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def available_run_names(pattern=None, history=None):
    """Return cached run names, optionally filtered by a regex pattern."""
    history = _get_history(history)
    names = sorted(history["run_name"].dropna().unique())
    if pattern is None:
        return names
    regex = re.compile(pattern)
    return [name for name in names if regex.search(name)]


def make_run(name, color=None, label=None, dash=None):
    """Small helper for explicit run specs."""
    return {"name": name, "color": color, "label": label, "dash": dash}


def save_plot(fig, name):
    """Save a Plotly figure as HTML under FIG_DIR, unless an absolute path is passed."""
    if not name:
        return None
    path = Path(name)
    if path.suffix.lower() != ".html":
        path = FIG_DIR / f"{safe_name(name)}.html"
    elif not path.is_absolute():
        path = FIG_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Saved plot: {path}")
    return path


def plot_runs(
    runs,
    *,
    metric=None,
    split="test",
    colors=None,
    labels=None,
    dashes=None,
    smooth=5,
    show_raw=True,
    multiply=100.0,
    title=None,
    yaxis_title="Survival (%)",
    xaxis_title="Steps (M)",
    y_range=None,
    width=1300,
    height=550,
    save_name=None,
    show=False,
    history=None,
):
    """
    Plot one chart with all requested runs.

    `runs` can be:
    - {"run name": "#color", ...}
    - [("run name", "#color", "label", "dash"), ...]
    - [{"name": "run name", "color": "#color", "label": "label", "dash": "dash"}, ...]
    """
    history = _get_history(history)
    specs = _normalize_runs(runs, colors=colors, labels=labels, dashes=dashes)
    metric_candidates = _metric_candidates(metric=metric, split=split)

    fig = go.Figure()
    added = _add_run_traces(
        fig,
        specs,
        metric_candidates,
        history=history,
        smooth=smooth,
        show_raw=show_raw,
        multiply=multiply,
        legend_seen=set(),
    )
    if added == 0:
        fig.add_annotation(
            text="No data found for the requested runs and metric.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=width,
        height=height,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    fig.update_xaxes(title_text=xaxis_title)
    fig.update_yaxes(title_text=yaxis_title, range=y_range)
    save_plot(fig, save_name)
    if show:
        fig.show()
        return None
    return fig


def plot_run_groups(
    groups,
    *,
    metric=None,
    split="test",
    colors=None,
    labels=None,
    dashes=None,
    smooth=5,
    show_raw=True,
    multiply=100.0,
    title=None,
    yaxis_title="Survival (%)",
    xaxis_title="Steps (M)",
    y_range=None,
    ncols=2,
    subplot_height=420,
    width=1500,
    save_name=None,
    show=False,
    history=None,
):
    """
    Plot several run groups as subplots.

    `groups` can be a dict like {"subplot title": runs, ...}, where each `runs`
    value accepts the same formats as `plot_runs`.
    """
    history = _get_history(history)
    group_items = _normalize_groups(groups)
    if not group_items:
        raise ValueError("Pass at least one subplot group.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(group_items) / ncols))
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[title for title, _ in group_items],
        shared_xaxes=False,
        shared_yaxes=False,
    )
    metric_candidates = _metric_candidates(metric=metric, split=split)
    added = 0
    legend_layouts = {}

    for idx, (_, group_runs) in enumerate(group_items, start=1):
        row = int(np.ceil(idx / ncols))
        col = ((idx - 1) % ncols) + 1
        specs = _normalize_runs(group_runs, colors=colors, labels=labels, dashes=dashes)
        legend_id = _legend_id(idx)
        legend_layouts[legend_id] = _subplot_legend_layout(fig, row, col, ncols)
        added += _add_run_traces(
            fig,
            specs,
            metric_candidates,
            history=history,
            smooth=smooth,
            show_raw=show_raw,
            multiply=multiply,
            row=row,
            col=col,
            legend_seen=set(),
            legend_id=legend_id,
            legend_group_prefix=f"subplot{idx}:",
        )
        fig.update_xaxes(title_text=xaxis_title, row=row, col=col)
        fig.update_yaxes(title_text=yaxis_title, range=y_range, row=row, col=col)

    if added == 0:
        fig.add_annotation(
            text="No data found for the requested runs and metric.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    layout = {
        "title": title,
        "template": "plotly_white",
        "width": width,
        "height": max(520, subplot_height * nrows),
        "hovermode": "x unified",
        "showlegend": True,
        "margin": {"l": 70, "r": 30, "t": 90, "b": 60},
    }
    layout.update(legend_layouts)
    fig.update_layout(**layout)
    save_plot(fig, save_name)
    if show:
        fig.show()
        return None
    return fig


def plot_run_means(
    mean_runs,
    *,
    metric=None,
    split="test",
    colors=None,
    dashes=None,
    smooth=5,
    show_members=False,
    show_std=True,
    min_members=1,
    multiply=100.0,
    title=None,
    yaxis_title="Survival (%)",
    xaxis_title="Steps (M)",
    y_range=None,
    width=1300,
    height=550,
    save_name=None,
    show=False,
    history=None,
):
    """
    Plot averaged curves. Each entry in `mean_runs` is one curve.

    Example spec:
    {
        "entropy_decay_sto": {"runs": ["run_s0_stoch", "run_s1_stoch"], "color": "#1f77b4"},
        "entropy_decay_det": {"runs": ["run_s0_det", "run_s1_det"], "color": "#1f77b4", "dash": "dash"},
        "no_entropy_decay_sto": {"runs": ["run_s0_stoch", "run_s1_stoch"], "color": "#ff7f0e"},
    }
    """
    history = _get_history(history)
    specs = _normalize_mean_specs(mean_runs, colors=colors, dashes=dashes)
    metric_candidates = _metric_candidates(metric=metric, split=split)

    fig = go.Figure()
    added = _add_mean_traces(
        fig,
        specs,
        metric_candidates,
        history=history,
        smooth=smooth,
        show_members=show_members,
        show_std=show_std,
        min_members=min_members,
        multiply=multiply,
        legend_seen=set(),
    )
    if added == 0:
        fig.add_annotation(
            text="No data found for the requested mean curves.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=width,
        height=height,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    fig.update_xaxes(title_text=xaxis_title)
    fig.update_yaxes(title_text=yaxis_title, range=y_range)
    save_plot(fig, save_name)
    if show:
        fig.show()
        return None
    return fig


def plot_run_mean_groups(
    groups,
    *,
    metric=None,
    split="test",
    colors=None,
    dashes=None,
    smooth=5,
    show_members=False,
    show_std=True,
    min_members=1,
    multiply=100.0,
    title=None,
    yaxis_title="Survival (%)",
    xaxis_title="Steps (M)",
    y_range=None,
    ncols=2,
    subplot_height=420,
    width=1500,
    save_name=None,
    show=False,
    history=None,
):
    """
    Plot mean curves in subplots.

    `groups` is a dict like {"subplot title": mean_runs, ...}. Each `mean_runs`
    value accepts the same format as `plot_run_means`.
    """
    history = _get_history(history)
    group_items = _normalize_groups(groups)
    if not group_items:
        raise ValueError("Pass at least one subplot group.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(group_items) / ncols))
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[title for title, _ in group_items],
        shared_xaxes=False,
        shared_yaxes=False,
    )
    metric_candidates = _metric_candidates(metric=metric, split=split)
    added = 0
    legend_layouts = {}

    for idx, (_, mean_runs) in enumerate(group_items, start=1):
        row = int(np.ceil(idx / ncols))
        col = ((idx - 1) % ncols) + 1
        specs = _normalize_mean_specs(mean_runs, colors=colors, dashes=dashes)
        legend_id = _legend_id(idx)
        legend_layouts[legend_id] = _subplot_legend_layout(fig, row, col, ncols)
        added += _add_mean_traces(
            fig,
            specs,
            metric_candidates,
            history=history,
            smooth=smooth,
            show_members=show_members,
            show_std=show_std,
            min_members=min_members,
            multiply=multiply,
            row=row,
            col=col,
            legend_seen=set(),
            legend_id=legend_id,
            legend_group_prefix=f"mean_subplot{idx}:",
        )
        fig.update_xaxes(title_text=xaxis_title, row=row, col=col)
        fig.update_yaxes(title_text=yaxis_title, range=y_range, row=row, col=col)

    if added == 0:
        fig.add_annotation(
            text="No data found for the requested mean curves.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    layout = {
        "title": title,
        "template": "plotly_white",
        "width": width,
        "height": max(520, subplot_height * nrows),
        "hovermode": "x unified",
        "showlegend": True,
        "margin": {"l": 70, "r": 30, "t": 90, "b": 60},
    }
    layout.update(legend_layouts)
    fig.update_layout(**layout)
    save_plot(fig, save_name)
    if show:
        fig.show()
        return None
    return fig


def _get_history(history=None):
    if history is not None:
        return history
    if "history_df" not in globals():
        raise NameError("Run the W&B history loading cell first so `history_df` exists.")
    return history_df


def _metric_candidates(metric=None, split="test"):
    if metric is None:
        return SURVIVAL_METRIC_CANDIDATES.get(split, SURVIVAL_METRIC_CANDIDATES["test"])
    if isinstance(metric, str):
        return [metric]
    return list(metric)


def _normalize_groups(groups):
    if isinstance(groups, dict):
        return list(groups.items())
    normalized = []
    for item in groups:
        if isinstance(item, dict):
            normalized.append((item["title"], item["runs"]))
        else:
            title, runs = item
            normalized.append((title, runs))
    return normalized


def _legend_id(index):
    return "legend" if index == 1 else f"legend{index}"


def _subplot_legend_layout(fig, row, col, ncols):
    x_domain, y_domain = _subplot_domains(fig, row, col, ncols)
    return {
        "x": x_domain[0] + 0.01,
        "y": y_domain[1] - 0.02,
        "xanchor": "left",
        "yanchor": "top",
        "orientation": "v",
        "bgcolor": "rgba(255,255,255,0.62)",
        "bordercolor": "rgba(0,0,0,0.10)",
        "borderwidth": 1,
        "font": {"size": 9},
        "itemsizing": "constant",
    }


def _subplot_domains(fig, row, col, ncols):
    axis_index = (row - 1) * ncols + col
    suffix = "" if axis_index == 1 else str(axis_index)
    x_axis = getattr(fig.layout, f"xaxis{suffix}")
    y_axis = getattr(fig.layout, f"yaxis{suffix}")
    return x_axis.domain, y_axis.domain


def mean_curve(
    runs,
    *,
    color=None,
    dash=None,
    label=None,
    width=3,
    opacity=1.0,
    show_std=None,
    std_alpha=0.14,
    show_members=None,
    member_alpha=0.18,
):
    """Create one averaged-curve spec for `plot_run_means`."""
    spec = {
        "runs": runs,
        "color": color,
        "dash": dash,
        "width": width,
        "opacity": opacity,
        "show_std": show_std,
        "std_alpha": std_alpha,
        "show_members": show_members,
        "member_alpha": member_alpha,
    }
    if label is not None:
        spec["label"] = label
    return spec


def mean_curves_from_prefixes(
    entropy_prefix,
    no_entropy_prefix,
    *,
    seeds=(0, 1, 2),
    entropy_label="entropy decay",
    no_entropy_label="no entropy decay",
    entropy_color="#1f77b4",
    no_entropy_color="#ff7f0e",
    stoch_suffix="stoch",
    det_suffix="det",
    stoch_dash="solid",
    det_dash="dash",
    **style,
):
    """Build four mean curves from exact W&B run-name prefixes."""
    return {
        f"{entropy_label} sto": mean_curve(
            _seeded_run_names(entropy_prefix, seeds, stoch_suffix),
            color=entropy_color,
            dash=stoch_dash,
            **style,
        ),
        f"{entropy_label} det": mean_curve(
            _seeded_run_names(entropy_prefix, seeds, det_suffix),
            color=entropy_color,
            dash=det_dash,
            **style,
        ),
        f"{no_entropy_label} sto": mean_curve(
            _seeded_run_names(no_entropy_prefix, seeds, stoch_suffix),
            color=no_entropy_color,
            dash=stoch_dash,
            **style,
        ),
        f"{no_entropy_label} det": mean_curve(
            _seeded_run_names(no_entropy_prefix, seeds, det_suffix),
            color=no_entropy_color,
            dash=det_dash,
            **style,
        ),
    }


def _seeded_run_names(prefix, seeds, suffix):
    return [f"{prefix}_s{seed}_{suffix}" for seed in seeds]


def sto_det_mean_specs(
    runs,
    *,
    color="#1f77b4",
    det_color=None,
    stoch_label="sto",
    det_label="det",
    stoch_dash="solid",
    det_dash="dash",
    **style,
):
    """Optional helper: split a mixed run list into stochastic and deterministic mean specs."""
    run_names = _as_run_name_list(runs)
    stoch_runs = [name for name in run_names if _is_stoch_run(name)]
    det_runs = [name for name in run_names if _is_det_run(name)]
    specs = {}
    if stoch_runs:
        specs[stoch_label] = mean_curve(stoch_runs, color=color, dash=stoch_dash, **style)
    if det_runs:
        specs[det_label] = mean_curve(det_runs, color=det_color or color, dash=det_dash, **style)
    return specs


def _normalize_mean_specs(mean_runs, colors=None, dashes=None):
    if isinstance(mean_runs, dict):
        raw_specs = [_coerce_mean_spec(label, value) for label, value in mean_runs.items()]
    else:
        raw_specs = [_coerce_mean_spec_from_item(item) for item in mean_runs]

    specs = []
    for idx, spec in enumerate(raw_specs):
        label = spec["label"]
        spec["runs"] = _as_run_name_list(spec["runs"])
        spec["color"] = spec.get("color") or _lookup(colors, label, idx) or DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        spec["dash"] = spec.get("dash") or _lookup(dashes, label, idx) or _auto_mean_dash(label)
        spec["width"] = spec.get("width", spec.get("line_width", 3))
        spec["opacity"] = spec.get("opacity", 1.0)
        spec["std_alpha"] = spec.get("std_alpha", 0.14)
        spec["member_alpha"] = spec.get("member_alpha", 0.18)
        specs.append(spec)
    return specs


def _coerce_mean_spec(label, value):
    if isinstance(value, dict):
        spec = dict(value)
        spec.setdefault("label", label)
        if "runs" not in spec:
            style_keys = {
                "label", "name", "color", "dash", "width", "line_width", "opacity",
                "show_std", "std_alpha", "show_members", "member_alpha",
            }
            if any(key in spec for key in style_keys):
                raise ValueError(f"Mean curve {label!r} has style keys but no `runs` list.")
            spec["runs"] = list(value.keys())
    else:
        spec = {"label": label, "runs": value}
    if "name" in spec and "label" not in spec:
        spec["label"] = spec["name"]
    return spec


def _coerce_mean_spec_from_item(item):
    if isinstance(item, dict):
        spec = dict(item)
        if "name" in spec and "label" not in spec:
            spec["label"] = spec["name"]
        if "label" not in spec or "runs" not in spec:
            raise ValueError("Mean specs need `label` and `runs` keys.")
        return spec
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        spec = {"label": item[0], "runs": item[1]}
        if len(item) > 2:
            spec["color"] = item[2]
        if len(item) > 3:
            spec["dash"] = item[3]
        if len(item) > 4:
            spec["width"] = item[4]
        return spec
    raise ValueError(f"Could not understand mean spec: {item!r}")


def _as_run_name_list(runs):
    if isinstance(runs, str):
        return [runs]
    if isinstance(runs, dict):
        return list(runs.keys())
    return list(runs)


def _auto_mean_dash(label):
    return "dash" if "det" in str(label).lower() else "solid"


def _is_det_run(name):
    lower = str(name).lower()
    return lower.endswith("_det") or "_det_" in lower


def _is_stoch_run(name):
    lower = str(name).lower()
    return lower.endswith("_stoch") or "_stoch_" in lower or lower.endswith("_sto") or "_sto_" in lower


def _mean_metric_frame(history, spec, metric_candidates, min_members=1, smooth=None):
    frames = []
    metrics_used = []
    for run_name in spec["runs"]:
        data, metric = _run_metric_frame(history, run_name, metric_candidates)
        if data.empty:
            continue
        member = data[["step", "step_millions", "value"]].copy()
        member["run_name"] = run_name
        frames.append(member)
        metrics_used.append(metric)

    if not frames:
        print(f"Skipping {spec['label']}: no usable runs found")
        return pd.DataFrame(), pd.DataFrame(), []

    if len(frames) < len(spec["runs"]):
        print(f"{spec['label']}: using {len(frames)} / {len(spec['runs'])} runs")

    members = pd.concat(frames, ignore_index=True).sort_values(["run_name", "step"]).copy()
    members["smoothed_value"] = members.groupby("run_name", sort=False)["value"].transform(
        lambda values: _smooth_values(pd.to_numeric(values, errors="coerce"), smooth)
    )
    # Smooth each seed/member first, then average smoothed values at exact logged steps only.
    # If a run stops early, it simply stops contributing; no interpolation is performed.
    stats = members.groupby("step", as_index=False).agg(
        mean=("smoothed_value", "mean"),
        std=("smoothed_value", "std"),
        n=("smoothed_value", "count"),
    )
    stats = stats[stats["n"] >= int(min_members)].copy()
    if stats.empty:
        print(f"Skipping {spec['label']}: no step has at least {min_members} member(s)")
        return pd.DataFrame(), members, metrics_used
    stats["std"] = stats["std"].fillna(0.0)
    stats["step_millions"] = stats["step"] / 1_000_000
    return stats.sort_values("step"), members.sort_values(["run_name", "step"]), metrics_used


def _add_mean_traces(
    fig,
    specs,
    metric_candidates,
    *,
    history,
    smooth,
    show_members,
    show_std,
    min_members,
    multiply,
    row=None,
    col=None,
    legend_seen=None,
    legend_id="legend",
    legend_group_prefix="",
):
    legend_seen = legend_seen if legend_seen is not None else set()
    added = 0
    add_kwargs = {} if row is None else {"row": row, "col": col}

    for spec in specs:
        label = spec["label"]
        color = spec["color"]
        dash = spec["dash"]
        width = spec["width"]
        opacity = spec["opacity"]
        member_alpha = spec["member_alpha"]
        std_alpha = spec["std_alpha"]
        use_members = show_members if spec.get("show_members") is None else spec["show_members"]
        use_std = show_std if spec.get("show_std") is None else spec["show_std"]
        stats, members, metrics_used = _mean_metric_frame(history, spec, metric_candidates, min_members=min_members, smooth=smooth)
        if stats.empty:
            continue

        legend_group = f"{legend_group_prefix}{label}"
        if use_members:
            for run_name, member in members.groupby("run_name"):
                fig.add_trace(
                    go.Scatter(
                        x=member["step_millions"],
                        y=_scale_values(member["smoothed_value"], multiply),
                        mode="lines",
                        name=f"{label} member",
                        legendgroup=legend_group,
                        legend=legend_id,
                        showlegend=False,
                        opacity=member_alpha,
                        line={"color": color, "width": spec.get("member_width", 1), "dash": dash},
                        hovertemplate=(
                            f"<b>{run_name}</b><br>"
                            "step: %{x:.2f}M<br>"
                            "smoothed value: %{y:.3f}<extra></extra>"
                        ),
                    ),
                    **add_kwargs,
                )
                added += 1

        x = stats["step_millions"]
        mean_y = _scale_values(stats["mean"], multiply)
        std_y = _scale_values(stats["std"], multiply)

        if use_std and stats["n"].max() > 1:
            upper = mean_y + std_y
            lower = mean_y - std_y
            fig.add_trace(
                go.Scatter(
                    x=list(x) + list(x[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    mode="lines",
                    name=f"{label} std",
                    legendgroup=legend_group,
                    legend=legend_id,
                    showlegend=False,
                    line={"color": "rgba(0,0,0,0)", "width": 0},
                    fill="toself",
                    fillcolor=_color_with_alpha(color, std_alpha),
                    hoverinfo="skip",
                ),
                **add_kwargs,
            )
            added += 1

        legend_key = (label, color, dash)
        show_mean_legend = legend_key not in legend_seen
        legend_seen.add(legend_key)
        metric_label = ", ".join(sorted(set(metrics_used)))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_y,
                mode="lines",
                name=label,
                legendgroup=legend_group,
                legend=legend_id,
                showlegend=show_mean_legend,
                customdata=stats["n"],
                opacity=opacity,
                line={"color": color, "width": width, "dash": dash},
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"metric: {metric_label}<br>"
                    "runs at step: %{customdata}<br>"
                    "step: %{x:.2f}M<br>"
                    "mean of smoothed seeds: %{y:.3f}<extra></extra>"
                ),
            ),
            **add_kwargs,
        )
        added += 1
    return added


def _color_with_alpha(color, alpha):
    color = str(color)
    match = re.fullmatch(r"#?([0-9A-Fa-f]{6})", color)
    if not match:
        return f"rgba(127,127,127,{alpha})"
    value = match.group(1)
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def _normalize_runs(runs, colors=None, labels=None, dashes=None):
    if isinstance(runs, dict):
        run_items = [{"name": name, "color": color} for name, color in runs.items()]
    else:
        run_items = list(runs)

    specs = []
    for idx, item in enumerate(run_items):
        spec = _coerce_run_spec(item)
        name = spec["name"]
        spec["color"] = spec.get("color") or _lookup(colors, name, idx) or DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        spec["label"] = spec.get("label") or _lookup(labels, name, idx) or name
        spec["dash"] = spec.get("dash") or _lookup(dashes, name, idx) or _auto_dash(name)
        specs.append(spec)
    return specs


def _coerce_run_spec(item):
    if isinstance(item, str):
        return {"name": item}
    if isinstance(item, dict):
        if "name" in item:
            return dict(item)
        if len(item) == 1:
            name, color = next(iter(item.items()))
            return {"name": name, "color": color}
        raise ValueError("Run dictionaries need a `name` key, or exactly one {name: color} item.")
    if isinstance(item, (tuple, list)) and item:
        spec = {"name": item[0]}
        if len(item) > 1:
            spec["color"] = item[1]
        if len(item) > 2:
            spec["label"] = item[2]
        if len(item) > 3:
            spec["dash"] = item[3]
        return spec
    raise ValueError(f"Could not understand run spec: {item!r}")


def _lookup(values, name, idx):
    if values is None:
        return None
    if isinstance(values, dict):
        return values.get(name)
    values = list(values)
    if not values:
        return None
    return values[idx % len(values)]


def _auto_dash(name):
    lower = str(name).lower()
    return "dash" if lower.endswith("_det") or "_det_" in lower else "solid"


def _run_metric_frame(history, run_name, metric_candidates):
    run_history = history[history["run_name"] == run_name]
    if run_history.empty:
        print(f"Skipping {run_name}: run not found in history_df")
        return pd.DataFrame(), None

    for metric in metric_candidates:
        metric_history = run_history[run_history["metric"] == metric].copy()
        if metric_history.empty:
            continue
        metric_history["step"] = pd.to_numeric(metric_history["step"], errors="coerce")
        metric_history["value"] = pd.to_numeric(metric_history["value"], errors="coerce")
        metric_history = metric_history.dropna(subset=["step", "value"])
        if metric_history.empty:
            continue
        metric_history = metric_history.groupby("step", as_index=False)["value"].mean()
        metric_history["step_millions"] = metric_history["step"] / 1_000_000
        return metric_history.sort_values("step"), metric

    print(f"Skipping {run_name}: none of these metrics were found: {metric_candidates}")
    return pd.DataFrame(), None


def _smooth_values(values, smooth):
    if smooth is None or int(smooth) <= 1:
        return values
    return values.rolling(int(smooth), min_periods=1).mean()


def _scale_values(values, multiply):
    values = pd.to_numeric(values, errors="coerce")
    if multiply is not None:
        values = values * float(multiply)
    return values


def _add_run_traces(
    fig,
    specs,
    metric_candidates,
    *,
    history,
    smooth,
    show_raw,
    multiply,
    row=None,
    col=None,
    legend_seen=None,
    legend_id="legend",
    legend_group_prefix="",
):
    legend_seen = legend_seen if legend_seen is not None else set()
    added = 0
    add_kwargs = {} if row is None else {"row": row, "col": col}

    for spec in specs:
        run_name = spec["name"]
        label = spec["label"]
        color = spec["color"]
        dash = spec["dash"]
        data, metric = _run_metric_frame(history, run_name, metric_candidates)
        if data.empty:
            continue

        x = data["step_millions"]
        y = _scale_values(data["value"], multiply)
        hover = (
            "<b>%{fullData.name}</b><br>"
            "metric: " + metric + "<br>"
            "step: %{x:.2f}M<br>"
            "smoothed value: %{y:.3f}<extra></extra>"
        )
        legend_group = f"{legend_group_prefix}{label}"

        if show_raw:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"{label} raw",
                    legendgroup=legend_group,
                    legend=legend_id,
                    showlegend=False,
                    opacity=0.22,
                    line={"color": color, "width": 1, "dash": dash},
                    hovertemplate=hover,
                ),
                **add_kwargs,
            )
            added += 1

        legend_key = (label, color, dash)
        show_smooth_legend = legend_key not in legend_seen
        legend_seen.add(legend_key)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=_smooth_values(y, smooth),
                mode="lines",
                name=label,
                legendgroup=legend_group,
                legend=legend_id,
                showlegend=show_smooth_legend,
                line={"color": color, "width": 2.8, "dash": dash},
                hovertemplate=hover,
            ),
            **add_kwargs,
        )
        added += 1
    return added


def plot_entropy_decay_comparison():
    """Plot the entropy-decay vs no-entropy-decay comparison marked #KEEP."""
    mean_groups = {
        "A1": mean_curves_from_prefixes(
            "noval20_mlp_a1_entropy_decay",
            "noval20_mlp_a1_no_entropy_decay",
        ),
        "A3": mean_curves_from_prefixes(
            "noval20_mlp_a3_entropy_decay",
            "noval20_mlp_a3_no_entropy_decay",
        ),
        "A3 logit": mean_curves_from_prefixes(
            "noval20_mlp_a3_logit_decay",
            "noval20_mlp_a3_logit_no_entropy_decay",
        ),
        "A4": mean_curves_from_prefixes(
            "noval20_mlp_a4_entropy_decay",
            "noval20_mlp_a4_no_entropy_decay",
            seeds=(1, 2),
        ),
    }

    return plot_run_mean_groups(
        mean_groups,
        split="test",
        smooth=5,
        title="Entropy Decay vs No Entropy Decay",
        ncols=2,
        show_members=True,
    )


def plot_gine_best2_baseline_comparisons():
    # GINE best-search 2: baseline 00 against every other seed-averaged family.
    # Run the W&B download/cache cells first so history_df contains these histories.

    GINE_BEST2_SEEDS = (0, 1, 2)
    GINE_BEST2_BASELINE_PREFIX = "best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update"
    GINE_BEST2_BASELINE_LABEL = "00 baseline"
    GINE_BEST2_BASELINE_COLOR = "#1f77b4"
    GINE_BEST2_COMPARE_COLOR = "#ff7f0e"

    GINE_BEST2_COMPARISONS = [
        (
            "000 vs 00: concat-flat off vs on",
            "best_000_shared_actor_gnn_gine_a4_no_concat_flat_critic_gnn_legacy_update",
            "000 no concat-flat",
        ),
        (
            "00 vs 01: legacy vs optimized critic update",
            "best_01_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_optcritic",
            "01 optimized critic update",
        ),
        (
            "00 vs 02: GNN critic vs MLP critic",
            "best_02_shared_actor_gnn_gine_a4_concat_flat_critic_mlp_legacy_update",
            "02 MLP critic",
        ),
        (
            "00 vs 03: shared vs non-shared actor GNN",
            "best_03_nonshared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update",
            "03 non-shared actor GNN",
        ),
        (
            "00 vs 04: standard vs light GINE",
            "best_04_shared_actor_gnn_light_gine_a4_concat_flat_critic_gnn_legacy_update",
            "04 light GINE",
        ),
        (
            "00 vs 05: constant entropy vs entropy decay",
            "best_05_shared_actor_gnn_gine_a4_entropy_decay_concat_flat_critic_gnn_legacy_update",
            "05 entropy decay",
        ),
        (
            "00 vs 06: with vs without node ID",
            "best_06_shared_actor_gnn_gine_a4_no_node_id_concat_flat_critic_gnn_legacy_update",
            "06 no node ID",
        ),
        (
            "00 vs 07: GINE vs GAT",
            "best_07_shared_actor_gnn_gat_a4_concat_flat_critic_gnn_legacy_update",
            "07 GAT",
        ),
        (
            "00 vs 08: GINE vs weighted GCN",
            "best_08_shared_actor_gnn_weighted_gcn_a4_concat_flat_critic_gnn_legacy_update",
            "08 weighted GCN",
        ),
        (
            "00 vs 09: full concat GNN critic vs light no-concat MLP critic",
            "best_09_shared_actor_gnn_light_gine_a4_no_concat_flat_critic_mlp_optcritic",
            "09 light no-concat MLP critic",
        ),
        (
            "00 vs 10: full concat GNN critic vs light concat MLP critic",
            "best_10_shared_actor_gnn_light_gine_a4_concat_flat_critic_mlp_optcritic",
            "10 light concat MLP critic",
        ),
    ]


    def gine_best2_seeded_runs(prefix, seeds=GINE_BEST2_SEEDS):
        return [f"{prefix}_s{seed}" for seed in seeds]


    def resolve_gine_best2_seeded_runs(prefix, seeds=GINE_BEST2_SEEDS, history=None):
        """Resolve exact config names to run names available in history_df, with a small fallback."""
        requested = gine_best2_seeded_runs(prefix, seeds=seeds)
        history = _get_history(history)
        available = available_run_names(history=history)
        available_set = set(available)
        resolved = []
        missing = []

        for name in requested:
            if name in available_set:
                resolved.append(name)
                continue
            substring_matches = [candidate for candidate in available if name in candidate or candidate in name]
            if substring_matches:
                resolved.append(sorted(substring_matches, key=lambda candidate: (len(candidate), candidate))[0])
            else:
                missing.append(name)

        if missing:
            print(f"Missing {len(missing)} seed run(s) for {prefix}:")
            for name in missing:
                print(f"  - {name}")
        return resolved


    def gine_best2_mean_specs(other_prefix, other_label, seeds=GINE_BEST2_SEEDS):
        baseline_runs = resolve_gine_best2_seeded_runs(GINE_BEST2_BASELINE_PREFIX, seeds=seeds)
        other_runs = resolve_gine_best2_seeded_runs(other_prefix, seeds=seeds)
        return {
            GINE_BEST2_BASELINE_LABEL: mean_curve(
                baseline_runs,
                color=GINE_BEST2_BASELINE_COLOR,
                dash="solid",
                width=3,
                member_alpha=0.18,
                std_alpha=0.10,
            ),
            other_label: mean_curve(
                other_runs,
                color=GINE_BEST2_COMPARE_COLOR,
                dash="solid",
                width=3,
                member_alpha=0.18,
                std_alpha=0.12,
            ),
        }


    gine_best2_mean_groups = {}
    for title, other_prefix, other_label in GINE_BEST2_COMPARISONS:
        specs = gine_best2_mean_specs(other_prefix, other_label)
        if all(spec["runs"] for spec in specs.values()):
            gine_best2_mean_groups[title] = specs
        else:
            print(f"Skipping {title}: at least one side has no resolved runs.")

    if not gine_best2_mean_groups:
        raise RuntimeError("No GINE best-search 2 comparisons could be built from history_df.")

    fig_gine_best2 = plot_run_mean_groups(
        gine_best2_mean_groups,
        split="test",
        smooth=5,
        title="configs/gine_s0_s1_s2: baseline 00 vs each alternative",
        ncols=3,
        subplot_height=360,
        width=1700,
        y_range=[0, 105],
        show_members=True,
        show_std=True,
        save_name="gine_best_search2_baseline_comparisons",
    )
    fig_gine_best2

    return fig_gine_best2


def plot_adaptive_intervention_budget_7_comparisons(baseline_source="gine_best2"):
    """Plot adaptive_intervention_budget_7 conditions against the shared main baseline."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    AIB_CONFIG_DIR = TASK_DIR / "configs" / "adaptive_intervention_budget_7"
    AIB_BASELINE_SOURCES = {
        "gine_best2": {
            "prefix": "best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update",
            "label": "main baseline: GINE 00",
        },
        "gine_best_search_2": {
            "prefix": "best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update",
            "label": "main baseline: GINE 00",
        },
        "heuristic_vs_gate": {
            "prefix": "hvg_00_baseline",
            "label": "main baseline: HVG baseline",
        },
        "hvg": {
            "prefix": "hvg_00_baseline",
            "label": "main baseline: HVG baseline",
        },
        "phase4_sparse_control_16": {
            "prefix": "sparse16_flat_p000",
            "label": "main baseline: Sparse16 flat p0.000",
        },
        "phase4_sparse": {
            "prefix": "sparse16_flat_p000",
            "label": "main baseline: Sparse16 flat p0.000",
        },
    }
    baseline_source_key = str(baseline_source).strip().lower()
    if baseline_source_key not in AIB_BASELINE_SOURCES:
        raise ValueError(
            "baseline_source must be one of "
            f"{sorted(AIB_BASELINE_SOURCES)}; got {baseline_source!r}."
        )
    AIB_BASELINE_SPEC = AIB_BASELINE_SOURCES[baseline_source_key]
    AIB_BASELINE_LABEL = AIB_BASELINE_SPEC["label"]
    AIB_BASELINE_COLOR = "#1f77b4"
    AIB_COMPARE_COLOR = "#ff7f0e"
    AIB_GATE_COLOR = "#d62728"
    AIB_FAMILY_LABELS = {
        "aib_00_flat_local_t020": "flat local target 0.20",
        "aib_01_flat_local_t010": "flat local target 0.10",
        "aib_02_flat_local_t035": "flat local target 0.35",
        "aib_03_gate_hgreedy_sep_local_t020": "gate h-greedy separate entropy target 0.20",
    }
    AIB_FAMILY_ORDER = {
        "aib_00_flat_local_t020": 0,
        "aib_01_flat_local_t010": 1,
        "aib_02_flat_local_t035": 2,
        "aib_03_gate_hgreedy_sep_local_t020": 3,
    }

    def _aib_family_from_stem(stem):
        return re.sub(r"_s\d+$", "", str(stem))

    def _aib_target_token(value):
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "?"

    def _aib_read_expected_configs(config_dir=AIB_CONFIG_DIR):
        rows = []
        for config_path in sorted(Path(config_dir).glob("*.toml")):
            cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
            args = cfg.get("args", {})
            run = cfg.get("run", {})
            stem = config_path.stem
            seed_match = re.search(r"_s(\d+)$", stem)
            family = _aib_family_from_stem(stem)
            target = args.get("intervention_budget_target")
            gated = bool(args.get("intervention_gate", False))
            label = AIB_FAMILY_LABELS.get(family)
            if label is None:
                gate_prefix = "gate" if gated else "flat"
                label = f"{gate_prefix} target {_aib_target_token(target)}"
            rows.append({
                "config_stem": stem,
                "family": family,
                "family_label": label,
                "family_order": AIB_FAMILY_ORDER.get(family, 99),
                "seed": int(seed_match.group(1)) if seed_match else args.get("seed"),
                "expected_run_name": run.get("name") or args.get("exp_tag") or stem,
                "config_path": str(config_path),
                "intervention_gate": gated,
                "intervention_budget_target": target,
                "intervention_budget_cost_mode": args.get("intervention_budget_cost_mode"),
                "intervention_gate_eval_mode": args.get("intervention_gate_eval_mode"),
                "total_timesteps": args.get("total_timesteps"),
            })
        expected = pd.DataFrame(rows)
        if expected.empty:
            raise FileNotFoundError(f"No TOML configs found under {config_dir}")
        return expected.sort_values(["family_order", "seed", "expected_run_name"]).reset_index(drop=True)

    def _aib_resolve_runs(expected_configs, family, history=None):
        history = _get_history(history)
        available = available_run_names(history=history)
        available_set = set(available)
        resolved = []
        missing = []
        family_configs = expected_configs[expected_configs["family"] == family].sort_values("seed")
        for name in family_configs["expected_run_name"]:
            if name in available_set:
                resolved.append(name)
                continue
            substring_matches = [candidate for candidate in available if str(name) in candidate or candidate in str(name)]
            if substring_matches:
                resolved.append(sorted(substring_matches, key=lambda candidate: (len(candidate), candidate))[0])
            else:
                missing.append(name)
        return resolved, missing

    def _aib_resolve_seeded_prefix(prefix, seeds=(0, 1, 2), history=None):
        history = _get_history(history)
        available = available_run_names(history=history)
        available_set = set(available)
        resolved = []
        missing = []
        for name in [f"{prefix}_s{seed}" for seed in seeds]:
            if name in available_set:
                resolved.append(name)
                continue
            substring_matches = [candidate for candidate in available if name in candidate or candidate in name]
            if substring_matches:
                resolved.append(sorted(substring_matches, key=lambda candidate: (len(candidate), candidate))[0])
            else:
                missing.append(name)
        return resolved, missing

    AIB_EXPECTED_CONFIGS = _aib_read_expected_configs()
    baseline_runs, baseline_missing = _aib_resolve_seeded_prefix(AIB_BASELINE_SPEC["prefix"])
    if not baseline_runs:
        raise RuntimeError(
            f"No cached shared baseline runs found for {AIB_BASELINE_SPEC['prefix']}. "
            "Load/download that baseline folder first."
        )

    aib_mean_groups = {}
    unresolved_rows = [
        {"family": baseline_source_key, "expected_run_name": name, "reason": "missing shared baseline run"}
        for name in baseline_missing
    ]
    comparison_families = (
        AIB_EXPECTED_CONFIGS[["family", "family_label", "family_order"]]
        .drop_duplicates()
        .sort_values(["family_order", "family"])
    )
    for item in comparison_families.itertuples(index=False):
        compare_runs, missing = _aib_resolve_runs(AIB_EXPECTED_CONFIGS, item.family)
        unresolved_rows.extend(
            {"family": item.family, "expected_run_name": name, "reason": "missing comparison run"}
            for name in missing
        )
        if not compare_runs:
            print(f"Skipping {item.family_label}: no cached runs found.")
            continue
        compare_color = AIB_GATE_COLOR if "gate" in item.family else AIB_COMPARE_COLOR
        aib_mean_groups[f"{item.family_label} vs {AIB_BASELINE_LABEL}"] = {
            AIB_BASELINE_LABEL: mean_curve(
                baseline_runs,
                color=AIB_BASELINE_COLOR,
                dash="solid",
                width=3,
                member_alpha=0.14,
                std_alpha=0.10,
            ),
            item.family_label: mean_curve(
                compare_runs,
                color=compare_color,
                dash="solid",
                width=3,
                member_alpha=0.14,
                std_alpha=0.12,
            ),
        }

    if not aib_mean_groups:
        raise RuntimeError(
            "No adaptive_intervention_budget_7 comparison groups could be built from history_df."
        )

    fig_aib = plot_run_mean_groups(
        aib_mean_groups,
        split="test",
        smooth=5,
        title=(
            "configs/adaptive_intervention_budget_7: adaptive budget vs "
            f"{AIB_BASELINE_LABEL}"
        ),
        ncols=2,
        subplot_height=380,
        width=1500,
        y_range=[0, 105],
        show_members=True,
        show_std=True,
        save_name=f"adaptive_intervention_budget_7_vs_{baseline_source_key}",
    )
    return {
        "fig": fig_aib,
        "baseline_source": baseline_source_key,
        "baseline_runs": baseline_runs,
        "expected_configs": AIB_EXPECTED_CONFIGS,
        "missing_runs": pd.DataFrame(unresolved_rows),
        "mean_groups": aib_mean_groups,
    }



def plot_gine_neighbors_vs_original():
    # Compare original GINE s0/s1/s2 configs against the include-neighbor reruns.
    # Run after the W&B history-loading cell and the plotting-helper cell above.
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    from IPython.display import display

    GINE_NEIGHBOR_BASE_DIR = TASK_DIR / "configs" / "gine_s0_s1_s2"
    GINE_NEIGHBOR_INCLUDE_DIR = TASK_DIR / "configs" / "gine_s0_s1_s2_include_neighbors"
    GINE_NEIGHBOR_SEEDS = (0, 1, 2)
    GINE_NEIGHBOR_SPLIT = "test"
    GINE_NEIGHBOR_SMOOTH = 5
    GINE_NEIGHBOR_BASE_LABEL = "no neighbors"
    GINE_NEIGHBOR_INCLUDE_LABEL = "include neighbors"
    GINE_NEIGHBOR_BASE_COLOR = "#1f77b4"
    GINE_NEIGHBOR_INCLUDE_COLOR = "#d62728"

    # Keep this on so copied configs with the same exp_tag can still be separated by W&B config.
    # The enrichment only touches candidate runs matching these config stems.
    GINE_NEIGHBOR_ENRICH_FROM_WANDB = True

    # Fill only if a W&B run is not recoverable from the config stem/exp_tag.
    # Keys are config stems; values can be a W&B run name or run id.
    GINE_NEIGHBOR_RUN_NAME_OVERRIDES = {
        "base": {},
        "include_neighbors": {},
    }

    GINE_NEIGHBOR_INCLUDE_NAME_PATTERNS = (
        "{stem}_include_neighbors",
        "{stem}_gnn_include_neighbors",
        "{stem}_with_neighbors",
        "{stem}_neighbors",
        "include_neighbors_{stem}",
        "{stem}",
    )

    GINE_NEIGHBOR_FAMILY_LABELS = {
        "best_000_shared_actor_gnn_gine_a4_no_concat_flat_critic_gnn_legacy_update": "000 no concat-flat",
        "best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update": "00 baseline",
        "best_01_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_optcritic": "01 optimized critic",
        "best_02_shared_actor_gnn_gine_a4_concat_flat_critic_mlp_legacy_update": "02 MLP critic",
        "best_03_nonshared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update": "03 non-shared actor",
        "best_04_shared_actor_gnn_light_gine_a4_concat_flat_critic_gnn_legacy_update": "04 light GINE",
        "best_05_shared_actor_gnn_gine_a4_entropy_decay_concat_flat_critic_gnn_legacy_update": "05 entropy decay",
        "best_06_shared_actor_gnn_gine_a4_no_node_id_concat_flat_critic_gnn_legacy_update": "06 no node ID",
        "best_07_shared_actor_gnn_gat_a4_concat_flat_critic_gnn_legacy_update": "07 GAT",
        "best_08_shared_actor_gnn_weighted_gcn_a4_concat_flat_critic_gnn_legacy_update": "08 weighted GCN",
    }


    def _gine_neighbor_read_configs(config_dir, variant):
        rows = []
        for path in sorted(Path(config_dir).glob("*.toml")):
            cfg = tomllib.loads(path.read_text(encoding="utf-8"))
            args = cfg.get("args", {})
            run = cfg.get("run", {})
            stem = path.stem
            seed_match = re.search(r"_s(\d+)$", stem)
            rows.append({
                "variant": variant,
                "config_path": str(path),
                "stem": stem,
                "family": re.sub(r"_s\d+$", "", stem),
                "seed": int(seed_match.group(1)) if seed_match else args.get("seed"),
                "exp_tag": args.get("exp_tag"),
                "run_label": run.get("name"),
                "config_gnn_include_neighbors": args.get("gnn_include_neighbors"),
                "config_total_timesteps": args.get("total_timesteps"),
            })
        return pd.DataFrame(rows)


    def _gine_neighbor_unique(values):
        seen = set()
        result = []
        for value in values:
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except TypeError:
                pass
            text = str(value)
            if text and text not in seen:
                result.append(text)
                seen.add(text)
        return result


    def _gine_neighbor_bool(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        if isinstance(value, bool):
            return value
        lower = str(value).strip().lower()
        if lower in {"1", "true", "yes", "y", "on"}:
            return True
        if lower in {"0", "false", "no", "n", "off"}:
            return False
        return None


    def _gine_neighbor_number(value):
        try:
            if pd.isna(value):
                return np.nan
            return float(value)
        except (TypeError, ValueError):
            return np.nan


    def _gine_neighbor_inserted_names(name):
        match = re.match(r"^(best_\d+)(_.+)$", str(name))
        if not match:
            return []
        return [f"{match.group(1)}_neighbors{match.group(2)}"]


    def _gine_neighbor_candidate_names(row, variant):
        overrides = GINE_NEIGHBOR_RUN_NAME_OVERRIDES.get(variant, {})
        candidates = []
        override = overrides.get(row["stem"])
        if override:
            candidates.append(override)

        sources = _gine_neighbor_unique([row.get("stem"), row.get("exp_tag"), row.get("run_label")])
        if variant == "include_neighbors":
            for source in sources:
                candidates.extend(_gine_neighbor_inserted_names(source))
                candidates.extend(pattern.format(stem=source) for pattern in GINE_NEIGHBOR_INCLUDE_NAME_PATTERNS)
        candidates.extend(sources)
        return _gine_neighbor_unique(candidates)


    def _gine_neighbor_run_catalog(history=None):
        history = _get_history(history)
        if history.empty:
            raise RuntimeError("history_df is empty. Run the W&B history-loading cell first.")

        catalog = history[["run_name", "run_id"]].dropna().drop_duplicates().copy()
        catalog["run_name"] = catalog["run_name"].astype(str)
        catalog["run_id"] = catalog["run_id"].astype(str)

        if "runs_df" in globals() and isinstance(runs_df, pd.DataFrame) and not runs_df.empty:
            meta = runs_df.copy()
            if "id" in meta.columns:
                meta = meta.rename(columns={"id": "run_id", "name": "run_name_from_runs_df"})
                keep = [
                    col for col in [
                        "run_id", "run_name_from_runs_df", "exp_tag", "created_at", "global_step",
                        "gnn_include_neighbors", "total_timesteps", "seed",
                    ] if col in meta.columns
                ]
                catalog = catalog.merge(meta[keep].drop_duplicates("run_id"), on="run_id", how="left")
                if "run_name_from_runs_df" in catalog.columns:
                    catalog["run_name"] = catalog["run_name"].fillna(catalog["run_name_from_runs_df"])
                    catalog = catalog.drop(columns=["run_name_from_runs_df"])

        for col in ["exp_tag", "created_at", "global_step", "gnn_include_neighbors", "total_timesteps", "seed"]:
            if col not in catalog.columns:
                catalog[col] = np.nan
        return catalog


    def _gine_neighbor_initial_candidate_mask(catalog, configs):
        all_names = set()
        for _, row in configs.iterrows():
            for variant in ["base", "include_neighbors"]:
                all_names.update(_gine_neighbor_candidate_names(row, variant))
        mask = pd.Series(False, index=catalog.index)
        for col in ["run_name", "run_id", "exp_tag"]:
            if col in catalog.columns:
                values = catalog[col].fillna("").astype(str)
                mask = mask | values.isin(all_names)
                for name in all_names:
                    if name:
                        mask = mask | values.str.contains(re.escape(name), na=False)
        return mask


    def _gine_neighbor_enrich_catalog_from_wandb(catalog, candidate_mask):
        if not GINE_NEIGHBOR_ENRICH_FROM_WANDB:
            return catalog
        api_obj = globals().get("api")
        if api_obj is None and globals().get("USE_LOCAL_CACHE_ONLY", False):
            return catalog
        if api_obj is None and "wandb" in globals() and wandb is not None:
            api_obj = wandb.Api(timeout=globals().get("WANDB_API_TIMEOUT", 300))
        if api_obj is None:
            print("W&B config enrichment skipped: no `api`/`wandb` object is available.")
            return catalog

        candidate_ids = sorted(catalog.loc[candidate_mask, "run_id"].dropna().astype(str).unique())
        if not candidate_ids:
            return catalog

        print(f"Enriching {len(candidate_ids)} candidate run(s) from W&B config for neighbor matching...")
        updates = []
        for idx, run_id in enumerate(candidate_ids, start=1):
            try:
                wb_run = api_obj.run(f"{ENTITY}/{PROJECT}/{run_id}")
                cfg = dict(wb_run.config)
                summary = dict(wb_run.summary)
                updates.append({
                    "run_id": run_id,
                    "run_name_wandb": wb_run.name,
                    "created_at_wandb": getattr(wb_run, "created_at", None),
                    "exp_tag_wandb": cfg.get("exp_tag"),
                    "gnn_include_neighbors_wandb": cfg.get("gnn_include_neighbors"),
                    "total_timesteps_wandb": cfg.get("total_timesteps"),
                    "seed_wandb": cfg.get("seed"),
                    "global_step_wandb": summary.get("charts/global_step"),
                })
            except Exception as exc:
                print(f"  {idx:>2}/{len(candidate_ids)} skipped {run_id}: {type(exc).__name__}: {exc}")

        if not updates:
            return catalog

        enriched = catalog.merge(pd.DataFrame(updates), on="run_id", how="left")
        for base_col, wb_col in [
            ("run_name", "run_name_wandb"),
            ("created_at", "created_at_wandb"),
            ("exp_tag", "exp_tag_wandb"),
            ("gnn_include_neighbors", "gnn_include_neighbors_wandb"),
            ("total_timesteps", "total_timesteps_wandb"),
            ("seed", "seed_wandb"),
            ("global_step", "global_step_wandb"),
        ]:
            if wb_col in enriched.columns:
                enriched[base_col] = enriched[base_col].where(enriched[base_col].notna(), enriched[wb_col])
                enriched = enriched.drop(columns=[wb_col])
        return enriched


    def _gine_neighbor_score_match(match, row, variant, candidates):
        desired_neighbors = variant == "include_neighbors"
        run_name = str(match.get("run_name", ""))
        run_id = str(match.get("run_id", ""))
        exp_tag = str(match.get("exp_tag", ""))
        text = f"{run_name} {run_id} {exp_tag}".lower()
        score = 0

        if run_name in candidates or run_id in candidates or exp_tag in candidates:
            score += 100
        if str(row["stem"]) in text:
            score += 40
        if "neighbor" in text:
            score += 40 if desired_neighbors else -40

        matched_flag = _gine_neighbor_bool(match.get("gnn_include_neighbors"))
        if matched_flag is not None:
            score += 220 if matched_flag == desired_neighbors else -500

        matched_total = _gine_neighbor_number(match.get("total_timesteps"))
        config_total = _gine_neighbor_number(row.get("config_total_timesteps"))
        if not np.isnan(matched_total) and not np.isnan(config_total):
            score += 80 if int(matched_total) == int(config_total) else -120

        matched_seed = _gine_neighbor_number(match.get("seed"))
        config_seed = _gine_neighbor_number(row.get("seed"))
        if not np.isnan(matched_seed) and not np.isnan(config_seed):
            score += 20 if int(matched_seed) == int(config_seed) else -20

        global_step = _gine_neighbor_number(match.get("global_step"))
        if not np.isnan(global_step):
            if desired_neighbors:
                score += 10 if global_step >= 20_000_000 else 0
            else:
                score += 10 if global_step <= 16_000_000 else -10
        return score


    def _gine_neighbor_sorted_matches(catalog, row, variant):
        candidates = _gine_neighbor_candidate_names(row, variant)
        mask = pd.Series(False, index=catalog.index)
        for col in ["run_name", "run_id", "exp_tag"]:
            values = catalog[col].fillna("").astype(str)
            mask = mask | values.isin(candidates)
            for name in candidates:
                if name:
                    mask = mask | values.str.contains(re.escape(name), na=False)

        matches = catalog[mask].copy()
        if matches.empty:
            return matches

        matches["match_score"] = matches.apply(
            lambda match: _gine_neighbor_score_match(match, row, variant, candidates),
            axis=1,
        )
        matches["created_at_sort"] = pd.to_datetime(matches["created_at"], errors="coerce")
        ascending_created = variant != "include_neighbors"
        return matches.sort_values(
            ["match_score", "created_at_sort", "run_id"],
            ascending=[False, ascending_created, True],
            na_position="last",
        )


    def _gine_neighbor_resolve_configs(configs, catalog):
        resolved_rows = []
        unresolved_rows = []
        base_run_ids_by_stem = {}

        ordered = configs.sort_values(["stem", "variant"]).copy()
        ordered["variant_order"] = ordered["variant"].map({"base": 0, "include_neighbors": 1}).fillna(2)
        ordered = ordered.sort_values(["stem", "variant_order"])

        for _, row in ordered.iterrows():
            variant = row["variant"]
            matches = _gine_neighbor_sorted_matches(catalog, row, variant)
            if matches.empty:
                unresolved_rows.append({
                    "variant": variant,
                    "family": row["family"],
                    "seed": row["seed"],
                    "stem": row["stem"],
                    "reason": "no matching W&B run name/id/exp_tag in history_df",
                })
                continue

            if variant == "include_neighbors" and row["stem"] in base_run_ids_by_stem:
                base_run_id = base_run_ids_by_stem[row["stem"]]
                non_base_matches = matches[matches["run_id"].astype(str) != str(base_run_id)]
                choice_pool = non_base_matches if not non_base_matches.empty else matches.iloc[0:0]
            else:
                choice_pool = matches

            if choice_pool.empty:
                unresolved_rows.append({
                    "variant": variant,
                    "family": row["family"],
                    "seed": row["seed"],
                    "stem": row["stem"],
                    "reason": "only matched the same run_id as the no-neighbor run",
                })
                continue

            choice = choice_pool.iloc[0]
            alias_prefix = "base" if variant == "base" else "neighbors"
            alias = f"{alias_prefix} | {row['stem']}"
            resolved_rows.append({
                "variant": variant,
                "family": row["family"],
                "seed": row["seed"],
                "stem": row["stem"],
                "alias": alias,
                "run_name": choice.get("run_name"),
                "run_id": choice.get("run_id"),
                "exp_tag": choice.get("exp_tag"),
                "match_score": choice.get("match_score"),
                "gnn_include_neighbors": choice.get("gnn_include_neighbors"),
                "total_timesteps": choice.get("total_timesteps"),
                "global_step": choice.get("global_step"),
            })
            if variant == "base":
                base_run_ids_by_stem[row["stem"]] = choice.get("run_id")

        return pd.DataFrame(resolved_rows), pd.DataFrame(unresolved_rows)


    def _gine_neighbor_alias_history(history, resolved):
        frames = []
        for row in resolved.itertuples(index=False):
            run_history = history[history["run_id"].astype(str) == str(row.run_id)].copy()
            if run_history.empty:
                print(f"No history rows found for resolved run_id {row.run_id} ({row.alias})")
                continue
            run_history["run_name"] = row.alias
            frames.append(run_history)
        if not frames:
            return pd.DataFrame(columns=history.columns)
        return pd.concat(frames, ignore_index=True)


    def _gine_neighbor_family_label(family):
        return GINE_NEIGHBOR_FAMILY_LABELS.get(family, family.replace("_", " "))


    def _gine_neighbor_final_survival(alias, history):
        data, metric = _run_metric_frame(history, alias, _metric_candidates(metric=None, split=GINE_NEIGHBOR_SPLIT))
        if data.empty:
            return {"metric": None, "final_step_m": np.nan, "final_survival_pct": np.nan}
        final = data.sort_values("step").iloc[-1]
        return {
            "metric": metric,
            "final_step_m": float(final["step"] / 1_000_000),
            "final_survival_pct": float(final["value"] * 100.0),
        }


    GINE_NEIGHBOR_CONFIGS = pd.concat([
        _gine_neighbor_read_configs(GINE_NEIGHBOR_BASE_DIR, "base"),
        _gine_neighbor_read_configs(GINE_NEIGHBOR_INCLUDE_DIR, "include_neighbors"),
    ], ignore_index=True)

    GINE_NEIGHBOR_RUN_CATALOG = _gine_neighbor_run_catalog()
    _candidate_mask = _gine_neighbor_initial_candidate_mask(GINE_NEIGHBOR_RUN_CATALOG, GINE_NEIGHBOR_CONFIGS)
    GINE_NEIGHBOR_RUN_CATALOG = _gine_neighbor_enrich_catalog_from_wandb(GINE_NEIGHBOR_RUN_CATALOG, _candidate_mask)

    GINE_NEIGHBOR_RESOLVED_RUNS, GINE_NEIGHBOR_UNRESOLVED_RUNS = _gine_neighbor_resolve_configs(
        GINE_NEIGHBOR_CONFIGS,
        GINE_NEIGHBOR_RUN_CATALOG,
    )

    if GINE_NEIGHBOR_RESOLVED_RUNS.empty:
        if not GINE_NEIGHBOR_UNRESOLVED_RUNS.empty:
            # display(GINE_NEIGHBOR_UNRESOLVED_RUNS.sort_values(["variant", "family", "seed"]))
            pass
        raise RuntimeError(
            "No GINE neighbor-comparison runs were resolved. Run/download the histories for "
            "gine_s0_s1_s2 and gine_s0_s1_s2_include_neighbors first, then rerun this cell."
        )

    GINE_NEIGHBOR_HISTORY = _gine_neighbor_alias_history(_get_history(), GINE_NEIGHBOR_RESOLVED_RUNS)

    _final_rows = []
    for row in GINE_NEIGHBOR_RESOLVED_RUNS.itertuples(index=False):
        final = _gine_neighbor_final_survival(row.alias, GINE_NEIGHBOR_HISTORY)
        _final_rows.append({**row._asdict(), **final})
    GINE_NEIGHBOR_RESOLVED_RUNS = pd.DataFrame(_final_rows)

    _summary_rows = []
    for family, family_runs in GINE_NEIGHBOR_RESOLVED_RUNS.groupby("family"):
        base = family_runs[family_runs["variant"] == "base"]
        neighbors = family_runs[family_runs["variant"] == "include_neighbors"]
        _summary_rows.append({
            "family": _gine_neighbor_family_label(family),
            "no_neighbor_runs": len(base),
            "include_neighbor_runs": len(neighbors),
            "no_neighbors_final_mean_pct": base["final_survival_pct"].mean(),
            "include_neighbors_final_mean_pct": neighbors["final_survival_pct"].mean(),
            "delta_include_minus_no_neighbors_pct": neighbors["final_survival_pct"].mean() - base["final_survival_pct"].mean(),
        })
    GINE_NEIGHBOR_SUMMARY = pd.DataFrame(_summary_rows).sort_values(
        "delta_include_minus_no_neighbors_pct",
        ascending=False,
        na_position="last",
    )

    if not GINE_NEIGHBOR_UNRESOLVED_RUNS.empty:
        print("Unresolved config runs. These are usually runs that are not downloaded yet, or W&B names/exp_tags collide without config metadata.")
        # display(GINE_NEIGHBOR_UNRESOLVED_RUNS.sort_values(["variant", "family", "seed"]))

    # print("Resolved runs:")
    # display(GINE_NEIGHBOR_RESOLVED_RUNS.sort_values(["family", "variant", "seed"])[[
    #     "family", "variant", "seed", "alias", "run_name", "run_id", "match_score",
    #     "gnn_include_neighbors", "total_timesteps", "global_step", "final_step_m", "final_survival_pct",
    # ]])

    # print("Final survival summary:")
    # display(GINE_NEIGHBOR_SUMMARY)

    GINE_NEIGHBOR_MEAN_GROUPS = {}
    for family in sorted(GINE_NEIGHBOR_CONFIGS["family"].unique()):
        family_runs = GINE_NEIGHBOR_RESOLVED_RUNS[GINE_NEIGHBOR_RESOLVED_RUNS["family"] == family]
        base_aliases = family_runs[family_runs["variant"] == "base"].sort_values("seed")["alias"].tolist()
        neighbor_aliases = family_runs[family_runs["variant"] == "include_neighbors"].sort_values("seed")["alias"].tolist()
        if not base_aliases or not neighbor_aliases:
            print(f"Skipping {_gine_neighbor_family_label(family)}: missing one side of the comparison.")
            continue
        GINE_NEIGHBOR_MEAN_GROUPS[_gine_neighbor_family_label(family)] = {
            GINE_NEIGHBOR_BASE_LABEL: mean_curve(
                base_aliases,
                color=GINE_NEIGHBOR_BASE_COLOR,
                dash="solid",
                width=3,
                member_alpha=0.14,
                std_alpha=0.10,
            ),
            GINE_NEIGHBOR_INCLUDE_LABEL: mean_curve(
                neighbor_aliases,
                color=GINE_NEIGHBOR_INCLUDE_COLOR,
                dash="solid",
                width=3,
                member_alpha=0.14,
                std_alpha=0.12,
            ),
        }

    if not GINE_NEIGHBOR_MEAN_GROUPS:
        raise RuntimeError(
            "No complete no-neighbor vs include-neighbor groups could be built. "
            "Make sure histories for both folders are present in history_df. If the copied configs reused the same W&B names, "
            "keep GINE_NEIGHBOR_ENRICH_FROM_WANDB=True or fill GINE_NEIGHBOR_RUN_NAME_OVERRIDES."
        )

    fig_gine_neighbors_vs_base = plot_run_mean_groups(
        GINE_NEIGHBOR_MEAN_GROUPS,
        split=GINE_NEIGHBOR_SPLIT,
        smooth=GINE_NEIGHBOR_SMOOTH,
        title="configs/gine_s0_s1_s2 vs configs/gine_s0_s1_s2_include_neighbors: include neighbor nodes vs original local subgraphs",
        ncols=2,
        subplot_height=360,
        width=1550,
        y_range=[0, 105],
        show_members=True,
        show_std=True,
        history=GINE_NEIGHBOR_HISTORY,
        save_name="gine_s0_s1_s2_include_neighbors_vs_original",
    )
    fig_gine_neighbors_vs_base

    return {
        "fig": fig_gine_neighbors_vs_base,
        "configs": GINE_NEIGHBOR_CONFIGS,
        "resolved_runs": GINE_NEIGHBOR_RESOLVED_RUNS,
        "unresolved_runs": GINE_NEIGHBOR_UNRESOLVED_RUNS,
        "summary": GINE_NEIGHBOR_SUMMARY,
        "history": GINE_NEIGHBOR_HISTORY,
        "mean_groups": GINE_NEIGHBOR_MEAN_GROUPS,
    }



def plot_phase4_sparse_control_16_survival():
    # Survival-only baseline comparisons for phase4_sparse_control_16.
    # Style mirrors the earlier mean-comparison cells: thin seed members, thick condition averages.
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    try:
        from IPython.display import display
    except ImportError:
        display = print

    PHASE4_SPARSE_CONFIG_DIR = TASK_DIR / "configs" / "phase4_sparse_control_16"
    PHASE4_SPARSE_SURVIVAL_SMOOTH = 5
    PHASE4_SPARSE_SURVIVAL_SAVE_NAME = "phase4_sparse_control_16_survival_baseline_comparisons"
    PHASE4_SPARSE_BASELINE_DESIGN = "flat"
    PHASE4_SPARSE_BASELINE_PENALTY = 0.0
    PHASE4_SPARSE_BASELINE_COLOR = "#1f77b4"
    PHASE4_SPARSE_COMPARE_COLOR = "#ff7f0e"
    PHASE4_SPARSE_MEMBER_ALPHA = 0.18
    PHASE4_SPARSE_MEMBER_WIDTH = 1.1
    PHASE4_SPARSE_MEAN_WIDTH = 4.0
    PHASE4_SPARSE_SHOW_STD = True
    PHASE4_SPARSE_STD_ALPHA = 0.10
    PHASE4_SPARSE_NCOLS = 3
    PHASE4_SPARSE_SURVIVAL_CANDIDATES = [
        "test/charts/episodic_survival",
        "test/episodic_survival",
        "charts/episodic_survival",
    ]
    PHASE4_SPARSE_DESIGN_ORDER = {"flat": 0, "gated": 1}


    def _phase4_sparse_seed_from_name(name):
        match = re.search(r"_s(\d+)$", str(name))
        return int(match.group(1)) if match else np.nan


    def _phase4_sparse_design_from_name(name, args):
        if "intervention_gate" in args:
            return "gated" if bool(args.get("intervention_gate")) else "flat"
        match = re.search(r"sparse16_(flat|gated)_", str(name))
        return match.group(1) if match else "unknown"


    def _phase4_sparse_penalty_from_name(name, args):
        penalty = args.get("intervention_penalty")
        if penalty is not None:
            return float(penalty)
        match = re.search(r"_p(\d+)_", str(name))
        return int(match.group(1)) / 1000.0 if match else np.nan


    def _phase4_sparse_condition_key(design, penalty):
        return f"{design}|{float(penalty):.6f}"


    def _phase4_sparse_condition_label(design, penalty):
        return f"{design} p{float(penalty):.3f}"


    def _phase4_sparse_color_with_alpha(color, alpha):
        match = re.fullmatch(r"#?([0-9A-Fa-f]{6})", str(color))
        if not match:
            return f"rgba(127,127,127,{alpha})"
        value = match.group(1)
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
        return f"rgba({red},{green},{blue},{alpha})"


    def phase4_sparse_read_expected_configs(config_dir=PHASE4_SPARSE_CONFIG_DIR):
        rows = []
        for config_path in sorted(Path(config_dir).glob("*.toml")):
            cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
            args = cfg.get("args", {})
            run = cfg.get("run", {})
            stem = config_path.stem
            seed = args.get("seed")
            if seed is None:
                seed = _phase4_sparse_seed_from_name(stem)
            design = _phase4_sparse_design_from_name(stem, args)
            penalty = _phase4_sparse_penalty_from_name(stem, args)
            expected_run_name = str(run.get("name") or args.get("exp_tag") or stem).strip()
            rows.append({
                "config_stem": stem,
                "expected_run_name": expected_run_name,
                "config_path": str(config_path),
                "design": design,
                "penalty": penalty,
                "seed": int(seed) if not pd.isna(seed) else np.nan,
                "intervention_gate": bool(args.get("intervention_gate", design == "gated")),
                "total_timesteps": args.get("total_timesteps"),
            })
        expected = pd.DataFrame(rows)
        if expected.empty:
            raise FileNotFoundError(f"No TOML configs found under {config_dir}")
        expected["condition_key"] = expected.apply(
            lambda row: _phase4_sparse_condition_key(row["design"], row["penalty"]), axis=1
        )
        expected["condition_label"] = expected.apply(
            lambda row: _phase4_sparse_condition_label(row["design"], row["penalty"]), axis=1
        )
        expected["run_label"] = expected.apply(
            lambda row: f"{row['condition_label']} s{int(row['seed'])}", axis=1
        )
        expected["design_order"] = expected["design"].map(PHASE4_SPARSE_DESIGN_ORDER).fillna(99)
        return expected.sort_values(["penalty", "design_order", "seed", "expected_run_name"]).reset_index(drop=True)


    def phase4_sparse_load_cached_histories(expected_configs, cache_index_path=CACHE_INDEX_PATH):
        if not Path(cache_index_path).exists():
            raise FileNotFoundError(
                f"Missing full-history cache index: {cache_index_path}. Run the history loading cell first."
            )

        cache_index = pd.read_csv(cache_index_path)
        if cache_index.empty:
            raise RuntimeError(f"Full-history cache index is empty: {cache_index_path}")
        cache_index["name_raw"] = cache_index["name"]
        cache_index["name"] = cache_index["name"].astype(str).str.strip()
        cache_index = cache_index.drop_duplicates("name", keep="last")

        coverage = expected_configs.merge(
            cache_index,
            left_on="expected_run_name",
            right_on="name",
            how="left",
        )
        coverage["cache_exists"] = coverage["history_parquet"].apply(
            lambda value: isinstance(value, str) and Path(value).exists()
        ) | coverage["history_csv"].apply(
            lambda value: isinstance(value, str) and Path(value).exists()
        )

        histories = []
        for _, row in coverage[coverage["cache_exists"]].iterrows():
            parquet_path = Path(row["history_parquet"]) if isinstance(row.get("history_parquet"), str) else None
            csv_path = Path(row["history_csv"]) if isinstance(row.get("history_csv"), str) else None
            if parquet_path is not None and parquet_path.exists():
                history = pd.read_parquet(parquet_path)
            elif csv_path is not None and csv_path.exists():
                history = pd.read_csv(csv_path)
            else:
                continue

            history = history.copy()
            history["run_name"] = row["expected_run_name"]
            history["run_id"] = row.get("id")
            step_col = "_step" if "_step" in history.columns else "step"
            if step_col not in history.columns:
                print(f"Skipping {row['expected_run_name']}: no _step/step column in cached history")
                continue
            history["step"] = pd.to_numeric(history[step_col], errors="coerce")
            history["step_millions"] = history["step"] / 1_000_000.0
            for col in [
                "config_stem", "expected_run_name", "design", "penalty", "seed", "condition_key",
                "condition_label", "run_label", "intervention_gate", "total_timesteps",
            ]:
                history[col] = row[col]
            histories.append(history)

        if histories:
            history = pd.concat(histories, ignore_index=True, sort=False)
        else:
            history = pd.DataFrame()
        coverage["cache_status"] = np.where(coverage["cache_exists"], "cached", "missing")
        coverage = coverage.sort_values(["penalty", "design_order", "seed", "expected_run_name"]).reset_index(drop=True)
        return coverage, history


    def phase4_sparse_build_survival_long(history):
        rows = []
        if history.empty:
            return pd.DataFrame()
        for run_name, run_history in history.groupby("run_name", sort=False):
            metric_col = next(
                (col for col in PHASE4_SPARSE_SURVIVAL_CANDIDATES if col in run_history.columns and run_history[col].notna().any()),
                None,
            )
            if metric_col is None:
                continue
            frame = run_history[[
                "run_name", "run_id", "run_label", "condition_key", "condition_label", "design",
                "penalty", "seed", "step", "step_millions",
            ]].copy()
            frame["episodic_survival"] = pd.to_numeric(run_history[metric_col], errors="coerce") * 100.0
            frame["source_metric"] = metric_col
            rows.append(frame.dropna(subset=["step", "step_millions", "episodic_survival"]))
        if not rows:
            return pd.DataFrame()
        survival = pd.concat(rows, ignore_index=True)
        survival = survival.sort_values(["condition_key", "seed", "step"])
        # Smooth individual seed/member curves first; averages are computed from these smoothed values.
        survival["member_value"] = survival.groupby("run_name", sort=False)["episodic_survival"].transform(
            lambda series: series.rolling(PHASE4_SPARSE_SURVIVAL_SMOOTH, min_periods=1).mean()
        )
        return survival


    def phase4_sparse_condition_stats(survival):
        if survival.empty:
            return pd.DataFrame()
        # Average smoothed seed values at exact logged steps only; shorter runs stop contributing, no interpolation.
        stats = (
            survival.groupby(["condition_key", "condition_label", "design", "penalty", "step", "step_millions"], dropna=False)
            .agg(
                mean=("member_value", "mean"),
                std=("member_value", "std"),
                n=("member_value", "count"),
            )
            .reset_index()
            .sort_values(["condition_key", "step"])
        )
        stats["std"] = stats["std"].fillna(0.0)
        # `mean_smoothed`/`std_smoothed` are kept as plotting names, but no second smoothing is applied.
        stats["mean_smoothed"] = stats["mean"]
        stats["std_smoothed"] = stats["std"]
        return stats


    def phase4_sparse_comparison_conditions(coverage, survival):
        baseline_key = _phase4_sparse_condition_key(PHASE4_SPARSE_BASELINE_DESIGN, PHASE4_SPARSE_BASELINE_PENALTY)
        cached_keys = set(survival["condition_key"].dropna().unique()) if not survival.empty else set()
        rows = (
            coverage[["condition_key", "condition_label", "design", "penalty", "design_order"]]
            .drop_duplicates()
            .sort_values(["penalty", "design_order", "condition_label"])
        )
        comparisons = []
        skipped = []
        for row in rows.itertuples(index=False):
            if row.condition_key == baseline_key:
                continue
            if row.condition_key in cached_keys:
                comparisons.append(row)
            else:
                skipped.append(row.condition_label)
        return baseline_key, comparisons, skipped


    def _phase4_sparse_add_condition_traces(
        fig,
        survival,
        stats,
        condition_key,
        label,
        color,
        row,
        col,
        legend_seen,
        *,
        role,
    ):
        members = survival[survival["condition_key"] == condition_key]
        condition_stats = stats[stats["condition_key"] == condition_key]
        if members.empty or condition_stats.empty:
            return 0

        added = 0
        legendgroup = f"{role}:{condition_key}"
        for run_name, member in members.groupby("run_name", sort=False):
            member = member.sort_values("step")
            seed = int(member["seed"].iloc[0]) if member["seed"].notna().any() else "?"
            fig.add_trace(
                go.Scatter(
                    x=member["step_millions"],
                    y=member["member_value"],
                    mode="lines",
                    name=f"{label} seed {seed}",
                    legendgroup=legendgroup,
                    showlegend=False,
                    opacity=PHASE4_SPARSE_MEMBER_ALPHA,
                    line={"color": color, "width": PHASE4_SPARSE_MEMBER_WIDTH},
                    hovertemplate=(
                        f"<b>{run_name}</b><br>"
                        "step=%{x:.2f}M<br>"
                        "smoothed seed survival=%{y:.2f}%<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            added += 1

        condition_stats = condition_stats.sort_values("step")
        if PHASE4_SPARSE_SHOW_STD and condition_stats["n"].max() > 1:
            upper = condition_stats["mean_smoothed"] + condition_stats["std_smoothed"]
            lower = condition_stats["mean_smoothed"] - condition_stats["std_smoothed"]
            fig.add_trace(
                go.Scatter(
                    x=list(condition_stats["step_millions"]) + list(condition_stats["step_millions"].iloc[::-1]),
                    y=list(upper) + list(lower.iloc[::-1]),
                    mode="lines",
                    name=f"{label} std",
                    legendgroup=legendgroup,
                    showlegend=False,
                    line={"color": "rgba(0,0,0,0)", "width": 0},
                    fill="toself",
                    fillcolor=_phase4_sparse_color_with_alpha(color, PHASE4_SPARSE_STD_ALPHA),
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            added += 1

        showlegend = label not in legend_seen
        legend_seen.add(label)
        fig.add_trace(
            go.Scatter(
                x=condition_stats["step_millions"],
                y=condition_stats["mean_smoothed"],
                mode="lines",
                name=label,
                legendgroup=legendgroup,
                showlegend=showlegend,
                customdata=np.stack([condition_stats["n"], condition_stats["mean"], condition_stats["std"]], axis=-1),
                line={"color": color, "width": PHASE4_SPARSE_MEAN_WIDTH},
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "step=%{x:.2f}M<br>"
                    "mean of smoothed seeds=%{y:.2f}%<br>"
                    "mean=%{customdata[1]:.2f}%<br>"
                    "std=%{customdata[2]:.2f}%<br>"
                    "runs at step=%{customdata[0]}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        added += 1
        return added


    def phase4_sparse_plot_survival_baseline_comparisons(
        survival,
        coverage,
        save_name=PHASE4_SPARSE_SURVIVAL_SAVE_NAME,
    ):
        if survival.empty:
            raise RuntimeError("No episodic survival data was available in the cached histories.")

        stats = phase4_sparse_condition_stats(survival)
        baseline_key, comparisons, skipped = phase4_sparse_comparison_conditions(coverage, survival)
        if baseline_key not in set(survival["condition_key"].dropna().unique()):
            raise RuntimeError(
                "The baseline condition has no cached survival data. "
                f"Expected {_phase4_sparse_condition_label(PHASE4_SPARSE_BASELINE_DESIGN, PHASE4_SPARSE_BASELINE_PENALTY)}."
            )
        if not comparisons:
            raise RuntimeError("No cached alternative conditions were available to compare against the baseline.")

        ncols = min(PHASE4_SPARSE_NCOLS, len(comparisons))
        nrows = int(np.ceil(len(comparisons) / ncols))
        baseline_label = _phase4_sparse_condition_label(PHASE4_SPARSE_BASELINE_DESIGN, PHASE4_SPARSE_BASELINE_PENALTY)
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[f"{item.condition_label} vs {baseline_label}" for item in comparisons],
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.07,
            vertical_spacing=0.13,
        )

        legend_seen = set()
        added = 0
        for idx, item in enumerate(comparisons, start=1):
            row = int(np.ceil(idx / ncols))
            col = ((idx - 1) % ncols) + 1
            added += _phase4_sparse_add_condition_traces(
                fig,
                survival,
                stats,
                baseline_key,
                f"baseline: {baseline_label}",
                PHASE4_SPARSE_BASELINE_COLOR,
                row,
                col,
                legend_seen,
                role="baseline",
            )
            added += _phase4_sparse_add_condition_traces(
                fig,
                survival,
                stats,
                item.condition_key,
                item.condition_label,
                PHASE4_SPARSE_COMPARE_COLOR,
                row,
                col,
                legend_seen,
                role="compare",
            )
            fig.update_xaxes(title_text="Steps (M)", row=row, col=col)
            fig.update_yaxes(title_text="Episodic survival (%)", range=[0, 105], row=row, col=col)

        missing = coverage.loc[~coverage["cache_exists"], "expected_run_name"].tolist()
        missing_text = "Missing cached histories: " + ", ".join(missing) if missing else "All expected runs are cached."
        skipped_text = "Skipped uncached conditions: " + ", ".join(skipped) if skipped else "Every non-baseline condition has at least one cached run."
        run_max_steps = (
            survival.groupby(["condition_key", "condition_label", "run_name", "seed"], dropna=False, as_index=False)
            .agg(max_step_m=("step_millions", "max"))
        )
        if run_max_steps.empty:
            short_text = ""
        else:
            condition_max = run_max_steps.groupby("condition_key", dropna=False)["max_step_m"].transform("max")
            short_runs = run_max_steps[run_max_steps["max_step_m"] < 0.80 * condition_max].copy()
            if short_runs.empty:
                short_text = ""
            else:
                examples = ", ".join(
                    f"{row.run_name} {row.max_step_m:.2f}M"
                    for row in short_runs.sort_values(["condition_label", "seed"]).head(6).itertuples(index=False)
                )
                more = f", +{len(short_runs) - 6} more" if len(short_runs) > 6 else ""
                short_text = f"<br>Shorter cached histories vs same condition: {examples}{more}."
        fig.add_annotation(
            text=(
                f"Thin lines are smoothed individual seeds; thick lines average the smoothed seed values. "
                f"Smooth={PHASE4_SPARSE_SURVIVAL_SMOOTH} logged points.<br>"
                f"Baseline is {baseline_label}. {missing_text}<br>{skipped_text}{short_text}"
            ),
            x=0,
            y=1.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font={"size": 12, "color": "#374151"},
        )
        fig.update_layout(
            title="configs/phase4_sparse_control_16: episodic survival baseline comparisons",
            template="plotly_white",
            width=1700,
            height=max(760, 360 * nrows),
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.20, "xanchor": "left", "x": 0},
            margin={"l": 80, "r": 30, "t": 165, "b": 70},
        )
        if added == 0:
            fig.add_annotation(
                text="No data found for the requested baseline comparisons.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
        if "save_plot" in globals():
            save_plot(fig, save_name)
        else:
            output_path = FIG_DIR / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', str(save_name)).strip('._-')}.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_path, include_plotlyjs="cdn")
            print(f"Saved plot: {output_path}")
        if globals().get("SHOW_FIGURES", True):
            fig.show()
        return fig


    PHASE4_SPARSE_EXPECTED_CONFIGS = phase4_sparse_read_expected_configs()
    PHASE4_SPARSE_COVERAGE, PHASE4_SPARSE_HISTORY = phase4_sparse_load_cached_histories(PHASE4_SPARSE_EXPECTED_CONFIGS)
    PHASE4_SPARSE_SURVIVAL_LONG = phase4_sparse_build_survival_long(PHASE4_SPARSE_HISTORY)
    PHASE4_SPARSE_SURVIVAL_STATS = phase4_sparse_condition_stats(PHASE4_SPARSE_SURVIVAL_LONG)

    print(
        f"phase4_sparse_control_16 cache coverage: "
        f"{int(PHASE4_SPARSE_COVERAGE['cache_exists'].sum())}/{len(PHASE4_SPARSE_COVERAGE)} runs cached"
    )
    # display(PHASE4_SPARSE_COVERAGE[[
    #     "expected_run_name", "condition_label", "seed", "cache_status", "rows", "id",
    # ]])

    PHASE4_SPARSE_SURVIVAL_FIG = phase4_sparse_plot_survival_baseline_comparisons(
        PHASE4_SPARSE_SURVIVAL_LONG,
        PHASE4_SPARSE_COVERAGE,
    )
    PHASE4_SPARSE_COMPARISON_FIG = PHASE4_SPARSE_SURVIVAL_FIG

    return {
        "fig": PHASE4_SPARSE_SURVIVAL_FIG,
        "comparison_fig": PHASE4_SPARSE_COMPARISON_FIG,
        "expected_configs": PHASE4_SPARSE_EXPECTED_CONFIGS,
        "coverage": PHASE4_SPARSE_COVERAGE,
        "history": PHASE4_SPARSE_HISTORY,
        "survival_long": PHASE4_SPARSE_SURVIVAL_LONG,
        "survival_stats": PHASE4_SPARSE_SURVIVAL_STATS,
    }



def plot_heuristic_vs_gate_survival():
    # Survival-only baseline comparisons for heuristic_vs_gate_s0_s1_s2.
    # Thin lines are individual seeds; thick lines are seed averages.
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    try:
        from IPython.display import display
    except ImportError:
        display = print

    HVG_CONFIG_DIR = TASK_DIR / "configs" / "heuristic_vs_gate_s0_s1_s2"
    HVG_BASELINE_FAMILY = "hvg_00_baseline"
    HVG_SURVIVAL_SMOOTH = 5
    HVG_SURVIVAL_SAVE_NAME = "heuristic_vs_gate_s0_s1_s2_survival_baseline_comparisons"
    HVG_DOWNLOAD_REGEX = r"^hvg_(?:00_baseline|01_eval_rho090|02_gate_final_map|03_gate_hierarchical|04_eval_local_rho090)_s[0-2]$"
    HVG_BASELINE_COLOR = "#1f77b4"
    HVG_COMPARE_COLOR = "#ff7f0e"
    HVG_MEMBER_ALPHA = 0.18
    HVG_MEMBER_WIDTH = 1.1
    HVG_MEAN_WIDTH = 4.0
    HVG_SHOW_STD = True
    HVG_STD_ALPHA = 0.10
    HVG_NCOLS = 2
    HVG_SURVIVAL_CANDIDATES = [
        "test/charts/episodic_survival",
        "test/episodic_survival",
        "charts/episodic_survival",
        "validation/episodic_survival",
        "train_eval/charts/episodic_survival",
        "train_eval/episodic_survival",
    ]
    HVG_FAMILY_LABELS = {
        "hvg_00_baseline": "baseline",
        "hvg_01_eval_rho090": "global rho heuristic",
        "hvg_04_eval_local_rho090": "local rho heuristic",
        "hvg_02_gate_final_map": "gate final-action MAP",
        "hvg_03_gate_hierarchical": "gate hierarchical greedy",
    }
    HVG_FAMILY_ORDER = {
        "hvg_00_baseline": 0,
        "hvg_01_eval_rho090": 1,
        "hvg_04_eval_local_rho090": 2,
        "hvg_02_gate_final_map": 3,
        "hvg_03_gate_hierarchical": 4,
    }


    def _hvg_seed_from_name(name):
        match = re.search(r"_s(\d+)$", str(name))
        return int(match.group(1)) if match else np.nan


    def _hvg_family_from_name(name):
        return re.sub(r"_s\d+$", "", str(name))


    def _hvg_color_with_alpha(color, alpha):
        match = re.fullmatch(r"#?([0-9A-Fa-f]{6})", str(color))
        if not match:
            return f"rgba(127,127,127,{alpha})"
        value = match.group(1)
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
        return f"rgba({red},{green},{blue},{alpha})"


    def _hvg_survival_scale(values):
        numeric = pd.to_numeric(values, errors="coerce")
        max_value = numeric.max(skipna=True)
        if pd.isna(max_value):
            return 100.0
        return 100.0 if max_value <= 1.5 else 1.0


    def hvg_read_expected_configs(config_dir=HVG_CONFIG_DIR):
        rows = []
        for config_path in sorted(Path(config_dir).glob("*.toml")):
            cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
            args = cfg.get("args", {})
            run = cfg.get("run", {})
            stem = config_path.stem
            family = _hvg_family_from_name(stem)
            seed = args.get("seed")
            if seed is None:
                seed = _hvg_seed_from_name(stem)
            expected_run_name = run.get("name") or args.get("exp_tag") or stem
            rows.append({
                "config_stem": stem,
                "expected_run_name": expected_run_name,
                "config_path": str(config_path),
                "family": family,
                "family_label": HVG_FAMILY_LABELS.get(family, family.replace("_", " ")),
                "family_order": HVG_FAMILY_ORDER.get(family, 99),
                "seed": int(seed) if not pd.isna(seed) else np.nan,
                "intervention_gate": bool(args.get("intervention_gate", False)),
                "heuristic_type": args.get("heuristic_type"),
                "eval_action_heuristic": args.get("eval_action_heuristic"),
                "intervention_gate_eval_mode": args.get("intervention_gate_eval_mode"),
                "total_timesteps": args.get("total_timesteps"),
            })
        expected = pd.DataFrame(rows)
        if expected.empty:
            raise FileNotFoundError(f"No TOML configs found under {config_dir}")
        return expected.sort_values(["family_order", "seed", "expected_run_name"]).reset_index(drop=True)


    def hvg_load_cached_histories(expected_configs, cache_index_path=CACHE_INDEX_PATH):
        if not Path(cache_index_path).exists():
            raise FileNotFoundError(
                f"Missing full-history cache index: {cache_index_path}. Run the history loading cell first."
            )

        cache_index = pd.read_csv(cache_index_path)
        if cache_index.empty:
            raise RuntimeError(f"Full-history cache index is empty: {cache_index_path}")
        cache_index["name"] = cache_index["name"].astype(str)
        cache_index = cache_index.drop_duplicates("name", keep="last")

        coverage = expected_configs.merge(
            cache_index,
            left_on="expected_run_name",
            right_on="name",
            how="left",
        )
        coverage["cache_exists"] = coverage["history_parquet"].apply(
            lambda value: isinstance(value, str) and Path(value).exists()
        ) | coverage["history_csv"].apply(
            lambda value: isinstance(value, str) and Path(value).exists()
        )

        histories = []
        for _, row in coverage[coverage["cache_exists"]].iterrows():
            parquet_path = Path(row["history_parquet"]) if isinstance(row.get("history_parquet"), str) else None
            csv_path = Path(row["history_csv"]) if isinstance(row.get("history_csv"), str) else None
            if parquet_path is not None and parquet_path.exists():
                history = pd.read_parquet(parquet_path)
            elif csv_path is not None and csv_path.exists():
                history = pd.read_csv(csv_path)
            else:
                continue

            history = history.copy()
            history["run_name"] = row["expected_run_name"]
            history["run_id"] = row.get("id")
            step_col = "_step" if "_step" in history.columns else "step"
            if step_col not in history.columns:
                print(f"Skipping {row['expected_run_name']}: no _step/step column in cached history")
                continue
            history["step"] = pd.to_numeric(history[step_col], errors="coerce")
            history["step_millions"] = history["step"] / 1_000_000.0
            for col in [
                "config_stem", "expected_run_name", "family", "family_label", "family_order",
                "seed", "intervention_gate", "heuristic_type", "eval_action_heuristic",
                "intervention_gate_eval_mode", "total_timesteps",
            ]:
                history[col] = row[col]
            histories.append(history)

        if histories:
            history = pd.concat(histories, ignore_index=True, sort=False)
        else:
            history = pd.DataFrame()
        coverage["cache_status"] = np.where(coverage["cache_exists"], "cached", "missing")
        coverage = coverage.sort_values(["family_order", "seed", "expected_run_name"]).reset_index(drop=True)
        return coverage, history


    def hvg_build_survival_long(history):
        rows = []
        if history.empty:
            return pd.DataFrame()
        for run_name, run_history in history.groupby("run_name", sort=False):
            metric_col = next(
                (col for col in HVG_SURVIVAL_CANDIDATES if col in run_history.columns and run_history[col].notna().any()),
                None,
            )
            if metric_col is None:
                continue
            values = pd.to_numeric(run_history[metric_col], errors="coerce")
            frame = run_history[[
                "run_name", "run_id", "family", "family_label", "family_order", "seed", "step", "step_millions",
            ]].copy()
            frame["episodic_survival"] = values * _hvg_survival_scale(values)
            frame["source_metric"] = metric_col
            rows.append(frame.dropna(subset=["step", "step_millions", "episodic_survival"]))
        if not rows:
            return pd.DataFrame()
        survival = pd.concat(rows, ignore_index=True)
        survival = survival.sort_values(["family_order", "seed", "step"])
        # Smooth individual seed/member curves first; averages are computed from these smoothed values.
        survival["member_value"] = survival.groupby("run_name", sort=False)["episodic_survival"].transform(
            lambda series: series.rolling(HVG_SURVIVAL_SMOOTH, min_periods=1).mean()
        )
        return survival


    def hvg_survival_stats(survival):
        if survival.empty:
            return pd.DataFrame()
        # Average smoothed seed values at exact logged steps only; shorter runs stop contributing, no interpolation.
        stats = (
            survival.groupby(["family", "family_label", "family_order", "step", "step_millions"], dropna=False)
            .agg(
                mean=("member_value", "mean"),
                std=("member_value", "std"),
                n=("member_value", "count"),
            )
            .reset_index()
            .sort_values(["family_order", "step"])
        )
        stats["std"] = stats["std"].fillna(0.0)
        # `mean_smoothed`/`std_smoothed` are kept as plotting names, but no second smoothing is applied.
        stats["mean_smoothed"] = stats["mean"]
        stats["std_smoothed"] = stats["std"]
        return stats


    def _hvg_add_family_traces(fig, survival, stats, family, label, color, row, col, legend_seen, role):
        members = survival[survival["family"] == family]
        family_stats = stats[stats["family"] == family]
        if members.empty or family_stats.empty:
            return 0

        added = 0
        legendgroup = f"{role}:{family}"
        for run_name, member in members.groupby("run_name", sort=False):
            member = member.sort_values("step")
            seed = int(member["seed"].iloc[0]) if member["seed"].notna().any() else "?"
            fig.add_trace(
                go.Scatter(
                    x=member["step_millions"],
                    y=member["member_value"],
                    mode="lines",
                    name=f"{label} seed {seed}",
                    legendgroup=legendgroup,
                    showlegend=False,
                    opacity=HVG_MEMBER_ALPHA,
                    line={"color": color, "width": HVG_MEMBER_WIDTH},
                    hovertemplate=(
                        f"<b>{run_name}</b><br>"
                        "step=%{x:.2f}M<br>"
                        "smoothed seed survival=%{y:.2f}%<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            added += 1

        family_stats = family_stats.sort_values("step")
        if HVG_SHOW_STD and family_stats["n"].max() > 1:
            upper = family_stats["mean_smoothed"] + family_stats["std_smoothed"]
            lower = family_stats["mean_smoothed"] - family_stats["std_smoothed"]
            fig.add_trace(
                go.Scatter(
                    x=list(family_stats["step_millions"]) + list(family_stats["step_millions"].iloc[::-1]),
                    y=list(upper) + list(lower.iloc[::-1]),
                    mode="lines",
                    name=f"{label} std",
                    legendgroup=legendgroup,
                    showlegend=False,
                    line={"color": "rgba(0,0,0,0)", "width": 0},
                    fill="toself",
                    fillcolor=_hvg_color_with_alpha(color, HVG_STD_ALPHA),
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            added += 1

        showlegend = label not in legend_seen
        legend_seen.add(label)
        fig.add_trace(
            go.Scatter(
                x=family_stats["step_millions"],
                y=family_stats["mean_smoothed"],
                mode="lines",
                name=label,
                legendgroup=legendgroup,
                showlegend=showlegend,
                customdata=np.stack([family_stats["n"], family_stats["mean"], family_stats["std"]], axis=-1),
                line={"color": color, "width": HVG_MEAN_WIDTH},
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "step=%{x:.2f}M<br>"
                    "mean of smoothed seeds=%{y:.2f}%<br>"
                    "mean=%{customdata[1]:.2f}%<br>"
                    "std=%{customdata[2]:.2f}%<br>"
                    "runs at step=%{customdata[0]}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        added += 1
        return added


    def hvg_plot_survival_baseline_comparisons(survival, coverage, save_name=HVG_SURVIVAL_SAVE_NAME):
        if survival.empty:
            print("No cached episodic survival data for heuristic_vs_gate_s0_s1_s2 yet.")
            print("Download with RUN_NAME_REGEX =", repr(HVG_DOWNLOAD_REGEX))
            return None

        stats = hvg_survival_stats(survival)
        available_families = set(survival["family"].dropna().unique())
        if HVG_BASELINE_FAMILY not in available_families:
            print(f"No cached baseline data for {HVG_BASELINE_FAMILY}; cannot build baseline comparisons yet.")
            print("Download with RUN_NAME_REGEX =", repr(HVG_DOWNLOAD_REGEX))
            return None

        comparisons = (
            coverage[["family", "family_label", "family_order"]]
            .drop_duplicates()
            .query("family != @HVG_BASELINE_FAMILY")
            .sort_values(["family_order", "family_label"])
        )
        comparisons = comparisons[comparisons["family"].isin(available_families)]
        if comparisons.empty:
            print("No cached non-baseline heuristic/gate variants to compare yet.")
            print("Download with RUN_NAME_REGEX =", repr(HVG_DOWNLOAD_REGEX))
            return None

        ncols = min(HVG_NCOLS, len(comparisons))
        nrows = int(np.ceil(len(comparisons) / ncols))
        baseline_label = HVG_FAMILY_LABELS.get(HVG_BASELINE_FAMILY, HVG_BASELINE_FAMILY)
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[f"{row.family_label} vs {baseline_label}" for row in comparisons.itertuples(index=False)],
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.08,
            vertical_spacing=0.13,
        )

        legend_seen = set()
        for idx, item in enumerate(comparisons.itertuples(index=False), start=1):
            row = int(np.ceil(idx / ncols))
            col = ((idx - 1) % ncols) + 1
            _hvg_add_family_traces(
                fig, survival, stats, HVG_BASELINE_FAMILY, f"baseline: {baseline_label}",
                HVG_BASELINE_COLOR, row, col, legend_seen, role="baseline",
            )
            _hvg_add_family_traces(
                fig, survival, stats, item.family, item.family_label,
                HVG_COMPARE_COLOR, row, col, legend_seen, role="compare",
            )
            fig.update_xaxes(title_text="Steps (M)", row=row, col=col)
            fig.update_yaxes(title_text="Episodic survival (%)", range=[0, 105], row=row, col=col)

        missing = coverage.loc[~coverage["cache_exists"], "expected_run_name"].tolist()
        missing_text = "Missing cached histories: " + ", ".join(missing) if missing else "All expected runs are cached."
        fig.add_annotation(
            text=(
                f"Thin lines are smoothed individual seeds; thick lines average the smoothed seed values. "
                f"Smooth={HVG_SURVIVAL_SMOOTH} logged points.<br>{missing_text}"
            ),
            x=0,
            y=1.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font={"size": 12, "color": "#374151"},
        )
        fig.update_layout(
            title="configs/heuristic_vs_gate_s0_s1_s2: episodic survival baseline comparisons",
            template="plotly_white",
            width=1500,
            height=max(760, 380 * nrows),
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.20, "xanchor": "left", "x": 0},
            margin={"l": 80, "r": 30, "t": 165, "b": 70},
        )
        if "save_plot" in globals():
            save_plot(fig, save_name)
        else:
            output_path = FIG_DIR / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', str(save_name)).strip('._-')}.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_path, include_plotlyjs="cdn")
            print(f"Saved plot: {output_path}")
        if globals().get("SHOW_FIGURES", True):
            fig.show()
        return fig


    HVG_EXPECTED_CONFIGS = hvg_read_expected_configs()
    HVG_COVERAGE, HVG_HISTORY = hvg_load_cached_histories(HVG_EXPECTED_CONFIGS)
    HVG_SURVIVAL_LONG = hvg_build_survival_long(HVG_HISTORY)
    HVG_SURVIVAL_STATS = hvg_survival_stats(HVG_SURVIVAL_LONG)

    print(
        f"heuristic_vs_gate_s0_s1_s2 cache coverage: "
        f"{int(HVG_COVERAGE['cache_exists'].sum())}/{len(HVG_COVERAGE)} runs cached"
    )
    # display(HVG_COVERAGE[[
    #     "expected_run_name", "family_label", "seed", "cache_status", "rows", "id",
    # ]])

    HVG_SURVIVAL_BASELINE_FIG = hvg_plot_survival_baseline_comparisons(
        HVG_SURVIVAL_LONG,
        HVG_COVERAGE,
    )

    return {
        "fig": HVG_SURVIVAL_BASELINE_FIG,
        "expected_configs": HVG_EXPECTED_CONFIGS,
        "coverage": HVG_COVERAGE,
        "history": HVG_HISTORY,
        "survival_long": HVG_SURVIVAL_LONG,
        "survival_stats": HVG_SURVIVAL_STATS,
    }
