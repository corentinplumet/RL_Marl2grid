"""Comparison-dashboard helpers for A0, sparse, AIB, heuristic, and BC runs.

The notebooks in this folder should stay small. This module does the slightly
boring work: resolve config folders, read cached W&B histories, extract the
metric columns we care about, and build Plotly figures/tables.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .run_data import RUN_DATA_DIR, scan_run_data
except ImportError:  # pragma: no cover - direct notebook/script execution fallback
    from run_data import RUN_DATA_DIR, scan_run_data

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install plotly first, for example: pip install plotly") from exc

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(value: Any) -> None:
        print(value)


pd.set_option("display.max_columns", 180)
pd.set_option("display.max_rows", 180)


AGENTS = ("agent_0", "agent_1", "agent_2")
JOINT_COUNTS = (0, 1, 2, 3)

A0_HVG_LABELS = {
    "a0_hvg_00_baseline": "baseline",
    "a0_hvg_01_eval_rho090": "global rho heuristic",
    "a0_hvg_04_eval_local_rho090": "local rho heuristic",
    "a0_hvg_02_gate_final_map": "gate final-action MAP",
    "a0_hvg_03_gate_hierarchical": "gate hierarchical greedy",
    "hvg_00_baseline": "baseline",
    "hvg_01_eval_rho090": "global rho heuristic",
    "hvg_04_eval_local_rho090": "local rho heuristic",
    "hvg_02_gate_final_map": "gate final-action MAP",
    "hvg_03_gate_hierarchical": "gate hierarchical greedy",
}
A0_HVG_ORDER = {
    "a0_hvg_00_baseline": 0,
    "a0_hvg_01_eval_rho090": 1,
    "a0_hvg_04_eval_local_rho090": 2,
    "a0_hvg_02_gate_final_map": 3,
    "a0_hvg_03_gate_hierarchical": 4,
    "hvg_00_baseline": 0,
    "hvg_01_eval_rho090": 1,
    "hvg_04_eval_local_rho090": 2,
    "hvg_02_gate_final_map": 3,
    "hvg_03_gate_hierarchical": 4,
}
AIB_LABELS = {
    "a0_aib_00_flat_local_t020": "AIB flat local target 0.20",
    "a0_aib_01_flat_local_t010": "AIB flat local target 0.10",
    "a0_aib_02_flat_local_t035": "AIB flat local target 0.35",
    "a0_aib_03_gate_hgreedy_sep_local_t020": "AIB gate h-greedy target 0.20",
    "a0_aib_04_flat_nonidle_t020": "AIB flat non-idle target 0.20",
    "aib_00_flat_local_t020": "AIB flat local target 0.20",
    "aib_01_flat_local_t010": "AIB flat local target 0.10",
    "aib_02_flat_local_t035": "AIB flat local target 0.35",
    "aib_03_gate_hgreedy_sep_local_t020": "AIB gate h-greedy target 0.20",
    "aib_04_flat_nonidle_t020": "AIB flat non-idle target 0.20",
}
AIB_ORDER = {
    "a0_aib_00_flat_local_t020": 0,
    "a0_aib_01_flat_local_t010": 1,
    "a0_aib_02_flat_local_t035": 2,
    "a0_aib_03_gate_hgreedy_sep_local_t020": 3,
    "a0_aib_04_flat_nonidle_t020": 4,
    "aib_00_flat_local_t020": 0,
    "aib_01_flat_local_t010": 1,
    "aib_02_flat_local_t035": 2,
    "aib_03_gate_hgreedy_sep_local_t020": 3,
    "aib_04_flat_nonidle_t020": 4,
}

SURVIVAL_CANDIDATES = {
    "test": [
        "test/charts/episodic_survival",
        "test/episodic_survival",
        "validation/episodic_survival",
    ],
    "train_eval": [
        "train_eval/charts/episodic_survival",
        "train_eval/episodic_survival",
    ],
    "eval": [
        "eval/charts/episodic_survival",
        "eval/episodic_survival",
        "charts/episodic_survival",
    ],
    "train": ["charts/episodic_survival"],
}


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
REPO_ROOT = TASK_DIR.parent
CONFIG_ROOT = TASK_DIR / "configs"
CACHE_DIR = TASK_DIR / "outputs" / "wandb_cache"
FULL_HISTORY_DIR = CACHE_DIR / "full_history"
CACHE_INDEX_PATH = CACHE_DIR / "full_history_cache_index.csv"
PERMANENT_RUN_DATA_DIR = RUN_DATA_DIR
FIG_DIR = TASK_DIR / "outputs" / "comparison_dashboard_figures"
FULL_TEST_DIR = TASK_DIR / "outputs" / "full_test_eval"
TEACHER_DATASET_ROOT = TASK_DIR / "outputs" / "teacher_student_datasets"
TEACHER_STUDENT_CHECKPOINT_DIR = TASK_DIR / "checkpoint" / "teacher_student"

for directory in [FIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-") or "plot"


def save_fig(fig: Optional[go.Figure], name: str, save: bool = True) -> Optional[Path]:
    if fig is None or not save:
        return None
    path = FIG_DIR / f"{safe_name(name)}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Saved: {path}")
    return path


def show_or_return(fig: Optional[go.Figure], show: bool = True) -> Optional[go.Figure]:
    if fig is not None and show:
        fig.show()
    return fig


def _path_or_none(value: Any) -> Optional[Path]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    path = Path(value).expanduser()
    return path if path.exists() else None


def _relative_to(path: Path, root: Path) -> Optional[Path]:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def _run_data_group_dir(path: Path) -> Optional[Path]:
    relative = _relative_to(path, PERMANENT_RUN_DATA_DIR)
    if relative is None:
        return None
    parts = relative.parts
    if not parts:
        return PERMANENT_RUN_DATA_DIR
    return (PERMANENT_RUN_DATA_DIR / parts[0]).resolve()


def _history_source_folder(row: Dict[str, Any] | pd.Series) -> Optional[Path]:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    history_source_folder = _path_or_none(row.get("history_source_folder"))
    if history_source_folder is not None:
        return history_source_folder.resolve()
    history_source_path = _path_or_none(row.get("history_source_path"))
    if history_source_path is not None:
        return history_source_path.parent.resolve()
    run_data_dir = _path_or_none(row.get("run_data_dir"))
    if run_data_dir is not None:
        return run_data_dir.resolve()
    run_dir = _path_or_none(row.get("run_dir"))
    if run_dir is not None:
        return run_dir.resolve()
    for column in ("history_parquet", "history_csv"):
        history_path = _path_or_none(row.get(column))
        if history_path is None:
            continue
        run_data_group = _run_data_group_dir(history_path)
        if run_data_group is not None:
            return run_data_group
        if _relative_to(history_path, FULL_HISTORY_DIR) is not None:
            return FULL_HISTORY_DIR.resolve()
        return history_path.parent.resolve()
    return None


def print_history_source_folders(
    rows: pd.DataFrame | Sequence[Dict[str, Any]],
    *,
    label: str = "History source folders used to build curves",
    limit: int = 12,
) -> List[Path]:
    if rows is None or len(rows) == 0:
        print(f"{label}: none")
        return []
    iterator = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
    counts: Dict[Path, int] = {}
    for row in iterator:
        folder = _history_source_folder(row)
        if folder is None:
            continue
        counts[folder] = counts.get(folder, 0) + 1
    if not counts:
        print(f"{label}: no local history files found")
        return []
    ordered = sorted(counts.items(), key=lambda item: (str(item[0]), item[1]))
    total = sum(counts.values())
    print(f"{label}: {len(ordered)} folder(s), {total} run(s)")
    for folder, count in ordered[:limit]:
        suffix = "s" if count != 1 else ""
        print(f"  - {folder} ({count} run{suffix})")
    if len(ordered) > limit:
        print(f"  ... {len(ordered) - limit} more folder(s)")
    return [folder for folder, _ in ordered]


def print_plot_source_folders(
    data: pd.DataFrame,
    *,
    label: str = "Plot source folders used to build curves",
) -> List[Path]:
    if data is None or data.empty:
        print(f"{label}: none")
        return []
    source_cols = [
        column
        for column in [
            "run_name",
            "run_id",
            "history_source_folder",
            "history_source_path",
            "run_data_dir",
            "run_dir",
            "history_parquet",
            "history_csv",
        ]
        if column in data.columns
    ]
    if not any(
        column in source_cols
        for column in [
            "history_source_folder",
            "history_source_path",
            "run_data_dir",
            "run_dir",
            "history_parquet",
            "history_csv",
        ]
    ):
        print(f"{label}: no local history source metadata found")
        return []
    return print_history_source_folders(data[source_cols].drop_duplicates(), label=label)


def _category_colors(labels: Sequence[Any]) -> Dict[str, str]:
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet
    return {str(label): palette[idx % len(palette)] for idx, label in enumerate(labels)}


def _seed_list(values: pd.Series) -> List[int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return sorted(numeric.astype(int).unique().tolist())


def aggregate_agent_seed_means(
    df: pd.DataFrame,
    *,
    source_col: str,
    value_col: str,
    agent_col: str = "agent",
    family_col: Optional[str] = None,
    seed_col: Optional[str] = None,
    extra_mean_cols: Sequence[str] = (),
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    data = df.copy()
    family_col = family_col or f"{source_col}_family"
    seed_col = seed_col or f"{source_col}_seed"
    if family_col not in data.columns or seed_col not in data.columns:
        data = add_seed_group_columns(
            data,
            source_col=source_col,
            family_col=family_col,
            seed_col=seed_col,
        )
    if family_col not in data.columns:
        data[family_col] = data[source_col].astype(str)
    if seed_col not in data.columns:
        data[seed_col] = np.nan

    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[value_col])
    if data.empty:
        return data

    group_cols = [family_col, agent_col]
    agg_spec: Dict[str, Any] = {
        "mean": (value_col, "mean"),
        "std": (value_col, "std"),
        "min": (value_col, "min"),
        "max": (value_col, "max"),
        "n_seeds": (source_col, "nunique"),
        "seeds": (seed_col, _seed_list),
        "members": (source_col, lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),
    }
    if "n" in data.columns:
        agg_spec["total_n"] = ("n", "sum")
        agg_spec["mean_n"] = ("n", "mean")
    for column in extra_mean_cols:
        if column in data.columns and column != value_col:
            data[column] = pd.to_numeric(data[column], errors="coerce")
            agg_spec[f"{column}_mean"] = (column, "mean")

    grouped = data.groupby(group_cols, dropna=False, observed=True, as_index=False).agg(**agg_spec)
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["plot_label"] = grouped[family_col].astype(str)
    return grouped.sort_values(["plot_label", agent_col]).reset_index(drop=True)


def plot_agent_seed_mean_bar(
    df: pd.DataFrame,
    *,
    source_col: str,
    value_col: str,
    title: str,
    y_title: Optional[str] = None,
    agent_col: str = "agent",
    family_col: Optional[str] = None,
    seed_col: Optional[str] = None,
    extra_mean_cols: Sequence[str] = (),
    save_name: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    height: int = 620,
    width: int = 1250,
    y_range: Optional[Sequence[float]] = None,
    add_zero_line: bool = False,
    x_tickangle: Optional[float] = None,
) -> Optional[go.Figure]:
    agg = aggregate_agent_seed_means(
        df,
        source_col=source_col,
        value_col=value_col,
        agent_col=agent_col,
        family_col=family_col,
        seed_col=seed_col,
        extra_mean_cols=extra_mean_cols,
    )
    if agg.empty:
        print(f"No data for {title}")
        return None

    run_order = agg["plot_label"].dropna().astype(str).drop_duplicates().tolist()
    agents = [agent for agent in AGENTS if agent in set(agg[agent_col].astype(str))]
    if not agents:
        agents = sorted(agg[agent_col].dropna().astype(str).unique().tolist())
    colors = _category_colors(run_order)
    patterns = {"agent_0": "", "agent_1": "/", "agent_2": "\\"}

    extra_hover_cols = [f"{column}_mean" for column in extra_mean_cols if f"{column}_mean" in agg.columns]
    fig = go.Figure()
    for agent in agents:
        agent_data = agg[agg[agent_col].astype(str) == str(agent)].copy()
        agent_data = pd.DataFrame({"plot_label": run_order}).merge(
            agent_data,
            on="plot_label",
            how="left",
        )
        custom_cols = ["std", "min", "max", "n_seeds", "seeds", "members", *extra_hover_cols]
        if "total_n" in agent_data.columns:
            custom_cols.extend(["total_n", "mean_n"])
        customdata = np.stack(
            [
                (
                    agent_data[column].fillna(0.0)
                    if pd.api.types.is_numeric_dtype(agent_data[column])
                    else agent_data[column].astype(str)
                )
                for column in custom_cols
            ],
            axis=-1,
        )
        hover_lines = [
            "run=%{x}",
            f"agent={agent}",
            f"{y_title or value_col}=%{{y:.4f}}",
            "std across seeds=%{customdata[0]:.4f}",
            "min=%{customdata[1]:.4f}",
            "max=%{customdata[2]:.4f}",
            "n seeds=%{customdata[3]}",
            "seeds=%{customdata[4]}",
            "members=%{customdata[5]}",
        ]
        for idx, column in enumerate(extra_hover_cols, start=6):
            hover_lines.append(f"{column}=%{{customdata[{idx}]:.4f}}")
        if "total_n" in agent_data.columns:
            total_idx = 6 + len(extra_hover_cols)
            hover_lines.append(f"total_n=%{{customdata[{total_idx}]:.0f}}")
            hover_lines.append(f"mean_n=%{{customdata[{total_idx + 1}]:.0f}}")
        fig.add_trace(
            go.Bar(
                x=agent_data["plot_label"],
                y=agent_data["mean"],
                name=str(agent),
                legendgroup=str(agent),
                marker={
                    "color": [colors[str(label)] for label in agent_data["plot_label"]],
                    "line": {"color": "rgba(0,0,0,0.35)", "width": 0.7},
                    "pattern": {"shape": patterns.get(str(agent), "")},
                },
                error_y={"type": "data", "array": agent_data["std"].fillna(0.0), "visible": True},
                customdata=customdata,
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            )
        )

    if add_zero_line:
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    fig.update_yaxes(title_text=y_title or value_col, range=list(y_range) if y_range else None)
    fig.update_xaxes(
        title_text=source_col.replace("_", " "),
        categoryorder="array",
        categoryarray=run_order,
        tickangle=x_tickangle,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        barmode="group",
        height=height,
        width=width,
        bargap=0.18,
        bargroupgap=0.04,
        legend={"title": {"text": "agent"}, "orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},
        margin={"l": 70, "r": 210, "t": 90, "b": 120},
    )
    if save_name:
        save_fig(fig, save_name, save=save)
    return show_or_return(fig, show=show)


def _as_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _seed_from_name(name: str) -> float:
    match = re.search(r"_s(\d+)$", str(name))
    return float(match.group(1)) if match else np.nan


def _family_from_name(name: str) -> str:
    return re.sub(r"_s\d+$", "", str(name))


def _family_seed_from_label(label: Any) -> Tuple[str, float]:
    text = str(label)
    name = Path(text).name
    stem = Path(name).stem if Path(name).suffix else name
    return _family_from_name(stem), _seed_from_name(stem)


def add_seed_group_columns(
    df: pd.DataFrame,
    *,
    source_col: str,
    family_col: str,
    seed_col: str,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty or source_col not in out.columns:
        return out
    pairs = out[source_col].map(_family_seed_from_label)
    out[family_col] = pairs.map(lambda item: item[0])
    out[seed_col] = pairs.map(lambda item: item[1])
    return out


def _read_toml(path: Path) -> Dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def resolve_config_folder(folder: str | Path) -> Path:
    path = Path(folder).expanduser()
    candidates = [
        path,
        CONFIG_ROOT / path,
        TASK_DIR / path,
        REPO_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve config folder: {folder}")


def _condition_from_config(stem: str, folder_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    family = _family_from_name(stem)
    seed = _seed_from_name(stem)
    gate = _as_bool(args.get("intervention_gate"))
    eval_heuristic = str(args.get("eval_action_heuristic", "none") or "none")
    intervention_penalty = _as_float(args.get("intervention_penalty"), 0.0)
    safe_intervention_penalty = _as_float(args.get("safe_intervention_penalty"), 0.0)
    adaptive_budget = bool(_as_bool(args.get("adaptive_intervention_budget")) or False)

    condition = family
    condition_order = 999.0
    experiment = folder_name
    design = "gate" if gate else "flat"
    mechanism = "baseline"

    sparse_match = re.search(r"(?:a0_)?sparse16_(flat|gated)_p(\d{3})$", family)
    if family in A0_HVG_LABELS:
        condition = A0_HVG_LABELS[family]
        condition_order = float(A0_HVG_ORDER[family])
        if eval_heuristic != "none":
            mechanism = f"eval heuristic: {eval_heuristic}"
        elif gate:
            mechanism = f"gate: {args.get('intervention_gate_eval_mode', 'unknown')}"
        else:
            mechanism = "baseline"
    elif sparse_match:
        design = sparse_match.group(1)
        penalty = int(sparse_match.group(2)) / 1000.0
        condition = f"sparse16 {design} p{penalty:.3f}"
        condition_order = 10.0 + (0.0 if design == "flat" else 10.0) + penalty
        mechanism = "fixed intervention penalty"
    elif family in AIB_LABELS:
        condition = AIB_LABELS[family]
        condition_order = 50.0 + float(AIB_ORDER[family])
        mechanism = "adaptive intervention budget"
    elif folder_name.startswith("a0_"):
        condition = family.replace("_", " ")

    if adaptive_budget:
        mechanism = "adaptive intervention budget"
    elif intervention_penalty > 0 or safe_intervention_penalty > 0:
        mechanism = "fixed intervention penalty"

    return {
        "run_name": stem,
        "expected_run_name": stem,
        "experiment": experiment,
        "family": family,
        "condition": condition,
        "condition_order": condition_order,
        "seed": seed,
        "design": design,
        "mechanism": mechanism,
        "intervention_gate": bool(gate),
        "intervention_gate_eval_mode": args.get("intervention_gate_eval_mode"),
        "eval_action_heuristic": eval_heuristic,
        "eval_action_rho_threshold": _as_float(args.get("eval_action_rho_threshold")),
        "intervention_penalty": intervention_penalty,
        "safe_intervention_penalty": safe_intervention_penalty,
        "safe_intervention_rho_threshold": _as_float(args.get("safe_intervention_rho_threshold")),
        "adaptive_intervention_budget": adaptive_budget,
        "intervention_budget_cost_mode": args.get("intervention_budget_cost_mode"),
        "intervention_budget_target": _as_float(args.get("intervention_budget_target")),
        "intervention_budget_lr": _as_float(args.get("intervention_budget_lr")),
        "intervention_budget_rho_threshold": _as_float(args.get("intervention_budget_rho_threshold")),
        "total_timesteps": _as_float(args.get("total_timesteps")),
        "eval_freq": _as_float(args.get("eval_freq")),
        "n_envs": _as_float(args.get("n_envs")),
        "n_steps": _as_float(args.get("n_steps")),
    }


def read_expected_configs(config_folders: Sequence[str | Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for folder in config_folders:
        folder_path = resolve_config_folder(folder)
        for path in sorted(folder_path.glob("*.toml")):
            cfg = _read_toml(path)
            args = cfg.get("args", {})
            row = _condition_from_config(path.stem, folder_path.name, args)
            row["config_path"] = str(path)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["experiment", "condition_order", "seed", "run_name"])


def load_cache_index(path: Path = CACHE_INDEX_PATH) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    if "name" in df.columns and "run_name" not in df.columns:
        df = df.rename(columns={"name": "run_name"})
    if "id" in df.columns and "run_id" not in df.columns:
        df = df.rename(columns={"id": "run_id"})
    scanned_cache = _scan_full_history_cache(FULL_HISTORY_DIR)
    scanned_run_data = scan_run_data(PERMANENT_RUN_DATA_DIR, include_legacy=False)
    if df.empty and scanned_cache.empty and scanned_run_data.empty:
        raise FileNotFoundError(
            f"Missing readable W&B histories. Expected permanent run data under "
            f"{PERMANENT_RUN_DATA_DIR}, or cache index {path}, or metadata under "
            f"{FULL_HISTORY_DIR}."
        )
    pieces = [piece for piece in [df, scanned_cache, scanned_run_data] if not piece.empty]
    if pieces:
        df = pd.concat(pieces, ignore_index=True, sort=False)
        df = df.drop_duplicates(subset=["run_name", "run_id"], keep="last")
    for column in ["history_parquet", "history_csv"]:
        if column not in df.columns:
            df[column] = None
        df[column] = df.apply(
            lambda row: _resolve_cached_history_path(
                row.get(column),
                row.get("run_name"),
                row.get("run_id"),
                "history.parquet" if column == "history_parquet" else "history.csv.gz",
            ),
            axis=1,
        )
    df["has_history"] = df.apply(
        lambda row: bool(row["history_parquet"] and row["history_parquet"].exists())
        or bool(row["history_csv"] and row["history_csv"].exists()),
        axis=1,
    )
    return df[df["has_history"]].reset_index(drop=True)


def _resolve_cached_history_path(
    value: Any,
    run_name: Any,
    run_id: Any,
    filename: str,
) -> Optional[Path]:
    if isinstance(value, Path):
        return value if value.exists() else None
    if isinstance(value, str) and value:
        path = Path(value)
        if path.exists():
            return path
        run_folder = FULL_HISTORY_DIR / f"{run_name}__{run_id}"
        candidate = run_folder / path.name
        if candidate.exists():
            return candidate
    if run_name and run_id:
        run_folder = FULL_HISTORY_DIR / f"{run_name}__{run_id}"
        candidate = run_folder / filename
        if candidate.exists():
            return candidate
    return Path(value) if isinstance(value, str) and value else None


def _scan_full_history_cache(root: Path = FULL_HISTORY_DIR) -> pd.DataFrame:
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for meta_path in sorted(root.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_name = meta.get("name") or meta.get("run_name")
        run_id = meta.get("id") or meta.get("run_id")
        if not run_name or not run_id:
            continue
        rows.append(
            {
                "run_name": run_name,
                "run_id": run_id,
                "artifact": meta.get("artifact_resolved")
                or meta.get("artifact_requested")
                or meta.get("artifact"),
                "history_parquet": _resolve_cached_history_path(
                    meta.get("history_parquet"),
                    run_name,
                    run_id,
                    "history.parquet",
                ),
                "history_csv": _resolve_cached_history_path(
                    meta.get("history_csv"),
                    run_name,
                    run_id,
                    "history.csv.gz",
                ),
                "rows": meta.get("rows"),
                "columns": meta.get("columns"),
                "state": meta.get("state"),
            }
        )
    return pd.DataFrame(rows)


def select_cached_runs(
    config_folders: Optional[Sequence[str | Path]] = None,
    exact_run_names: Optional[Sequence[str]] = None,
    regex: Optional[str] = None,
) -> pd.DataFrame:
    cache = load_cache_index()
    expected = (
        read_expected_configs(config_folders)
        if config_folders
        else pd.DataFrame()
    )
    selected_parts = []
    if not expected.empty:
        selected_parts.append(
            cache.merge(expected, on="run_name", how="inner", suffixes=("", "_cfg"))
        )
    if exact_run_names:
        names = set(map(str, exact_run_names))
        selected_parts.append(cache[cache["run_name"].astype(str).isin(names)].copy())
    if regex:
        pattern = re.compile(regex)
        selected_parts.append(cache[cache["run_name"].astype(str).map(lambda x: bool(pattern.search(x)))].copy())
    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True, sort=False)
        selected = selected.drop_duplicates(subset=["run_id", "run_name"])
    else:
        selected = cache.copy()

    if "condition" not in selected.columns:
        selected["family"] = selected["run_name"].map(_family_from_name)
        selected["condition"] = selected["family"].str.replace("_", " ", regex=False)
        selected["condition_order"] = 999.0
        selected["seed"] = selected["run_name"].map(_seed_from_name)
        selected["experiment"] = "selected"
        selected["design"] = "unknown"
        selected["mechanism"] = "unknown"
    return selected.sort_values(["experiment", "condition_order", "seed", "run_name"]).reset_index(drop=True)


def cache_coverage(config_folders: Sequence[str | Path]) -> pd.DataFrame:
    expected = read_expected_configs(config_folders)
    cache = load_cache_index()
    coverage = expected.merge(
        cache[["run_name", "run_id", "rows", "columns"]],
        on="run_name",
        how="left",
    )
    coverage["cached"] = coverage["run_id"].notna()
    return coverage.sort_values(["experiment", "condition_order", "seed", "run_name"])


def read_cached_history(row: pd.Series | Dict[str, Any]) -> pd.DataFrame:
    parquet_path = row.get("history_parquet")
    csv_path = row.get("history_csv")
    if parquet_path and Path(parquet_path).exists():
        try:
            history = pd.read_parquet(parquet_path)
        except Exception as exc:
            if not (csv_path and Path(csv_path).exists()):
                raise
            print(
                f"parquet read failed for {row.get('run_name')}: "
                f"{type(exc).__name__}; falling back to CSV"
            )
            history = pd.read_csv(csv_path)
    elif csv_path and Path(csv_path).exists():
        history = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No cached history for {row.get('run_name')}")
    history = history.copy()
    history["run_name"] = row["run_name"]
    history["run_id"] = row["run_id"]
    if "_step" not in history.columns:
        if "charts/global_step" in history.columns:
            history["_step"] = history["charts/global_step"]
        elif "global_step" in history.columns:
            history["_step"] = history["global_step"]
        elif "step" in history.columns:
            history["_step"] = history["step"]
        else:
            history["_step"] = np.arange(len(history), dtype=float)
    history["_step"] = pd.to_numeric(history["_step"], errors="coerce")
    history["step_millions"] = history["_step"] / 1_000_000.0
    return history.dropna(subset=["_step"])


def load_cached_histories(
    selected_runs: pd.DataFrame,
    *,
    step_max_m: Optional[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    frames = []
    rows = selected_runs.to_dict("records")
    for idx, row in enumerate(rows, start=1):
        if verbose:
            print(f"[{idx:>3}/{len(rows)}] loading {row['run_name']}", flush=True)
        try:
            history = read_cached_history(row)
            source_folder = _history_source_folder(row)
            source_path = (
                _path_or_none(row.get("history_parquet"))
                or _path_or_none(row.get("history_csv"))
            )
            if source_folder is not None:
                history["history_source_folder"] = str(source_folder)
            if source_path is not None:
                history["history_source_path"] = str(source_path)
            frames.append(history)
        except Exception as exc:
            print(f"skipped {row.get('run_name')}: {type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True, sort=False)
    if step_max_m is not None:
        history = history[history["step_millions"] <= float(step_max_m)].copy()

    meta_cols = [
        "run_name",
        "run_id",
        "experiment",
        "family",
        "condition",
        "condition_order",
        "seed",
        "design",
        "mechanism",
        "intervention_gate",
        "intervention_gate_eval_mode",
        "eval_action_heuristic",
        "eval_action_rho_threshold",
        "intervention_penalty",
        "safe_intervention_penalty",
        "adaptive_intervention_budget",
        "intervention_budget_cost_mode",
        "intervention_budget_target",
        "intervention_budget_lr",
        "intervention_budget_rho_threshold",
    ]
    for column in meta_cols:
        if column not in selected_runs.columns:
            selected_runs[column] = np.nan
    return history.merge(selected_runs[meta_cols], on=["run_name", "run_id"], how="left")


def _first_available(history: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in history.columns and history[column].notna().any():
            return column
    return None


def _coalesce_columns(history: pd.DataFrame, candidates: Sequence[str]) -> Tuple[pd.Series, Optional[str]]:
    out = pd.Series(np.nan, index=history.index, dtype=float)
    used = []
    for column in candidates:
        if column in history.columns:
            values = pd.to_numeric(history[column], errors="coerce")
            mask = out.isna() & values.notna()
            out.loc[mask] = values.loc[mask]
            if values.notna().any():
                used.append(column)
    return out, (", ".join(used) if used else None)


def survival_curve(
    history: pd.DataFrame,
    split: str = "test",
    *,
    smooth: int = 1,
) -> pd.DataFrame:
    candidates = SURVIVAL_CANDIDATES.get(split, [split])
    values, used = _coalesce_columns(history, candidates)
    source_cols = [
        column
        for column in ["history_source_folder", "history_source_path"]
        if column in history.columns
    ]
    out = history[
        [
            "run_name",
            "run_id",
            "experiment",
            "family",
            "condition",
            "condition_order",
            "seed",
            "design",
            "mechanism",
            "_step",
            "step_millions",
            *source_cols,
        ]
    ].copy()
    out["metric"] = f"{split}_survival"
    out["metric_column"] = used
    out["value"] = values
    out = out.dropna(subset=["value"])
    if smooth and smooth > 1 and not out.empty:
        out["value_smooth"] = out.groupby("run_id", dropna=False)["value"].transform(
            lambda s: s.rolling(int(smooth), min_periods=1).mean()
        )
    else:
        out["value_smooth"] = out["value"]
    return out


def agent_metric_curve(
    history: pd.DataFrame,
    *,
    metric: str,
    source: str = "eval",
    split: str = "test",
    smooth: int = 1,
) -> pd.DataFrame:
    rows = []
    for agent in AGENTS:
        if metric == "action0":
            candidates = (
                [f"{split}/explain/frac_action_0_{agent}"]
                if source == "eval"
                else [f"train/frac_action_0_{agent}"]
            )
            label = "action 0 fraction"
        elif metric == "nonidle":
            candidates = (
                [f"{split}/explain/action_nonidle_{agent}"]
                if source == "eval"
                else [f"train/explain/action_nonidle_{agent}", f"train/frac_action_0_{agent}"]
            )
            label = "non-idle fraction"
        elif metric == "illegal":
            candidates = [f"train/illegal_action_rate_{agent}"]
            label = "illegal action rate"
        elif metric == "gate_prob_intervene":
            candidates = [f"train/intervention_gate_prob_intervene_{agent}"]
            label = "gate P(intervene)"
        elif metric == "gate_actual_intervene":
            candidates = [
                f"train/intervention_gate_intervene_frac_{agent}",
                f"train/explain/gate_intervened_{agent}",
            ]
            label = "gate actual intervention fraction"
        elif metric == "gate_entropy":
            candidates = [f"train/intervention_gate_entropy_{agent}"]
            label = "gate entropy"
        else:
            raise ValueError(f"Unknown agent metric: {metric}")

        values, used = _coalesce_columns(history, candidates)
        if metric == "nonidle" and used == f"train/frac_action_0_{agent}":
            values = 1.0 - values
        source_cols = [
            column
            for column in ["history_source_folder", "history_source_path"]
            if column in history.columns
        ]
        df = history[
            [
                "run_name",
                "run_id",
                "experiment",
                "family",
                "condition",
                "condition_order",
                "seed",
                "design",
                "mechanism",
                "_step",
                "step_millions",
                *source_cols,
            ]
        ].copy()
        df["agent"] = agent
        df["metric"] = metric
        df["metric_label"] = label
        df["metric_column"] = used
        df["value"] = values
        rows.append(df.dropna(subset=["value"]))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if smooth and smooth > 1 and not out.empty:
        out["value_smooth"] = out.groupby(["run_id", "agent"], dropna=False)["value"].transform(
            lambda s: s.rolling(int(smooth), min_periods=1).mean()
        )
    else:
        out["value_smooth"] = out.get("value", pd.Series(dtype=float))
    return out


def heuristic_curve(
    history: pd.DataFrame,
    *,
    split: str = "test",
    smooth: int = 1,
) -> pd.DataFrame:
    specs = [
        ("force_noop_any_agent", [f"{split}/heuristic/force_noop_any_agent_frac", f"{split}/heuristic/force_noop_frac"], None),
        ("force_noop_all_agents", [f"{split}/heuristic/force_noop_all_agents_frac"], None),
    ]
    for agent in AGENTS:
        specs.extend(
            [
                (f"force_noop_{agent}", [f"{split}/heuristic/force_noop_frac_{agent}"], agent),
                (f"policy_nonidle_{agent}", [f"{split}/heuristic/policy_nonidle_{agent}"], agent),
                (f"blocked_nonidle_{agent}", [f"{split}/heuristic/blocked_nonidle_{agent}"], agent),
                (f"mean_pre_action_rho_{agent}", [f"{split}/heuristic/mean_pre_action_max_rho_{agent}"], agent),
            ]
        )
    rows = []
    source_cols = [
        column
        for column in ["history_source_folder", "history_source_path"]
        if column in history.columns
    ]
    for metric, candidates, agent in specs:
        values, used = _coalesce_columns(history, candidates)
        df = history[
            [
                "run_name",
                "run_id",
                "experiment",
                "family",
                "condition",
                "condition_order",
                "seed",
                "design",
                "mechanism",
                "_step",
                "step_millions",
                *source_cols,
            ]
        ].copy()
        df["agent"] = agent or "joint"
        df["metric"] = metric
        df["metric_column"] = used
        df["value"] = values
        rows.append(df.dropna(subset=["value"]))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if smooth and smooth > 1 and not out.empty:
        out["value_smooth"] = out.groupby(["run_id", "metric"], dropna=False)["value"].transform(
            lambda s: s.rolling(int(smooth), min_periods=1).mean()
        )
    else:
        out["value_smooth"] = out.get("value", pd.Series(dtype=float))
    return out


def joint_nonidle_curve(history: pd.DataFrame, *, smooth: int = 1) -> pd.DataFrame:
    rows = []
    source_cols = [
        column
        for column in ["history_source_folder", "history_source_path"]
        if column in history.columns
    ]
    for count in JOINT_COUNTS:
        col = f"train/non_idle_agents_count_{count}_frac"
        if col not in history.columns:
            continue
        df = history[
            [
                "run_name",
                "run_id",
                "experiment",
                "family",
                "condition",
                "condition_order",
                "seed",
                "design",
                "mechanism",
                "_step",
                "step_millions",
                *source_cols,
            ]
        ].copy()
        df["non_idle_agents"] = count
        df["metric"] = col
        df["value"] = pd.to_numeric(history[col], errors="coerce")
        rows.append(df.dropna(subset=["value"]))
    for metric in ["train/frac_any_non_idle", "train/frac_multi_agent_non_idle", "train/non_idle_agents_mean"]:
        if metric not in history.columns:
            continue
        df = history[
            [
                "run_name",
                "run_id",
                "experiment",
                "family",
                "condition",
                "condition_order",
                "seed",
                "design",
                "mechanism",
                "_step",
                "step_millions",
                *source_cols,
            ]
        ].copy()
        df["non_idle_agents"] = metric
        df["metric"] = metric
        df["value"] = pd.to_numeric(history[metric], errors="coerce")
        rows.append(df.dropna(subset=["value"]))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if smooth and smooth > 1 and not out.empty:
        out["value_smooth"] = out.groupby(["run_id", "metric"], dropna=False)["value"].transform(
            lambda s: s.rolling(int(smooth), min_periods=1).mean()
        )
    else:
        out["value_smooth"] = out.get("value", pd.Series(dtype=float))
    return out


def aib_curve(history: pd.DataFrame, *, smooth: int = 1) -> pd.DataFrame:
    specs = [
        ("lambda_mean", "train/intervention_budget_lambda_mean", None),
        ("cost_mean", "train/intervention_budget_cost_mean", None),
        ("cost_violation_mean", "train/intervention_budget_cost_violation_mean", None),
        ("penalty_mean", "train/intervention_budget_penalty_mean", None),
        ("safety_weight_mean", "train/intervention_budget_safety_weight_mean", None),
    ]
    for agent in AGENTS:
        specs.extend(
            [
                ("lambda", f"train/intervention_budget_lambda_{agent}", agent),
                ("cost", f"train/intervention_budget_cost_{agent}", agent),
                ("cost_violation", f"train/intervention_budget_cost_violation_{agent}", agent),
                ("penalty_mean", f"train/intervention_budget_penalty_mean_{agent}", agent),
                ("safety_weight", f"train/intervention_budget_safety_weight_{agent}", agent),
                ("rate_when_budget_costly", f"train/intervention_rate_when_budget_costly_{agent}", agent),
                ("rate_when_budget_free", f"train/intervention_rate_when_budget_free_{agent}", agent),
            ]
        )
    rows = []
    source_cols = [
        column
        for column in ["history_source_folder", "history_source_path"]
        if column in history.columns
    ]
    for metric, col, agent in specs:
        if col not in history.columns:
            continue
        df = history[
            [
                "run_name",
                "run_id",
                "experiment",
                "family",
                "condition",
                "condition_order",
                "seed",
                "design",
                "mechanism",
                "_step",
                "step_millions",
                *source_cols,
            ]
        ].copy()
        df["agent"] = agent or "joint"
        df["metric"] = metric
        df["metric_column"] = col
        df["value"] = pd.to_numeric(history[col], errors="coerce")
        rows.append(df.dropna(subset=["value"]))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if smooth and smooth > 1 and not out.empty:
        out["value_smooth"] = out.groupby(["run_id", "agent", "metric"], dropna=False)["value"].transform(
            lambda s: s.rolling(int(smooth), min_periods=1).mean()
        )
    else:
        out["value_smooth"] = out.get("value", pd.Series(dtype=float))
    return out


def sparse_penalty_curve(history: pd.DataFrame, *, smooth: int = 1) -> pd.DataFrame:
    specs = [
        ("penalty_mean", "train/intervention_penalty_mean", None),
        ("penalty_total", "train/intervention_penalty_total", None),
        ("safe_state_frac", "train/safe_state_frac", None),
    ]
    for agent in AGENTS:
        specs.extend(
            [
                ("penalty_mean", f"train/intervention_penalty_mean_{agent}", agent),
                ("rate_when_safe", f"train/intervention_rate_when_safe_{agent}", agent),
                ("rate_when_hazard", f"train/intervention_rate_when_hazard_{agent}", agent),
            ]
        )
    rows = []
    source_cols = [
        column
        for column in ["history_source_folder", "history_source_path"]
        if column in history.columns
    ]
    for metric, col, agent in specs:
        if col not in history.columns:
            continue
        df = history[
            [
                "run_name",
                "run_id",
                "experiment",
                "family",
                "condition",
                "condition_order",
                "seed",
                "design",
                "mechanism",
                "_step",
                "step_millions",
                *source_cols,
            ]
        ].copy()
        df["agent"] = agent or "joint"
        df["metric"] = metric
        df["metric_column"] = col
        df["value"] = pd.to_numeric(history[col], errors="coerce")
        rows.append(df.dropna(subset=["value"]))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if smooth and smooth > 1 and not out.empty:
        out["value_smooth"] = out.groupby(["run_id", "agent", "metric"], dropna=False)["value"].transform(
            lambda s: s.rolling(int(smooth), min_periods=1).mean()
        )
    else:
        out["value_smooth"] = out.get("value", pd.Series(dtype=float))
    return out


def last_window(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    last_n: int = 5,
    group_cols: Sequence[str] = ("run_id",),
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = (
        df.sort_values([*group_cols, "_step"])
        .groupby(list(group_cols), dropna=False, observed=True)
        .tail(int(last_n))
        .copy()
    )
    return out


def per_run_final(
    df: pd.DataFrame,
    *,
    last_n: int = 5,
    group_cols: Sequence[str] = ("run_id",),
    value_col: str = "value",
) -> pd.DataFrame:
    window = last_window(df, last_n=last_n, group_cols=group_cols, value_col=value_col)
    if window.empty:
        return window
    id_cols = [
        "run_name",
        "run_id",
        "experiment",
        "family",
        "condition",
        "condition_order",
        "seed",
        "design",
        "mechanism",
        "history_source_folder",
        "history_source_path",
    ]
    extra = [c for c in ["agent", "metric", "metric_label", "non_idle_agents"] if c in window.columns]
    by = [c for c in [*id_cols, *extra] if c in window.columns]
    return (
        window.groupby(by, dropna=False, observed=True, as_index=False)
        .agg(
            final_value=(value_col, "mean"),
            final_step_millions=("step_millions", "max"),
            n_logged_points=(value_col, "count"),
        )
        .sort_values(["condition_order", "seed", *extra])
    )


def seed_aggregate(
    per_run: pd.DataFrame,
    *,
    value_col: str = "final_value",
    extra_group_cols: Sequence[str] = (),
) -> pd.DataFrame:
    if per_run.empty:
        return per_run.copy()
    group_cols = [
        "experiment",
        "family",
        "condition",
        "condition_order",
        "design",
        "mechanism",
        *extra_group_cols,
    ]
    group_cols = [c for c in group_cols if c in per_run.columns]
    return (
        per_run.groupby(group_cols, dropna=False, observed=True, as_index=False)
        .agg(
            mean=(value_col, "mean"),
            std=(value_col, "std"),
            min=(value_col, "min"),
            max=(value_col, "max"),
            n_seeds=("run_id", "nunique"),
            seeds=("seed", lambda s: sorted(pd.Series(s).dropna().astype(int).unique().tolist())),
            runs=("run_name", lambda s: sorted(pd.Series(s).dropna().astype(str).unique().tolist())),
        )
        .assign(std=lambda d: d["std"].fillna(0.0))
        .sort_values(["condition_order", *list(extra_group_cols)])
    )


def _aggregate_timeseries(
    df: pd.DataFrame,
    *,
    value_col: str = "value_smooth",
    extra_group_cols: Sequence[str] = (),
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    group_cols = [
        "experiment",
        "family",
        "condition",
        "condition_order",
        "design",
        "mechanism",
        *extra_group_cols,
        "_step",
        "step_millions",
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    return (
        df.groupby(group_cols, dropna=False, observed=True, as_index=False)
        .agg(
            mean=(value_col, "mean"),
            std=(value_col, "std"),
            n_seeds=("run_id", "nunique"),
            seeds=("seed", lambda s: sorted(pd.Series(s).dropna().astype(int).unique().tolist())),
        )
        .assign(std=lambda d: d["std"].fillna(0.0))
        .sort_values(["condition_order", *list(extra_group_cols), "step_millions"])
    )


def plot_timeseries(
    df: pd.DataFrame,
    *,
    title: str,
    y_title: str,
    color_col: str = "condition",
    facet_col: Optional[str] = None,
    extra_group_cols: Sequence[str] = (),
    save_name: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    height: int = 620,
) -> Optional[go.Figure]:
    if df.empty:
        print(f"No data for {title}")
        return None
    agg = _aggregate_timeseries(df, extra_group_cols=extra_group_cols)
    if facet_col:
        fig = px.line(
            agg,
            x="step_millions",
            y="mean",
            color=color_col,
            facet_row=facet_col,
            hover_data=["std", "n_seeds", "seeds", "design", "mechanism"],
            labels={"step_millions": "steps (M)", "mean": y_title, color_col: "condition"},
            title=title,
        )
        fig.update_yaxes(matches=None)
    else:
        fig = px.line(
            agg,
            x="step_millions",
            y="mean",
            color=color_col,
            hover_data=["std", "n_seeds", "seeds", "design", "mechanism"],
            labels={"step_millions": "steps (M)", "mean": y_title, color_col: "condition"},
            title=title,
        )
    fig.update_layout(template="plotly_white", height=height, width=1450, hovermode="x unified")
    fig.update_traces(line={"width": 2.6})
    print_plot_source_folders(df)
    if save_name:
        save_fig(fig, save_name, save=save)
    return show_or_return(fig, show=show)


def plot_final_bar(
    per_run: pd.DataFrame,
    *,
    title: str,
    y_title: str,
    extra_group_cols: Sequence[str] = (),
    x_col: str = "condition",
    color_col: Optional[str] = None,
    save_name: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    height: int = 620,
) -> Optional[go.Figure]:
    if per_run.empty:
        print(f"No data for {title}")
        return None
    agg = seed_aggregate(per_run, extra_group_cols=extra_group_cols)
    if color_col is None and extra_group_cols:
        color_col = extra_group_cols[0]
    fig = px.bar(
        agg,
        x=x_col,
        y="mean",
        color=color_col,
        error_y="std",
        barmode="group",
        hover_data=["std", "min", "max", "n_seeds", "seeds", "runs"],
        labels={x_col: "condition", "mean": y_title},
        title=title,
    )
    # Add seed points with a light jitter so the mean does not hide instability.
    seed_points = per_run.copy()
    fig.add_trace(
        go.Scatter(
            x=seed_points[x_col],
            y=seed_points["final_value"],
            mode="markers",
            marker={"size": 7, "color": "rgba(35,35,35,0.55)"},
            name="seed runs",
            showlegend=True,
            customdata=np.stack(
                [
                    seed_points["run_name"].astype(str),
                    seed_points["seed"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "run=%{customdata[0]}<br>"
                "seed=%{customdata[1]}<br>"
                "value=%{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(template="plotly_white", height=height, width=1450)
    fig.update_xaxes(categoryorder="array", categoryarray=agg.sort_values("condition_order")[x_col].astype(str).unique())
    print_plot_source_folders(per_run)
    if save_name:
        save_fig(fig, save_name, save=save)
    return show_or_return(fig, show=show)


def action_survival_tradeoff(
    history: pd.DataFrame,
    *,
    action_source: str = "eval",
    split: str = "test",
    last_n: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    survival = per_run_final(survival_curve(history, split=split), last_n=last_n)
    action0 = per_run_final(
        agent_metric_curve(history, metric="action0", source=action_source, split=split),
        last_n=last_n,
        group_cols=("run_id", "agent"),
    )
    if survival.empty or action0.empty:
        return pd.DataFrame(), pd.DataFrame()
    action0_mean = (
        action0.groupby(
            ["run_id", "run_name", "experiment", "family", "condition", "condition_order", "seed", "design", "mechanism"],
            dropna=False,
            observed=True,
            as_index=False,
        )
        .agg(mean_action0=("final_value", "mean"))
    )
    merged = survival.merge(
        action0_mean,
        on=["run_id", "run_name", "experiment", "family", "condition", "condition_order", "seed", "design", "mechanism"],
        how="inner",
    ).rename(columns={"final_value": "survival"})
    agg = (
        merged.groupby(["experiment", "family", "condition", "condition_order", "design", "mechanism"], dropna=False, observed=True, as_index=False)
        .agg(
            survival_mean=("survival", "mean"),
            survival_std=("survival", "std"),
            action0_mean=("mean_action0", "mean"),
            action0_std=("mean_action0", "std"),
            n_seeds=("run_id", "nunique"),
            seeds=("seed", lambda s: sorted(pd.Series(s).dropna().astype(int).unique().tolist())),
            runs=("run_name", lambda s: sorted(pd.Series(s).dropna().astype(str).unique().tolist())),
        )
        .fillna({"survival_std": 0.0, "action0_std": 0.0})
        .sort_values("condition_order")
    )
    return merged, agg


def plot_action_survival_tradeoff(
    history: pd.DataFrame,
    *,
    action_source: str = "eval",
    split: str = "test",
    last_n: int = 5,
    save_name: Optional[str] = None,
    save: bool = True,
    show: bool = True,
) -> Optional[go.Figure]:
    per_run, agg = action_survival_tradeoff(
        history, action_source=action_source, split=split, last_n=last_n
    )
    if agg.empty:
        print("No action/survival tradeoff data.")
        return None
    fig = px.scatter(
        agg,
        x="action0_mean",
        y="survival_mean",
        color="condition",
        size="n_seeds",
        error_x="action0_std",
        error_y="survival_std",
        hover_data=["design", "mechanism", "seeds", "runs"],
        labels={
            "action0_mean": f"mean action 0 fraction ({action_source})",
            "survival_mean": f"mean {split} survival",
        },
        title=f"Action sparsity vs survival, last {last_n} logged points",
    )
    fig.add_trace(
        go.Scatter(
            x=per_run["mean_action0"],
            y=per_run["survival"],
            mode="markers",
            marker={"size": 7, "color": "rgba(40,40,40,0.45)"},
            name="seed runs",
            customdata=np.stack([per_run["run_name"].astype(str), per_run["seed"].astype(str)], axis=-1),
            hovertemplate="run=%{customdata[0]}<br>seed=%{customdata[1]}<br>action0=%{x:.4f}<br>survival=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(template="plotly_white", height=650, width=1150)
    print_plot_source_folders(per_run)
    if save_name:
        save_fig(fig, save_name, save=save)
    return show_or_return(fig, show=show)


def metric_availability(history: pd.DataFrame) -> pd.DataFrame:
    patterns = {
        "test_survival": r"^test/(charts/)?episodic_survival$",
        "train_eval_survival": r"^train_eval/(charts/)?episodic_survival$",
        "train_action0": r"^train/frac_action_0_agent_\d+$",
        "eval_action0": r"^(test|train_eval|eval)/explain/frac_action_0_agent_\d+$",
        "heuristic_override": r"^(test|train_eval|eval)/heuristic/",
        "joint_nonidle": r"^train/non_idle_agents_",
        "illegal": r"^train/illegal_action_rate_agent_\d+$",
        "gate": r"^train/intervention_gate_",
        "aib": r"^train/intervention_budget_",
    }
    rows = []
    for run_id, run in history.groupby("run_id", dropna=False):
        cols = [c for c in run.columns if run[c].notna().any()]
        row = {
            "run_name": run["run_name"].iloc[0],
            "run_id": run_id,
            "condition": run["condition"].iloc[0],
            "seed": run["seed"].iloc[0],
            "history_rows": len(run),
            "max_step_m": run["step_millions"].max(),
        }
        for name, pattern in patterns.items():
            rx = re.compile(pattern)
            row[name] = sum(bool(rx.search(str(c))) for c in cols)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition", "seed", "run_name"])


def load_full_test_results(path: Path = FULL_TEST_DIR) -> pd.DataFrame:
    path = Path(path)
    rows = []
    for json_path in sorted(path.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"path": str(json_path), "bad": True, "error": repr(exc)})
            continue
        checkpoint = Path(str(data.get("checkpoint", "")))
        stem = checkpoint.stem
        rows.append(
            {
                "path": str(json_path),
                "bad": False,
                "checkpoint": str(checkpoint),
                "checkpoint_stem": stem,
                "run_like": re.sub(r"^(best_test_|final_)", "", stem),
                "checkpoint_global_step": data.get("checkpoint_global_step"),
                "split": data.get("split"),
                "eval_episodes": data.get("eval_episodes"),
                "eval_action_heuristic": data.get("eval_action_heuristic", "none"),
                "eval_action_rho_threshold": data.get("eval_action_rho_threshold"),
                "obs_normalization": data.get("obs_normalization"),
                "survival_frac": data.get("survival_frac"),
                "survival_percent": data.get("survival_percent"),
                "intervention_gate": data.get("intervention_gate"),
                "intervention_gate_eval_mode": data.get("intervention_gate_eval_mode"),
            }
        )
    return pd.DataFrame(rows)


def plot_full_test_results(results: pd.DataFrame, *, show: bool = True, save: bool = True) -> Optional[go.Figure]:
    if results.empty or "survival_percent" not in results.columns:
        print("No full-test result JSONs found.")
        return None
    data = results[~results.get("bad", False)].copy()
    if data.empty:
        print("No readable full-test result JSONs found.")
        return None
    fig = px.bar(
        data.sort_values("survival_percent", ascending=False),
        x="run_like",
        y="survival_percent",
        color="eval_action_heuristic",
        hover_data=[
            "checkpoint_stem",
            "checkpoint_global_step",
            "split",
            "eval_episodes",
            "obs_normalization",
            "eval_action_rho_threshold",
        ],
        labels={"run_like": "checkpoint/run", "survival_percent": "full-test survival (%)"},
        title="Standalone full-test evaluation summaries",
    )
    fig.update_layout(template="plotly_white", height=650, width=1450)
    fig.update_xaxes(tickangle=-35)
    save_fig(fig, "full_test_eval_survival_summary", save=save)
    return show_or_return(fig, show=show)


def _dataset_agent_rows(summary: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
    rows = []
    agent_summary = summary.get("agent_summary") or summary.get("agent_summaries") or {}
    for agent, metrics in agent_summary.items():
        rows.append(
            {
                "dataset": dataset_name,
                "agent": agent,
                "n": metrics.get("n"),
                "teacher_action0_frac": metrics.get("teacher_action_0_frac", metrics.get("teacher_action0_frac")),
                "teacher_nonidle_frac": metrics.get("teacher_nonidle_frac"),
                "policy_nonidle_frac": metrics.get("policy_nonidle_frac"),
                "force_noop_frac": metrics.get("force_noop_frac"),
                "was_overwritten_frac": metrics.get("was_overwritten_frac"),
                "overwrite_to_action0_frac": metrics.get("overwrite_to_action0_frac"),
                "obs_nan_or_inf": metrics.get("obs_nan_or_inf"),
            }
        )
    return rows


def load_teacher_dataset_summaries(root: Path = TEACHER_DATASET_ROOT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    dataset_rows = []
    agent_rows = []
    if not root.exists():
        return pd.DataFrame(), pd.DataFrame()
    metadata_paths = list(root.glob("metadata_exports/*_metadata.json"))
    metadata_paths.extend(root.glob("*/metadata/metadata.json"))
    metadata_paths.extend(path for path in root.glob("*/metadata.json"))
    seen = set()
    for metadata_path in sorted(metadata_paths):
        metadata_path = metadata_path.resolve()
        if metadata_path in seen:
            continue
        seen.add(metadata_path)
        if metadata_path.parent.name == "metadata_exports":
            dataset_name = re.sub(r"_metadata\.json$", "", metadata_path.name)
            dataset_dir = root / dataset_name
        elif metadata_path.parent.name == "metadata":
            dataset_dir = metadata_path.parent.parent
            dataset_name = dataset_dir.name
        else:
            dataset_dir = metadata_path.parent
            dataset_name = dataset_dir.name
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            dataset_rows.append({"dataset": dataset_name, "bad": True, "error": repr(exc)})
            continue
        shard_count = 0
        shard_dir = dataset_dir / "shards"
        if shard_dir.exists():
            shard_count = len(list(shard_dir.glob("shard_*.npz")))
        elif dataset_dir.exists():
            shard_count = len(list(dataset_dir.glob("shard_*.npz")))
        else:
            shard_count = metadata.get("n_shards")
        dataset_rows.append(
            {
                "dataset": dataset_name,
                "bad": False,
                "path": str(dataset_dir),
                "metadata_path": str(metadata_path),
                "status": metadata.get("status"),
                "checkpoint": metadata.get("checkpoint"),
                "checkpoint_global_step": metadata.get("checkpoint_global_step"),
                "split": metadata.get("split"),
                "eval_action_heuristic": metadata.get("eval_action_heuristic"),
                "eval_action_rho_threshold": metadata.get("eval_action_rho_threshold"),
                "n_env_steps": metadata.get("n_env_steps"),
                "n_agent_examples": metadata.get("n_agent_examples"),
                "n_completed_episodes": metadata.get("n_completed_episodes"),
                "n_unique_chronic_fingerprints": metadata.get("n_unique_chronic_fingerprints"),
                "n_shards": shard_count,
            }
        )
        agent_rows.extend(_dataset_agent_rows(metadata, dataset_name))
    dataset_df = add_seed_group_columns(
        pd.DataFrame(dataset_rows),
        source_col="dataset",
        family_col="dataset_family",
        seed_col="dataset_seed",
    )
    agent_df = add_seed_group_columns(
        pd.DataFrame(agent_rows),
        source_col="dataset",
        family_col="dataset_family",
        seed_col="dataset_seed",
    )
    return dataset_df, agent_df


def aggregate_teacher_dataset_agents(agent_df: pd.DataFrame) -> pd.DataFrame:
    if agent_df.empty:
        return agent_df.copy()
    data = add_seed_group_columns(
        agent_df,
        source_col="dataset",
        family_col="dataset_family",
        seed_col="dataset_seed",
    )
    metrics = [
        "teacher_action0_frac",
        "teacher_nonidle_frac",
        "policy_nonidle_frac",
        "force_noop_frac",
        "was_overwritten_frac",
        "overwrite_to_action0_frac",
        "obs_nan_or_inf",
    ]
    agg_spec: Dict[str, Any] = {
        "n_seed_datasets": ("dataset", "nunique"),
        "datasets": ("dataset", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),
        "seeds": ("dataset_seed", _seed_list),
    }
    if "n" in data.columns:
        agg_spec["total_n"] = ("n", "sum")
        agg_spec["mean_n"] = ("n", "mean")
    for metric in metrics:
        if metric in data.columns:
            data[metric] = pd.to_numeric(data[metric], errors="coerce")
            agg_spec[metric] = (metric, "mean")
            agg_spec[f"{metric}_std"] = (metric, "std")
    out = data.groupby(["dataset_family", "agent"], dropna=False, observed=True, as_index=False).agg(**agg_spec)
    std_cols = [col for col in out.columns if col.endswith("_std")]
    if std_cols:
        out[std_cols] = out[std_cols].fillna(0.0)
    return out.sort_values(["dataset_family", "agent"]).reset_index(drop=True)


def plot_teacher_dataset_balance(agent_df: pd.DataFrame, *, show: bool = True, save: bool = True) -> List[go.Figure]:
    figs = []
    if agent_df.empty:
        print("No teacher dataset agent summaries found.")
        return figs
    for metric, title in [
        ("teacher_action0_frac", "Teacher dataset action-0 fraction"),
        ("teacher_nonidle_frac", "Teacher dataset non-idle fraction"),
        ("was_overwritten_frac", "Teacher heuristic overwrite fraction"),
        ("force_noop_frac", "Teacher force-noop fraction"),
    ]:
        if metric not in agent_df.columns:
            continue
        fig = plot_agent_seed_mean_bar(
            agent_df,
            source_col="dataset",
            family_col="dataset_family",
            seed_col="dataset_seed",
            value_col=metric,
            title=f"{title}: seed mean by dataset family and agent",
            y_title=metric,
            extra_mean_cols=["policy_nonidle_frac", "overwrite_to_action0_frac"],
            save_name=f"teacher_dataset_{metric}",
            save=save,
            show=show,
            height=560,
            width=1250,
            y_range=[0, 1],
            x_tickangle=-30,
        )
        if fig is not None:
            figs.append(fig)
    return figs


def _load_torch_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import torch
    except ImportError:
        print("torch is not installed; cannot inspect student checkpoints.")
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_teacher_student_checkpoints(
    checkpoint_dir: Path = TEACHER_STUDENT_CHECKPOINT_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_rows = []
    epoch_rows = []
    if not checkpoint_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    for path in sorted(checkpoint_dir.glob("*.tar")):
        record = _load_torch_checkpoint(path)
        if record is None:
            continue
        ts = record.get("teacher_student", {}) or record.get("training_state", {}).get("teacher_student", {})
        args = ts.get("args", {}) if isinstance(ts, dict) else {}
        dataset_meta = ts.get("dataset_metadata", {}) if isinstance(ts, dict) else {}
        ckpt_rows.append(
            {
                "checkpoint": path.name,
                "path": str(path),
                "global_step": record.get("global_step"),
                "dataset": Path(str(ts.get("dataset", ""))).name if isinstance(ts, dict) else None,
                "teacher_checkpoint": Path(str(ts.get("teacher_checkpoint", ""))).name if isinstance(ts, dict) else None,
                "teacher_checkpoint_global_step": ts.get("teacher_checkpoint_global_step") if isinstance(ts, dict) else None,
                "epochs": args.get("epochs"),
                "batch_size": args.get("batch_size"),
                "balanced_nonidle_frac": args.get("balanced_nonidle_frac"),
                "nonidle_weight": args.get("nonidle_weight"),
                "aux_intervention_loss": args.get("aux_intervention_loss"),
                "aux_weight": args.get("aux_weight"),
                "soft_distillation_loss": args.get("soft_distillation_loss"),
                "soft_weight": args.get("soft_weight"),
                "soft_temperature": args.get("soft_temperature"),
                "student_training_objective": ts.get("student_training_objective"),
                "dataset_n_env_steps": dataset_meta.get("n_env_steps"),
                "dataset_n_agent_examples": dataset_meta.get("n_agent_examples"),
            }
        )
        for epoch_record in ts.get("history", []) if isinstance(ts, dict) else []:
            for phase in ["train", "eval"]:
                for agent, metrics in epoch_record.get(phase, {}).items():
                    row = {
                        "checkpoint": path.name,
                        "dataset": Path(str(ts.get("dataset", ""))).name,
                        "epoch": epoch_record.get("epoch"),
                        "optimizer_steps": epoch_record.get("optimizer_steps"),
                        "phase": phase,
                        "agent": agent,
                    }
                    row.update(metrics)
                    epoch_rows.append(row)
    checkpoint_df = add_seed_group_columns(
        pd.DataFrame(ckpt_rows),
        source_col="checkpoint",
        family_col="checkpoint_family",
        seed_col="checkpoint_seed",
    )
    if "dataset" in checkpoint_df.columns:
        checkpoint_df = add_seed_group_columns(
            checkpoint_df,
            source_col="dataset",
            family_col="dataset_family",
            seed_col="dataset_seed",
        )
    epoch_df = add_seed_group_columns(
        pd.DataFrame(epoch_rows),
        source_col="checkpoint",
        family_col="checkpoint_family",
        seed_col="checkpoint_seed",
    )
    if "dataset" in epoch_df.columns:
        epoch_df = add_seed_group_columns(
            epoch_df,
            source_col="dataset",
            family_col="dataset_family",
            seed_col="dataset_seed",
        )
    return checkpoint_df, epoch_df


def plot_student_bc_metrics(epoch_df: pd.DataFrame, *, show: bool = True, save: bool = True) -> List[go.Figure]:
    figs = []
    if epoch_df.empty:
        print("No teacher-student BC epoch metrics found.")
        return figs
    eval_df = epoch_df[epoch_df["phase"].eq("eval")].copy()
    for metric in ["accuracy", "pred_nonidle_frac", "false_noop_rate", "false_intervention_rate"]:
        if metric not in eval_df.columns:
            continue
        fig = px.line(
            eval_df,
            x="epoch",
            y=metric,
            color="checkpoint",
            facet_row="agent",
            markers=True,
            hover_data=["dataset", "optimizer_steps"],
            title=f"Student BC eval {metric}",
        )
        fig.update_yaxes(matches=None)
        fig.update_layout(template="plotly_white", height=820, width=1400)
        save_fig(fig, f"student_bc_eval_{metric}", save=save)
        show_or_return(fig, show=show)
        figs.append(fig)
    return figs
