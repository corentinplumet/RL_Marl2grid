"""Action-distribution analysis helpers extracted from action_distribution_plots.ipynb.

The original notebook is intentionally left intact. This module keeps the same metric
logic behind reusable functions so the notebook in this folder can stay small.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:
    raise ImportError("Install plotly first, for example: pip install plotly") from exc

try:
    from IPython.display import display
except Exception:  # pragma: no cover - useful outside notebooks
    def display(value):
        print(value)

pd.set_option("display.max_columns", 180)
pd.set_option("display.max_rows", 180)


def _seed_list(values):
    return sorted(pd.Series(values).dropna().astype(int).unique().tolist())


def find_task_dir(start=None):
    """Locate the Topology_Task directory from a notebook, repo root, or subfolder."""
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "main.py").is_file():
            return candidate
        task_dir = candidate / "Topology_Task"
        if (task_dir / "main.py").is_file():
            return task_dir
    raise RuntimeError("Could not locate Topology_Task/main.py from the current working directory.")


TASK_DIR = find_task_dir()
CONFIG_ROOT = TASK_DIR / "configs"
DEFAULT_EXPERIMENT_FOLDERS = {
    "intervention_gate_15": CONFIG_ROOT / "intervention_gate_15",
    "phase4_sparse_control_16": CONFIG_ROOT / "phase4_sparse_control_16",
    "heuristic_vs_gate_s0_s1_s2": CONFIG_ROOT / "heuristic_vs_gate_s0_s1_s2",
    "adaptive_intervention_budget_7": CONFIG_ROOT / "adaptive_intervention_budget_7",
}
CACHE_DIR = TASK_DIR / "outputs" / "wandb_cache"
CACHE_INDEX_PATH = CACHE_DIR / "full_history_cache_index.csv"
FIG_DIR = TASK_DIR / "outputs" / "action_distribution_figures"
TRACE_CACHE_DIR = CACHE_DIR / "action_trace_tables"
for directory in [FIG_DIR, TRACE_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

AIB_EXPERIMENT = "adaptive_intervention_budget_7"
AIB_TITLE = "Adaptive intervention budget 7"
AIB_FAMILY_LABELS = {
    "aib_00_flat_local_t020": "AIB flat local target 0.20",
    "aib_01_flat_local_t010": "AIB flat local target 0.10",
    "aib_02_flat_local_t035": "AIB flat local target 0.35",
    "aib_03_gate_hgreedy_sep_local_t020": "AIB gate h-greedy target 0.20",
    "aib_04_flat_nonidle_t020": "AIB flat non-idle target 0.20",
}
AIB_FAMILY_ORDER = {
    "aib_00_flat_local_t020": 0,
    "aib_01_flat_local_t010": 1,
    "aib_02_flat_local_t035": 2,
    "aib_03_gate_hgreedy_sep_local_t020": 3,
    "aib_04_flat_nonidle_t020": 4,
}

_CELL_CONFIG = 'FINAL_WINDOW_STEPS = 10\nSMOOTH_WINDOW = 5\nPLOT_STEP_MAX_M = None\nSAVE_FIGURES = True\nSHOW_FIGURES = True\n\n# Leave as None to use the first available comparison. After running the comparison-spec cell,\n# set this to one of COMPARISON_SPECS["comparison_title"].\nSELECTED_COMPARISON = None\n\n# Optional exact action-id traces. Only useful for runs launched with trace_rollout_actions=true.\nFETCH_TRACE_TABLES_FROM_WANDB = False\nENTITY = os.getenv("WANDB_ENTITY", "corentin-plumet-epfl")\nPROJECT = os.getenv("WANDB_PROJECT", "Grid2Op")\nWANDB_API_TIMEOUT = 300\nTRACE_TOP_K_ACTION_IDS = 10\nTRACE_TABLE_KEYS = (\n    "train/rollout_action_trace",\n    "train/rollout_action_trace_table",\n    "test/rollout_action_trace",\n    "test/rollout_action_trace_table",\n    "eval/rollout_action_trace",\n    "eval/rollout_action_trace_table",\n)\n\nIG15_LABELS = {\n    "ig_00_phase2_base": "IG15 p000 constant entropy",\n    "ig_01_entropy_decay": "IG15 p000 entropy decay",\n    "ig_02_topo001_entropy_decay": "IG15 topo 0.001 + entropy decay",\n    "ig_03_topo005_entropy_decay": "IG15 topo 0.005 + entropy decay",\n    "ig_04_topo010_entropy_decay": "IG15 topo 0.010 + entropy decay",\n}\nSPARSE16_PENALTY_LABELS = {\n    0.000: "p0.000",\n    0.001: "p0.001",\n    0.003: "p0.003",\n    0.010: "p0.010",\n}\n\nHVG_LABELS = {\n    "hvg_00_baseline": "HVG baseline",\n    "hvg_01_eval_rho090": "HVG global rho 0.90",\n    "hvg_04_eval_local_rho090": "HVG local rho 0.90",\n    "hvg_02_gate_final_map": "HVG gate final MAP",\n    "hvg_03_gate_hierarchical": "HVG gate hierarchical",\n}\nHVG_ORDER = {\n    "hvg_00_baseline": 0,\n    "hvg_01_eval_rho090": 1,\n    "hvg_04_eval_local_rho090": 2,\n    "hvg_02_gate_final_map": 3,\n    "hvg_03_gate_hierarchical": 4,\n}\n'
_CELL_CONFIG_CACHE = 'def safe_name(text):\n    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")\n    return text or "plot"\n\n\ndef save_figure(fig, name):\n    if not SAVE_FIGURES or fig is None:\n        return None\n    path = FIG_DIR / f"{safe_name(name)}.html"\n    fig.write_html(path, include_plotlyjs="cdn")\n    print(f"Saved: {path}")\n    return path\n\n\nSEED_POINT_MARKER = {\n    "size": 8,\n    "symbol": "circle",\n    "opacity": 0.9,\n    "line": {"color": "white", "width": 0.9},\n}\n\n\ndef _bar_trace_color_map(fig):\n    colors = {}\n    if fig is None:\n        return colors\n    for trace in fig.data:\n        if getattr(trace, "type", None) == "bar" and getattr(trace, "name", None) is not None:\n            colors[str(trace.name)] = trace.marker.color\n    return colors\n\n\ndef _customdata_from_columns(data, columns):\n    if not columns:\n        return None\n    arrays = []\n    for column in columns:\n        values = data[column]\n        if values.dtype == "object":\n            values = values.astype(str)\n        arrays.append(values.to_numpy())\n    return np.stack(arrays, axis=-1)\n\n\ndef _ordered_unique(values):\n    out = []\n    seen = set()\n    for value in values:\n        key = str(value)\n        if key not in seen:\n            seen.add(key)\n            out.append(key)\n    return out\n\n\ndef _seed_jitter_positions(n_points, width):\n    if n_points <= 1:\n        return np.zeros(n_points)\n    spread = min(width * 0.28, 0.08)\n    return np.linspace(-spread, spread, n_points)\n\n\ndef add_seed_point_overlay(fig, seed_data, *, x_col, y_col, group_col="condition_label", customdata_cols=None, hovertemplate=None):\n    if fig is None or seed_data is None or seed_data.empty:\n        return fig\n\n    bar_traces = [trace for trace in fig.data if getattr(trace, "type", None) == "bar"]\n    if not bar_traces:\n        return fig\n\n    grouped_axis = x_col != group_col\n    category_order = _ordered_unique(\n        value\n        for trace in bar_traces\n        for value in list(trace.x)\n    )\n    group_order = _ordered_unique(trace.name for trace in bar_traces)\n    category_to_position = {category: idx for idx, category in enumerate(category_order)}\n\n    if grouped_axis:\n        cluster_width = 0.82\n        group_width = cluster_width / max(len(group_order), 1)\n        group_to_offset = {\n            group: (idx - (len(group_order) - 1) / 2.0) * group_width\n            for idx, group in enumerate(group_order)\n        }\n        bar_width = group_width * 0.86\n    else:\n        group_to_offset = {group: 0.0 for group in group_order}\n        bar_width = 0.64\n\n    colors = _bar_trace_color_map(fig)\n    for trace in bar_traces:\n        group_name = str(trace.name)\n        original_x = [str(value) for value in list(trace.x)]\n        trace.hovertext = original_x\n        if trace.hovertemplate:\n            trace.hovertemplate = trace.hovertemplate.replace("%{x}", "%{hovertext}")\n        trace.x = [category_to_position[value] + group_to_offset.get(group_name, 0.0) for value in original_x]\n        trace.width = bar_width\n\n    seed_points = seed_data.copy()\n    seed_points["__category_label"] = seed_points[x_col].astype(str)\n    seed_points["__group_label"] = seed_points[group_col].astype(str)\n    seed_points["__bar_x"] = seed_points["__category_label"].map(category_to_position)\n    seed_points["__bar_x"] = seed_points["__bar_x"] + seed_points["__group_label"].map(group_to_offset).fillna(0.0)\n    seed_points = seed_points.dropna(subset=["__bar_x", y_col])\n    if seed_points.empty:\n        fig.update_xaxes(tickmode="array", tickvals=list(range(len(category_order))), ticktext=category_order)\n        return fig\n\n    adjusted_hovertemplate = hovertemplate.replace("%{x}", "%{text}") if hovertemplate else None\n    group_keys = ["__group_label", "__category_label"] if grouped_axis else ["__group_label"]\n    plotted_groups = []\n    for group_value, group_data in seed_points.groupby(group_keys, dropna=False, sort=False):\n        if grouped_axis:\n            group_name = str(group_value[0])\n            category_name = str(group_value[1])\n        else:\n            group_name = str(group_value[0] if isinstance(group_value, tuple) else group_value)\n            category_name = None\n        sort_cols = [column for column in ["seed", "run_name"] if column in group_data.columns]\n        group_data = group_data.sort_values(sort_cols).copy() if sort_cols else group_data.copy()\n        group_data["__dot_x"] = group_data["__bar_x"].to_numpy() + _seed_jitter_positions(len(group_data), bar_width)\n        marker = dict(SEED_POINT_MARKER)\n        marker["color"] = colors.get(group_name, "rgba(45,45,45,0.78)")\n        trace_name = f"{group_name} seeds" if category_name is None else f"{group_name} {category_name} seeds"\n        plotted_groups.append(group_name)\n        fig.add_trace(\n            go.Scatter(\n                x=group_data["__dot_x"],\n                y=group_data[y_col],\n                text=group_data["__category_label"],\n                mode="markers",\n                name=trace_name,\n                legendgroup=group_name,\n                showlegend=False,\n                marker=marker,\n                customdata=_customdata_from_columns(group_data, customdata_cols or []),\n                hovertemplate=adjusted_hovertemplate,\n            )\n        )\n\n    fig.update_xaxes(\n        tickmode="array",\n        tickvals=list(range(len(category_order))),\n        ticktext=category_order,\n    )\n    return fig\n\ndef _as_float(value, default=np.nan):\n    if value is None:\n        return default\n    try:\n        return float(value)\n    except (TypeError, ValueError):\n        return default\n\n\ndef _as_bool(value):\n    if isinstance(value, bool):\n        return value\n    if value is None or (isinstance(value, float) and np.isnan(value)):\n        return None\n    lower = str(value).strip().lower()\n    if lower in {"1", "true", "yes", "y", "on"}:\n        return True\n    if lower in {"0", "false", "no", "n", "off"}:\n        return False\n    return None\n\n\ndef _seed_from_stem(stem):\n    match = re.search(r"_s(\\d+)$", stem)\n    return int(match.group(1)) if match else np.nan\n\n\ndef _family_from_stem(stem):\n    return re.sub(r"_s\\d+$", "", stem)\n\n\ndef _sparse_penalty_from_family(family):\n    match = re.search(r"_p(\\d{3})$", family)\n    if not match:\n        return np.nan\n    return int(match.group(1)) / 1000.0\n\n\ndef _sparse_design_from_family(family):\n    match = re.match(r"^sparse16_(flat|gated)_p\\d{3}$", family)\n    return match.group(1) if match else None\n\n\ndef _metadata_from_config(path, experiment):\n    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))\n    args = cfg.get("args", {})\n    stem = Path(path).stem\n    family = _family_from_stem(stem)\n    seed = _seed_from_stem(stem)\n    gate_enabled = _as_bool(args.get("intervention_gate"))\n    topology_reward = _as_float(args.get("topology_reward_weight"), 0.0)\n    intervention_penalty = _as_float(args.get("intervention_penalty"), 0.0)\n    entropy_initial = _as_float(args.get("entropy_coef"), np.nan)\n    entropy_final = _as_float(args.get("entropy_coef_final"), np.nan)\n    entropy_schedule = "decay" if entropy_final < entropy_initial else "constant"\n    eval_action_heuristic = args.get("eval_action_heuristic") or "none"\n    eval_action_rho_threshold = _as_float(args.get("eval_action_rho_threshold"), np.nan)\n    intervention_gate_eval_mode = args.get("intervention_gate_eval_mode") or "none"\n\n    if experiment == "intervention_gate_15":\n        family_label = IG15_LABELS.get(family, family.replace("_", " "))\n        design = "gated"\n        control_axis = "topology_reward"\n        control_value = topology_reward\n        control_label = f"topo {topology_reward:g}"\n        comparison_group = "IG15"\n    elif experiment == "phase4_sparse_control_16":\n        design = _sparse_design_from_family(family) or ("gated" if gate_enabled else "flat")\n        control_axis = "intervention_penalty"\n        control_value = intervention_penalty if not np.isnan(intervention_penalty) else _sparse_penalty_from_family(family)\n        control_label = SPARSE16_PENALTY_LABELS.get(control_value, f"p{control_value:g}")\n        family_label = f"Sparse16 {design} {control_label}"\n        comparison_group = "Sparse16"\n    elif experiment == "heuristic_vs_gate_s0_s1_s2":\n        eval_action_heuristic_normalized = str(eval_action_heuristic).strip().lower()\n        family_label = HVG_LABELS.get(family, family.replace("_", " "))\n        if gate_enabled:\n            design = "gated"\n        elif eval_action_heuristic_normalized != "none":\n            design = "eval_heuristic"\n        else:\n            design = "baseline"\n        control_axis = "variant_order"\n        control_value = HVG_ORDER.get(family, np.nan)\n        control_label = family_label.replace("HVG ", "")\n        comparison_group = "HVG"\n    else:\n        family_label = family.replace("_", " ")\n        design = "gated" if gate_enabled else "flat"\n        control_axis = "unknown"\n        control_value = np.nan\n        control_label = "unknown"\n        comparison_group = experiment\n\n    return {\n        "expected_run_name": stem,\n        "config_path": str(path),\n        "experiment": experiment,\n        "comparison_group": comparison_group,\n        "family": family,\n        "family_label": family_label,\n        "seed": seed,\n        "design": design,\n        "gate_enabled": gate_enabled,\n        "control_axis": control_axis,\n        "control_value": control_value,\n        "control_label": control_label,\n        "eval_action_heuristic": eval_action_heuristic,\n        "eval_action_rho_threshold": eval_action_rho_threshold,\n        "intervention_gate_eval_mode": intervention_gate_eval_mode,\n        "topology_reward_weight": topology_reward,\n        "intervention_penalty": intervention_penalty,\n        "entropy_coef": entropy_initial,\n        "entropy_coef_final": entropy_final,\n        "entropy_schedule": entropy_schedule,\n        "total_timesteps": args.get("total_timesteps"),\n        "eval_freq": args.get("eval_freq"),\n        "n_steps": args.get("n_steps"),\n        "n_envs": args.get("n_envs"),\n        "rollout_action_samples": int(args.get("n_steps", 0)) * int(args.get("n_envs", 0)),\n    }\n\n\ndef read_expected_configs():\n    rows = []\n    for experiment, folder in EXPERIMENT_FOLDERS.items():\n        if not folder.exists():\n            print(f"Missing config folder: {folder}")\n            continue\n        for path in sorted(folder.glob("*.toml")):\n            rows.append(_metadata_from_config(path, experiment))\n    if not rows:\n        raise RuntimeError("No expected config TOMLs found.")\n    return pd.DataFrame(rows)\n\n\ndef load_cache_index(path=CACHE_INDEX_PATH):\n    if not Path(path).exists():\n        raise FileNotFoundError(f"Missing {path}. Run the W&B history cache notebook first.")\n    index = pd.read_csv(path)\n    index = index.rename(columns={"name": "run_name", "id": "run_id"}).copy()\n    for col in ["history_parquet", "history_csv"]:\n        if col not in index.columns:\n            index[col] = None\n    index["history_parquet"] = index["history_parquet"].apply(lambda value: Path(value) if pd.notna(value) else None)\n    index["history_csv"] = index["history_csv"].apply(lambda value: Path(value) if pd.notna(value) else None)\n    index["has_history"] = index.apply(\n        lambda row: bool(row["history_parquet"] and row["history_parquet"].exists())\n        or bool(row["history_csv"] and row["history_csv"].exists()),\n        axis=1,\n    )\n    return index[index["has_history"]].reset_index(drop=True)\n\n\nEXPECTED_CONFIGS = read_expected_configs()\ncache_index = load_cache_index()\nselected_runs = cache_index.merge(\n    EXPECTED_CONFIGS,\n    left_on="run_name",\n    right_on="expected_run_name",\n    how="inner",\n)\ncoverage = EXPECTED_CONFIGS.merge(\n    selected_runs[["expected_run_name", "run_id", "rows", "columns"]],\n    on="expected_run_name",\n    how="left",\n)\ncoverage["cached"] = coverage["run_id"].notna()\n\nprint(f"Expected configs: {len(EXPECTED_CONFIGS)}")\nprint(f"Cached expected runs: {coverage[\'cached\'].sum()} / {len(coverage)}")\n# display(coverage.groupby(["experiment", "family_label"], dropna=False).agg(\n#     expected=("expected_run_name", "count"),\n#     cached=("cached", "sum"),\n#     seeds=("seed", lambda values: sorted(pd.Series(values).dropna().astype(int).unique())),\n# ).reset_index())\n\nmissing = coverage[~coverage["cached"]].sort_values(["experiment", "family_label", "seed"])\nif not missing.empty:\n    print("Missing cached histories for these expected configs:")\n    display(missing[["experiment", "expected_run_name", "family_label", "seed", "control_label", "design"]])\n'
_CELL_HISTORIES = 'def read_cached_history(row):\n    parquet_path = row.get("history_parquet")\n    csv_path = row.get("history_csv")\n    if parquet_path and Path(parquet_path).exists():\n        history = pd.read_parquet(parquet_path)\n    elif csv_path and Path(csv_path).exists():\n        history = pd.read_csv(csv_path)\n    else:\n        raise FileNotFoundError(f"No cached history file for {row[\'run_name\']} ({row[\'run_id\']})")\n    history = history.copy()\n    history["run_name"] = row["run_name"]\n    history["run_id"] = row["run_id"]\n    if "_step" not in history.columns:\n        if "charts/global_step" in history.columns:\n            history["_step"] = history["charts/global_step"]\n        elif "step" in history.columns:\n            history["_step"] = history["step"]\n        else:\n            history["_step"] = np.arange(len(history), dtype=float)\n    history["_step"] = pd.to_numeric(history["_step"], errors="coerce")\n    history["step_millions"] = history["_step"] / 1_000_000\n    return history\n\n\ndef load_cached_histories(selected):\n    frames = []\n    total = len(selected)\n    for idx, row in enumerate(selected.to_dict("records"), start=1):\n        print(f"[{idx:>3}/{total}] loading {row[\'run_name\']}", flush=True)\n        try:\n            frames.append(read_cached_history(row))\n        except Exception as exc:\n            print(f"    skipped: {type(exc).__name__}: {exc}")\n    if not frames:\n        raise RuntimeError("No histories could be loaded from selected cached runs.")\n    history = pd.concat(frames, ignore_index=True, sort=False).dropna(subset=["_step"])\n    if PLOT_STEP_MAX_M is not None:\n        history = history[history["step_millions"] <= float(PLOT_STEP_MAX_M)]\n    meta_cols = [\n        "run_name", "run_id", "experiment", "comparison_group", "family", "family_label", "seed",\n        "design", "gate_enabled", "control_axis", "control_value", "control_label",\n        "topology_reward_weight", "intervention_penalty", "entropy_schedule",\n        "n_steps", "n_envs", "rollout_action_samples",\n    ]\n    return history.merge(selected[meta_cols], on=["run_name", "run_id"], how="left").reset_index(drop=True)\n\n\nhistory_wide = load_cached_histories(selected_runs)\nprint(f"Loaded history shape: {history_wide.shape}")\n\nACTION_COLUMN_HINTS = [\n    "frac_action_0", "entropy_agent", "intervention_gate", "nonidle_action_entropy",\n    "non_idle_agents", "illegal_action",\n]\nmetric_availability = []\nfor row in selected_runs.to_dict("records"):\n    run_history = history_wide[history_wide["run_id"] == row["run_id"]]\n    cols = list(run_history.columns)\n    available = {hint: sum(hint in str(col) for col in cols) for hint in ACTION_COLUMN_HINTS}\n    metric_availability.append({\n        "run_name": row["run_name"],\n        "family_label": row["family_label"],\n        "seed": row["seed"],\n        "history_rows": len(run_history),\n        **available,\n    })\nmetric_availability = pd.DataFrame(metric_availability)\nprint("Action metric availability by run:")\n# display(metric_availability.sort_values(["family_label", "seed"]))\n'
_CELL_FEATURES = 'ID_COLS = [\n    "run_name", "run_id", "experiment", "comparison_group", "family", "family_label", "seed",\n    "design", "gate_enabled", "control_axis", "control_value", "control_label",\n    "topology_reward_weight", "intervention_penalty", "entropy_schedule",\n    "n_steps", "n_envs", "rollout_action_samples", "_step", "step_millions",\n]\n\nAGENT_FEATURE_PATTERNS = [\n    {\n        "feature_group": "agent_action0",\n        "label_prefix": "action 0",\n        "pattern": re.compile(r"^train/frac_action_0_(agent_\\d+)$"),\n        "transform": lambda values: values,\n    },\n    {\n        "feature_group": "agent_non_idle",\n        "label_prefix": "non-idle",\n        "pattern": re.compile(r"^train/frac_action_0_(agent_\\d+)$"),\n        "transform": lambda values: 1.0 - values,\n    },\n    {\n        "feature_group": "agent_entropy",\n        "label_prefix": "entropy",\n        "pattern": re.compile(r"^train/entropy_(agent_\\d+)$"),\n        "transform": lambda values: values,\n    },\n    {\n        "feature_group": "agent_illegal",\n        "label_prefix": "illegal",\n        "pattern": re.compile(r"^train/illegal_action_rate_(agent_\\d+)$"),\n        "transform": lambda values: values,\n    },\n    {\n        "feature_group": "gate_intervene_frac",\n        "label_prefix": "gate intervene frac",\n        "pattern": re.compile(r"^train/intervention_gate_intervene_frac_(agent_\\d+)$"),\n        "transform": lambda values: values,\n    },\n    {\n        "feature_group": "gate_do_nothing_frac",\n        "label_prefix": "gate do-nothing frac",\n        "pattern": re.compile(r"^train/intervention_gate_do_nothing_frac_(agent_\\d+)$"),\n        "transform": lambda values: values,\n    },\n    {\n        "feature_group": "gate_prob_intervene",\n        "label_prefix": "gate prob intervene",\n        "pattern": re.compile(r"^train/intervention_gate_prob_intervene_(agent_\\d+)$"),\n        "transform": lambda values: values,\n    },\n    {\n        "feature_group": "gate_entropy",\n        "label_prefix": "gate entropy",\n        "pattern": re.compile(r"^train/intervention_gate_entropy_(agent_\\d+)$"),\n        "transform": lambda values: values,\n    },\n    {\n        "feature_group": "nonidle_action_entropy",\n        "label_prefix": "non-idle action entropy",\n        "pattern": re.compile(r"^train/nonidle_action_entropy_(agent_\\d+)$"),\n        "transform": lambda values: values,\n    },\n]\n\nSUMMARY_FEATURES = {\n    "frac_any_non_idle": "train/frac_any_non_idle",\n    "frac_multi_agent_non_idle": "train/frac_multi_agent_non_idle",\n    "non_idle_agents_mean": "train/non_idle_agents_mean",\n    "non_idle_agents_std": "train/non_idle_agents_std",\n}\nNON_IDLE_COUNT_PATTERN = re.compile(r"^train/non_idle_agents_count_(\\d+)_frac$")\n\nFEATURE_GROUP_ORDER = {\n    "agent_non_idle": 0,\n    "agent_action0": 1,\n    "gate_intervene_frac": 2,\n    "gate_prob_intervene": 3,\n    "gate_do_nothing_frac": 4,\n    "gate_entropy": 5,\n    "nonidle_action_entropy": 6,\n    "joint_non_idle": 7,\n    "summary_non_idle": 8,\n    "agent_illegal": 9,\n    "agent_entropy": 10,\n}\n\n\ndef _melt_agent_feature(history, spec):\n    cols = [col for col in history.columns if spec["pattern"].match(str(col))]\n    if not cols:\n        return pd.DataFrame()\n    data = history[ID_COLS + cols].melt(\n        id_vars=ID_COLS,\n        value_vars=cols,\n        var_name="metric",\n        value_name="raw_value",\n    )\n    data["raw_value"] = pd.to_numeric(data["raw_value"], errors="coerce")\n    data = data.dropna(subset=["raw_value"])\n    if data.empty:\n        return data\n    data["entity"] = data["metric"].str.extract(spec["pattern"].pattern)[0]\n    data["value"] = spec["transform"](data["raw_value"])\n    data["feature_group"] = spec["feature_group"]\n    data["feature_key"] = data["feature_group"] + ":" + data["entity"].astype(str)\n    data["feature_label"] = spec["label_prefix"] + " " + data["entity"].astype(str)\n    return data.drop(columns=["raw_value"])\n\n\ndef _melt_joint_non_idle(history):\n    cols = [col for col in history.columns if NON_IDLE_COUNT_PATTERN.match(str(col))]\n    if not cols:\n        return pd.DataFrame()\n    data = history[ID_COLS + cols].melt(\n        id_vars=ID_COLS,\n        value_vars=cols,\n        var_name="metric",\n        value_name="value",\n    )\n    data["value"] = pd.to_numeric(data["value"], errors="coerce")\n    data = data.dropna(subset=["value"])\n    if data.empty:\n        return data\n    data["entity"] = data["metric"].str.extract(NON_IDLE_COUNT_PATTERN.pattern)[0].astype(int)\n    data["feature_group"] = "joint_non_idle"\n    data["feature_key"] = "joint_non_idle:" + data["entity"].astype(str)\n    data["feature_label"] = data["entity"].astype(str) + " agents act"\n    return data\n\n\ndef _melt_summary_features(history):\n    available = {label: col for label, col in SUMMARY_FEATURES.items() if col in history.columns}\n    if not available:\n        return pd.DataFrame()\n    data = history[ID_COLS + list(available.values())].melt(\n        id_vars=ID_COLS,\n        value_vars=list(available.values()),\n        var_name="metric",\n        value_name="value",\n    )\n    data["value"] = pd.to_numeric(data["value"], errors="coerce")\n    data = data.dropna(subset=["value"])\n    if data.empty:\n        return data\n    metric_to_label = {col: label for label, col in available.items()}\n    data["entity"] = data["metric"].map(metric_to_label)\n    data["feature_group"] = "summary_non_idle"\n    data["feature_key"] = "summary_non_idle:" + data["entity"].astype(str)\n    data["feature_label"] = data["entity"].astype(str).str.replace("_", " ")\n    return data\n\n\ndef build_feature_long(history):\n    frames = []\n    for spec in AGENT_FEATURE_PATTERNS:\n        frames.append(_melt_agent_feature(history, spec))\n    frames.append(_melt_joint_non_idle(history))\n    frames.append(_melt_summary_features(history))\n    frames = [frame for frame in frames if frame is not None and not frame.empty]\n    if not frames:\n        raise RuntimeError("No action-distribution feature metrics were found in the selected histories.")\n    feature_long = pd.concat(frames, ignore_index=True, sort=False)\n    feature_long["feature_group_order"] = feature_long["feature_group"].map(FEATURE_GROUP_ORDER).fillna(99)\n    return feature_long\n\n\ndef final_window_average(data, value_col, group_cols, window=FINAL_WINDOW_STEPS):\n    sorted_data = data.sort_values(group_cols + ["_step"])\n    tail = sorted_data.groupby(group_cols, dropna=False).tail(int(window))\n    return tail.groupby(group_cols, dropna=False, as_index=False)[value_col].mean()\n\n\ndef rolling_mean_by_group(data, value_col, group_cols, window=SMOOTH_WINDOW):\n    out = data.sort_values(group_cols + ["_step"]).copy()\n    if window is None or int(window) <= 1:\n        out[f"{value_col}_smooth"] = out[value_col]\n    else:\n        out[f"{value_col}_smooth"] = out.groupby(group_cols, dropna=False)[value_col].transform(\n            lambda values: values.rolling(int(window), min_periods=1).mean()\n        )\n    return out\n\n\nFEATURE_LONG = build_feature_long(history_wide)\n# print(f"Feature rows: {len(FEATURE_LONG):,}")\n# display(FEATURE_LONG.groupby(["experiment", "family_label", "feature_group"], dropna=False).size().reset_index(name="rows"))\n'
_CELL_PROFILES = 'PROFILE_GROUP_COLS = [\n    "run_name", "run_id", "experiment", "comparison_group", "family", "family_label", "seed",\n    "design", "gate_enabled", "control_axis", "control_value", "control_label",\n    "topology_reward_weight", "intervention_penalty", "entropy_schedule",\n    "feature_group", "feature_group_order", "feature_key", "feature_label", "entity",\n]\n\nprofile_by_run = final_window_average(FEATURE_LONG, "value", PROFILE_GROUP_COLS)\nACTION_PROFILE = profile_by_run.groupby([\n    "experiment", "comparison_group", "family", "family_label", "design", "gate_enabled",\n    "control_axis", "control_value", "control_label", "topology_reward_weight",\n    "intervention_penalty", "entropy_schedule", "feature_group", "feature_group_order",\n    "feature_key", "feature_label", "entity",\n], dropna=False, as_index=False).agg(\n    value=("value", "mean"),\n    std=("value", "std"),\n    runs=("run_id", "nunique"),\n    seeds=("seed", lambda values: sorted(pd.Series(values).dropna().astype(int).unique())),\n)\n\n# print(f"Final-window profile rows: {len(ACTION_PROFILE):,}")\n# display(ACTION_PROFILE.groupby(["experiment", "family_label", "feature_group"], dropna=False).agg(\n#     features=("feature_key", "nunique"),\n#     runs=("runs", "max"),\n# ).reset_index())\n'
_CELL_ACTION0_ALL = 'ACTION0_AVERAGE_LAST_N_LOGGED = 5\n\n\ndef action0_final_rows():\n    data = FEATURE_LONG[FEATURE_LONG["feature_group"] == "agent_action0"].copy()\n    if data.empty:\n        return data\n\n    # Average the last few cached training values for each run and agent.\n    data = data.sort_values(["run_id", "entity", "_step"])\n    data = data.groupby(["run_id", "entity"], dropna=False).tail(int(ACTION0_AVERAGE_LAST_N_LOGGED)).copy()\n\n    data["rollout_action_samples"] = pd.to_numeric(data["rollout_action_samples"], errors="coerce")\n    missing_samples = data["rollout_action_samples"].isna() | (data["rollout_action_samples"] <= 0)\n    if missing_samples.any():\n        fallback_samples = (\n            pd.to_numeric(data.get("n_steps"), errors="coerce").fillna(0)\n            * pd.to_numeric(data.get("n_envs"), errors="coerce").fillna(0)\n        )\n        data.loc[missing_samples, "rollout_action_samples"] = fallback_samples[missing_samples]\n\n    summary_cols = [\n        "run_name", "run_id", "experiment", "comparison_group", "family", "family_label",\n        "seed", "design", "control_axis", "control_value", "control_label", "entity",\n        "n_steps", "n_envs", "rollout_action_samples",\n    ]\n    data = data.groupby(summary_cols, dropna=False, as_index=False).agg(\n        final_fraction_action0=("value", "mean"),\n        std_last_fraction_action0=("value", "std"),\n        final_step_millions=("step_millions", "max"),\n        averaged_logged_points=("value", "count"),\n    )\n    data["std_last_fraction_action0"] = data["std_last_fraction_action0"].fillna(0.0)\n    data["final_action0_count"] = (data["final_fraction_action0"] * data["rollout_action_samples"]).round().astype("Int64")\n    data["std_last_action0_count"] = (data["std_last_fraction_action0"] * data["rollout_action_samples"]).round().astype("Int64")\n    data["agent"] = data["entity"].astype(str)\n    data["run_label"] = data["run_name"]\n    data["folder_label"] = data["experiment"].map({\n        "intervention_gate_15": "Intervention gate 15",\n        "phase4_sparse_control_16": "Phase4 sparse control 16",\n        "heuristic_vs_gate_s0_s1_s2": "Heuristic vs gate s0/s1/s2",\n    }).fillna(data["experiment"])\n    data["agent"] = pd.Categorical(data["agent"], categories=["agent_0", "agent_1", "agent_2"], ordered=True)\n    return data.dropna(subset=["final_fraction_action0"])\n\n\nACTION0_FINAL_LONG = action0_final_rows()\nACTION0_COUNT_LONG = ACTION0_FINAL_LONG  # Backward-compatible name for exported table code.\n\nACTION0_COUNT_SUMMARY = ACTION0_FINAL_LONG[[\n    "experiment",\n    "run_name",\n    "family_label",\n    "seed",\n    "agent",\n    "final_fraction_action0",\n    "final_action0_count",\n    "final_step_millions",\n    "rollout_action_samples",\n]].sort_values(["experiment", "family_label", "seed", "agent"]).reset_index(drop=True)\n\n# print(f"Action-0 fraction averaged over the last {ACTION0_AVERAGE_LAST_N_LOGGED} logged points by run and agent:")\n# display(ACTION0_COUNT_SUMMARY)\n\n\ndef plot_action0_fraction_all_runs(experiment, title, save_name):\n    data = ACTION0_FINAL_LONG[ACTION0_FINAL_LONG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No final action-0 data for {experiment}")\n        return None\n\n    data = data.sort_values(["family_label", "seed", "agent"])\n    fig = px.bar(\n        data,\n        x="agent",\n        y="final_fraction_action0",\n        color="run_label",\n        barmode="group",\n        hover_data=[\n            "family_label",\n            "seed",\n            "final_action0_count",\n            "final_step_millions",\n            "rollout_action_samples",\n        ],\n        labels={\n            "agent": "agent",\n            "final_fraction_action0": f"mean action 0 fraction over last {ACTION0_AVERAGE_LAST_N_LOGGED} logged",\n            "run_label": "run",\n        },\n        title=title,\n    )\n    fig.update_yaxes(range=[0, 1], title_text="action 0 fraction")\n    fig.update_layout(\n        template="plotly_white",\n        height=620,\n        width=1450,\n        bargap=0.18,\n        bargroupgap=0.04,\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 70, "r": 340, "t": 90, "b": 70},\n    )\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\nfig_action0_fraction_ig15_all_runs = plot_action0_fraction_all_runs(\n    "intervention_gate_15",\n    f"Intervention gate 15: action 0 fraction by agent for all runs (mean of last {ACTION0_AVERAGE_LAST_N_LOGGED} logged)",\n    "final_action0_fraction_by_agent_all_runs_intervention_gate_15",\n)\nfig_action0_fraction_sparse16_all_runs = plot_action0_fraction_all_runs(\n    "phase4_sparse_control_16",\n    f"Phase4 sparse control 16: action 0 fraction by agent for all runs (mean of last {ACTION0_AVERAGE_LAST_N_LOGGED} logged)",\n    "final_action0_fraction_by_agent_all_runs_phase4_sparse_control_16",\n)\nfig_action0_fraction_hvg_all_runs = plot_action0_fraction_all_runs(\n    "heuristic_vs_gate_s0_s1_s2",\n    f"Heuristic vs gate s0/s1/s2: action 0 fraction by agent for all runs (mean of last {ACTION0_AVERAGE_LAST_N_LOGGED} logged)",\n    "final_action0_fraction_by_agent_all_runs_heuristic_vs_gate_s0_s1_s2",\n)\n'
_CELL_ACTION0_SEED = 'def _seed_list(values):\n    return sorted(pd.Series(values).dropna().astype(int).unique().tolist())\n\n\ndef seed_aggregated_action0_rows():\n    if ACTION0_FINAL_LONG.empty:\n        return pd.DataFrame()\n    grouped = ACTION0_FINAL_LONG.groupby(\n        [\n            "experiment",\n            "comparison_group",\n            "family",\n            "family_label",\n            "design",\n            "control_axis",\n            "control_value",\n            "control_label",\n            "agent",\n        ],\n        dropna=False,\n        observed=True,\n        as_index=False,\n    ).agg(\n        mean_final_fraction_action0=("final_fraction_action0", "mean"),\n        std_final_fraction_action0=("final_fraction_action0", "std"),\n        mean_final_action0_count=("final_action0_count", "mean"),\n        std_final_action0_count=("final_action0_count", "std"),\n        n_seeds=("seed", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    grouped["std_final_fraction_action0"] = grouped["std_final_fraction_action0"].fillna(0.0)\n    grouped["std_final_action0_count"] = grouped["std_final_action0_count"].fillna(0.0)\n    grouped["agent"] = pd.Categorical(grouped["agent"], categories=["agent_0", "agent_1", "agent_2"], ordered=True)\n    grouped["condition_label"] = grouped["family_label"]\n    return grouped.sort_values(["experiment", "control_value", "design", "family_label", "agent"])\n\n\nACTION0_SEED_AGG = seed_aggregated_action0_rows()\n# print(f"Seed-aggregated action-0 fraction by condition and agent (run mean of last {ACTION0_AVERAGE_LAST_N_LOGGED} logged):")\n# display(ACTION0_SEED_AGG)\n\n\ndef plot_action0_fraction_seed_aggregated(experiment, title, save_name):\n    data = ACTION0_SEED_AGG[ACTION0_SEED_AGG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No seed-aggregated final action-0 data for {experiment}")\n        return None\n\n    data = data.sort_values(["control_value", "design", "condition_label", "agent"])\n    fig = px.bar(\n        data,\n        x="agent",\n        y="mean_final_fraction_action0",\n        color="condition_label",\n        error_y="std_final_fraction_action0",\n        barmode="group",\n        hover_data={\n            "condition_label": True,\n            "control_label": True,\n            "design": True,\n            "n_seeds": True,\n            "seeds": True,\n            "mean_final_action0_count": ":.0f",\n            "std_final_action0_count": ":.0f",\n            "mean_final_fraction_action0": ":.4f",\n            "std_final_fraction_action0": ":.4f",\n        },\n        labels={\n            "agent": "agent",\n            "mean_final_fraction_action0": "mean final action 0 fraction",\n            "condition_label": "condition",\n        },\n        title=title,\n    )\n\n    seed_points = ACTION0_FINAL_LONG[ACTION0_FINAL_LONG["experiment"] == experiment].copy()\n    if not seed_points.empty:\n        seed_points["condition_label"] = seed_points["family_label"]\n        seed_points = seed_points.sort_values(["control_value", "design", "condition_label", "agent", "seed"])\n        add_seed_point_overlay(\n            fig,\n            seed_points,\n            x_col="agent",\n            y_col="final_fraction_action0",\n            group_col="condition_label",\n            customdata_cols=[\n                "run_name",\n                "seed",\n                "condition_label",\n                "final_fraction_action0",\n                "final_action0_count",\n                "averaged_logged_points",\n            ],\n            hovertemplate=(\n                "seed run=%{customdata[0]}<br>"\n                "seed=%{customdata[1]}<br>"\n                "condition=%{customdata[2]}<br>"\n                "agent=%{x}<br>"\n                "seed action-0 fraction=%{y:.4f}<br>"\n                "action-0 count=%{customdata[4]}<br>"\n                "logged points=%{customdata[5]}<extra></extra>"\n            ),\n        )\n\n    fig.update_yaxes(range=[0, 1], title_text="mean final action 0 fraction")\n    fig.update_layout(\n        template="plotly_white",\n        height=620,\n        width=1450,\n        bargap=0.18,\n        bargroupgap=0.04,\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 70, "r": 340, "t": 90, "b": 70},\n    )\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\nfig_action0_fraction_ig15_seed_agg = plot_action0_fraction_seed_aggregated(\n    "intervention_gate_15",\n    f"Intervention gate 15: seed-aggregated action 0 fraction by agent (run mean of last {ACTION0_AVERAGE_LAST_N_LOGGED} logged)",\n    "seed_aggregated_final_action0_fraction_by_agent_intervention_gate_15",\n)\nfig_action0_fraction_sparse16_seed_agg = plot_action0_fraction_seed_aggregated(\n    "phase4_sparse_control_16",\n    f"Phase4 sparse control 16: seed-aggregated action 0 fraction by agent (run mean of last {ACTION0_AVERAGE_LAST_N_LOGGED} logged)",\n    "seed_aggregated_final_action0_fraction_by_agent_phase4_sparse_control_16",\n)\nfig_action0_fraction_hvg_seed_agg = plot_action0_fraction_seed_aggregated(\n    "heuristic_vs_gate_s0_s1_s2",\n    f"Heuristic vs gate s0/s1/s2: seed-aggregated action 0 fraction by agent (run mean of last {ACTION0_AVERAGE_LAST_N_LOGGED} logged)",\n    "seed_aggregated_final_action0_fraction_by_agent_heuristic_vs_gate_s0_s1_s2",\n)\n'
_CELL_SURVIVAL_ACTION = 'SURVIVAL_ACTION_BEHAVIOR_SMOOTH_WINDOW = 5\nSURVIVAL_METRIC_CANDIDATES = [\n    "test/charts/episodic_survival",\n    "test/episodic_survival",\n    "validation/episodic_survival",\n    "charts/episodic_survival",\n    "train_eval/charts/episodic_survival",\n    "train_eval/episodic_survival",\n]\n\ntry:\n    _seed_list\nexcept NameError:\n    def _seed_list(values):\n        return sorted(pd.Series(values).dropna().astype(int).unique().tolist())\n\n\ndef _survival_scale(values):\n    numeric = pd.to_numeric(values, errors="coerce")\n    max_value = numeric.max(skipna=True)\n    if pd.isna(max_value):\n        return 100.0\n    return 100.0 if max_value <= 1.5 else 1.0\n\n\ndef seed_aggregated_survival_rows():\n    frames = []\n    for run_id, run_history in history_wide.groupby("run_id", sort=False):\n        metric = next(\n            (\n                candidate for candidate in SURVIVAL_METRIC_CANDIDATES\n                if candidate in run_history.columns and run_history[candidate].notna().any()\n            ),\n            None,\n        )\n        if metric is None:\n            continue\n\n        values = pd.to_numeric(run_history[metric], errors="coerce")\n        frame = run_history[[\n            "run_name",\n            "run_id",\n            "experiment",\n            "comparison_group",\n            "family",\n            "family_label",\n            "seed",\n            "design",\n            "control_axis",\n            "control_value",\n            "control_label",\n            "entropy_schedule",\n            "_step",\n            "step_millions",\n        ]].copy()\n        frame["survival_pct"] = values * _survival_scale(values)\n        frame["metric_used"] = metric\n        frames.append(frame.dropna(subset=["survival_pct", "_step", "step_millions"]))\n\n    if not frames:\n        return pd.DataFrame()\n\n    survival = pd.concat(frames, ignore_index=True, sort=False)\n    grouped = survival.groupby(\n        [\n            "experiment",\n            "comparison_group",\n            "family",\n            "family_label",\n            "design",\n            "control_axis",\n            "control_value",\n            "control_label",\n            "entropy_schedule",\n            "_step",\n            "step_millions",\n        ],\n        dropna=False,\n        as_index=False,\n    ).agg(\n        mean_survival_pct=("survival_pct", "mean"),\n        std_survival_pct=("survival_pct", "std"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n        metrics_used=("metric_used", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    grouped["std_survival_pct"] = grouped["std_survival_pct"].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    grouped = grouped.sort_values(["experiment", "control_value", "design", "family_label", "_step"])\n\n    if SURVIVAL_ACTION_BEHAVIOR_SMOOTH_WINDOW and int(SURVIVAL_ACTION_BEHAVIOR_SMOOTH_WINDOW) > 1:\n        grouped["mean_survival_pct_smooth"] = grouped.groupby(\n            ["experiment", "family"],\n            dropna=False,\n        )["mean_survival_pct"].transform(\n            lambda values: values.rolling(int(SURVIVAL_ACTION_BEHAVIOR_SMOOTH_WINDOW), min_periods=1).mean()\n        )\n        grouped["std_survival_pct_smooth"] = grouped.groupby(\n            ["experiment", "family"],\n            dropna=False,\n        )["std_survival_pct"].transform(\n            lambda values: values.rolling(int(SURVIVAL_ACTION_BEHAVIOR_SMOOTH_WINDOW), min_periods=1).mean()\n        )\n    else:\n        grouped["mean_survival_pct_smooth"] = grouped["mean_survival_pct"]\n        grouped["std_survival_pct_smooth"] = grouped["std_survival_pct"]\n    return grouped\n\n\nSURVIVAL_SEED_AGG = seed_aggregated_survival_rows()\n# print("Seed-aggregated survival time series:")\n# display(SURVIVAL_SEED_AGG.groupby(["experiment", "family_label"], dropna=False).agg(\n#     points=("_step", "count"),\n#     n_seeds=("n_seeds", "max"),\n#     seeds=("seeds", "first"),\n#     metrics=("metrics_used", "first"),\n# ).reset_index())\n\n\nACTION_BEHAVIOR_SMOOTH_WINDOW = 5\n\n\ndef seed_aggregated_action_behavior_rows():\n    behavior = FEATURE_LONG[FEATURE_LONG["feature_group"].isin(["agent_action0", "agent_non_idle"])].copy()\n    if behavior.empty:\n        return pd.DataFrame()\n\n    # First collapse the three agents inside each run, so each run contributes one\n    # action-0 and one non-idle curve to the seed aggregate.\n    per_run = behavior.groupby(\n        [\n            "run_name",\n            "run_id",\n            "experiment",\n            "comparison_group",\n            "family",\n            "family_label",\n            "seed",\n            "design",\n            "control_axis",\n            "control_value",\n            "control_label",\n            "entropy_schedule",\n            "feature_group",\n            "_step",\n            "step_millions",\n        ],\n        dropna=False,\n        as_index=False,\n    ).agg(\n        run_mean_value=("value", "mean"),\n        agents=("entity", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n\n    grouped = per_run.groupby(\n        [\n            "experiment",\n            "comparison_group",\n            "family",\n            "family_label",\n            "design",\n            "control_axis",\n            "control_value",\n            "control_label",\n            "entropy_schedule",\n            "feature_group",\n            "_step",\n            "step_millions",\n        ],\n        dropna=False,\n        as_index=False,\n    ).agg(\n        mean_value=("run_mean_value", "mean"),\n        std_value=("run_mean_value", "std"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    grouped["std_value"] = grouped["std_value"].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    grouped["metric_label"] = grouped["feature_group"].map({\n        "agent_action0": "action 0 fraction",\n        "agent_non_idle": "non-idle fraction",\n    })\n    grouped = grouped.sort_values(["experiment", "control_value", "design", "family_label", "feature_group", "_step"])\n\n    if ACTION_BEHAVIOR_SMOOTH_WINDOW and int(ACTION_BEHAVIOR_SMOOTH_WINDOW) > 1:\n        grouped["mean_value_smooth"] = grouped.groupby(\n            ["experiment", "family", "feature_group"],\n            dropna=False,\n        )["mean_value"].transform(lambda values: values.rolling(int(ACTION_BEHAVIOR_SMOOTH_WINDOW), min_periods=1).mean())\n        grouped["std_value_smooth"] = grouped.groupby(\n            ["experiment", "family", "feature_group"],\n            dropna=False,\n        )["std_value"].transform(lambda values: values.rolling(int(ACTION_BEHAVIOR_SMOOTH_WINDOW), min_periods=1).mean())\n    else:\n        grouped["mean_value_smooth"] = grouped["mean_value"]\n        grouped["std_value_smooth"] = grouped["std_value"]\n    return grouped\n\n\nACTION_BEHAVIOR_SEED_AGG = seed_aggregated_action_behavior_rows()\n# print("Seed-aggregated action behavior time series:")\n# display(ACTION_BEHAVIOR_SEED_AGG.groupby(["experiment", "family_label", "metric_label"], dropna=False).agg(\n#     points=("_step", "count"),\n#     n_seeds=("n_seeds", "max"),\n#     seeds=("seeds", "first"),\n# ).reset_index())\n\n\ndef plot_survival_vs_action_behavior(experiment, title, save_name):\n    survival = SURVIVAL_SEED_AGG[SURVIVAL_SEED_AGG["experiment"] == experiment].copy()\n    behavior = ACTION_BEHAVIOR_SEED_AGG[ACTION_BEHAVIOR_SEED_AGG["experiment"] == experiment].copy()\n    if survival.empty or behavior.empty:\n        print(f"Missing survival or action-behavior data for {experiment}")\n        return None\n\n    conditions = pd.concat([\n        survival[["family", "condition_label", "control_value", "design"]],\n        behavior[["family", "condition_label", "control_value", "design"]],\n    ], ignore_index=True).drop_duplicates()\n    conditions = conditions.sort_values(["control_value", "design", "condition_label"])\n    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24\n    colors = {row.family: palette[idx % len(palette)] for idx, row in enumerate(conditions.itertuples(index=False))}\n\n    fig = make_subplots(\n        rows=3,\n        cols=1,\n        shared_xaxes=True,\n        vertical_spacing=0.055,\n        subplot_titles=(\n            "Episodic survival",\n            "Mean action-0 fraction across agents",\n            "Mean non-idle fraction across agents",\n        ),\n    )\n\n    for _, row in conditions.iterrows():\n        family = row["family"]\n        label = row["condition_label"]\n        color = colors[family]\n\n        survival_data = survival[survival["family"] == family].sort_values("step_millions")\n        if not survival_data.empty:\n            fig.add_trace(\n                go.Scatter(\n                    x=survival_data["step_millions"],\n                    y=survival_data["mean_survival_pct_smooth"],\n                    mode="lines",\n                    name=label,\n                    legendgroup=family,\n                    showlegend=True,\n                    line={"color": color, "width": 3},\n                    customdata=np.stack([\n                        survival_data["n_seeds"],\n                        survival_data["seeds"].astype(str),\n                        survival_data["metrics_used"].astype(str),\n                    ], axis=-1),\n                    hovertemplate=(\n                        f"<b>{label}</b><br>"\n                        "step=%{x:.2f}M<br>"\n                        "survival=%{y:.2f}%<br>"\n                        "seeds=%{customdata[1]}<br>"\n                        "n_seeds=%{customdata[0]}<br>"\n                        "metric=%{customdata[2]}<extra></extra>"\n                    ),\n                ),\n                row=1,\n                col=1,\n            )\n\n        for feature_group, row_idx in [("agent_action0", 2), ("agent_non_idle", 3)]:\n            metric_data = behavior[(behavior["family"] == family) & (behavior["feature_group"] == feature_group)].sort_values("step_millions")\n            if metric_data.empty:\n                continue\n            fig.add_trace(\n                go.Scatter(\n                    x=metric_data["step_millions"],\n                    y=metric_data["mean_value_smooth"],\n                    mode="lines",\n                    name=label,\n                    legendgroup=family,\n                    showlegend=False,\n                    line={"color": color, "width": 2.6},\n                    customdata=np.stack([\n                        metric_data["n_seeds"],\n                        metric_data["seeds"].astype(str),\n                        metric_data["metric_label"].astype(str),\n                    ], axis=-1),\n                    hovertemplate=(\n                        f"<b>{label}</b><br>"\n                        "step=%{x:.2f}M<br>"\n                        "%{customdata[2]}=%{y:.4f}<br>"\n                        "seeds=%{customdata[1]}<br>"\n                        "n_seeds=%{customdata[0]}<extra></extra>"\n                    ),\n                ),\n                row=row_idx,\n                col=1,\n            )\n\n    fig.update_layout(\n        title=title,\n        template="plotly_white",\n        height=940,\n        width=1450,\n        hovermode="x unified",\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 80, "r": 340, "t": 95, "b": 70},\n    )\n    fig.update_yaxes(title_text="survival (%)", range=[0, 105], row=1, col=1)\n    fig.update_yaxes(title_text="action 0 fraction", range=[0, 1], row=2, col=1)\n    fig.update_yaxes(title_text="non-idle fraction", range=[0, 1], row=3, col=1)\n    fig.update_xaxes(title_text="steps (M)", row=3, col=1)\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\nfig_survival_vs_action_ig15 = plot_survival_vs_action_behavior(\n    "intervention_gate_15",\n    "Intervention gate 15: survival vs action behavior over training",\n    "survival_vs_action_behavior_intervention_gate_15",\n)\nfig_survival_vs_action_sparse16 = plot_survival_vs_action_behavior(\n    "phase4_sparse_control_16",\n    "Phase4 sparse control 16: survival vs action behavior over training",\n    "survival_vs_action_behavior_phase4_sparse_control_16",\n)\nfig_survival_vs_action_hvg = plot_survival_vs_action_behavior(\n    "heuristic_vs_gate_s0_s1_s2",\n    "Heuristic vs gate s0/s1/s2: survival vs action behavior over training",\n    "survival_vs_action_behavior_heuristic_vs_gate_s0_s1_s2",\n)\n'
_CELL_ENTROPY_ACTION0 = 'ENTROPY_ACTION0_SMOOTH_WINDOW = 5\nENTROPY_ACTION0_FINAL_LAST_N_LOGGED = 5\n\n\ndef entropy_action0_run_rows():\n    data = FEATURE_LONG[FEATURE_LONG["feature_group"].isin(["agent_action0", "agent_entropy"])].copy()\n    if data.empty:\n        return pd.DataFrame()\n\n    group_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "feature_group",\n        "_step",\n        "step_millions",\n    ]\n    per_run = data.groupby(group_cols, dropna=False, as_index=False).agg(\n        run_mean_value=("value", "mean"),\n        run_std_agent_value=("value", "std"),\n        n_agents_observed=("entity", "nunique"),\n        agents=("entity", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    per_run["run_std_agent_value"] = per_run["run_std_agent_value"].fillna(0.0)\n    per_run["metric_label"] = per_run["feature_group"].map({\n        "agent_action0": "action 0 fraction",\n        "agent_entropy": "policy entropy",\n    })\n    return per_run.sort_values(["experiment", "control_value", "design", "family_label", "feature_group", "_step"])\n\n\ndef seed_aggregated_entropy_action0_rows():\n    if ENTROPY_ACTION0_RUN_LONG.empty:\n        return pd.DataFrame()\n\n    group_cols = [\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "feature_group",\n        "metric_label",\n        "_step",\n        "step_millions",\n    ]\n    grouped = ENTROPY_ACTION0_RUN_LONG.groupby(group_cols, dropna=False, as_index=False).agg(\n        mean_value=("run_mean_value", "mean"),\n        std_value=("run_mean_value", "std"),\n        mean_agent_spread=("run_std_agent_value", "mean"),\n        min_agents_observed=("n_agents_observed", "min"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    grouped["std_value"] = grouped["std_value"].fillna(0.0)\n    grouped["mean_agent_spread"] = grouped["mean_agent_spread"].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    grouped = grouped.sort_values(["experiment", "control_value", "design", "family_label", "feature_group", "_step"])\n\n    if ENTROPY_ACTION0_SMOOTH_WINDOW and int(ENTROPY_ACTION0_SMOOTH_WINDOW) > 1:\n        for column in ["mean_value", "std_value", "mean_agent_spread"]:\n            grouped[f"{column}_smooth"] = grouped.groupby(\n                ["experiment", "family", "feature_group"],\n                dropna=False,\n            )[column].transform(lambda values: values.rolling(int(ENTROPY_ACTION0_SMOOTH_WINDOW), min_periods=1).mean())\n    else:\n        for column in ["mean_value", "std_value", "mean_agent_spread"]:\n            grouped[f"{column}_smooth"] = grouped[column]\n    return grouped\n\n\ndef seed_aggregated_final_entropy_action0_rows():\n    if ENTROPY_ACTION0_RUN_LONG.empty:\n        return pd.DataFrame()\n\n    data = ENTROPY_ACTION0_RUN_LONG.sort_values(["run_id", "feature_group", "_step"]).copy()\n    data = data.groupby(["run_id", "feature_group"], dropna=False).tail(int(ENTROPY_ACTION0_FINAL_LAST_N_LOGGED))\n    per_run_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "feature_group",\n    ]\n    per_run = data.groupby(per_run_cols, dropna=False, as_index=False).agg(\n        final_value=("run_mean_value", "mean"),\n        final_agent_spread=("run_std_agent_value", "mean"),\n        final_step_millions=("step_millions", "max"),\n        min_agents_observed=("n_agents_observed", "min"),\n        averaged_logged_points=("run_mean_value", "count"),\n    )\n\n    index_cols = [col for col in per_run_cols if col != "feature_group"]\n    values = per_run.pivot_table(index=index_cols, columns="feature_group", values="final_value", aggfunc="mean").reset_index()\n    spreads = per_run.pivot_table(index=index_cols, columns="feature_group", values="final_agent_spread", aggfunc="mean").reset_index()\n    counts = per_run.groupby(index_cols, dropna=False, as_index=False).agg(\n        final_step_millions=("final_step_millions", "max"),\n        min_agents_observed=("min_agents_observed", "min"),\n        min_averaged_logged_points=("averaged_logged_points", "min"),\n    )\n\n    for column in ["agent_action0", "agent_entropy"]:\n        if column not in values.columns:\n            values[column] = np.nan\n        if column not in spreads.columns:\n            spreads[column] = np.nan\n    values = values.rename(columns={\n        "agent_action0": "final_action0_fraction",\n        "agent_entropy": "final_entropy",\n    })\n    spreads = spreads.rename(columns={\n        "agent_action0": "final_action0_agent_spread",\n        "agent_entropy": "final_entropy_agent_spread",\n    })\n    spread_cols = index_cols + ["final_action0_agent_spread", "final_entropy_agent_spread"]\n    per_run_wide = values.merge(spreads[spread_cols], on=index_cols, how="left").merge(counts, on=index_cols, how="left")\n    per_run_wide = per_run_wide.dropna(subset=["final_action0_fraction", "final_entropy"], how="any")\n    if per_run_wide.empty:\n        return pd.DataFrame()\n\n    grouped = per_run_wide.groupby(\n        [\n            "experiment",\n            "comparison_group",\n            "family",\n            "family_label",\n            "design",\n            "control_axis",\n            "control_value",\n            "control_label",\n            "entropy_schedule",\n        ],\n        dropna=False,\n        as_index=False,\n    ).agg(\n        mean_final_action0_fraction=("final_action0_fraction", "mean"),\n        std_final_action0_fraction=("final_action0_fraction", "std"),\n        mean_final_entropy=("final_entropy", "mean"),\n        std_final_entropy=("final_entropy", "std"),\n        mean_final_action0_agent_spread=("final_action0_agent_spread", "mean"),\n        mean_final_entropy_agent_spread=("final_entropy_agent_spread", "mean"),\n        mean_final_step_millions=("final_step_millions", "mean"),\n        min_agents_observed=("min_agents_observed", "min"),\n        min_averaged_logged_points=("min_averaged_logged_points", "min"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    for column in ["std_final_action0_fraction", "std_final_entropy"]:\n        grouped[column] = grouped[column].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    return grouped.sort_values(["experiment", "control_value", "design", "family_label"])\n\n\nENTROPY_ACTION0_RUN_LONG = entropy_action0_run_rows()\nENTROPY_ACTION0_SEED_AGG = seed_aggregated_entropy_action0_rows()\nENTROPY_ACTION0_FINAL_SEED_AGG = seed_aggregated_final_entropy_action0_rows()\n\n# print("Seed-aggregated action-0 vs entropy time series:")\n# if ENTROPY_ACTION0_SEED_AGG.empty:\n#     print("No action-0/entropy metrics found.")\n# else:\n#     display(ENTROPY_ACTION0_SEED_AGG.groupby(["experiment", "family_label", "metric_label"], dropna=False).agg(\n#         points=("_step", "count"),\n#         n_seeds=("n_seeds", "max"),\n#         seeds=("seeds", "first"),\n#         min_agents_observed=("min_agents_observed", "min"),\n#     ).reset_index())\n\n# print(f"Seed-aggregated final action-0 vs entropy, averaged over last {ENTROPY_ACTION0_FINAL_LAST_N_LOGGED} logged points:")\n# if ENTROPY_ACTION0_FINAL_SEED_AGG.empty:\n#     print("No final action-0/entropy summary available.")\n# else:\n#     display(ENTROPY_ACTION0_FINAL_SEED_AGG[[\n#         "experiment",\n#         "family_label",\n#         "mean_final_action0_fraction",\n#         "std_final_action0_fraction",\n#         "mean_final_entropy",\n#         "std_final_entropy",\n#         "n_seeds",\n#         "seeds",\n#     ]])\n\n\ndef plot_entropy_action0_timeseries(experiment, title, save_name):\n    data = ENTROPY_ACTION0_SEED_AGG[ENTROPY_ACTION0_SEED_AGG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No action-0/entropy time-series data for {experiment}")\n        return None\n\n    conditions = data[["family", "condition_label", "control_value", "design"]].drop_duplicates()\n    conditions = conditions.sort_values(["control_value", "design", "condition_label"])\n    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24\n    colors = {row.family: palette[idx % len(palette)] for idx, row in enumerate(conditions.itertuples(index=False))}\n\n    fig = make_subplots(\n        rows=2,\n        cols=1,\n        shared_xaxes=True,\n        vertical_spacing=0.075,\n        subplot_titles=(\n            "Mean action-0 fraction across agents",\n            "Mean policy entropy across agents",\n        ),\n    )\n    for _, condition in conditions.iterrows():\n        family = condition["family"]\n        label = condition["condition_label"]\n        color = colors[family]\n        for feature_group, row_idx in [("agent_action0", 1), ("agent_entropy", 2)]:\n            metric_data = data[(data["family"] == family) & (data["feature_group"] == feature_group)].sort_values("step_millions")\n            if metric_data.empty:\n                continue\n            fig.add_trace(\n                go.Scatter(\n                    x=metric_data["step_millions"],\n                    y=metric_data["mean_value_smooth"],\n                    mode="lines",\n                    name=label,\n                    legendgroup=family,\n                    showlegend=row_idx == 1,\n                    line={"color": color, "width": 3 if row_idx == 1 else 2.8},\n                    customdata=np.stack([\n                        metric_data["std_value_smooth"],\n                        metric_data["mean_agent_spread_smooth"],\n                        metric_data["n_seeds"],\n                        metric_data["seeds"].astype(str),\n                        metric_data["metric_label"].astype(str),\n                    ], axis=-1),\n                    hovertemplate=(\n                        f"<b>{label}</b><br>"\n                        "step=%{x:.2f}M<br>"\n                        "%{customdata[4]}=%{y:.4f}<br>"\n                        "std across seeds=%{customdata[0]:.4f}<br>"\n                        "mean agent spread=%{customdata[1]:.4f}<br>"\n                        "seeds=%{customdata[3]}<br>"\n                        "n_seeds=%{customdata[2]}<extra></extra>"\n                    ),\n                ),\n                row=row_idx,\n                col=1,\n            )\n\n    fig.update_layout(\n        title=title,\n        template="plotly_white",\n        height=780,\n        width=1450,\n        hovermode="x unified",\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 80, "r": 360, "t": 95, "b": 70},\n    )\n    fig.update_yaxes(title_text="action-0 fraction", range=[0, 1], row=1, col=1)\n    fig.update_yaxes(title_text="entropy", row=2, col=1)\n    fig.update_xaxes(title_text="steps (M)", row=2, col=1)\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\ndef plot_final_action0_entropy_scatter(experiment, title, save_name):\n    data = ENTROPY_ACTION0_FINAL_SEED_AGG[ENTROPY_ACTION0_FINAL_SEED_AGG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No final action-0/entropy data for {experiment}")\n        return None\n\n    data = data.sort_values(["control_value", "design", "condition_label"])\n    fig = px.scatter(\n        data,\n        x="mean_final_action0_fraction",\n        y="mean_final_entropy",\n        color="condition_label",\n        symbol="design",\n        size="n_seeds",\n        size_max=18,\n        error_x="std_final_action0_fraction",\n        error_y="std_final_entropy",\n        hover_data={\n            "condition_label": True,\n            "control_label": True,\n            "design": True,\n            "mean_final_action0_fraction": ":.4f",\n            "std_final_action0_fraction": ":.4f",\n            "mean_final_entropy": ":.4f",\n            "std_final_entropy": ":.4f",\n            "mean_final_action0_agent_spread": ":.4f",\n            "mean_final_entropy_agent_spread": ":.4f",\n            "n_seeds": True,\n            "seeds": True,\n            "min_averaged_logged_points": True,\n        },\n        labels={\n            "mean_final_action0_fraction": "mean final action-0 fraction",\n            "mean_final_entropy": "mean final entropy",\n            "condition_label": "condition",\n        },\n        title=title,\n    )\n    fig.update_xaxes(range=[0, 1], title_text="action-0 fraction")\n    fig.update_yaxes(title_text="policy entropy")\n    fig.update_traces(marker={"line": {"width": 1, "color": "white"}})\n    fig.update_layout(\n        template="plotly_white",\n        height=640,\n        width=1450,\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 80, "r": 360, "t": 90, "b": 80},\n    )\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\nfig_entropy_action0_timeseries_ig15 = plot_entropy_action0_timeseries(\n    "intervention_gate_15",\n    "Intervention gate 15: action-0 fraction vs entropy over training",\n    "entropy_action0_timeseries_intervention_gate_15",\n)\nfig_entropy_action0_timeseries_sparse16 = plot_entropy_action0_timeseries(\n    "phase4_sparse_control_16",\n    "Phase4 sparse control 16: action-0 fraction vs entropy over training",\n    "entropy_action0_timeseries_phase4_sparse_control_16",\n)\nfig_entropy_action0_timeseries_hvg = plot_entropy_action0_timeseries(\n    "heuristic_vs_gate_s0_s1_s2",\n    "Heuristic vs gate s0/s1/s2: action-0 fraction vs entropy over training",\n    "entropy_action0_timeseries_heuristic_vs_gate_s0_s1_s2",\n)\nfig_entropy_action0_final_scatter_ig15 = plot_final_action0_entropy_scatter(\n    "intervention_gate_15",\n    f"Intervention gate 15: final action-0 confidence map (mean of last {ENTROPY_ACTION0_FINAL_LAST_N_LOGGED} logged)",\n    "final_entropy_action0_confidence_intervention_gate_15",\n)\nfig_entropy_action0_final_scatter_sparse16 = plot_final_action0_entropy_scatter(\n    "phase4_sparse_control_16",\n    f"Phase4 sparse control 16: final action-0 confidence map (mean of last {ENTROPY_ACTION0_FINAL_LAST_N_LOGGED} logged)",\n    "final_entropy_action0_confidence_phase4_sparse_control_16",\n)\nfig_entropy_action0_final_scatter_hvg = plot_final_action0_entropy_scatter(\n    "heuristic_vs_gate_s0_s1_s2",\n    f"Heuristic vs gate s0/s1/s2: final action-0 confidence map (mean of last {ENTROPY_ACTION0_FINAL_LAST_N_LOGGED} logged)",\n    "final_entropy_action0_confidence_heuristic_vs_gate_s0_s1_s2",\n)\n'
_CELL_AGENT_IMBALANCE = 'AGENT_IMBALANCE_SMOOTH_WINDOW = 5\nAGENT_IMBALANCE_FINAL_LAST_N_LOGGED = 5\n\n\ndef agent_imbalance_run_rows():\n    data = FEATURE_LONG[FEATURE_LONG["feature_group"] == "agent_non_idle"].copy()\n    if data.empty:\n        return pd.DataFrame()\n\n    group_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "_step",\n        "step_millions",\n    ]\n    grouped = data.groupby(group_cols, dropna=False, as_index=False).agg(\n        min_non_idle=("value", "min"),\n        max_non_idle=("value", "max"),\n        mean_non_idle=("value", "mean"),\n        std_agent_non_idle=("value", "std"),\n        n_agents_observed=("entity", "nunique"),\n        agents=("entity", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    grouped["std_agent_non_idle"] = grouped["std_agent_non_idle"].fillna(0.0)\n    grouped["non_idle_imbalance"] = grouped["max_non_idle"] - grouped["min_non_idle"]\n    return grouped.sort_values(["experiment", "control_value", "design", "family_label", "_step"])\n\n\ndef seed_aggregated_agent_imbalance_rows():\n    if AGENT_IMBALANCE_RUN_LONG.empty:\n        return pd.DataFrame()\n\n    group_cols = [\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "_step",\n        "step_millions",\n    ]\n    grouped = AGENT_IMBALANCE_RUN_LONG.groupby(group_cols, dropna=False, as_index=False).agg(\n        mean_imbalance=("non_idle_imbalance", "mean"),\n        std_imbalance=("non_idle_imbalance", "std"),\n        mean_non_idle=("mean_non_idle", "mean"),\n        mean_min_non_idle=("min_non_idle", "mean"),\n        mean_max_non_idle=("max_non_idle", "mean"),\n        min_agents_observed=("n_agents_observed", "min"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    grouped["std_imbalance"] = grouped["std_imbalance"].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    grouped = grouped.sort_values(["experiment", "control_value", "design", "family_label", "_step"])\n\n    if AGENT_IMBALANCE_SMOOTH_WINDOW and int(AGENT_IMBALANCE_SMOOTH_WINDOW) > 1:\n        for column in ["mean_imbalance", "std_imbalance", "mean_non_idle", "mean_min_non_idle", "mean_max_non_idle"]:\n            grouped[f"{column}_smooth"] = grouped.groupby(\n                ["experiment", "family"],\n                dropna=False,\n            )[column].transform(lambda values: values.rolling(int(AGENT_IMBALANCE_SMOOTH_WINDOW), min_periods=1).mean())\n    else:\n        for column in ["mean_imbalance", "std_imbalance", "mean_non_idle", "mean_min_non_idle", "mean_max_non_idle"]:\n            grouped[f"{column}_smooth"] = grouped[column]\n    return grouped\n\n\ndef seed_aggregated_final_agent_imbalance_rows():\n    if AGENT_IMBALANCE_RUN_LONG.empty:\n        return pd.DataFrame()\n\n    data = AGENT_IMBALANCE_RUN_LONG.sort_values(["run_id", "_step"]).copy()\n    data = data.groupby(["run_id"], dropna=False).tail(int(AGENT_IMBALANCE_FINAL_LAST_N_LOGGED))\n    per_run_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n    ]\n    per_run = data.groupby(per_run_cols, dropna=False, as_index=False).agg(\n        final_imbalance=("non_idle_imbalance", "mean"),\n        final_mean_non_idle=("mean_non_idle", "mean"),\n        final_min_non_idle=("min_non_idle", "mean"),\n        final_max_non_idle=("max_non_idle", "mean"),\n        final_step_millions=("step_millions", "max"),\n        min_agents_observed=("n_agents_observed", "min"),\n        averaged_logged_points=("non_idle_imbalance", "count"),\n    )\n\n    grouped = per_run.groupby(\n        [\n            "experiment",\n            "comparison_group",\n            "family",\n            "family_label",\n            "design",\n            "control_axis",\n            "control_value",\n            "control_label",\n            "entropy_schedule",\n        ],\n        dropna=False,\n        as_index=False,\n    ).agg(\n        mean_final_imbalance=("final_imbalance", "mean"),\n        std_final_imbalance=("final_imbalance", "std"),\n        mean_final_non_idle=("final_mean_non_idle", "mean"),\n        mean_final_min_non_idle=("final_min_non_idle", "mean"),\n        mean_final_max_non_idle=("final_max_non_idle", "mean"),\n        mean_final_step_millions=("final_step_millions", "mean"),\n        min_agents_observed=("min_agents_observed", "min"),\n        min_averaged_logged_points=("averaged_logged_points", "min"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    grouped["std_final_imbalance"] = grouped["std_final_imbalance"].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    return grouped.sort_values(["experiment", "control_value", "design", "family_label"])\n\n\nAGENT_IMBALANCE_RUN_LONG = agent_imbalance_run_rows()\nAGENT_IMBALANCE_SEED_AGG = seed_aggregated_agent_imbalance_rows()\nAGENT_IMBALANCE_FINAL_SEED_AGG = seed_aggregated_final_agent_imbalance_rows()\n\n# print("Seed-aggregated non-idle imbalance time series:")\n# if AGENT_IMBALANCE_SEED_AGG.empty:\n#     print("No agent non-idle metrics found for imbalance plots.")\n# else:\n#     display(AGENT_IMBALANCE_SEED_AGG.groupby(["experiment", "family_label"], dropna=False).agg(\n#         points=("_step", "count"),\n#         n_seeds=("n_seeds", "max"),\n#         seeds=("seeds", "first"),\n#         min_agents_observed=("min_agents_observed", "min"),\n#     ).reset_index())\n\n# print(f"Seed-aggregated final non-idle imbalance, averaged over last {AGENT_IMBALANCE_FINAL_LAST_N_LOGGED} logged points:")\n# if AGENT_IMBALANCE_FINAL_SEED_AGG.empty:\n#     print("No final agent imbalance summary available.")\n# else:\n#     display(AGENT_IMBALANCE_FINAL_SEED_AGG[[\n#         "experiment",\n#         "family_label",\n#         "mean_final_imbalance",\n#         "std_final_imbalance",\n#         "mean_final_non_idle",\n#         "mean_final_min_non_idle",\n#         "mean_final_max_non_idle",\n#         "n_seeds",\n#         "seeds",\n#     ]])\n\n\ndef plot_agent_imbalance_timeseries(experiment, title, save_name):\n    data = AGENT_IMBALANCE_SEED_AGG[AGENT_IMBALANCE_SEED_AGG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No agent imbalance time-series data for {experiment}")\n        return None\n\n    conditions = data[["family", "condition_label", "control_value", "design"]].drop_duplicates()\n    conditions = conditions.sort_values(["control_value", "design", "condition_label"])\n    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24\n    colors = {row.family: palette[idx % len(palette)] for idx, row in enumerate(conditions.itertuples(index=False))}\n\n    fig = go.Figure()\n    for _, condition in conditions.iterrows():\n        family = condition["family"]\n        label = condition["condition_label"]\n        condition_data = data[data["family"] == family].sort_values("step_millions")\n        if condition_data.empty:\n            continue\n        fig.add_trace(\n            go.Scatter(\n                x=condition_data["step_millions"],\n                y=condition_data["mean_imbalance_smooth"],\n                mode="lines",\n                name=label,\n                line={"color": colors[family], "width": 3},\n                customdata=np.stack([\n                    condition_data["std_imbalance_smooth"],\n                    condition_data["mean_non_idle_smooth"],\n                    condition_data["mean_min_non_idle_smooth"],\n                    condition_data["mean_max_non_idle_smooth"],\n                    condition_data["n_seeds"],\n                    condition_data["seeds"].astype(str),\n                ], axis=-1),\n                hovertemplate=(\n                    f"<b>{label}</b><br>"\n                    "step=%{x:.2f}M<br>"\n                    "imbalance=%{y:.4f}<br>"\n                    "std=%{customdata[0]:.4f}<br>"\n                    "mean non-idle=%{customdata[1]:.4f}<br>"\n                    "min agent=%{customdata[2]:.4f}<br>"\n                    "max agent=%{customdata[3]:.4f}<br>"\n                    "seeds=%{customdata[5]}<br>"\n                    "n_seeds=%{customdata[4]}<extra></extra>"\n                ),\n            )\n        )\n\n    fig.update_layout(\n        title=title,\n        template="plotly_white",\n        height=620,\n        width=1450,\n        hovermode="x unified",\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 80, "r": 360, "t": 90, "b": 70},\n    )\n    fig.update_yaxes(title_text="max(non-idle) - min(non-idle)", range=[0, 1])\n    fig.update_xaxes(title_text="steps (M)")\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\ndef _final_agent_imbalance_seed_points(experiment):\n    data = AGENT_IMBALANCE_RUN_LONG[AGENT_IMBALANCE_RUN_LONG["experiment"] == experiment].copy()\n    if data.empty:\n        return pd.DataFrame()\n    data = data.sort_values(["run_id", "_step"])\n    data = data.groupby(["run_id"], dropna=False).tail(int(AGENT_IMBALANCE_FINAL_LAST_N_LOGGED))\n    per_run_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n    ]\n    seed_points = data.groupby(per_run_cols, dropna=False, as_index=False).agg(\n        seed_final_imbalance=("non_idle_imbalance", "mean"),\n        seed_final_non_idle=("mean_non_idle", "mean"),\n        seed_final_min_non_idle=("min_non_idle", "mean"),\n        seed_final_max_non_idle=("max_non_idle", "mean"),\n        final_step_millions=("step_millions", "max"),\n        averaged_logged_points=("non_idle_imbalance", "count"),\n    )\n    seed_points["condition_label"] = seed_points["family_label"]\n    return seed_points.sort_values(["control_value", "design", "condition_label", "seed"])\n\n\ndef plot_final_agent_imbalance(experiment, title, save_name):\n    data = AGENT_IMBALANCE_FINAL_SEED_AGG[AGENT_IMBALANCE_FINAL_SEED_AGG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No final agent imbalance data for {experiment}")\n        return None\n\n    data = data.sort_values(["control_value", "design", "condition_label"])\n    fig = px.bar(\n        data,\n        x="condition_label",\n        y="mean_final_imbalance",\n        color="condition_label",\n        error_y="std_final_imbalance",\n        hover_data={\n            "condition_label": True,\n            "control_label": True,\n            "design": True,\n            "mean_final_imbalance": ":.4f",\n            "std_final_imbalance": ":.4f",\n            "mean_final_non_idle": ":.4f",\n            "mean_final_min_non_idle": ":.4f",\n            "mean_final_max_non_idle": ":.4f",\n            "n_seeds": True,\n            "seeds": True,\n            "min_averaged_logged_points": True,\n        },\n        labels={\n            "condition_label": "condition",\n            "mean_final_imbalance": "mean final imbalance",\n        },\n        title=title,\n    )\n\n    seed_points = _final_agent_imbalance_seed_points(experiment)\n    add_seed_point_overlay(\n        fig,\n        seed_points,\n        x_col="condition_label",\n        y_col="seed_final_imbalance",\n        group_col="condition_label",\n        customdata_cols=[\n            "run_name",\n            "seed",\n            "condition_label",\n            "seed_final_non_idle",\n            "seed_final_min_non_idle",\n            "seed_final_max_non_idle",\n            "averaged_logged_points",\n        ],\n        hovertemplate=(\n            "seed run=%{customdata[0]}<br>"\n            "seed=%{customdata[1]}<br>"\n            "condition=%{customdata[2]}<br>"\n            "seed imbalance=%{y:.4f}<br>"\n            "mean non-idle=%{customdata[3]:.4f}<br>"\n            "min agent=%{customdata[4]:.4f}<br>"\n            "max agent=%{customdata[5]:.4f}<br>"\n            "logged points=%{customdata[6]}<extra></extra>"\n        ),\n    )\n\n    fig.update_yaxes(title_text="max(non-idle) - min(non-idle)", range=[0, 1])\n    fig.update_xaxes(tickangle=25)\n    fig.update_layout(\n        template="plotly_white",\n        height=640,\n        width=1450,\n        showlegend=False,\n        bargap=0.22,\n        margin={"l": 80, "r": 80, "t": 90, "b": 150},\n    )\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\nfig_agent_imbalance_timeseries_ig15 = plot_agent_imbalance_timeseries(\n    "intervention_gate_15",\n    "Intervention gate 15: agent non-idle imbalance over training",\n    "agent_non_idle_imbalance_timeseries_intervention_gate_15",\n)\nfig_agent_imbalance_timeseries_sparse16 = plot_agent_imbalance_timeseries(\n    "phase4_sparse_control_16",\n    "Phase4 sparse control 16: agent non-idle imbalance over training",\n    "agent_non_idle_imbalance_timeseries_phase4_sparse_control_16",\n)\nfig_agent_imbalance_timeseries_hvg = plot_agent_imbalance_timeseries(\n    "heuristic_vs_gate_s0_s1_s2",\n    "Heuristic vs gate s0/s1/s2: agent non-idle imbalance over training",\n    "agent_non_idle_imbalance_timeseries_heuristic_vs_gate_s0_s1_s2",\n)\nfig_agent_imbalance_final_ig15 = plot_final_agent_imbalance(\n    "intervention_gate_15",\n    f"Intervention gate 15: final agent non-idle imbalance (mean of last {AGENT_IMBALANCE_FINAL_LAST_N_LOGGED} logged)",\n    "final_agent_non_idle_imbalance_intervention_gate_15",\n)\nfig_agent_imbalance_final_sparse16 = plot_final_agent_imbalance(\n    "phase4_sparse_control_16",\n    f"Phase4 sparse control 16: final agent non-idle imbalance (mean of last {AGENT_IMBALANCE_FINAL_LAST_N_LOGGED} logged)",\n    "final_agent_non_idle_imbalance_phase4_sparse_control_16",\n)\nfig_agent_imbalance_final_hvg = plot_final_agent_imbalance(\n    "heuristic_vs_gate_s0_s1_s2",\n    f"Heuristic vs gate s0/s1/s2: final agent non-idle imbalance (mean of last {AGENT_IMBALANCE_FINAL_LAST_N_LOGGED} logged)",\n    "final_agent_non_idle_imbalance_heuristic_vs_gate_s0_s1_s2",\n)\n'
_CELL_JOINT_COORDINATION = 'JOINT_COORDINATION_SMOOTH_WINDOW = 5\nJOINT_COORDINATION_FINAL_LAST_N_LOGGED = 5\nJOINT_AGENT_COUNT_LABELS = {\n    0: "0 agents act",\n    1: "1 agent acts",\n    2: "2 agents act",\n    3: "3 agents act",\n}\nJOINT_AGENT_COUNT_ORDER = list(JOINT_AGENT_COUNT_LABELS.values())\n\n\ndef joint_coordination_run_rows():\n    data = FEATURE_LONG[FEATURE_LONG["feature_group"] == "joint_non_idle"].copy()\n    if data.empty:\n        return pd.DataFrame()\n\n    data["n_agents_act"] = pd.to_numeric(data["entity"], errors="coerce")\n    data = data.dropna(subset=["n_agents_act", "value", "_step"])\n    data["n_agents_act"] = data["n_agents_act"].astype(int)\n    data = data[data["n_agents_act"].isin(JOINT_AGENT_COUNT_LABELS.keys())].copy()\n    data["coordination_label"] = data["n_agents_act"].map(JOINT_AGENT_COUNT_LABELS)\n    data["coordination_label"] = pd.Categorical(\n        data["coordination_label"],\n        categories=JOINT_AGENT_COUNT_ORDER,\n        ordered=True,\n    )\n    return data.sort_values(["experiment", "control_value", "design", "family_label", "n_agents_act", "_step"])\n\n\ndef seed_aggregated_joint_coordination_rows():\n    if JOINT_COORDINATION_RUN_LONG.empty:\n        return pd.DataFrame()\n\n    group_cols = [\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "n_agents_act",\n        "coordination_label",\n        "_step",\n        "step_millions",\n    ]\n    grouped = JOINT_COORDINATION_RUN_LONG.groupby(\n        group_cols,\n        dropna=False,\n        observed=True,\n        as_index=False,\n    ).agg(\n        mean_fraction=("value", "mean"),\n        std_fraction=("value", "std"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    grouped["std_fraction"] = grouped["std_fraction"].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    grouped["coordination_label"] = pd.Categorical(\n        grouped["coordination_label"],\n        categories=JOINT_AGENT_COUNT_ORDER,\n        ordered=True,\n    )\n    grouped = grouped.sort_values(["experiment", "control_value", "design", "family_label", "n_agents_act", "_step"])\n\n    if JOINT_COORDINATION_SMOOTH_WINDOW and int(JOINT_COORDINATION_SMOOTH_WINDOW) > 1:\n        grouped["mean_fraction_smooth"] = grouped.groupby(\n            ["experiment", "family", "n_agents_act"],\n            dropna=False,\n            observed=True,\n        )["mean_fraction"].transform(lambda values: values.rolling(int(JOINT_COORDINATION_SMOOTH_WINDOW), min_periods=1).mean())\n        grouped["std_fraction_smooth"] = grouped.groupby(\n            ["experiment", "family", "n_agents_act"],\n            dropna=False,\n            observed=True,\n        )["std_fraction"].transform(lambda values: values.rolling(int(JOINT_COORDINATION_SMOOTH_WINDOW), min_periods=1).mean())\n    else:\n        grouped["mean_fraction_smooth"] = grouped["mean_fraction"]\n        grouped["std_fraction_smooth"] = grouped["std_fraction"]\n    return grouped\n\n\ndef seed_aggregated_final_joint_coordination_rows():\n    if JOINT_COORDINATION_RUN_LONG.empty:\n        return pd.DataFrame()\n\n    data = JOINT_COORDINATION_RUN_LONG.sort_values(["run_id", "n_agents_act", "_step"]).copy()\n    data = data.groupby(["run_id", "n_agents_act"], dropna=False, observed=True).tail(int(JOINT_COORDINATION_FINAL_LAST_N_LOGGED))\n    per_run_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "n_agents_act",\n        "coordination_label",\n    ]\n    per_run = data.groupby(per_run_cols, dropna=False, observed=True, as_index=False).agg(\n        final_fraction=("value", "mean"),\n        final_step_millions=("step_millions", "max"),\n        averaged_logged_points=("value", "count"),\n    )\n\n    grouped = per_run.groupby(\n        [\n            "experiment",\n            "comparison_group",\n            "family",\n            "family_label",\n            "design",\n            "control_axis",\n            "control_value",\n            "control_label",\n            "entropy_schedule",\n            "n_agents_act",\n            "coordination_label",\n        ],\n        dropna=False,\n        observed=True,\n        as_index=False,\n    ).agg(\n        mean_final_fraction=("final_fraction", "mean"),\n        std_final_fraction=("final_fraction", "std"),\n        mean_final_step_millions=("final_step_millions", "mean"),\n        min_averaged_logged_points=("averaged_logged_points", "min"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    grouped["std_final_fraction"] = grouped["std_final_fraction"].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    grouped["coordination_label"] = pd.Categorical(\n        grouped["coordination_label"],\n        categories=JOINT_AGENT_COUNT_ORDER,\n        ordered=True,\n    )\n    return grouped.sort_values(["experiment", "control_value", "design", "family_label", "n_agents_act"])\n\n\ndef joint_coordination_coverage_rows():\n    keep_cols = [\n        "experiment",\n        "run_name",\n        "run_id",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_label",\n    ]\n    if selected_runs.empty:\n        return pd.DataFrame(columns=keep_cols + ["has_joint_coordination", "joint_coordination_rows", "joint_coordination_bins"])\n\n    if JOINT_COORDINATION_RUN_LONG.empty:\n        available = pd.DataFrame(columns=["run_id", "run_name", "joint_coordination_rows", "joint_coordination_bins"])\n    else:\n        available = JOINT_COORDINATION_RUN_LONG.groupby(["run_id", "run_name"], dropna=False, as_index=False).agg(\n            joint_coordination_rows=("value", "count"),\n            joint_coordination_bins=("coordination_label", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n            first_step_millions=("step_millions", "min"),\n            last_step_millions=("step_millions", "max"),\n        )\n    out = selected_runs[keep_cols].merge(available, on=["run_id", "run_name"], how="left")\n    out["joint_coordination_rows"] = out["joint_coordination_rows"].fillna(0).astype(int)\n    out["has_joint_coordination"] = out["joint_coordination_rows"] > 0\n    return out.sort_values(["experiment", "family_label", "seed"])\n\n\nJOINT_COORDINATION_RUN_LONG = joint_coordination_run_rows()\nJOINT_COORDINATION_SEED_AGG = seed_aggregated_joint_coordination_rows()\nJOINT_COORDINATION_FINAL_SEED_AGG = seed_aggregated_final_joint_coordination_rows()\nJOINT_COORDINATION_COVERAGE = joint_coordination_coverage_rows()\n\n# print("Joint coordination metric coverage for cached runs:")\n# if JOINT_COORDINATION_COVERAGE.empty:\n#     print("No cached runs found in the selected cache.")\n# else:\n#     display(JOINT_COORDINATION_COVERAGE[[\n#         "experiment",\n#         "run_name",\n#         "family_label",\n#         "seed",\n#         "has_joint_coordination",\n#         "joint_coordination_rows",\n#         "joint_coordination_bins",\n#     ]])\n\n# missing_joint_coordination = JOINT_COORDINATION_COVERAGE[~JOINT_COORDINATION_COVERAGE["has_joint_coordination"]]\n# if not missing_joint_coordination.empty:\n#     print("Cached runs with no non-empty joint coordination rows:")\n#     display(missing_joint_coordination[["experiment", "run_name", "family_label", "seed"]])\n\n# print("Seed-aggregated joint action coordination time series:")\n# if JOINT_COORDINATION_SEED_AGG.empty:\n#     print("No joint non-idle count metrics found.")\n# else:\n#     display(JOINT_COORDINATION_SEED_AGG.groupby(["experiment", "family_label", "coordination_label"], dropna=False, observed=True).agg(\n#         points=("_step", "count"),\n#         n_seeds=("n_seeds", "max"),\n#         seeds=("seeds", "first"),\n#     ).reset_index())\n\n# print(f"Seed-aggregated final joint action coordination, averaged over last {JOINT_COORDINATION_FINAL_LAST_N_LOGGED} logged points:")\n# if JOINT_COORDINATION_FINAL_SEED_AGG.empty:\n#     print("No final joint coordination summary available.")\n# else:\n#     display(JOINT_COORDINATION_FINAL_SEED_AGG[[\n#         "experiment",\n#         "family_label",\n#         "coordination_label",\n#         "mean_final_fraction",\n#         "std_final_fraction",\n#         "n_seeds",\n#         "seeds",\n#     ]])\n\n\ndef plot_joint_coordination_timeseries(experiment, title, save_name):\n    data = JOINT_COORDINATION_SEED_AGG[JOINT_COORDINATION_SEED_AGG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No joint coordination time-series data for {experiment}")\n        return None\n\n    counts = [count for count in JOINT_AGENT_COUNT_LABELS if count in set(data["n_agents_act"].astype(int))]\n    conditions = data[["family", "condition_label", "control_value", "design"]].drop_duplicates()\n    conditions = conditions.sort_values(["control_value", "design", "condition_label"])\n    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24\n    colors = {row.family: palette[idx % len(palette)] for idx, row in enumerate(conditions.itertuples(index=False))}\n\n    fig = make_subplots(\n        rows=len(counts),\n        cols=1,\n        shared_xaxes=True,\n        vertical_spacing=0.055,\n        subplot_titles=[JOINT_AGENT_COUNT_LABELS[count] for count in counts],\n    )\n\n    for row_idx, count in enumerate(counts, start=1):\n        count_data = data[data["n_agents_act"].astype(int) == int(count)]\n        for _, condition in conditions.iterrows():\n            family = condition["family"]\n            label = condition["condition_label"]\n            condition_data = count_data[count_data["family"] == family].sort_values("step_millions")\n            if condition_data.empty:\n                continue\n            fig.add_trace(\n                go.Scatter(\n                    x=condition_data["step_millions"],\n                    y=condition_data["mean_fraction_smooth"],\n                    mode="lines",\n                    name=label,\n                    legendgroup=family,\n                    showlegend=row_idx == 1,\n                    line={"color": colors[family], "width": 2.8},\n                    customdata=np.stack([\n                        condition_data["n_seeds"],\n                        condition_data["seeds"].astype(str),\n                        condition_data["std_fraction_smooth"],\n                    ], axis=-1),\n                    hovertemplate=(\n                        f"<b>{label}</b><br>"\n                        f"coordination={JOINT_AGENT_COUNT_LABELS[count]}<br>"\n                        "step=%{x:.2f}M<br>"\n                        "fraction=%{y:.4f}<br>"\n                        "std=%{customdata[2]:.4f}<br>"\n                        "seeds=%{customdata[1]}<br>"\n                        "n_seeds=%{customdata[0]}<extra></extra>"\n                    ),\n                ),\n                row=row_idx,\n                col=1,\n            )\n        fig.update_yaxes(title_text="fraction", range=[0, 1], row=row_idx, col=1)\n\n    fig.update_layout(\n        title=title,\n        template="plotly_white",\n        height=1080,\n        width=1450,\n        hovermode="x unified",\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 80, "r": 360, "t": 100, "b": 70},\n    )\n    fig.update_xaxes(title_text="steps (M)", row=len(counts), col=1)\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\ndef _final_joint_coordination_seed_points(experiment):\n    data = JOINT_COORDINATION_RUN_LONG[JOINT_COORDINATION_RUN_LONG["experiment"] == experiment].copy()\n    if data.empty:\n        return pd.DataFrame()\n    data = data.sort_values(["run_id", "n_agents_act", "_step"])\n    data = data.groupby(["run_id", "n_agents_act"], dropna=False, observed=True).tail(int(JOINT_COORDINATION_FINAL_LAST_N_LOGGED))\n    per_run_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "n_agents_act",\n        "coordination_label",\n    ]\n    seed_points = data.groupby(per_run_cols, dropna=False, observed=True, as_index=False).agg(\n        seed_final_fraction=("value", "mean"),\n        final_step_millions=("step_millions", "max"),\n        averaged_logged_points=("value", "count"),\n    )\n    seed_points["condition_label"] = seed_points["family_label"]\n    seed_points["coordination_label"] = pd.Categorical(\n        seed_points["coordination_label"],\n        categories=JOINT_AGENT_COUNT_ORDER,\n        ordered=True,\n    )\n    return seed_points.sort_values(["n_agents_act", "control_value", "design", "condition_label", "seed"])\n\n\ndef plot_final_joint_coordination(experiment, title, save_name):\n    data = JOINT_COORDINATION_FINAL_SEED_AGG[JOINT_COORDINATION_FINAL_SEED_AGG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No final joint coordination data for {experiment}")\n        return None\n\n    data["coordination_label"] = pd.Categorical(\n        data["coordination_label"],\n        categories=JOINT_AGENT_COUNT_ORDER,\n        ordered=True,\n    )\n    fig = px.bar(\n        data.sort_values(["n_agents_act", "control_value", "design", "condition_label"]),\n        x="coordination_label",\n        y="mean_final_fraction",\n        color="condition_label",\n        error_y="std_final_fraction",\n        barmode="group",\n        hover_data={\n            "condition_label": True,\n            "control_label": True,\n            "design": True,\n            "mean_final_fraction": ":.4f",\n            "std_final_fraction": ":.4f",\n            "n_seeds": True,\n            "seeds": True,\n            "min_averaged_logged_points": True,\n        },\n        labels={\n            "coordination_label": "agents acting in same step",\n            "mean_final_fraction": "mean final fraction",\n            "condition_label": "condition",\n        },\n        title=title,\n    )\n\n    seed_points = _final_joint_coordination_seed_points(experiment)\n    add_seed_point_overlay(\n        fig,\n        seed_points,\n        x_col="coordination_label",\n        y_col="seed_final_fraction",\n        group_col="condition_label",\n        customdata_cols=[\n            "run_name",\n            "seed",\n            "condition_label",\n            "coordination_label",\n            "seed_final_fraction",\n            "averaged_logged_points",\n        ],\n        hovertemplate=(\n            "seed run=%{customdata[0]}<br>"\n            "seed=%{customdata[1]}<br>"\n            "condition=%{customdata[2]}<br>"\n            "coordination=%{customdata[3]}<br>"\n            "seed fraction=%{y:.4f}<br>"\n            "logged points=%{customdata[5]}<extra></extra>"\n        ),\n    )\n\n    fig.update_yaxes(title_text="fraction of environment steps", range=[0, 1])\n    fig.update_layout(\n        template="plotly_white",\n        height=640,\n        width=1450,\n        bargap=0.18,\n        bargroupgap=0.04,\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 80, "r": 360, "t": 90, "b": 80},\n    )\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\nfig_joint_coordination_timeseries_ig15 = plot_joint_coordination_timeseries(\n    "intervention_gate_15",\n    "Intervention gate 15: joint action coordination over training",\n    "joint_action_coordination_timeseries_intervention_gate_15",\n)\nfig_joint_coordination_timeseries_sparse16 = plot_joint_coordination_timeseries(\n    "phase4_sparse_control_16",\n    "Phase4 sparse control 16: joint action coordination over training",\n    "joint_action_coordination_timeseries_phase4_sparse_control_16",\n)\nfig_joint_coordination_timeseries_hvg = plot_joint_coordination_timeseries(\n    "heuristic_vs_gate_s0_s1_s2",\n    "Heuristic vs gate s0/s1/s2: joint action coordination over training",\n    "joint_action_coordination_timeseries_heuristic_vs_gate_s0_s1_s2",\n)\nfig_joint_coordination_final_ig15 = plot_final_joint_coordination(\n    "intervention_gate_15",\n    f"Intervention gate 15: final joint action coordination (mean of last {JOINT_COORDINATION_FINAL_LAST_N_LOGGED} logged)",\n    "final_joint_action_coordination_intervention_gate_15",\n)\nfig_joint_coordination_final_sparse16 = plot_final_joint_coordination(\n    "phase4_sparse_control_16",\n    f"Phase4 sparse control 16: final joint action coordination (mean of last {JOINT_COORDINATION_FINAL_LAST_N_LOGGED} logged)",\n    "final_joint_action_coordination_phase4_sparse_control_16",\n)\nfig_joint_coordination_final_hvg = plot_final_joint_coordination(\n    "heuristic_vs_gate_s0_s1_s2",\n    f"Heuristic vs gate s0/s1/s2: final joint action coordination (mean of last {JOINT_COORDINATION_FINAL_LAST_N_LOGGED} logged)",\n    "final_joint_action_coordination_heuristic_vs_gate_s0_s1_s2",\n)\n'
_CELL_GATE_CALIBRATION = 'GATE_CALIBRATION_SMOOTH_WINDOW = 5\nGATE_CALIBRATION_FINAL_LAST_N_LOGGED = 5\n\n\ndef gate_calibration_run_rows():\n    gate = FEATURE_LONG[FEATURE_LONG["feature_group"].isin(["gate_prob_intervene", "gate_intervene_frac"])].copy()\n    if gate.empty:\n        return pd.DataFrame()\n\n    gate = gate[\n        gate["design"].astype(str).eq("gated")\n        | gate["gate_enabled"].map(lambda value: value is True).fillna(False)\n    ].copy()\n    if gate.empty:\n        return pd.DataFrame()\n\n    index_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "gate_enabled",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "entity",\n        "_step",\n        "step_millions",\n    ]\n    wide = gate.pivot_table(\n        index=index_cols,\n        columns="feature_group",\n        values="value",\n        aggfunc="mean",\n    ).reset_index()\n    wide.columns.name = None\n    for column in ["gate_prob_intervene", "gate_intervene_frac"]:\n        if column not in wide.columns:\n            wide[column] = np.nan\n    wide = wide.dropna(subset=["gate_prob_intervene", "gate_intervene_frac"], how="all")\n    if wide.empty:\n        return pd.DataFrame()\n\n    wide["prob_minus_actual"] = wide["gate_prob_intervene"] - wide["gate_intervene_frac"]\n    wide["actual_minus_prob"] = wide["gate_intervene_frac"] - wide["gate_prob_intervene"]\n    wide["agent"] = wide["entity"].astype(str)\n    wide["agent"] = pd.Categorical(wide["agent"], categories=["agent_0", "agent_1", "agent_2"], ordered=True)\n    return wide.sort_values(["experiment", "control_value", "design", "family_label", "agent", "_step"])\n\n\ndef seed_aggregated_gate_calibration_rows():\n    if GATE_CALIBRATION_RUN_LONG.empty:\n        return pd.DataFrame()\n\n    group_cols = [\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "entity",\n        "agent",\n        "_step",\n        "step_millions",\n    ]\n    grouped = GATE_CALIBRATION_RUN_LONG.groupby(\n        group_cols,\n        dropna=False,\n        observed=True,\n        as_index=False,\n    ).agg(\n        mean_gate_prob=("gate_prob_intervene", "mean"),\n        std_gate_prob=("gate_prob_intervene", "std"),\n        mean_intervene_frac=("gate_intervene_frac", "mean"),\n        std_intervene_frac=("gate_intervene_frac", "std"),\n        mean_prob_minus_actual=("prob_minus_actual", "mean"),\n        std_prob_minus_actual=("prob_minus_actual", "std"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    for column in ["std_gate_prob", "std_intervene_frac", "std_prob_minus_actual"]:\n        grouped[column] = grouped[column].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    grouped["agent"] = pd.Categorical(grouped["agent"].astype(str), categories=["agent_0", "agent_1", "agent_2"], ordered=True)\n    grouped = grouped.sort_values(["experiment", "control_value", "design", "family_label", "agent", "_step"])\n\n    smooth_cols = [\n        "mean_gate_prob",\n        "std_gate_prob",\n        "mean_intervene_frac",\n        "std_intervene_frac",\n        "mean_prob_minus_actual",\n        "std_prob_minus_actual",\n    ]\n    if GATE_CALIBRATION_SMOOTH_WINDOW and int(GATE_CALIBRATION_SMOOTH_WINDOW) > 1:\n        for column in smooth_cols:\n            grouped[f"{column}_smooth"] = grouped.groupby(\n                ["experiment", "family", "entity"],\n                dropna=False,\n                observed=True,\n            )[column].transform(lambda values: values.rolling(int(GATE_CALIBRATION_SMOOTH_WINDOW), min_periods=1).mean())\n    else:\n        for column in smooth_cols:\n            grouped[f"{column}_smooth"] = grouped[column]\n    return grouped\n\n\ndef seed_aggregated_final_gate_calibration_rows():\n    if GATE_CALIBRATION_RUN_LONG.empty:\n        return pd.DataFrame()\n\n    data = GATE_CALIBRATION_RUN_LONG.sort_values(["run_id", "agent", "_step"]).copy()\n    data = data.groupby(["run_id", "agent"], dropna=False, observed=True).tail(int(GATE_CALIBRATION_FINAL_LAST_N_LOGGED))\n    per_run_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "entity",\n        "agent",\n    ]\n    per_run = data.groupby(per_run_cols, dropna=False, observed=True, as_index=False).agg(\n        final_gate_prob=("gate_prob_intervene", "mean"),\n        final_intervene_frac=("gate_intervene_frac", "mean"),\n        final_prob_minus_actual=("prob_minus_actual", "mean"),\n        final_step_millions=("step_millions", "max"),\n        averaged_logged_points=("prob_minus_actual", "count"),\n    )\n\n    grouped = per_run.groupby(\n        [\n            "experiment",\n            "comparison_group",\n            "family",\n            "family_label",\n            "design",\n            "control_axis",\n            "control_value",\n            "control_label",\n            "entropy_schedule",\n            "entity",\n            "agent",\n        ],\n        dropna=False,\n        observed=True,\n        as_index=False,\n    ).agg(\n        mean_final_gate_prob=("final_gate_prob", "mean"),\n        std_final_gate_prob=("final_gate_prob", "std"),\n        mean_final_intervene_frac=("final_intervene_frac", "mean"),\n        std_final_intervene_frac=("final_intervene_frac", "std"),\n        mean_final_prob_minus_actual=("final_prob_minus_actual", "mean"),\n        std_final_prob_minus_actual=("final_prob_minus_actual", "std"),\n        mean_final_step_millions=("final_step_millions", "mean"),\n        min_averaged_logged_points=("averaged_logged_points", "min"),\n        n_seeds=("run_id", "nunique"),\n        seeds=("seed", _seed_list),\n        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n    )\n    for column in ["std_final_gate_prob", "std_final_intervene_frac", "std_final_prob_minus_actual"]:\n        grouped[column] = grouped[column].fillna(0.0)\n    grouped["condition_label"] = grouped["family_label"]\n    grouped["agent"] = pd.Categorical(grouped["agent"].astype(str), categories=["agent_0", "agent_1", "agent_2"], ordered=True)\n    return grouped.sort_values(["experiment", "control_value", "design", "family_label", "agent"])\n\n\ndef gate_calibration_coverage_rows():\n    gated_runs = selected_runs[\n        selected_runs["design"].astype(str).eq("gated")\n        | selected_runs["gate_enabled"].map(lambda value: value is True).fillna(False)\n    ].copy()\n    keep_cols = [\n        "experiment",\n        "run_name",\n        "run_id",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "gate_enabled",\n        "control_label",\n    ]\n    if gated_runs.empty:\n        return pd.DataFrame(columns=keep_cols + ["has_gate_calibration", "gate_calibration_rows", "gate_agents"])\n\n    if GATE_CALIBRATION_RUN_LONG.empty:\n        available = pd.DataFrame(columns=["run_id", "run_name", "gate_calibration_rows", "gate_agents", "first_step_millions", "last_step_millions"])\n    else:\n        available = GATE_CALIBRATION_RUN_LONG.groupby(["run_id", "run_name"], dropna=False, as_index=False).agg(\n            gate_calibration_rows=("prob_minus_actual", "count"),\n            gate_agents=("agent", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),\n            first_step_millions=("step_millions", "min"),\n            last_step_millions=("step_millions", "max"),\n        )\n    out = gated_runs[keep_cols].merge(available, on=["run_id", "run_name"], how="left")\n    out["gate_calibration_rows"] = out["gate_calibration_rows"].fillna(0).astype(int)\n    out["has_gate_calibration"] = out["gate_calibration_rows"] > 0\n    return out.sort_values(["experiment", "family_label", "seed"])\n\n\nGATE_CALIBRATION_RUN_LONG = gate_calibration_run_rows()\nGATE_CALIBRATION_SEED_AGG = seed_aggregated_gate_calibration_rows()\nGATE_CALIBRATION_FINAL_SEED_AGG = seed_aggregated_final_gate_calibration_rows()\nGATE_CALIBRATION_COVERAGE = gate_calibration_coverage_rows()\n\n# print("Gate calibration coverage for gated runs:")\n# if GATE_CALIBRATION_COVERAGE.empty:\n#     print("No gated runs found in the selected cache.")\n# else:\n#     display(GATE_CALIBRATION_COVERAGE[[\n#         "experiment",\n#         "run_name",\n#         "family_label",\n#         "seed",\n#         "has_gate_calibration",\n#         "gate_calibration_rows",\n#         "gate_agents",\n#     ]])\n\nmissing_gate_calibration = GATE_CALIBRATION_COVERAGE[~GATE_CALIBRATION_COVERAGE["has_gate_calibration"]]\n# if not missing_gate_calibration.empty:\n#     print("Gated cached runs with no non-empty gate probability/intervention-fraction rows:")\n#     display(missing_gate_calibration[["experiment", "run_name", "family_label", "seed"]])\n\n# print("Seed-aggregated gate probability vs actual intervention time series:")\n# if GATE_CALIBRATION_SEED_AGG.empty:\n#     print("No gated intervention probability/fraction metrics found.")\n# else:\n#     display(GATE_CALIBRATION_SEED_AGG.groupby(["experiment", "family_label", "agent"], dropna=False, observed=True).agg(\n#         points=("_step", "count"),\n#         n_seeds=("n_seeds", "max"),\n#         seeds=("seeds", "first"),\n#     ).reset_index())\n\n# print(f"Gate calibration final summary, averaged over last {GATE_CALIBRATION_FINAL_LAST_N_LOGGED} logged points:")\n# if GATE_CALIBRATION_FINAL_SEED_AGG.empty:\n#     print("No final gate calibration summary available.")\n# else:\n#     display(GATE_CALIBRATION_FINAL_SEED_AGG[[\n#         "experiment",\n#         "family_label",\n#         "agent",\n#         "mean_final_gate_prob",\n#         "mean_final_intervene_frac",\n#         "mean_final_prob_minus_actual",\n#         "std_final_prob_minus_actual",\n#         "n_seeds",\n#         "seeds",\n#     ]])\n\n\ndef plot_gate_probability_vs_intervention_timeseries(experiment, title, save_name):\n    data = GATE_CALIBRATION_SEED_AGG[GATE_CALIBRATION_SEED_AGG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No gate calibration data for {experiment}")\n        return None\n\n    agents = [agent for agent in ["agent_0", "agent_1", "agent_2"] if agent in set(data["agent"].astype(str))]\n    if not agents:\n        print(f"No agents found in gate calibration data for {experiment}")\n        return None\n\n    conditions = data[["family", "condition_label", "control_value", "design"]].drop_duplicates()\n    conditions = conditions.sort_values(["control_value", "design", "condition_label"])\n    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24\n    colors = {row.family: palette[idx % len(palette)] for idx, row in enumerate(conditions.itertuples(index=False))}\n\n    fig = make_subplots(\n        rows=len(agents),\n        cols=1,\n        shared_xaxes=True,\n        vertical_spacing=0.075,\n        subplot_titles=[f"{agent}: predicted gate probability vs actual intervention fraction" for agent in agents],\n    )\n\n    for row_idx, agent in enumerate(agents, start=1):\n        agent_data = data[data["agent"].astype(str) == agent]\n        for _, condition in conditions.iterrows():\n            family = condition["family"]\n            label = condition["condition_label"]\n            condition_data = agent_data[agent_data["family"] == family].sort_values("step_millions")\n            if condition_data.empty:\n                continue\n            customdata = np.stack([\n                condition_data["n_seeds"],\n                condition_data["seeds"].astype(str),\n                condition_data["mean_prob_minus_actual_smooth"],\n            ], axis=-1)\n            fig.add_trace(\n                go.Scatter(\n                    x=condition_data["step_millions"],\n                    y=condition_data["mean_gate_prob_smooth"],\n                    mode="lines",\n                    name=f"{label} predicted",\n                    legendgroup=f"{family}-predicted",\n                    showlegend=row_idx == 1,\n                    line={"color": colors[family], "width": 2.8},\n                    customdata=customdata,\n                    hovertemplate=(\n                        f"<b>{label}</b><br>"\n                        f"agent={agent}<br>"\n                        "step=%{x:.2f}M<br>"\n                        "gate prob intervene=%{y:.4f}<br>"\n                        "predicted - actual=%{customdata[2]:.4f}<br>"\n                        "seeds=%{customdata[1]}<br>"\n                        "n_seeds=%{customdata[0]}<extra></extra>"\n                    ),\n                ),\n                row=row_idx,\n                col=1,\n            )\n            fig.add_trace(\n                go.Scatter(\n                    x=condition_data["step_millions"],\n                    y=condition_data["mean_intervene_frac_smooth"],\n                    mode="lines",\n                    name=f"{label} actual",\n                    legendgroup=f"{family}-actual",\n                    showlegend=row_idx == 1,\n                    line={"color": colors[family], "width": 2.8, "dash": "dash"},\n                    customdata=customdata,\n                    hovertemplate=(\n                        f"<b>{label}</b><br>"\n                        f"agent={agent}<br>"\n                        "step=%{x:.2f}M<br>"\n                        "actual intervene frac=%{y:.4f}<br>"\n                        "predicted - actual=%{customdata[2]:.4f}<br>"\n                        "seeds=%{customdata[1]}<br>"\n                        "n_seeds=%{customdata[0]}<extra></extra>"\n                    ),\n                ),\n                row=row_idx,\n                col=1,\n            )\n        fig.update_yaxes(title_text="fraction", range=[0, 1], row=row_idx, col=1)\n\n    fig.update_layout(\n        title=title,\n        template="plotly_white",\n        height=880,\n        width=1450,\n        hovermode="x unified",\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 80, "r": 380, "t": 100, "b": 70},\n    )\n    fig.update_xaxes(title_text="steps (M)", row=len(agents), col=1)\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\ndef _final_gate_calibration_seed_points(experiment):\n    data = GATE_CALIBRATION_RUN_LONG[GATE_CALIBRATION_RUN_LONG["experiment"] == experiment].copy()\n    if data.empty:\n        return pd.DataFrame()\n    data = data.sort_values(["run_id", "agent", "_step"])\n    data = data.groupby(["run_id", "agent"], dropna=False, observed=True).tail(int(GATE_CALIBRATION_FINAL_LAST_N_LOGGED))\n    per_run_cols = [\n        "run_name",\n        "run_id",\n        "experiment",\n        "comparison_group",\n        "family",\n        "family_label",\n        "seed",\n        "design",\n        "control_axis",\n        "control_value",\n        "control_label",\n        "entropy_schedule",\n        "entity",\n        "agent",\n    ]\n    seed_points = data.groupby(per_run_cols, dropna=False, observed=True, as_index=False).agg(\n        seed_final_gate_prob=("gate_prob_intervene", "mean"),\n        seed_final_intervene_frac=("gate_intervene_frac", "mean"),\n        seed_final_prob_minus_actual=("prob_minus_actual", "mean"),\n        final_step_millions=("step_millions", "max"),\n        averaged_logged_points=("prob_minus_actual", "count"),\n    )\n    seed_points["condition_label"] = seed_points["family_label"]\n    seed_points["agent"] = pd.Categorical(seed_points["agent"].astype(str), categories=["agent_0", "agent_1", "agent_2"], ordered=True)\n    return seed_points.sort_values(["control_value", "design", "condition_label", "agent", "seed"])\n\n\ndef plot_gate_calibration_final_gap(experiment, title, save_name):\n    data = GATE_CALIBRATION_FINAL_SEED_AGG[GATE_CALIBRATION_FINAL_SEED_AGG["experiment"] == experiment].copy()\n    if data.empty:\n        print(f"No final gate calibration data for {experiment}")\n        return None\n\n    fig = px.bar(\n        data,\n        x="agent",\n        y="mean_final_prob_minus_actual",\n        color="condition_label",\n        error_y="std_final_prob_minus_actual",\n        barmode="group",\n        hover_data={\n            "condition_label": True,\n            "control_label": True,\n            "mean_final_gate_prob": ":.4f",\n            "mean_final_intervene_frac": ":.4f",\n            "mean_final_prob_minus_actual": ":.4f",\n            "std_final_prob_minus_actual": ":.4f",\n            "n_seeds": True,\n            "seeds": True,\n            "min_averaged_logged_points": True,\n        },\n        labels={\n            "agent": "agent",\n            "mean_final_prob_minus_actual": "predicted - actual intervention fraction",\n            "condition_label": "condition",\n        },\n        title=title,\n    )\n\n    seed_points = _final_gate_calibration_seed_points(experiment)\n    add_seed_point_overlay(\n        fig,\n        seed_points,\n        x_col="agent",\n        y_col="seed_final_prob_minus_actual",\n        group_col="condition_label",\n        customdata_cols=[\n            "run_name",\n            "seed",\n            "condition_label",\n            "seed_final_gate_prob",\n            "seed_final_intervene_frac",\n            "averaged_logged_points",\n        ],\n        hovertemplate=(\n            "seed run=%{customdata[0]}<br>"\n            "seed=%{customdata[1]}<br>"\n            "condition=%{customdata[2]}<br>"\n            "agent=%{x}<br>"\n            "seed predicted - actual=%{y:.4f}<br>"\n            "gate prob=%{customdata[3]:.4f}<br>"\n            "actual frac=%{customdata[4]:.4f}<br>"\n            "logged points=%{customdata[5]}<extra></extra>"\n        ),\n    )\n\n    fig.add_hline(y=0, line_width=1, line_color="black")\n    fig.update_layout(\n        template="plotly_white",\n        height=620,\n        width=1450,\n        bargap=0.18,\n        bargroupgap=0.04,\n        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},\n        margin={"l": 80, "r": 360, "t": 90, "b": 70},\n    )\n    fig.update_yaxes(title_text="predicted - actual intervention fraction")\n    save_figure(fig, save_name)\n    if SHOW_FIGURES:\n        fig.show()\n    return fig\n\n\nfig_gate_calibration_timeseries_ig15 = plot_gate_probability_vs_intervention_timeseries(\n    "intervention_gate_15",\n    "Intervention gate 15: gate probability vs actual intervention fraction over training",\n    "gate_probability_vs_actual_intervention_timeseries_intervention_gate_15",\n)\nfig_gate_calibration_timeseries_sparse16 = plot_gate_probability_vs_intervention_timeseries(\n    "phase4_sparse_control_16",\n    "Phase4 sparse control 16: gate probability vs actual intervention fraction over training",\n    "gate_probability_vs_actual_intervention_timeseries_phase4_sparse_control_16",\n)\nfig_gate_calibration_timeseries_hvg = plot_gate_probability_vs_intervention_timeseries(\n    "heuristic_vs_gate_s0_s1_s2",\n    "Heuristic vs gate s0/s1/s2: gate probability vs actual intervention fraction over training",\n    "gate_probability_vs_actual_intervention_timeseries_heuristic_vs_gate_s0_s1_s2",\n)\nfig_gate_calibration_final_gap_ig15 = plot_gate_calibration_final_gap(\n    "intervention_gate_15",\n    f"Intervention gate 15: final gate calibration gap by agent (mean of last {GATE_CALIBRATION_FINAL_LAST_N_LOGGED} logged)",\n    "final_gate_calibration_gap_intervention_gate_15",\n)\nfig_gate_calibration_final_gap_sparse16 = plot_gate_calibration_final_gap(\n    "phase4_sparse_control_16",\n    f"Phase4 sparse control 16: final gate calibration gap by agent (mean of last {GATE_CALIBRATION_FINAL_LAST_N_LOGGED} logged)",\n    "final_gate_calibration_gap_phase4_sparse_control_16",\n)\nfig_gate_calibration_final_gap_hvg = plot_gate_calibration_final_gap(\n    "heuristic_vs_gate_s0_s1_s2",\n    f"Heuristic vs gate s0/s1/s2: final gate calibration gap by agent (mean of last {GATE_CALIBRATION_FINAL_LAST_N_LOGGED} logged)",\n    "final_gate_calibration_gap_heuristic_vs_gate_s0_s1_s2",\n)\n'


def _experiment_folder_map(experiment_folders=None):
    if experiment_folders is None:
        return dict(DEFAULT_EXPERIMENT_FOLDERS)
    if isinstance(experiment_folders, dict):
        return {name: Path(path) for name, path in experiment_folders.items()}
    names = list(experiment_folders)
    return {name: CONFIG_ROOT / name for name in names}


def _annotate_aib_metadata(df):
    """Make AIB rows use readable labels and deterministic ordering."""
    if not isinstance(df, pd.DataFrame) or df.empty or "experiment" not in df.columns:
        return df
    if "family" not in df.columns:
        return df
    out = df.copy()
    mask = out["experiment"].astype(str).eq(AIB_EXPERIMENT)
    if not mask.any():
        return out

    family = out.loc[mask, "family"].astype(str)
    order = family.map(AIB_FAMILY_ORDER)
    labels = family.map(AIB_FAMILY_LABELS)
    fallback_labels = "AIB " + family.str.replace("_", " ", regex=False)

    if "comparison_group" in out.columns:
        out.loc[mask, "comparison_group"] = "AIB"
    if "family_label" in out.columns:
        out.loc[mask, "family_label"] = labels.fillna(fallback_labels)
    if "control_axis" in out.columns:
        out.loc[mask, "control_axis"] = "aib_variant"
    if "control_value" in out.columns:
        out.loc[mask, "control_value"] = order.fillna(99).astype(float)
    if "control_label" in out.columns:
        out.loc[mask, "control_label"] = labels.fillna(fallback_labels)
    if "design" in out.columns:
        gate_mask = family.str.contains("_gate_", regex=False)
        out.loc[mask, "design"] = np.where(gate_mask, "gated", "flat")
    return out


def _annotate_aib_context(context):
    for name in ["EXPECTED_CONFIGS", "selected_runs", "coverage"]:
        if name in context:
            context[name] = _annotate_aib_metadata(context[name])
    return context


def _base_context(experiment_folders=None):
    return {
        "Path": Path,
        "json": json,
        "os": os,
        "re": re,
        "Iterable": Iterable,
        "Sequence": Sequence,
        "np": np,
        "pd": pd,
        "tomllib": tomllib,
        "px": px,
        "go": go,
        "make_subplots": make_subplots,
        "display": display,
        "_seed_list": _seed_list,
        "TASK_DIR": TASK_DIR,
        "CONFIG_ROOT": CONFIG_ROOT,
        "EXPERIMENT_FOLDERS": _experiment_folder_map(experiment_folders),
        "CACHE_DIR": CACHE_DIR,
        "CACHE_INDEX_PATH": CACHE_INDEX_PATH,
        "FIG_DIR": FIG_DIR,
        "TRACE_CACHE_DIR": TRACE_CACHE_DIR,
    }


def _truthy(value):
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_final_summary_step_cutoff(data, final_summary_step_m):
    """Keep only rows at or before the step used as the final summary point."""
    if data is None or data.empty or final_summary_step_m is None:
        return data
    cutoff = float(final_summary_step_m)
    if "step_millions" in data.columns:
        steps = pd.to_numeric(data["step_millions"], errors="coerce")
    elif "_step" in data.columns:
        steps = pd.to_numeric(data["_step"], errors="coerce") / 1_000_000.0
    else:
        return data
    return data.loc[steps <= cutoff].copy()


def _final_summary_step_suffix():
    value = globals().get("FINAL_SUMMARY_STEP_M")
    if value is None:
        return ""
    return f" up to {float(value):g}M"


def _eval_split_priority(eval_split):
    requested = str(eval_split or "test").strip()
    priority = [requested, "test", "train_eval", "eval", "validation"]
    out = []
    for split in priority:
        if split and split not in out:
            out.append(split)
    return out


def _agent_ids_from_eval_history(history, split):
    patterns = [
        re.compile(rf"^{re.escape(split)}/explain/frac_action_0_(agent_\d+)$"),
        re.compile(rf"^{re.escape(split)}/explain/action_nonidle_(agent_\d+)$"),
        re.compile(rf"^{re.escape(split)}/heuristic/policy_nonidle_(agent_\d+)$"),
    ]
    agents = set()
    for col in history.columns:
        for pattern in patterns:
            match = pattern.match(str(col))
            if match:
                agents.add(match.group(1))
    return sorted(agents, key=lambda agent: int(agent.rsplit("_", 1)[-1]))


def _clear_training_action_feature_columns(history):
    prefixes = (
        "train/frac_action_0_",
        "train/explain/action_nonidle_",
        "train/entropy_",
        "train/illegal_action_rate_",
        "train/intervention_gate_",
        "train/nonidle_action_entropy_",
        "train/non_idle_agents_",
    )
    exact = {
        "train/frac_any_non_idle",
        "train/frac_multi_agent_non_idle",
        "train/non_idle_agents_mean",
        "train/non_idle_agents_std",
        "train/non_idle_agents_max",
    }
    for col in list(history.columns):
        if str(col).startswith(prefixes) or col in exact:
            history[col] = np.nan


def _copy_series(history, target, values, mask=None):
    if target not in history.columns:
        history[target] = np.nan
    numeric = pd.to_numeric(values, errors="coerce")
    update_mask = numeric.notna()
    if mask is not None:
        update_mask = update_mask & mask.fillna(False)
    if update_mask.any():
        history.loc[update_mask, target] = numeric.loc[update_mask]
    return bool(update_mask.any())


def _apply_eval_action_metric_source(history, *, eval_split="test"):
    """Map final evaluation action metrics onto the feature names used by the plots."""
    history = history.copy()
    _clear_training_action_feature_columns(history)

    split_used_by_agent = {}
    agents_seen = set()
    gate_enabled = (
        history["gate_enabled"].map(_truthy)
        if "gate_enabled" in history.columns
        else pd.Series(False, index=history.index)
    )

    for split in _eval_split_priority(eval_split):
        for agent in _agent_ids_from_eval_history(history, split):
            if agent in agents_seen:
                continue

            action0_col = f"{split}/explain/frac_action_0_{agent}"
            nonidle_col = f"{split}/explain/action_nonidle_{agent}"
            raw_action0 = (
                pd.to_numeric(history[action0_col], errors="coerce")
                if action0_col in history.columns
                else pd.Series(np.nan, index=history.index)
            )
            raw_nonidle = (
                pd.to_numeric(history[nonidle_col], errors="coerce")
                if nonidle_col in history.columns
                else pd.Series(np.nan, index=history.index)
            )
            action0_values = raw_action0.where(raw_action0.notna(), 1.0 - raw_nonidle)
            nonidle_values = raw_nonidle.where(raw_nonidle.notna(), 1.0 - raw_action0)

            if action0_values is None or action0_values.notna().sum() == 0:
                continue

            _copy_series(history, f"train/frac_action_0_{agent}", action0_values)
            _copy_series(history, f"train/explain/action_nonidle_{agent}", nonidle_values)

            gate_col = f"{split}/explain/gate_intervened_{agent}"
            gate_values = (
                pd.to_numeric(history[gate_col], errors="coerce")
                if gate_col in history.columns
                else nonidle_values
            )
            _copy_series(
                history,
                f"train/intervention_gate_intervene_frac_{agent}",
                gate_values,
                mask=gate_enabled,
            )
            _copy_series(
                history,
                f"train/intervention_gate_do_nothing_frac_{agent}",
                1.0 - gate_values,
                mask=gate_enabled,
            )

            agents_seen.add(agent)
            split_used_by_agent[agent] = split

    nonidle_cols = [
        f"train/explain/action_nonidle_{agent}"
        for agent in sorted(agents_seen, key=lambda value: int(value.rsplit("_", 1)[-1]))
        if f"train/explain/action_nonidle_{agent}" in history.columns
    ]
    if nonidle_cols:
        nonidle = history[nonidle_cols].apply(pd.to_numeric, errors="coerce")
        history["train/non_idle_agents_mean"] = nonidle.sum(axis=1, min_count=1)

    coverage_rows = []
    experiments = (
        sorted(history["experiment"].dropna().astype(str).unique().tolist())
        if "experiment" in history.columns
        else ["all"]
    )
    for experiment in experiments:
        experiment_mask = (
            history["experiment"].astype(str).eq(experiment)
            if "experiment" in history.columns
            else pd.Series(True, index=history.index)
        )
        for agent, split in split_used_by_agent.items():
            action0 = pd.to_numeric(
                history.get(f"train/frac_action_0_{agent}", pd.Series(dtype=float)),
                errors="coerce",
            )
            coverage_rows.append(
                {
                    "experiment": experiment,
                    "agent": agent,
                    "eval_split_used": split,
                    "action0_rows": int(action0.loc[experiment_mask].notna().sum()),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    if coverage.empty:
        print(
            "No final evaluation action metrics were found. Expected columns like "
            f"{eval_split}/explain/frac_action_0_agent_0.",
            flush=True,
        )
    else:
        available = coverage[coverage["action0_rows"] > 0]
        missing = sorted(
            set(coverage["experiment"].astype(str)) - set(available["experiment"].astype(str))
        )
        print(
            "Using final evaluation action metrics for action-distribution plots: "
            + ", ".join(
                f"{row.experiment}/{row.agent} from {row.eval_split_used} ({row.action0_rows} rows)"
                for row in available.itertuples(index=False)
            ),
            flush=True,
        )
        if missing:
            print(
                "No final evaluation action metrics for: " + ", ".join(missing),
                flush=True,
            )
    return history, coverage


def _cell_source_for_context(context, source, label=None):
    if label == "configs_and_cache":
        source = source.replace(
            '    index = index.rename(columns={"name": "run_name", "id": "run_id"}).copy()\n',
            (
                '    index = index.rename(columns={"name": "run_name", "id": "run_id"}).copy()\n'
                '    if "run_name" in index.columns:\n'
                '        index["run_name_raw"] = index["run_name"]\n'
                '        index["run_name"] = index["run_name"].astype(str).str.strip()\n'
            ),
        )
    source = source.replace(
        'def final_window_average(data, value_col, group_cols, window=FINAL_WINDOW_STEPS):\n'
        '    sorted_data = data.sort_values(group_cols + ["_step"])\n'
        '    tail = sorted_data.groupby(group_cols, dropna=False).tail(int(window))\n'
        '    return tail.groupby(group_cols, dropna=False, as_index=False)[value_col].mean()\n',
        'def final_window_average(data, value_col, group_cols, window=FINAL_WINDOW_STEPS):\n'
        '    filtered_data = _apply_final_summary_step_cutoff(data, FINAL_SUMMARY_STEP_M)\n'
        '    if filtered_data.empty:\n'
        '        return pd.DataFrame(columns=list(group_cols) + [value_col])\n'
        '    sorted_data = filtered_data.sort_values(group_cols + ["_step"])\n'
        '    tail = sorted_data.groupby(group_cols, dropna=False).tail(int(window))\n'
        '    return tail.groupby(group_cols, dropna=False, as_index=False)[value_col].mean()\n',
    )
    source = re.sub(
        r'(    data = data\.sort_values\([^\n]+\)(?:\.copy\(\))?\n)'
        r'(    data = data\.groupby\([^\n]+\.tail\(int\([A-Z0-9_]+\)\)(?:\.copy\(\))?\n)',
        r'\1    data = _apply_final_summary_step_cutoff(data, FINAL_SUMMARY_STEP_M)\n\2',
        source,
    )
    if context.get("ACTION_METRIC_SOURCE") != "eval":
        return source
    replacements = {
        "over training": "over evaluation",
        "cached training values": "cached evaluation values",
        "training values": "evaluation values",
        "Train": "Eval",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def _exec_cell(context, source, label):
    source = _cell_source_for_context(context, source, label=label)
    exec(compile(source, f"<action_distribution_metrics:{label}>", "exec"), context)
    return context


def load_action_distribution_context(
    *,
    experiment_folders=None,
    action_metric_source="eval",
    eval_split="test",
    save_figures=True,
    show_figures=False,
    show_tables=False,
    final_window_steps=None,
    smooth_window=None,
    plot_step_max_m=None,
    final_summary_step_m=None,
):
    """Load cached histories and build shared action-distribution tables."""
    context = _base_context(experiment_folders=experiment_folders)
    if not show_tables:
        context["display"] = lambda value: None
    _exec_cell(context, _CELL_CONFIG, "configuration")

    context["ACTION_METRIC_SOURCE"] = str(action_metric_source or "train").strip().lower()
    context["ACTION_EVAL_SPLIT"] = str(eval_split or "test").strip()
    context["SAVE_FIGURES"] = bool(save_figures)
    context["SHOW_FIGURES"] = bool(show_figures)
    context["FINAL_SUMMARY_STEP_M"] = final_summary_step_m
    context["_apply_final_summary_step_cutoff"] = _apply_final_summary_step_cutoff
    context["_final_summary_step_suffix"] = lambda: "" if final_summary_step_m is None else f" up to {float(final_summary_step_m):g}M"
    if final_window_steps is not None:
        context["FINAL_WINDOW_STEPS"] = int(final_window_steps)
    if smooth_window is not None:
        context["SMOOTH_WINDOW"] = int(smooth_window)
    if plot_step_max_m is not None:
        context["PLOT_STEP_MAX_M"] = plot_step_max_m

    _exec_cell(context, _CELL_CONFIG_CACHE, "configs_and_cache")
    _annotate_aib_context(context)
    if context["ACTION_METRIC_SOURCE"] == "eval":
        context["FIG_DIR"] = FIG_DIR / f"eval_{context['ACTION_EVAL_SPLIT']}"
        context["FIG_DIR"].mkdir(parents=True, exist_ok=True)
    _exec_cell(context, _CELL_HISTORIES, "histories")
    if context["ACTION_METRIC_SOURCE"] == "eval":
        history_wide, coverage = _apply_eval_action_metric_source(
            context["history_wide"],
            eval_split=context["ACTION_EVAL_SPLIT"],
        )
        context["history_wide"] = history_wide
        context["EVAL_ACTION_METRIC_COVERAGE"] = coverage
    _exec_cell(context, _CELL_FEATURES, "features")
    _exec_cell(context, _CELL_PROFILES, "profiles")
    return context


def _figures(context, names):
    return {name: context.get(name) for name in names if context.get(name) is not None}


def _plot_aib(context, fig_name, plotter_name, title, save_name):
    plotter = context.get(plotter_name)
    if not callable(plotter):
        return None
    fig = plotter(AIB_EXPERIMENT, title, save_name)
    context[fig_name] = fig
    return fig


def last5_logged_action0_fraction_by_agent_all_runs(context):
    _exec_cell(context, _CELL_ACTION0_ALL, "last5_action0_all_runs")
    _plot_aib(
        context,
        "fig_action0_fraction_aib_all_runs",
        "plot_action0_fraction_all_runs",
        f"{AIB_TITLE}: action 0 fraction by agent for all runs (mean of last 5 logged)",
        "final_action0_fraction_by_agent_all_runs_adaptive_intervention_budget_7",
    )
    return {
        "title": "Last-5logged action 0 fraction by agent for all runs",
        "figures": _figures(context, [
            "fig_action0_fraction_ig15_all_runs",
            "fig_action0_fraction_hvg_all_runs",
            "fig_action0_fraction_sparse16_all_runs",
            "fig_action0_fraction_aib_all_runs",
        ]),
        "tables": {
            "ACTION0_FINAL_LONG": context.get("ACTION0_FINAL_LONG"),
            "ACTION0_COUNT_LONG": context.get("ACTION0_COUNT_LONG"),
            "ACTION0_COUNT_SUMMARY": context.get("ACTION0_COUNT_SUMMARY"),
        },
    }


def seed_aggregated_last5_logged_action0_fraction_by_agent(context):
    if "ACTION0_FINAL_LONG" not in context:
        last5_logged_action0_fraction_by_agent_all_runs(context)
    _exec_cell(context, _CELL_ACTION0_SEED, "seed_aggregated_action0")
    _plot_aib(
        context,
        "fig_action0_fraction_aib_seed_agg",
        "plot_action0_fraction_seed_aggregated",
        f"{AIB_TITLE}: seed-aggregated action 0 fraction by agent (mean of last 5 logged)",
        "seed_aggregated_final_action0_fraction_by_agent_adaptive_intervention_budget_7",
    )
    return {
        "title": "Seed-Aggregated Last-5-Logged Action-0 Fraction By Agent",
        "figures": _figures(context, [
            "fig_action0_fraction_ig15_seed_agg",
            "fig_action0_fraction_hvg_seed_agg",
            "fig_action0_fraction_sparse16_seed_agg",
            "fig_action0_fraction_aib_seed_agg",
        ]),
        "tables": {"ACTION0_SEED_AGG": context.get("ACTION0_SEED_AGG")},
    }


def survival_vs_action0_non_idle_over_time(context):
    _exec_cell(context, _CELL_SURVIVAL_ACTION, "survival_vs_action_behavior")
    _plot_aib(
        context,
        "fig_survival_vs_action_aib",
        "plot_survival_vs_action_behavior",
        f"{AIB_TITLE}: survival vs action-0 / non-idle over evaluation",
        "survival_vs_action_behavior_adaptive_intervention_budget_7",
    )
    return {
        "title": "Survival vs Action-0 / Non-Idle Over Time",
        "figures": _figures(context, [
            "fig_survival_vs_action_ig15",
            "fig_survival_vs_action_hvg",
            "fig_survival_vs_action_sparse16",
            "fig_survival_vs_action_aib",
        ]),
        "tables": {
            "SURVIVAL_SEED_AGG": context.get("SURVIVAL_SEED_AGG"),
            "ACTION_BEHAVIOR_SEED_AGG": context.get("ACTION_BEHAVIOR_SEED_AGG"),
        },
    }


EXPERIMENT_TITLES = {
    "intervention_gate_15": "Intervention gate 15",
    "phase4_sparse_control_16": "Phase4 sparse control 16",
    "heuristic_vs_gate_s0_s1_s2": "Heuristic vs gate s0/s1/s2",
    AIB_EXPERIMENT: AIB_TITLE,
}


def _survival_scale_fallback(values):
    numeric = pd.to_numeric(values, errors="coerce")
    max_value = numeric.max(skipna=True)
    if pd.isna(max_value):
        return 100.0
    return 100.0 if max_value <= 1.5 else 1.0


def _final_survival_per_run_rows(context, last_n_logged=5):
    history = context.get("history_wide")
    if not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame()

    candidates = context.get("SURVIVAL_METRIC_CANDIDATES") or [
        "test/charts/episodic_survival",
        "test/episodic_survival",
        "validation/episodic_survival",
        "charts/episodic_survival",
        "train_eval/charts/episodic_survival",
        "train_eval/episodic_survival",
    ]
    scale_fn = context.get("_survival_scale") or _survival_scale_fallback
    final_summary_step_m = context.get("FINAL_SUMMARY_STEP_M")
    rows = []

    meta_cols = [
        "run_name",
        "run_id",
        "experiment",
        "comparison_group",
        "family",
        "family_label",
        "seed",
        "design",
        "control_axis",
        "control_value",
        "control_label",
        "entropy_schedule",
        "_step",
        "step_millions",
    ]
    meta_cols = [col for col in meta_cols if col in history.columns]

    for _, run_history in history.groupby("run_id", sort=False):
        metric = next(
            (
                candidate
                for candidate in candidates
                if candidate in run_history.columns and run_history[candidate].notna().any()
            ),
            None,
        )
        if metric is None:
            continue

        values = pd.to_numeric(run_history[metric], errors="coerce")
        frame = run_history[meta_cols].copy()
        frame["survival_pct"] = values * scale_fn(values)
        frame["metric_used"] = metric
        frame = frame.dropna(subset=["survival_pct", "_step", "step_millions"])
        frame = _apply_final_summary_step_cutoff(frame, final_summary_step_m)
        if frame.empty:
            continue

        tail = frame.sort_values("_step").tail(int(last_n_logged))
        first = tail.iloc[-1]
        rows.append(
            {
                "run_name": first.get("run_name"),
                "run_id": first.get("run_id"),
                "experiment": first.get("experiment"),
                "comparison_group": first.get("comparison_group"),
                "family": first.get("family"),
                "family_label": first.get("family_label"),
                "seed": first.get("seed"),
                "design": first.get("design"),
                "control_axis": first.get("control_axis"),
                "control_value": first.get("control_value"),
                "control_label": first.get("control_label"),
                "entropy_schedule": first.get("entropy_schedule"),
                "final_survival_pct": tail["survival_pct"].mean(),
                "std_final_survival_pct": tail["survival_pct"].std(),
                "final_survival_step_millions": tail["step_millions"].max(),
                "survival_logged_points": int(tail["survival_pct"].count()),
                "survival_metric_used": metric,
            }
        )

    out = pd.DataFrame(rows)
    if "std_final_survival_pct" in out.columns:
        out["std_final_survival_pct"] = out["std_final_survival_pct"].fillna(0.0)
    return out


def _action0_per_run_rows(context):
    action0 = context.get("ACTION0_FINAL_LONG")
    if not isinstance(action0, pd.DataFrame) or action0.empty:
        return pd.DataFrame()

    data = action0.copy()
    data["final_fraction_action0"] = pd.to_numeric(data["final_fraction_action0"], errors="coerce")
    data["final_action0_count"] = pd.to_numeric(data.get("final_action0_count"), errors="coerce")
    data["final_step_millions"] = pd.to_numeric(data.get("final_step_millions"), errors="coerce")

    group_cols = [
        "run_name",
        "run_id",
        "experiment",
        "comparison_group",
        "family",
        "family_label",
        "seed",
        "design",
        "control_axis",
        "control_value",
        "control_label",
        "entropy_schedule",
    ]
    group_cols = [col for col in group_cols if col in data.columns]
    out = data.groupby(group_cols, dropna=False, as_index=False).agg(
        final_action0_fraction=("final_fraction_action0", "mean"),
        std_agent_action0_fraction=("final_fraction_action0", "std"),
        final_action0_count=("final_action0_count", "sum"),
        final_action0_step_millions=("final_step_millions", "max"),
        n_agents=("agent", "nunique"),
    )
    out["std_agent_action0_fraction"] = out["std_agent_action0_fraction"].fillna(0.0)
    return out


def _action0_survival_tradeoff_rows(context, last_n_logged=5):
    action0 = _action0_per_run_rows(context)
    survival = _final_survival_per_run_rows(context, last_n_logged=last_n_logged)
    if action0.empty or survival.empty:
        return pd.DataFrame(), pd.DataFrame()

    keep_survival = [
        "run_id",
        "run_name",
        "final_survival_pct",
        "std_final_survival_pct",
        "final_survival_step_millions",
        "survival_logged_points",
        "survival_metric_used",
    ]
    keep_survival = [col for col in keep_survival if col in survival.columns]
    merged = action0.merge(survival[keep_survival], on=["run_id", "run_name"], how="inner")
    if merged.empty:
        return merged, pd.DataFrame()

    merged["condition_label"] = merged["family_label"]
    merged["step_gap_millions"] = (
        pd.to_numeric(merged["final_action0_step_millions"], errors="coerce")
        - pd.to_numeric(merged["final_survival_step_millions"], errors="coerce")
    ).abs()

    group_cols = [
        "experiment",
        "comparison_group",
        "family",
        "family_label",
        "condition_label",
        "design",
        "control_axis",
        "control_value",
        "control_label",
        "entropy_schedule",
    ]
    group_cols = [col for col in group_cols if col in merged.columns]
    grouped = merged.groupby(group_cols, dropna=False, as_index=False).agg(
        mean_action0_fraction=("final_action0_fraction", "mean"),
        std_action0_fraction=("final_action0_fraction", "std"),
        mean_survival_pct=("final_survival_pct", "mean"),
        std_survival_pct=("final_survival_pct", "std"),
        mean_final_step_millions=("final_survival_step_millions", "mean"),
        n_seeds=("run_id", "nunique"),
        seeds=("seed", _seed_list),
        runs=("run_name", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),
        metrics_used=("survival_metric_used", lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist())),
    )
    grouped["std_action0_fraction"] = grouped["std_action0_fraction"].fillna(0.0)
    grouped["std_survival_pct"] = grouped["std_survival_pct"].fillna(0.0)
    grouped = grouped.sort_values(["experiment", "control_value", "design", "family_label"])
    return merged, grouped


def _plot_action0_survival_tradeoff(context, run_rows, agg_rows, experiment):
    run_data = run_rows[run_rows["experiment"].astype(str).eq(experiment)].copy()
    agg_data = agg_rows[agg_rows["experiment"].astype(str).eq(experiment)].copy()
    if run_data.empty or agg_data.empty:
        return None

    conditions = agg_data[["family", "condition_label", "control_value", "design"]].drop_duplicates()
    conditions = conditions.sort_values(["control_value", "design", "condition_label"])
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    colors = {row.family: palette[idx % len(palette)] for idx, row in enumerate(conditions.itertuples(index=False))}
    title = f"{EXPERIMENT_TITLES.get(experiment, experiment)}: final action-0 vs episodic survival tradeoff"
    save_name = f"final_action0_vs_survival_tradeoff_{experiment}"

    fig = go.Figure()
    for _, condition in conditions.iterrows():
        family = condition["family"]
        label = condition["condition_label"]
        color = colors[family]
        seed_points = run_data[run_data["family"].astype(str).eq(str(family))].sort_values("seed")
        if not seed_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=seed_points["final_action0_fraction"],
                    y=seed_points["final_survival_pct"],
                    mode="markers",
                    name=f"{label} seeds",
                    legendgroup=family,
                    showlegend=False,
                    marker={
                        "color": color,
                        "size": 8,
                        "opacity": 0.55,
                        "line": {"color": "white", "width": 0.8},
                    },
                    customdata=np.stack(
                        [
                            seed_points["run_name"].astype(str),
                            seed_points["seed"],
                            seed_points["final_action0_count"],
                            seed_points["n_agents"],
                            seed_points["final_action0_step_millions"],
                            seed_points["final_survival_step_millions"],
                            seed_points["survival_metric_used"].astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "seed run=%{customdata[0]}<br>"
                        "seed=%{customdata[1]}<br>"
                        "action-0 fraction=%{x:.4f}<br>"
                        "episodic survival=%{y:.2f}%<br>"
                        "action-0 count=%{customdata[2]}<br>"
                        "agents=%{customdata[3]}<br>"
                        "action step=%{customdata[4]:.2f}M<br>"
                        "survival step=%{customdata[5]:.2f}M<br>"
                        "survival metric=%{customdata[6]}<extra></extra>"
                    ),
                )
            )

        mean_row = agg_data[agg_data["family"].astype(str).eq(str(family))].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[mean_row["mean_action0_fraction"]],
                y=[mean_row["mean_survival_pct"]],
                mode="markers",
                name=label,
                legendgroup=family,
                showlegend=True,
                marker={
                    "color": color,
                    "size": 17,
                    "symbol": "diamond",
                    "line": {"color": "black", "width": 1.1},
                },
                error_x={"type": "data", "array": [float(mean_row["std_action0_fraction"])]},
                error_y={"type": "data", "array": [float(mean_row["std_survival_pct"])]},
                customdata=np.array(
                    [[mean_row["n_seeds"], str(mean_row["seeds"]), str(mean_row["runs"]), str(mean_row["metrics_used"])]],
                    dtype=object,
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "mean action-0 fraction=%{x:.4f}<br>"
                    "mean episodic survival=%{y:.2f}%<br>"
                    "n_seeds=%{customdata[0]}<br>"
                    "seeds=%{customdata[1]}<br>"
                    "runs=%{customdata[2]}<br>"
                    "survival metric=%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.add_annotation(
        text="better: higher survival and more action 0",
        x=0.98,
        y=102,
        xref="x",
        yref="y",
        showarrow=True,
        ax=-90,
        ay=45,
        font={"size": 12, "color": "#333"},
        arrowcolor="#333",
    )
    source_label = "eval" if context.get("ACTION_METRIC_SOURCE") == "eval" else "train"
    final_step = context.get("FINAL_SUMMARY_STEP_M")
    subtitle = f"action source={source_label}; final window <= {final_step:g}M" if final_step is not None else f"action source={source_label}; final window = run end"
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}; seed dots, diamond/error bars = condition mean/std</sup>",
        template="plotly_white",
        height=680,
        width=1200,
        hovermode="closest",
        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.01},
        margin={"l": 80, "r": 340, "t": 105, "b": 70},
    )
    fig.update_xaxes(title_text="mean action-0 fraction across agents", range=[0, 1])
    fig.update_yaxes(title_text="episodic survival (%)", range=[0, 105])

    save_figure = context.get("save_figure")
    if callable(save_figure):
        save_figure(fig, save_name)
    if context.get("SHOW_FIGURES"):
        fig.show()
    return fig


def action0_survival_tradeoff(context, survival_last_n_logged=5):
    if "ACTION0_FINAL_LONG" not in context:
        last5_logged_action0_fraction_by_agent_all_runs(context)
    if "SURVIVAL_SEED_AGG" not in context or "ACTION_BEHAVIOR_SEED_AGG" not in context:
        survival_vs_action0_non_idle_over_time(context)

    run_rows, agg_rows = _action0_survival_tradeoff_rows(
        context,
        last_n_logged=survival_last_n_logged,
    )
    context["ACTION0_SURVIVAL_TRADEOFF_RUN_LONG"] = run_rows
    context["ACTION0_SURVIVAL_TRADEOFF_SEED_AGG"] = agg_rows

    figures = {}
    if run_rows.empty or agg_rows.empty:
        print("No overlapping action-0 and survival data available for tradeoff plots.")
    else:
        for experiment in sorted(run_rows["experiment"].dropna().astype(str).unique()):
            fig = _plot_action0_survival_tradeoff(context, run_rows, agg_rows, experiment)
            if fig is not None:
                figures[f"fig_action0_survival_tradeoff_{experiment}"] = fig

    return {
        "title": "Action-0 / Survival Tradeoff",
        "figures": figures,
        "tables": {
            "ACTION0_SURVIVAL_TRADEOFF_RUN_LONG": run_rows,
            "ACTION0_SURVIVAL_TRADEOFF_SEED_AGG": agg_rows,
        },
    }


def entropy_collapse_vs_action0_confidence(context):
    feature_long = context.get("FEATURE_LONG")
    has_entropy = (
        isinstance(feature_long, pd.DataFrame)
        and not feature_long.empty
        and "feature_group" in feature_long.columns
        and feature_long["feature_group"].eq("agent_entropy").any()
    )
    if context.get("ACTION_METRIC_SOURCE") == "eval" and not has_entropy:
        print("No evaluation entropy metrics found; skipping entropy/action-0 plots.")
        empty = pd.DataFrame()
        return {
            "title": "Entropy Collapse vs Action-0 Confidence",
            "figures": {},
            "tables": {
                "ENTROPY_ACTION0_RUN_LONG": empty,
                "ENTROPY_ACTION0_SEED_AGG": empty,
                "ENTROPY_ACTION0_FINAL_SEED_AGG": empty,
            },
        }
    _exec_cell(context, _CELL_ENTROPY_ACTION0, "entropy_vs_action0")
    _plot_aib(
        context,
        "fig_entropy_action0_timeseries_aib",
        "plot_entropy_action0_timeseries",
        f"{AIB_TITLE}: entropy collapse vs action-0 confidence over evaluation",
        "entropy_action0_timeseries_adaptive_intervention_budget_7",
    )
    _plot_aib(
        context,
        "fig_entropy_action0_final_scatter_aib",
        "plot_final_action0_entropy_scatter",
        f"{AIB_TITLE}: final entropy vs action-0 confidence",
        "final_entropy_action0_scatter_adaptive_intervention_budget_7",
    )
    return {
        "title": "Entropy Collapse vs Action-0 Confidence",
        "figures": _figures(context, [
            "fig_entropy_action0_timeseries_ig15",
            "fig_entropy_action0_timeseries_hvg",
            "fig_entropy_action0_timeseries_sparse16",
            "fig_entropy_action0_timeseries_aib",
            "fig_entropy_action0_final_scatter_ig15",
            "fig_entropy_action0_final_scatter_hvg",
            "fig_entropy_action0_final_scatter_sparse16",
            "fig_entropy_action0_final_scatter_aib",
        ]),
        "tables": {
            "ENTROPY_ACTION0_RUN_LONG": context.get("ENTROPY_ACTION0_RUN_LONG"),
            "ENTROPY_ACTION0_SEED_AGG": context.get("ENTROPY_ACTION0_SEED_AGG"),
            "ENTROPY_ACTION0_FINAL_SEED_AGG": context.get("ENTROPY_ACTION0_FINAL_SEED_AGG"),
        },
    }


def agent_non_idle_imbalance(context):
    _exec_cell(context, _CELL_AGENT_IMBALANCE, "agent_non_idle_imbalance")
    _plot_aib(
        context,
        "fig_agent_imbalance_timeseries_aib",
        "plot_agent_imbalance_timeseries",
        f"{AIB_TITLE}: agent non-idle imbalance over evaluation",
        "agent_non_idle_imbalance_timeseries_adaptive_intervention_budget_7",
    )
    _plot_aib(
        context,
        "fig_agent_imbalance_final_aib",
        "plot_final_agent_imbalance",
        f"{AIB_TITLE}: final agent non-idle imbalance (mean of last 5 logged)",
        "final_agent_non_idle_imbalance_adaptive_intervention_budget_7",
    )
    return {
        "title": "Agent Non-Idle Imbalance",
        "figures": _figures(context, [
            "fig_agent_imbalance_timeseries_ig15",
            "fig_agent_imbalance_timeseries_hvg",
            "fig_agent_imbalance_timeseries_sparse16",
            "fig_agent_imbalance_timeseries_aib",
            "fig_agent_imbalance_final_ig15",
            "fig_agent_imbalance_final_hvg",
            "fig_agent_imbalance_final_sparse16",
            "fig_agent_imbalance_final_aib",
        ]),
        "tables": {
            "AGENT_IMBALANCE_RUN_LONG": context.get("AGENT_IMBALANCE_RUN_LONG"),
            "AGENT_IMBALANCE_SEED_AGG": context.get("AGENT_IMBALANCE_SEED_AGG"),
            "AGENT_IMBALANCE_FINAL_SEED_AGG": context.get("AGENT_IMBALANCE_FINAL_SEED_AGG"),
        },
    }


def joint_action_coordination(context):
    feature_long = context.get("FEATURE_LONG")
    has_joint = (
        isinstance(feature_long, pd.DataFrame)
        and not feature_long.empty
        and "feature_group" in feature_long.columns
        and feature_long["feature_group"].eq("joint_non_idle").any()
    )
    if context.get("ACTION_METRIC_SOURCE") == "eval" and not has_joint:
        print(
            "No scalar evaluation joint-action metrics found; skipping joint coordination plots. "
            "Eval joint coordination needs eval non_idle_agents_count_* scalars or trace-table processing."
        )
        empty = pd.DataFrame()
        return {
            "title": "Joint Action Coordination",
            "figures": {},
            "tables": {
                "JOINT_COORDINATION_RUN_LONG": empty,
                "JOINT_COORDINATION_SEED_AGG": empty,
                "JOINT_COORDINATION_FINAL_SEED_AGG": empty,
                "JOINT_COORDINATION_COVERAGE": empty,
            },
        }
    _exec_cell(context, _CELL_JOINT_COORDINATION, "joint_action_coordination")
    _plot_aib(
        context,
        "fig_joint_coordination_timeseries_aib",
        "plot_joint_coordination_timeseries",
        f"{AIB_TITLE}: joint action coordination over evaluation",
        "joint_action_coordination_timeseries_adaptive_intervention_budget_7",
    )
    _plot_aib(
        context,
        "fig_joint_coordination_final_aib",
        "plot_final_joint_coordination",
        f"{AIB_TITLE}: final joint action coordination",
        "final_joint_action_coordination_adaptive_intervention_budget_7",
    )
    return {
        "title": "Joint Action Coordination",
        "figures": _figures(context, [
            "fig_joint_coordination_timeseries_ig15",
            "fig_joint_coordination_timeseries_hvg",
            "fig_joint_coordination_timeseries_sparse16",
            "fig_joint_coordination_timeseries_aib",
            "fig_joint_coordination_final_ig15",
            "fig_joint_coordination_final_hvg",
            "fig_joint_coordination_final_sparse16",
            "fig_joint_coordination_final_aib",
        ]),
        "tables": {
            "JOINT_COORDINATION_RUN_LONG": context.get("JOINT_COORDINATION_RUN_LONG"),
            "JOINT_COORDINATION_SEED_AGG": context.get("JOINT_COORDINATION_SEED_AGG"),
            "JOINT_COORDINATION_FINAL_SEED_AGG": context.get("JOINT_COORDINATION_FINAL_SEED_AGG"),
            "JOINT_COORDINATION_COVERAGE": context.get("JOINT_COORDINATION_COVERAGE"),
        },
    }


def gate_probability_vs_actual_intervention_fraction(context):
    feature_long = context.get("FEATURE_LONG")
    has_gate_prob = (
        isinstance(feature_long, pd.DataFrame)
        and not feature_long.empty
        and "feature_group" in feature_long.columns
        and feature_long["feature_group"].eq("gate_prob_intervene").any()
    )
    if context.get("ACTION_METRIC_SOURCE") == "eval" and not has_gate_prob:
        print(
            "No evaluation gate-probability metrics found; skipping gate probability vs actual plots. "
            "Eval logs final gate intervention fractions, but not gate probabilities."
        )
        empty = pd.DataFrame()
        return {
            "title": "Gate Probability vs Actual Intervention Fraction",
            "figures": {},
            "tables": {
                "GATE_CALIBRATION_RUN_LONG": empty,
                "GATE_CALIBRATION_SEED_AGG": empty,
                "GATE_CALIBRATION_FINAL_SEED_AGG": empty,
                "GATE_CALIBRATION_COVERAGE": empty,
            },
        }
    _exec_cell(context, _CELL_GATE_CALIBRATION, "gate_probability_vs_actual_intervention")
    _plot_aib(
        context,
        "fig_gate_calibration_timeseries_aib",
        "plot_gate_probability_vs_intervention_timeseries",
        f"{AIB_TITLE}: gate probability vs actual intervention fraction over evaluation",
        "gate_probability_vs_actual_intervention_timeseries_adaptive_intervention_budget_7",
    )
    _plot_aib(
        context,
        "fig_gate_calibration_final_gap_aib",
        "plot_gate_calibration_final_gap",
        f"{AIB_TITLE}: final gate calibration gap by agent",
        "final_gate_calibration_gap_adaptive_intervention_budget_7",
    )
    return {
        "title": "Gate Probability vs Actual Intervention Fraction",
        "figures": _figures(context, [
            "fig_gate_calibration_timeseries_ig15",
            "fig_gate_calibration_timeseries_hvg",
            "fig_gate_calibration_timeseries_sparse16",
            "fig_gate_calibration_timeseries_aib",
            "fig_gate_calibration_final_gap_ig15",
            "fig_gate_calibration_final_gap_hvg",
            "fig_gate_calibration_final_gap_sparse16",
            "fig_gate_calibration_final_gap_aib",
        ]),
        "tables": {
            "GATE_CALIBRATION_RUN_LONG": context.get("GATE_CALIBRATION_RUN_LONG"),
            "GATE_CALIBRATION_SEED_AGG": context.get("GATE_CALIBRATION_SEED_AGG"),
            "GATE_CALIBRATION_FINAL_SEED_AGG": context.get("GATE_CALIBRATION_FINAL_SEED_AGG"),
            "GATE_CALIBRATION_COVERAGE": context.get("GATE_CALIBRATION_COVERAGE"),
        },
    }


REQUESTED_METRIC_FUNCTIONS = [
    last5_logged_action0_fraction_by_agent_all_runs,
    seed_aggregated_last5_logged_action0_fraction_by_agent,
    survival_vs_action0_non_idle_over_time,
    action0_survival_tradeoff,
    entropy_collapse_vs_action0_confidence,
    agent_non_idle_imbalance,
    joint_action_coordination,
    gate_probability_vs_actual_intervention_fraction,
]


def run_requested_action_distribution_metrics(context=None, **load_kwargs):
    """Run all requested action-distribution metric sections and return their outputs."""
    context = context or load_action_distribution_context(**load_kwargs)
    return {func.__name__: func(context) for func in REQUESTED_METRIC_FUNCTIONS}


def display_metric_output(output, *, table_rows=20):
    """Convenience helper for notebook cells: display compact tables then figures."""
    print(output["title"])
    for table_name, table in output.get("tables", {}).items():
        if table is not None:
            print(f"{table_name}: {getattr(table, 'shape', '')}")
            if hasattr(table, "head"):
                display(table.head(table_rows))
    for fig in output.get("figures", {}).values():
        display(fig)


def display_metric_figures(output):
    """Display only figures for a metric output."""
    for fig in output.get("figures", {}).values():
        display(fig)
