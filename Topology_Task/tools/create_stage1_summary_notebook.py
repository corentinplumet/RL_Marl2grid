#!/usr/bin/env python3
"""Generate the combined Stage 1--1e graph-screening analysis notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    TASK_DIR
    / "analysis"
    / "metrics"
    / "notebooks"
    / "episodic_survival"
    / "gnn_graph_screening_stage1_all_stages_summary.ipynb"
)


def markdown(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def build_notebook():
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook["cells"] = [
        markdown(
            """
            # GNN graph screening: combined Stage 1--1e summary

            This notebook compares every configuration declared in Stage 1,
            Stage 1b, Stage 1c, Stage 1d, and Stage 1e.

            It provides:

            - cache and run-completion coverage;
            - stage-specific episodic-survival curves;
            - seed-aggregated curves for repeated configurations;
            - rankings at a common environment-step budget;
            - marginal summaries for every graph parameter;
            - matched comparisons in which only one parameter changes;
            - representation/normalization, structure, and direction heatmaps.

            The matched-effect table is the safest answer to “what helps?”.
            Marginal averages are also shown, but they can be confounded because
            the screening design is intentionally staged rather than fully crossed.
            """
        ),
        code(
            """
            from pathlib import Path
            import importlib
            import sys
            import tomllib

            import numpy as np
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            from IPython.display import display

            for candidate in [Path.cwd(), *Path.cwd().parents]:
                helpers = candidate / "helpers"
                repo_helpers = candidate / "Topology_Task" / "analysis" / "metrics" / "helpers"
                if helpers.exists() and (helpers / "wandb_metrics.py").exists():
                    sys.path.insert(0, str(helpers))
                    break
                if repo_helpers.exists() and (repo_helpers / "wandb_metrics.py").exists():
                    sys.path.insert(0, str(repo_helpers))
                    break
            else:
                raise FileNotFoundError(
                    "Could not locate Topology_Task/analysis/metrics/helpers"
                )

            import wandb_metrics as wm
            wm = importlib.reload(wm)
            import survival_comparison as sc
            sc = importlib.reload(sc)
            print("wandb_metrics:", wm.__file__)
            print("survival_comparison:", sc.__file__)
            print("task directory:", wm.TASK_DIR)
            """
        ),
        markdown(
            """
            ## Analysis controls

            Set `USE_LOCAL_CACHE_ONLY = False` to download missing or newer W&B
            histories. With local-only mode, active or not-yet-downloaded runs
            simply appear as missing in the coverage table.

            `COMPARISON_BUDGET_STEPS = None` uses the largest step reached by
            every selected run. Set it explicitly, for example to `8_000_000`,
            when the runs have completed. Runs that have not reached the selected
            budget are excluded from endpoint aggregation rather than being
            compared at an earlier checkpoint.
            """
        ),
        code(
            """
            USE_LOCAL_CACHE_ONLY = True
            SMOOTH_WINDOW = 5
            TARGET_BUDGET_STEPS = 8_000_000
            COMPARISON_BUDGET_STEPS = None
            DEDUPE_EQUIVALENT_SEED_RUNS = True

            STAGE_DIRS = {
                "Stage 1": wm.TASK_DIR / "configs" / "gnn_graph_screening"
                / "stage1_representation_normalization",
                "Stage 1b": wm.TASK_DIR / "configs" / "gnn_graph_screening"
                / "stage1b_normalization_confirmation",
                "Stage 1c": wm.TASK_DIR / "configs" / "gnn_graph_screening"
                / "stage1c_provisional_structure",
                "Stage 1d": wm.TASK_DIR / "configs" / "gnn_graph_screening"
                / "stage1d_heterogeneous_directions",
                "Stage 1e": wm.TASK_DIR / "configs" / "gnn_graph_screening"
                / "stage1e_heterogeneous_structure",
            }
            STAGE_ORDER = list(STAGE_DIRS)
            STAGE_RANK = {stage: rank for rank, stage in enumerate(STAGE_ORDER)}

            FACTOR_COLUMNS = [
                "graph_type",
                "normalization",
                "substation_edges",
                "substation_nodes",
                "readout",
                "encoder",
                "generator_direction",
                "load_direction",
                "line_direction",
                "summary_direction",
            ]
            """
        ),
        markdown(
            """
            ## Build the experiment catalog from TOML

            The notebook reads the configuration files directly instead of
            relying on filenames. Equivalent settings from different stages
            receive the same `factor_key`.
            """
        ),
        code(
            """
            def normalization_label(args):
                physical = bool(args.get("gnn_physical_scaling", False))
                running = bool(args.get("gnn_running_norm", False))
                return {
                    (False, False): "n0_none",
                    (True, False): "n1_physical",
                    (False, True): "n2_running",
                    (True, True): "n3_both",
                }[(physical, running)]


            def config_record(path, stage):
                with path.open("rb") as file:
                    config = tomllib.load(file)
                args = config["args"]
                run = config.get("run", {})
                readout = str(args.get("gnn_readout_aggr", "mean"))
                record = {
                    "stage": stage,
                    "stage_rank": STAGE_RANK[stage],
                    "config_path": str(path),
                    "config": path.name,
                    "run_name": str(run.get("name", path.stem)),
                    "seed": int(args.get("seed", 0)),
                    "graph_type": str(args.get("gnn_graph_type", "bus")),
                    "normalization": normalization_label(args),
                    "physical_scaling": bool(
                        args.get("gnn_physical_scaling", False)
                    ),
                    "running_normalization": bool(
                        args.get("gnn_running_norm", False)
                    ),
                    "substation_edges": bool(
                        args.get("gnn_add_substation_edges", False)
                    ),
                    "substation_nodes": bool(
                        args.get("gnn_add_substation_nodes", False)
                    ),
                    "readout": readout,
                    "virtual_node": readout == "virtual_node",
                    "encoder": str(args.get("gnn_type", "gine")),
                    "generator_direction": str(
                        args.get("gnn_generator_edge_direction", "bidirectional")
                    ),
                    "load_direction": str(
                        args.get("gnn_load_edge_direction", "bidirectional")
                    ),
                    "line_direction": str(
                        args.get("gnn_line_node_edge_direction", "bidirectional")
                    ),
                    "summary_direction": str(
                        args.get("gnn_summary_edge_direction", "bidirectional")
                    ),
                    "configured_steps": int(
                        args.get("total_timesteps", TARGET_BUDGET_STEPS)
                    ),
                    "time_limit_minutes": float(args.get("time_limit", np.nan)),
                }
                record["structure"] = "e{}n{}v{}".format(
                    int(record["substation_edges"]),
                    int(record["substation_nodes"]),
                    int(record["virtual_node"]),
                )
                record["factor_key"] = "|".join(
                    str(record[column]) for column in FACTOR_COLUMNS
                )
                record["config_label"] = (
                    f'{record["graph_type"]} · {record["normalization"]} · '
                    f'{record["structure"]} · {record["encoder"]}'
                )
                record["run_label"] = (
                    f'{record["config_label"]} · '
                    f'g={record["generator_direction"]} · '
                    f'l={record["load_direction"]} · s{record["seed"]}'
                )
                return record


            records = []
            for stage, config_dir in STAGE_DIRS.items():
                if not config_dir.exists():
                    print("Missing config folder:", config_dir)
                    continue
                records.extend(
                    config_record(path, stage)
                    for path in sorted(config_dir.glob("*.toml"))
                )

            config_catalog = pd.DataFrame(records)
            if config_catalog.empty:
                raise RuntimeError("No graph-screening TOML files were found.")

            print(f"Cataloged {len(config_catalog)} declared runs.")
            display(
                config_catalog.groupby("stage", observed=True)
                .agg(
                    configs=("run_name", "nunique"),
                    seeds=("seed", "nunique"),
                    graph_types=("graph_type", lambda x: ", ".join(sorted(set(x)))),
                )
                .reindex(STAGE_ORDER)
                .fillna(0)
            )
            display(
                config_catalog[
                    [
                        "stage",
                        "run_name",
                        "seed",
                        "graph_type",
                        "normalization",
                        "structure",
                        "generator_direction",
                        "load_direction",
                    ]
                ]
            )
            """
        ),
        markdown(
            """
            ## Load the selected W&B histories

            Only run names declared by the five config folders are selected.
            Missing runs remain visible in the coverage table.
            """
        ),
        code(
            """
            requested_run_names = config_catalog["run_name"].drop_duplicates().tolist()
            requested_run_name_set = set(requested_run_names)
            wm.configure_run_filter_from_names(requested_run_names)
            data = wm.load_wandb_data(use_local_cache_only=USE_LOCAL_CACHE_ONLY)

            runs_df = data.runs_df.copy()
            history_df = data.history_df.copy()
            if not history_df.empty:
                history_df = history_df[
                    history_df["run_name"].astype(str).isin(requested_run_name_set)
                ].copy()
            if "name" in runs_df:
                runs_df = runs_df[
                    runs_df["name"].astype(str).isin(requested_run_name_set)
                ].copy()

            found_run_names = set(history_df.get("run_name", pd.Series(dtype=str)))
            print(
                f"Loaded histories for {len(found_run_names)} / "
                f"{len(requested_run_names)} declared runs."
            )
            missing_run_names = sorted(requested_run_name_set - found_run_names)
            if missing_run_names:
                print("Missing histories:")
                for name in missing_run_names:
                    print("  ", name)
            """
        ),
        markdown(
            """
            ## Coverage and equivalent-run deduplication

            Some stages intentionally repeat a configuration and seed. For
            aggregation, the notebook keeps the repeat with the greatest
            observed step count. This prevents the same nominal seed from being
            counted twice. Set `DEDUPE_EQUIVALENT_SEED_RUNS = False` to inspect
            every repeat separately.
            """
        ),
        code(
            """
            if history_df.empty:
                observed_progress = pd.DataFrame(columns=["run_name", "observed_steps"])
            else:
                observed_progress = (
                    history_df.groupby("run_name", as_index=False)["step"]
                    .max()
                    .rename(columns={"step": "observed_steps"})
                )

            coverage = config_catalog.merge(
                observed_progress, on="run_name", how="left"
            )
            coverage["observed_steps_m"] = coverage["observed_steps"] / 1_000_000
            coverage["completion_pct"] = (
                100 * coverage["observed_steps"] / coverage["configured_steps"]
            )
            coverage["has_history"] = coverage["observed_steps"].notna()

            available_catalog = coverage[coverage["has_history"]].copy()
            if DEDUPE_EQUIVALENT_SEED_RUNS:
                analysis_catalog = (
                    available_catalog.sort_values(
                        ["factor_key", "seed", "observed_steps", "stage_rank"],
                        ascending=[True, True, False, False],
                    )
                    .drop_duplicates(["factor_key", "seed"], keep="first")
                    .copy()
                )
            else:
                analysis_catalog = available_catalog.copy()

            selected_names = set(analysis_catalog["run_name"])
            coverage["selected_for_aggregation"] = coverage["run_name"].isin(
                selected_names
            )
            display(
                coverage.sort_values(["stage_rank", "run_name"])[
                    [
                        "stage",
                        "run_name",
                        "observed_steps_m",
                        "completion_pct",
                        "selected_for_aggregation",
                    ]
                ]
                .round(2)
            )

            progress_plot = px.bar(
                coverage.sort_values(["stage_rank", "observed_steps_m"]),
                x="observed_steps_m",
                y="run_name",
                color="stage",
                orientation="h",
                hover_data=["graph_type", "normalization", "structure", "seed"],
                title="Declared run coverage",
                labels={
                    "observed_steps_m": "Observed environment steps (millions)",
                    "run_name": "Run",
                },
                height=max(600, 19 * len(coverage)),
            )
            progress_plot.add_vline(
                x=TARGET_BUDGET_STEPS / 1_000_000,
                line_dash="dash",
                annotation_text="8M target",
            )
            progress_plot.show()
            """
        ),
        markdown(
            """
            ## Extract test episodic-survival curves

            Each run uses the first available test-survival metric recognized by
            `wandb_metrics.py`. Values are converted to percentages and smoothed
            with a trailing five-evaluation mean by default.
            """
        ),
        code(
            """
            TEST_METRICS = wm.SURVIVAL_METRIC_CANDIDATES["test"]


            def survival_frame(run_name):
                run_history = history_df[history_df["run_name"] == run_name]
                for metric in TEST_METRICS:
                    frame = run_history[run_history["metric"] == metric][
                        ["step", "value"]
                    ].copy()
                    if frame.empty:
                        continue
                    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
                    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
                    frame = (
                        frame.dropna()
                        .groupby("step", as_index=False)["value"]
                        .mean()
                        .sort_values("step")
                    )
                    frame["survival_pct"] = 100 * frame["value"]
                    frame["smoothed_survival_pct"] = (
                        frame["survival_pct"]
                        .rolling(SMOOTH_WINDOW, min_periods=1)
                        .mean()
                    )
                    frame["metric"] = metric
                    frame["run_name"] = run_name
                    return frame
                return pd.DataFrame()


            survival_frames = [
                survival_frame(run_name)
                for run_name in analysis_catalog["run_name"].drop_duplicates()
            ]
            survival_frames = [frame for frame in survival_frames if not frame.empty]
            if survival_frames:
                survival_long = pd.concat(survival_frames, ignore_index=True)
                survival_long = survival_long.merge(
                    analysis_catalog[
                        [
                            "run_name",
                            "stage",
                            "stage_rank",
                            "seed",
                            "factor_key",
                            "config_label",
                            "run_label",
                            *FACTOR_COLUMNS,
                            "structure",
                        ]
                    ],
                    on="run_name",
                    how="left",
                )
            else:
                survival_long = pd.DataFrame()

            if survival_long.empty:
                raise RuntimeError(
                    "No test episodic-survival histories were found for the "
                    "selected runs."
                )

            max_survival_steps = survival_long.groupby("run_name")["step"].max()
            automatic_common_budget = int(max_survival_steps.min())
            comparison_budget = int(
                COMPARISON_BUDGET_STEPS
                if COMPARISON_BUDGET_STEPS is not None
                else automatic_common_budget
            )
            print(
                "Comparison budget:",
                f"{comparison_budget / 1_000_000:.3f}M steps",
                "(automatic common budget)"
                if COMPARISON_BUDGET_STEPS is None
                else "(user selected)",
            )


            def endpoint_at_budget(frame, budget):
                if frame["step"].max() < budget:
                    return np.nan
                eligible = frame[frame["step"] <= budget]
                return (
                    eligible["smoothed_survival_pct"].iloc[-1]
                    if not eligible.empty
                    else np.nan
                )


            endpoint_rows = []
            for run_name, frame in survival_long.groupby("run_name", sort=False):
                endpoint_rows.append(
                    {
                        "run_name": run_name,
                        "last_eval_step": frame["step"].max(),
                        "latest_5eval_survival_pct": frame[
                            "smoothed_survival_pct"
                        ].iloc[-1],
                        "best_5eval_survival_pct": frame[
                            "smoothed_survival_pct"
                        ].max(),
                        "comparison_survival_pct": endpoint_at_budget(
                            frame, comparison_budget
                        ),
                        "target_8m_survival_pct": endpoint_at_budget(
                            frame, TARGET_BUDGET_STEPS
                        ),
                    }
                )

            endpoint_df = analysis_catalog.merge(
                pd.DataFrame(endpoint_rows), on="run_name", how="left"
            )
            endpoint_df["last_eval_step_m"] = endpoint_df["last_eval_step"] / 1_000_000
            """
        ),
        markdown(
            """
            ## Reusable episodic-survival comparison function

            `sc.plot_survival_comparison` is the main function to reuse in your
            own notebooks. It accepts `survival_long` and lets you control:

            - **selection:** `filters`, `query`, `run_names`, and regular expressions;
            - **curves:** `group_by` decides which runs are averaged;
            - **names:** `label_by` or `label_map`;
            - **smoothing:** trailing, centered, exponential, or none;
            - **uncertainty:** standard deviation, standard error, 95% CI,
              min--max, interquartile range, or none;
            - **comparability:** minimum seed count, complete-run filtering,
              common horizons, and explicit step ranges;
            - **layout:** individual seed traces, facets, colors, line styles,
              highlights, axis ranges, and figure size;
            - **direct comparison:** absolute curves or difference from a
              selected baseline.

            The function returns a Plotly figure. With `return_data=True`, it
            also returns the exact aggregated dataframe used in the plot.
            """
        ),
        markdown(
            """
            ### Editable comparison playground

            This is the cell to modify most often. A curve is one unique
            combination of `PLOT_GROUP_BY`; all selected runs sharing that
            combination are treated as members/seeds of that curve.

            Use `PLOT_MIN_MEMBERS = "all"` to stop a mean curve as soon as one
            seed is missing. Use `1` to retain every available observation.
            """
        ),
        code(
            """
            PLOT_FILTERS = {
                "stage": "Stage 1b",
                # "graph_type": "bus",
                # "seed": [0, 1, 2],
            }
            PLOT_QUERY = None
            PLOT_RUN_NAMES = None
            PLOT_GROUP_BY = ["graph_type", "normalization"]
            PLOT_LABEL_BY = lambda row: (
                f'{row["graph_type"]} · {row["normalization"]}'
            )
            PLOT_SMOOTH = 5
            PLOT_SMOOTH_METHOD = "trailing"
            PLOT_UNCERTAINTY = "std"
            PLOT_MIN_MEMBERS = "all"
            PLOT_SHOW_MEMBERS = True
            PLOT_FACET_BY = None
            PLOT_END_STEP = TARGET_BUDGET_STEPS
            PLOT_HIGHLIGHT = []
            PLOT_COLORS = {
                "bus · n0_none": "#1f77b4",
                "bus · n1_physical": "#d62728",
                "heterogeneous · n0_none": "#2ca02c",
            }
            PLOT_DASHES = None

            custom_survival_figure, custom_survival_data = (
                sc.plot_survival_comparison(
                    survival_long,
                    filters=PLOT_FILTERS,
                    query=PLOT_QUERY,
                    run_names=PLOT_RUN_NAMES,
                    group_by=PLOT_GROUP_BY,
                    label_by=PLOT_LABEL_BY,
                    smooth=PLOT_SMOOTH,
                    smooth_method=PLOT_SMOOTH_METHOD,
                    uncertainty=PLOT_UNCERTAINTY,
                    min_members=PLOT_MIN_MEMBERS,
                    show_members=PLOT_SHOW_MEMBERS,
                    facet_by=PLOT_FACET_BY,
                    end_step=PLOT_END_STEP,
                    highlight=PLOT_HIGHLIGHT,
                    colors=PLOT_COLORS,
                    dashes=PLOT_DASHES,
                    budget_step=TARGET_BUDGET_STEPS,
                    title="Editable survival comparison",
                    return_data=True,
                )
            )
            custom_survival_figure.show()
            display(custom_survival_data.tail(12).round(2))
            """
        ),
        markdown(
            """
            ### Normalization effect relative to a baseline

            This view removes the common movement of the baseline curve. Values
            above zero mean that physical scaling is outperforming no
            normalization at that training step. The band combines uncertainty
            from both independently aggregated curves.
            """
        ),
        code(
            """
            normalization_delta_figure = sc.plot_survival_comparison(
                survival_long,
                filters={"stage": "Stage 1b", "graph_type": "bus"},
                group_by="normalization",
                label_by="normalization",
                smooth=5,
                uncertainty="ci95",
                min_members="all",
                comparison="difference",
                baseline={"normalization": "n0_none"},
                colors={
                    "n0_none": "#6b7280",
                    "n1_physical": "#d62728",
                },
                budget_step=TARGET_BUDGET_STEPS,
                y_range=None,
                title=(
                    "Stage 1b bus graph: survival difference from "
                    "no normalization"
                ),
            )
            normalization_delta_figure.show()
            """
        ),
        markdown(
            """
            ### Structure comparisons as small multiples

            Stage 1c and Stage 1e are shown in separate facets while keeping the
            same axes and plotting interface. These screens currently contain
            one seed per curve, so no uncertainty band is requested.
            """
        ),
        code(
            """
            structure_comparison_figure = sc.plot_survival_comparison(
                survival_long,
                filters={"stage": ["Stage 1c", "Stage 1e"]},
                group_by=["stage", "graph_type", "structure"],
                label_by=["graph_type", "structure"],
                facet_by="stage",
                facet_order=["Stage 1c", "Stage 1e"],
                smooth=5,
                uncertainty=None,
                min_members=1,
                show_points=False,
                budget_step=TARGET_BUDGET_STEPS,
                title="Structural graph-screening curves",
                ncols=2,
                height=560,
            )
            structure_comparison_figure.show()
            """
        ),
        markdown(
            """
            ### Heterogeneous edge-direction comparison

            Each curve reports the generator and load message directions. Use
            `highlight=[...]` in the call to emphasize a shortlist while keeping
            the remaining curves visible as context.
            """
        ),
        code(
            """
            direction_comparison_figure = sc.plot_survival_comparison(
                survival_long,
                filters={"stage": "Stage 1d"},
                group_by=["generator_direction", "load_direction"],
                label_by=lambda row: (
                    f'g={row["generator_direction"]} · '
                    f'l={row["load_direction"]}'
                ),
                smooth=5,
                uncertainty=None,
                min_members=1,
                budget_step=TARGET_BUDGET_STEPS,
                title="Stage 1d: heterogeneous message directions",
                width=1450,
                height=650,
            )
            direction_comparison_figure.show()
            """
        ),
        markdown(
            """
            ### Minimal copy-and-reuse template

            ```python
            from survival_comparison import plot_survival_comparison

            fig = plot_survival_comparison(
                survival_long,
                filters={"stage": "Stage 1e"},
                group_by=["graph_type", "structure"],
                label_by=["graph_type", "structure"],
                smooth=5,
                uncertainty=None,
                show_members=False,
                end_step=8_000_000,
                title="My comparison",
            )
            fig.show()
            ```

            For a completely separate notebook, use
            `sc.extract_survival_curves(history_df, catalog=my_catalog)` to build
            `survival_long` from the standard downloaded W&B history dataframe.
            """
        ),
        markdown(
            """
            ## Fair-budget ranking

            Only runs that reached the comparison budget receive a comparison
            value. This prevents a slow representation from being ranked using a
            later or earlier amount of experience.
            """
        ),
        code(
            """
            ranking_columns = [
                "stage",
                "run_name",
                "seed",
                "graph_type",
                "normalization",
                "structure",
                "encoder",
                "generator_direction",
                "load_direction",
                "last_eval_step_m",
                "comparison_survival_pct",
                "latest_5eval_survival_pct",
                "best_5eval_survival_pct",
                "target_8m_survival_pct",
            ]
            ranking = endpoint_df[ranking_columns].sort_values(
                "comparison_survival_pct", ascending=False, na_position="last"
            )
            display(ranking.round(2))

            ranked_available = ranking.dropna(
                subset=["comparison_survival_pct"]
            ).copy()
            if not ranked_available.empty:
                rank_plot = px.bar(
                    ranked_available.sort_values("comparison_survival_pct"),
                    x="comparison_survival_pct",
                    y="run_name",
                    color="stage",
                    orientation="h",
                    hover_data=[
                        "graph_type",
                        "normalization",
                        "structure",
                        "seed",
                        "generator_direction",
                        "load_direction",
                    ],
                    title=(
                        "Test survival at the common "
                        f"{comparison_budget / 1_000_000:.3f}M-step budget"
                    ),
                    labels={
                        "comparison_survival_pct": "Smoothed test survival (%)",
                        "run_name": "Run",
                    },
                    height=max(550, 20 * len(ranked_available)),
                )
                rank_plot.show()
            """
        ),
        markdown(
            """
            ## Stage-specific learning curves

            Change `STAGE_TO_PLOT` to inspect one stage without an unreadable
            all-run legend.
            """
        ),
        code(
            """
            STAGE_TO_PLOT = "Stage 1b"


            def plot_stage_curves(stage):
                frame = survival_long[survival_long["stage"] == stage].copy()
                if frame.empty:
                    print(f"No downloaded survival history for {stage}.")
                    return None
                figure = px.line(
                    frame,
                    x="step",
                    y="smoothed_survival_pct",
                    color="run_label",
                    hover_data=[
                        "run_name",
                        "graph_type",
                        "normalization",
                        "structure",
                        "generator_direction",
                        "load_direction",
                    ],
                    title=f"{stage}: five-evaluation smoothed test survival",
                    labels={
                        "step": "Environment steps",
                        "smoothed_survival_pct": "Test survival (%)",
                        "run_label": "Configuration",
                    },
                )
                figure.update_yaxes(range=[0, 105])
                figure.add_vline(x=comparison_budget, line_dash="dot")
                return figure


            stage_figure = plot_stage_curves(STAGE_TO_PLOT)
            if stage_figure is not None:
                stage_figure.show()
            """
        ),
        markdown(
            """
            ## Seed-aggregated learning curves

            Only logical configurations with more than one downloaded seed are
            shown. Thin error bars represent one standard deviation across seeds
            at each evaluation step.
            """
        ),
        code(
            """
            seed_curve_summary = (
                survival_long.groupby(
                    ["factor_key", "config_label", "step"], as_index=False
                )
                .agg(
                    mean_survival_pct=("smoothed_survival_pct", "mean"),
                    std_survival_pct=("smoothed_survival_pct", "std"),
                    n_seeds=("seed", "nunique"),
                )
            )
            multi_seed_keys = set(
                seed_curve_summary.loc[
                    seed_curve_summary["n_seeds"] > 1, "factor_key"
                ]
            )
            multi_seed_curves = seed_curve_summary[
                seed_curve_summary["factor_key"].isin(multi_seed_keys)
            ].copy()

            if multi_seed_curves.empty:
                print("No logical configuration currently has multiple downloaded seeds.")
            else:
                seed_figure = px.line(
                    multi_seed_curves,
                    x="step",
                    y="mean_survival_pct",
                    color="config_label",
                    error_y="std_survival_pct",
                    title="Seed-aggregated test survival",
                    labels={
                        "step": "Environment steps",
                        "mean_survival_pct": "Mean test survival (%)",
                        "config_label": "Configuration",
                    },
                )
                seed_figure.update_yaxes(range=[0, 105])
                seed_figure.show()
            """
        ),
        markdown(
            """
            ## Marginal parameter aggregation

            These summaries average all available endpoint observations sharing
            a parameter value. They are useful for exploration, but are not
            causal estimates because the remaining factors are not perfectly
            balanced.
            """
        ),
        code(
            """
            valid_endpoint_df = endpoint_df.dropna(
                subset=["comparison_survival_pct"]
            ).copy()


            def summarize_parameter(factor, frame=valid_endpoint_df):
                if factor not in frame:
                    raise KeyError(f"Unknown factor: {factor}")
                return (
                    frame.groupby(factor, dropna=False, as_index=False)
                    .agg(
                        mean_survival_pct=("comparison_survival_pct", "mean"),
                        std_survival_pct=("comparison_survival_pct", "std"),
                        median_survival_pct=("comparison_survival_pct", "median"),
                        observations=("run_name", "nunique"),
                        logical_configs=("factor_key", "nunique"),
                        seeds=("seed", "nunique"),
                    )
                    .sort_values("mean_survival_pct", ascending=False)
                )


            marginal_tables = []
            for factor in FACTOR_COLUMNS:
                summary = summarize_parameter(factor)
                summary.insert(0, "factor", factor)
                summary = summary.rename(columns={factor: "level"})
                marginal_tables.append(summary)
            marginal_summary = pd.concat(marginal_tables, ignore_index=True)
            display(marginal_summary.round(2))
            """
        ),
        markdown(
            """
            ### Editable single-parameter view

            Change `FACTOR_TO_AGGREGATE` and optional `FILTERS`. For example,
            `FILTERS = {"graph_type": "bus"}` isolates normalization effects
            within the bus representation.
            """
        ),
        code(
            """
            FACTOR_TO_AGGREGATE = "normalization"
            FILTERS = {}


            def apply_filters(frame, filters):
                selected = frame.copy()
                for column, value in filters.items():
                    values = value if isinstance(value, (list, tuple, set)) else [value]
                    selected = selected[selected[column].isin(values)]
                return selected


            filtered_endpoints = apply_filters(valid_endpoint_df, FILTERS)
            editable_summary = summarize_parameter(
                FACTOR_TO_AGGREGATE, filtered_endpoints
            )
            display(editable_summary.round(2))

            if not filtered_endpoints.empty:
                factor_plot = px.box(
                    filtered_endpoints,
                    x=FACTOR_TO_AGGREGATE,
                    y="comparison_survival_pct",
                    points="all",
                    color=FACTOR_TO_AGGREGATE,
                    hover_data=["run_name", "stage", "seed", "config_label"],
                    title=(
                        f"Endpoint survival grouped by {FACTOR_TO_AGGREGATE}"
                        + (f" with filters {FILTERS}" if FILTERS else "")
                    ),
                    labels={
                        "comparison_survival_pct": "Smoothed test survival (%)"
                    },
                )
                factor_plot.update_yaxes(range=[0, 105])
                factor_plot.show()
            """
        ),
        markdown(
            """
            ## Matched one-factor effects

            A matched pair has the same seed and identical values for every
            listed graph factor except the factor being tested. The reported
            delta is:

            `target level survival - reference level survival`.

            Positive values therefore favor the target. Effects with one pair
            are valid comparisons but remain provisional; `n_pairs` makes that
            uncertainty explicit.
            """
        ),
        code(
            """
            REFERENCE_LEVELS = {
                "graph_type": "bus",
                "normalization": "n0_none",
                "substation_edges": False,
                "substation_nodes": False,
                "readout": "mean",
                "encoder": "gine",
                "generator_direction": "bidirectional",
                "load_direction": "bidirectional",
                "line_direction": "bidirectional",
                "summary_direction": "bidirectional",
            }


            def matched_effects(frame, factor, reference):
                context_columns = [
                    column for column in FACTOR_COLUMNS if column != factor
                ] + ["seed"]
                base = frame[frame[factor] == reference][
                    [*context_columns, "comparison_survival_pct", "run_name"]
                ].rename(
                    columns={
                        "comparison_survival_pct": "reference_survival_pct",
                        "run_name": "reference_run",
                    }
                )
                rows = []
                for level in frame[factor].drop_duplicates():
                    if level == reference:
                        continue
                    target = frame[frame[factor] == level][
                        [*context_columns, "comparison_survival_pct", "run_name"]
                    ].rename(
                        columns={
                            "comparison_survival_pct": "target_survival_pct",
                            "run_name": "target_run",
                        }
                    )
                    paired = target.merge(base, on=context_columns, how="inner")
                    if paired.empty:
                        continue
                    paired["delta_survival_pct"] = (
                        paired["target_survival_pct"]
                        - paired["reference_survival_pct"]
                    )
                    paired["factor"] = factor
                    paired["level"] = level
                    paired["reference"] = reference
                    rows.append(paired)
                return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


            paired_frames = [
                matched_effects(valid_endpoint_df, factor, reference)
                for factor, reference in REFERENCE_LEVELS.items()
            ]
            paired_frames = [frame for frame in paired_frames if not frame.empty]
            paired_effect_details = (
                pd.concat(paired_frames, ignore_index=True)
                if paired_frames
                else pd.DataFrame()
            )

            if paired_effect_details.empty:
                print("No matched one-factor comparisons are available yet.")
                paired_effect_summary = pd.DataFrame()
            else:
                paired_effect_summary = (
                    paired_effect_details.groupby(
                        ["factor", "level", "reference"], as_index=False
                    )
                    .agg(
                        mean_delta_survival_pct=("delta_survival_pct", "mean"),
                        std_delta_survival_pct=("delta_survival_pct", "std"),
                        median_delta_survival_pct=("delta_survival_pct", "median"),
                        n_pairs=("delta_survival_pct", "size"),
                        win_rate=("delta_survival_pct", lambda x: (x > 0).mean()),
                    )
                    .sort_values("mean_delta_survival_pct", ascending=False)
                )
                paired_effect_summary["comparison"] = (
                    paired_effect_summary["factor"].astype(str)
                    + ": "
                    + paired_effect_summary["level"].astype(str)
                    + " vs "
                    + paired_effect_summary["reference"].astype(str)
                )
                display(paired_effect_summary.round(2))

                effect_plot_data = paired_effect_summary.copy()
                effect_plot_data["error"] = effect_plot_data[
                    "std_delta_survival_pct"
                ].fillna(0.0)
                effect_figure = px.bar(
                    effect_plot_data.sort_values("mean_delta_survival_pct"),
                    x="mean_delta_survival_pct",
                    y="comparison",
                    error_x="error",
                    color="factor",
                    orientation="h",
                    hover_data=["n_pairs", "win_rate", "median_delta_survival_pct"],
                    title=(
                        "Matched one-factor effects at "
                        f"{comparison_budget / 1_000_000:.3f}M steps"
                    ),
                    labels={
                        "mean_delta_survival_pct": (
                            "Mean change in test survival (percentage points)"
                        ),
                        "comparison": "Target versus reference",
                    },
                    height=max(500, 35 * len(effect_plot_data)),
                )
                effect_figure.add_vline(x=0, line_dash="dash")
                effect_figure.show()
            """
        ),
        markdown(
            """
            ## Targeted screening heatmaps

            These plots mirror the experimental design:

            1. representation versus normalization;
            2. structure versus representation;
            3. generator direction versus load direction.
            """
        ),
        code(
            """
            def show_heatmap(table, title, x_label, y_label, zmin=0, zmax=100):
                if table.empty or table.notna().sum().sum() == 0:
                    print(f"No data available for: {title}")
                    return None
                figure = px.imshow(
                    table,
                    text_auto=".1f",
                    aspect="auto",
                    zmin=zmin,
                    zmax=zmax,
                    color_continuous_scale="Viridis",
                    labels={
                        "x": x_label,
                        "y": y_label,
                        "color": "Survival (%)",
                    },
                    title=title,
                )
                figure.show()
                return figure


            base_mask = (
                (~valid_endpoint_df["substation_edges"])
                & (~valid_endpoint_df["substation_nodes"])
                & (valid_endpoint_df["readout"] == "mean")
                & (valid_endpoint_df["encoder"] == "gine")
                & (valid_endpoint_df["generator_direction"] == "bidirectional")
                & (valid_endpoint_df["load_direction"] == "bidirectional")
                & (valid_endpoint_df["line_direction"] == "bidirectional")
            )
            representation_normalization = (
                valid_endpoint_df[base_mask]
                .groupby(["graph_type", "normalization"])[
                    "comparison_survival_pct"
                ]
                .mean()
                .unstack("normalization")
            )
            show_heatmap(
                representation_normalization,
                "Representation × normalization",
                "Normalization",
                "Graph representation",
            )

            structure_mask = (
                valid_endpoint_df["graph_type"].isin(["bus", "heterogeneous"])
                & (valid_endpoint_df["normalization"] == "n0_none")
                & (valid_endpoint_df["encoder"] == "gine")
                & (valid_endpoint_df["generator_direction"] == "bidirectional")
                & (valid_endpoint_df["load_direction"] == "bidirectional")
            )
            structure_table = (
                valid_endpoint_df[structure_mask]
                .groupby(["graph_type", "structure"])["comparison_survival_pct"]
                .mean()
                .unstack("structure")
            )
            show_heatmap(
                structure_table,
                "Structure × representation",
                "Structure (edges / summary nodes / virtual readout)",
                "Graph representation",
            )

            direction_mask = (
                (valid_endpoint_df["graph_type"] == "heterogeneous")
                & (valid_endpoint_df["normalization"] == "n0_none")
                & (~valid_endpoint_df["substation_edges"])
                & (~valid_endpoint_df["substation_nodes"])
                & (valid_endpoint_df["readout"] == "mean")
                & (valid_endpoint_df["encoder"] == "gine")
            )
            direction_table = (
                valid_endpoint_df[direction_mask]
                .groupby(["generator_direction", "load_direction"])[
                    "comparison_survival_pct"
                ]
                .mean()
                .unstack("load_direction")
            )
            show_heatmap(
                direction_table,
                "Generator × load message direction",
                "Load direction",
                "Generator direction",
            )
            """
        ),
        markdown(
            """
            ## Logical-configuration ranking across seeds

            This is the table to use when selecting configurations for the next
            stage. Prefer configurations with multiple completed seeds; a
            high-scoring single-seed structural or directional result remains a
            screening lead, not a confirmed winner.
            """
        ),
        code(
            """
            logical_ranking = (
                valid_endpoint_df.groupby(
                    ["factor_key", "config_label", *FACTOR_COLUMNS, "structure"],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    mean_survival_pct=("comparison_survival_pct", "mean"),
                    std_survival_pct=("comparison_survival_pct", "std"),
                    median_survival_pct=("comparison_survival_pct", "median"),
                    n_seeds=("seed", "nunique"),
                    n_runs=("run_name", "nunique"),
                )
                .sort_values(
                    ["mean_survival_pct", "n_seeds"],
                    ascending=[False, False],
                )
            )
            display(logical_ranking.drop(columns=["factor_key"]).round(2))
            """
        ),
        markdown(
            """
            ## Optional CSV export

            Set `EXPORT_TABLES = True` to save the coverage, endpoint ranking,
            marginal summaries, matched effects, and logical ranking under
            `Topology_Task/outputs/gnn_graph_screening_summary`.
            """
        ),
        code(
            """
            EXPORT_TABLES = False

            if EXPORT_TABLES:
                export_dir = wm.TASK_DIR / "outputs" / "gnn_graph_screening_summary"
                export_dir.mkdir(parents=True, exist_ok=True)
                coverage.to_csv(export_dir / "coverage.csv", index=False)
                endpoint_df.to_csv(export_dir / "run_endpoints.csv", index=False)
                marginal_summary.to_csv(
                    export_dir / "marginal_parameter_summary.csv", index=False
                )
                logical_ranking.to_csv(
                    export_dir / "logical_configuration_ranking.csv", index=False
                )
                if not paired_effect_summary.empty:
                    paired_effect_summary.to_csv(
                        export_dir / "matched_parameter_effects.csv", index=False
                    )
                    paired_effect_details.to_csv(
                        export_dir / "matched_parameter_pairs.csv", index=False
                    )
                print("Saved summary tables to:", export_dir)
            """
        ),
    ]
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
