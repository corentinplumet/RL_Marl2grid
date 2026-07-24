"""Reusable episodic-survival extraction and comparison plots.

The main plotting function accepts a tidy dataframe with one row per
``(run_name, step)`` and arbitrary metadata columns.  This keeps it independent
from a particular experiment naming convention and makes it suitable for
notebooks that compare stages, graph representations, normalizations, or any
other catalogued parameter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots


DEFAULT_SURVIVAL_METRICS = {
    "test": (
        "test/charts/episodic_survival",
        "test/episodic_survival",
        "charts/episodic_survival",
    ),
    "train_eval": (
        "train_eval/charts/episodic_survival",
        "train_eval/episodic_survival",
    ),
    "validation": ("validation/episodic_survival",),
    "legacy": ("charts/episodic_survival",),
}

DEFAULT_COLORS = tuple(
    qualitative.Plotly
    + qualitative.Safe
    + qualitative.Dark24
    + qualitative.Light24
)
DEFAULT_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _smooth_series(values: pd.Series, window: int, method: str) -> pd.Series:
    window = max(1, int(window))
    method = str(method).lower()
    if window == 1 or method in {"none", "raw"}:
        return values.astype(float)
    if method == "trailing":
        return values.rolling(window, min_periods=1).mean()
    if method == "centered":
        return values.rolling(window, min_periods=1, center=True).mean()
    if method == "ewm":
        return values.ewm(span=window, adjust=False, min_periods=1).mean()
    raise ValueError(
        "smooth_method must be one of 'trailing', 'centered', 'ewm', or 'none'."
    )


def _metric_candidates(metric: str | Sequence[str] | None, split: str) -> list[str]:
    if metric is not None:
        return _as_list(metric)
    if split not in DEFAULT_SURVIVAL_METRICS:
        raise ValueError(
            f"Unknown split {split!r}. Choose from {sorted(DEFAULT_SURVIVAL_METRICS)} "
            "or pass metric explicitly."
        )
    return list(DEFAULT_SURVIVAL_METRICS[split])


def extract_survival_curves(
    history: pd.DataFrame,
    *,
    catalog: pd.DataFrame | None = None,
    split: str = "test",
    metric: str | Sequence[str] | None = None,
    run_col: str = "run_name",
    step_col: str = "step",
    metric_col: str = "metric",
    value_col: str = "value",
    value_scale: float = 100.0,
    smooth: int = 1,
    smooth_method: str = "trailing",
) -> pd.DataFrame:
    """Extract one survival curve per run from long or wide W&B history.

    Parameters
    ----------
    history:
        Either the long ``wandb_metrics.history_df`` form with ``metric`` and
        ``value`` columns, or a wide history dataframe whose metric names are
        columns.
    catalog:
        Optional run metadata.  It is merged by ``run_col`` and can contain
        columns such as stage, seed, graph type, normalization, or directions.
    metric:
        One metric name or an ordered list.  The first metric available for each
        run is used.  When omitted, candidates are selected from ``split``.
    value_scale:
        Multiply logged values by this amount.  Survival is normally logged in
        [0, 1], so the default converts it to percentage points.
    smooth:
        Per-run smoothing window.  Keep this at 1 when you want to choose the
        smoothing independently in each call to :func:`plot_survival_comparison`.
    """

    if history is None or history.empty:
        return pd.DataFrame(
            columns=[
                run_col,
                "step",
                "metric",
                "raw_survival_pct",
                "survival_pct",
            ]
        )
    if run_col not in history:
        raise KeyError(f"History is missing required run column {run_col!r}.")

    candidates = _metric_candidates(metric, split)
    long_form = metric_col in history.columns and value_col in history.columns
    wide_step_col = step_col
    if not long_form and wide_step_col not in history:
        for candidate in ("charts/global_step", "_step", "global_step"):
            if candidate in history:
                wide_step_col = candidate
                break
        else:
            raise KeyError(
                f"Wide history has no {step_col!r}, 'charts/global_step', "
                "'_step', or 'global_step' column."
            )

    frames: list[pd.DataFrame] = []
    for run_name, run_history in history.groupby(run_col, sort=False):
        selected_metric = None
        selected = pd.DataFrame()
        for candidate in candidates:
            if long_form:
                candidate_frame = run_history[
                    run_history[metric_col].astype(str).eq(candidate)
                ][[step_col, value_col]].copy()
                candidate_frame = candidate_frame.rename(
                    columns={step_col: "step", value_col: "raw_value"}
                )
            elif candidate in run_history:
                candidate_frame = run_history[
                    [wide_step_col, candidate]
                ].copy()
                candidate_frame = candidate_frame.rename(
                    columns={wide_step_col: "step", candidate: "raw_value"}
                )
            else:
                candidate_frame = pd.DataFrame()
            if not candidate_frame.empty:
                selected_metric = candidate
                selected = candidate_frame
                break

        if selected.empty:
            continue
        selected["step"] = pd.to_numeric(selected["step"], errors="coerce")
        selected["raw_value"] = pd.to_numeric(
            selected["raw_value"], errors="coerce"
        )
        selected = (
            selected.dropna(subset=["step", "raw_value"])
            .groupby("step", as_index=False)["raw_value"]
            .mean()
            .sort_values("step")
        )
        if selected.empty:
            continue
        selected[run_col] = run_name
        selected["metric"] = selected_metric
        selected["raw_survival_pct"] = selected["raw_value"] * float(value_scale)
        selected["survival_pct"] = _smooth_series(
            selected["raw_survival_pct"], smooth, smooth_method
        )
        frames.append(selected.drop(columns="raw_value"))

    if not frames:
        return pd.DataFrame(
            columns=[
                run_col,
                "step",
                "metric",
                "raw_survival_pct",
                "survival_pct",
            ]
        )

    curves = pd.concat(frames, ignore_index=True)
    if catalog is not None and not catalog.empty:
        if run_col not in catalog:
            raise KeyError(f"Catalog is missing run column {run_col!r}.")
        metadata = catalog.drop_duplicates(run_col)
        overlap = [
            column
            for column in metadata.columns
            if column in curves.columns and column != run_col
        ]
        if overlap:
            metadata = metadata.drop(columns=overlap)
        curves = curves.merge(metadata, on=run_col, how="left", validate="many_to_one")
    return curves.sort_values([run_col, "step"]).reset_index(drop=True)


def filter_survival_curves(
    curves: pd.DataFrame,
    *,
    filters: Mapping[str, Any] | None = None,
    query: str | None = None,
    run_names: Sequence[str] | None = None,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    run_col: str = "run_name",
) -> pd.DataFrame:
    """Filter curves using exact values, value lists, callables, or a query.

    Examples
    --------
    ``filters={"stage": "Stage 1b", "seed": [0, 1]}``

    A callable receives the complete column Series and must return a boolean
    Series, for example ``filters={"normalization": lambda s: s != "n0_none"}``.
    """

    selected = curves.copy()
    if filters:
        for column, rule in filters.items():
            if column not in selected:
                raise KeyError(
                    f"Cannot filter by {column!r}; available columns are "
                    f"{sorted(selected.columns)}."
                )
            if callable(rule):
                mask = rule(selected[column])
                if not isinstance(mask, pd.Series):
                    mask = selected[column].map(rule)
                selected = selected[pd.Series(mask, index=selected.index).fillna(False)]
            elif isinstance(rule, (list, tuple, set, frozenset, np.ndarray, pd.Index)):
                selected = selected[selected[column].isin(list(rule))]
            else:
                selected = selected[selected[column].eq(rule)]
    if query:
        selected = selected.query(query)
    if run_names is not None:
        selected = selected[selected[run_col].isin(list(run_names))]
    if include_regex:
        selected = selected[
            selected[run_col].astype(str).str.contains(include_regex, regex=True)
        ]
    if exclude_regex:
        selected = selected[
            ~selected[run_col].astype(str).str.contains(exclude_regex, regex=True)
        ]
    return selected.copy()


def _format_group_label(
    row: Mapping[str, Any],
    *,
    group_by: Sequence[str],
    label_by: str | Sequence[str] | Callable[[Mapping[str, Any]], str] | None,
) -> str:
    if callable(label_by):
        return str(label_by(row))
    columns = _as_list(label_by) if label_by is not None else list(group_by)
    if not columns:
        return "all selected runs"
    if len(columns) == 1:
        return str(row[columns[0]])
    return " · ".join(f"{column}={row[column]}" for column in columns)


def _hex_to_rgba(color: str, opacity: float) -> str:
    color = str(color)
    if re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return (
            f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, "
            f"{int(color[5:7], 16)}, {float(opacity):.4f})"
        )
    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f", {opacity})")
    return color


def _resolve_style(
    labels: Sequence[str],
    supplied: Mapping[str, str] | Sequence[str] | None,
    defaults: Sequence[str],
) -> dict[str, str]:
    labels = list(dict.fromkeys(str(label) for label in labels))
    if isinstance(supplied, Mapping):
        return {
            label: str(supplied.get(label, defaults[index % len(defaults)]))
            for index, label in enumerate(labels)
        }
    palette = list(supplied) if supplied is not None else list(defaults)
    if not palette:
        palette = list(defaults)
    return {label: str(palette[index % len(palette)]) for index, label in enumerate(labels)}


def _baseline_mask(frame: pd.DataFrame, baseline: str | Mapping[str, Any]) -> pd.Series:
    if isinstance(baseline, str):
        return frame["curve_label"].astype(str).eq(baseline)
    mask = pd.Series(True, index=frame.index)
    for column, value in baseline.items():
        if column not in frame:
            raise KeyError(
                f"Baseline uses unknown column {column!r}; available columns are "
                f"{sorted(frame.columns)}."
            )
        mask &= frame[column].eq(value)
    return mask


def _difference_from_baseline(
    summary: pd.DataFrame,
    *,
    baseline: str | Mapping[str, Any],
    facet_col: str | None,
    uncertainty: str | None,
) -> pd.DataFrame:
    baseline_rows = summary[_baseline_mask(summary, baseline)].copy()
    facet_keys = [facet_col] if facet_col else []
    identities = baseline_rows[[*facet_keys, "curve_label"]].drop_duplicates()
    counts = identities.groupby(facet_keys, dropna=False).size() if facet_keys else len(identities)
    invalid = bool((counts != 1).any()) if isinstance(counts, pd.Series) else counts != 1
    if baseline_rows.empty or invalid:
        raise ValueError(
            "The baseline must identify exactly one curve"
            + (" in every facet." if facet_col else ".")
        )

    baseline_table = baseline_rows[
        [*facet_keys, "step", "center", "std", "n_members", "curve_label"]
    ].rename(
        columns={
            "center": "baseline_center",
            "std": "baseline_std",
            "n_members": "baseline_n_members",
            "curve_label": "baseline_label",
        }
    )
    transformed = summary.merge(
        baseline_table,
        on=[*facet_keys, "step"],
        how="inner",
        validate="many_to_one",
    )
    transformed["center"] = (
        transformed["center"] - transformed["baseline_center"]
    )

    uncertainty = None if uncertainty is None else str(uncertainty).lower()
    if uncertainty in {"std", "sem", "ci95"}:
        variance = transformed["std"].fillna(0.0).pow(2)
        baseline_variance = transformed["baseline_std"].fillna(0.0).pow(2)
        if uncertainty in {"sem", "ci95"}:
            variance = variance / transformed["n_members"].clip(lower=1)
            baseline_variance = (
                baseline_variance
                / transformed["baseline_n_members"].clip(lower=1)
            )
        radius = np.sqrt(variance + baseline_variance)
        if uncertainty == "ci95":
            radius = 1.96 * radius
        transformed["lower"] = transformed["center"] - radius
        transformed["upper"] = transformed["center"] + radius
    else:
        transformed["lower"] = np.nan
        transformed["upper"] = np.nan

    is_baseline = transformed["curve_label"].eq(transformed["baseline_label"])
    transformed.loc[is_baseline, ["center", "lower", "upper"]] = 0.0
    return transformed


def prepare_survival_comparison(
    curves: pd.DataFrame,
    *,
    filters: Mapping[str, Any] | None = None,
    query: str | None = None,
    run_names: Sequence[str] | None = None,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    group_by: str | Sequence[str] = "run_name",
    label_by: str | Sequence[str] | Callable[[Mapping[str, Any]], str] | None = None,
    label_map: Mapping[str, str] | None = None,
    facet_by: str | None = None,
    member_col: str = "run_name",
    step_col: str = "step",
    value_col: str = "survival_pct",
    smooth: int = 5,
    smooth_method: str = "trailing",
    center: str = "mean",
    uncertainty: str | None = "std",
    min_members: int | str = 1,
    require_reached_step: float | None = None,
    start_step: float | None = None,
    end_step: float | None = None,
    common_horizon: bool = False,
    comparison: str = "absolute",
    baseline: str | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select, smooth, and aggregate survival curves for plotting.

    Returns ``(summary, members)``.  ``summary`` contains the center and
    uncertainty bounds for every plotted curve.  ``members`` contains the
    selected per-run curves after smoothing.
    """

    selected = filter_survival_curves(
        curves,
        filters=filters,
        query=query,
        run_names=run_names,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        run_col=member_col,
    )
    required = {member_col, step_col, value_col}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise KeyError(f"Survival data is missing required columns: {missing}")
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected[step_col] = pd.to_numeric(selected[step_col], errors="coerce")
    selected[value_col] = pd.to_numeric(selected[value_col], errors="coerce")
    selected = selected.dropna(subset=[member_col, step_col, value_col])
    selected = (
        selected.sort_values([member_col, step_col])
        .drop_duplicates([member_col, step_col], keep="last")
        .copy()
    )

    if require_reached_step is not None:
        completed = (
            selected.groupby(member_col)[step_col].max().loc[
                lambda values: values >= float(require_reached_step)
            ]
        )
        selected = selected[selected[member_col].isin(completed.index)]
    if start_step is not None:
        selected = selected[selected[step_col] >= float(start_step)]
    if end_step is not None:
        selected = selected[selected[step_col] <= float(end_step)]
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected["plot_value"] = selected.groupby(member_col, sort=False)[
        value_col
    ].transform(lambda values: _smooth_series(values, smooth, smooth_method))

    group_columns = _as_list(group_by)
    for column in group_columns:
        if column not in selected:
            raise KeyError(f"Unknown group_by column {column!r}.")
    if facet_by:
        if facet_by not in selected:
            raise KeyError(f"Unknown facet_by column {facet_by!r}.")
        if facet_by not in group_columns:
            group_columns.append(facet_by)
    if not group_columns:
        selected["_all_selected"] = "all selected runs"
        group_columns = ["_all_selected"]

    group_member_columns = [*group_columns, member_col]
    member_metadata = (
        selected[group_member_columns].drop_duplicates().groupby(
            group_columns, dropna=False, observed=True
        )[member_col].nunique().rename("expected_members").reset_index()
    )

    if common_horizon:
        member_horizons = (
            selected.groupby(group_member_columns, dropna=False, observed=True)[
                step_col
            ]
            .max()
            .rename("member_horizon")
            .reset_index()
        )
        common_horizons = (
            member_horizons.groupby(group_columns, dropna=False, observed=True)[
                "member_horizon"
            ]
            .min()
            .rename("common_horizon")
            .reset_index()
        )
        selected = selected.merge(common_horizons, on=group_columns, how="left")
        selected = selected[selected[step_col] <= selected["common_horizon"]]

    member_columns = [*group_columns, member_col, step_col]
    members = (
        selected.groupby(member_columns, dropna=False, observed=True, as_index=False)
        .agg(plot_value=("plot_value", "mean"))
        .sort_values([*group_columns, member_col, step_col])
    )
    # Preserve metadata used for labels, hover, and filters when constant per run.
    metadata_candidates = [
        column
        for column in selected.columns
        if column not in members.columns
        and column not in {value_col, "plot_value", "common_horizon"}
    ]
    metadata_columns = [
        column
        for column in metadata_candidates
        if selected.groupby(member_col, dropna=False)[column]
        .nunique(dropna=False)
        .le(1)
        .all()
    ]
    if metadata_columns:
        metadata = selected[
            [member_col, *metadata_columns]
        ].drop_duplicates(member_col)
        members = members.merge(metadata, on=member_col, how="left")

    grouped = members.groupby(
        [*group_columns, step_col], dropna=False, observed=True
    )["plot_value"]
    summary = grouped.agg(
        mean="mean",
        median="median",
        std="std",
        minimum="min",
        maximum="max",
        n_members="count",
    ).reset_index()
    quantiles = grouped.quantile([0.25, 0.75]).unstack().reset_index()
    quantiles = quantiles.rename(columns={0.25: "q25", 0.75: "q75"})
    summary = summary.merge(
        quantiles, on=[*group_columns, step_col], how="left"
    ).merge(member_metadata, on=group_columns, how="left")

    center = str(center).lower()
    if center not in {"mean", "median"}:
        raise ValueError("center must be 'mean' or 'median'.")
    summary["center"] = summary[center]
    uncertainty_key = None if uncertainty is None else str(uncertainty).lower()
    if uncertainty_key in {None, "none"}:
        summary["lower"] = np.nan
        summary["upper"] = np.nan
    elif uncertainty_key == "std":
        summary["lower"] = summary["center"] - summary["std"].fillna(0.0)
        summary["upper"] = summary["center"] + summary["std"].fillna(0.0)
    elif uncertainty_key == "sem":
        radius = summary["std"].fillna(0.0) / np.sqrt(
            summary["n_members"].clip(lower=1)
        )
        summary["lower"] = summary["center"] - radius
        summary["upper"] = summary["center"] + radius
    elif uncertainty_key == "ci95":
        radius = 1.96 * summary["std"].fillna(0.0) / np.sqrt(
            summary["n_members"].clip(lower=1)
        )
        summary["lower"] = summary["center"] - radius
        summary["upper"] = summary["center"] + radius
    elif uncertainty_key == "minmax":
        summary["lower"] = summary["minimum"]
        summary["upper"] = summary["maximum"]
    elif uncertainty_key == "iqr":
        summary["lower"] = summary["q25"]
        summary["upper"] = summary["q75"]
    else:
        raise ValueError(
            "uncertainty must be one of None, 'std', 'sem', 'ci95', "
            "'minmax', or 'iqr'."
        )

    if min_members == "all":
        summary = summary[
            summary["n_members"].eq(summary["expected_members"])
        ]
    else:
        minimum = int(min_members)
        if minimum < 1:
            raise ValueError("min_members must be a positive integer or 'all'.")
        summary = summary[summary["n_members"] >= minimum]

    label_rows = summary[group_columns].drop_duplicates().copy()
    label_rows["curve_label"] = [
        _format_group_label(
            row,
            group_by=[column for column in group_columns if column != facet_by],
            label_by=label_by,
        )
        for row in label_rows.to_dict("records")
    ]
    if label_map:
        label_rows["curve_label"] = label_rows["curve_label"].map(
            lambda label: str(label_map.get(label, label))
        )
    summary = summary.merge(label_rows, on=group_columns, how="left")
    members = members.merge(label_rows, on=group_columns, how="left")

    comparison = str(comparison).lower()
    if comparison == "difference":
        if baseline is None:
            raise ValueError("comparison='difference' requires baseline.")
        if center != "mean":
            raise ValueError(
                "Difference plots currently require center='mean' so their "
                "uncertainty can be combined consistently."
            )
        summary = _difference_from_baseline(
            summary,
            baseline=baseline,
            facet_col=facet_by,
            uncertainty=uncertainty_key,
        )
    elif comparison != "absolute":
        raise ValueError("comparison must be 'absolute' or 'difference'.")

    return (
        summary.sort_values([*( [facet_by] if facet_by else []), "curve_label", step_col]),
        members.sort_values([*( [facet_by] if facet_by else []), "curve_label", member_col, step_col]),
    )


def plot_survival_comparison(
    curves: pd.DataFrame,
    *,
    filters: Mapping[str, Any] | None = None,
    query: str | None = None,
    run_names: Sequence[str] | None = None,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    group_by: str | Sequence[str] = "run_name",
    label_by: str | Sequence[str] | Callable[[Mapping[str, Any]], str] | None = None,
    label_map: Mapping[str, str] | None = None,
    facet_by: str | None = None,
    facet_order: Sequence[Any] | None = None,
    member_col: str = "run_name",
    step_col: str = "step",
    value_col: str = "survival_pct",
    smooth: int = 5,
    smooth_method: str = "trailing",
    center: str = "mean",
    uncertainty: str | None = "std",
    min_members: int | str = 1,
    require_reached_step: float | None = None,
    start_step: float | None = None,
    end_step: float | None = None,
    common_horizon: bool = False,
    comparison: str = "absolute",
    baseline: str | Mapping[str, Any] | None = None,
    show_members: bool = False,
    show_points: bool = False,
    colors: Mapping[str, str] | Sequence[str] | None = None,
    dashes: Mapping[str, str] | Sequence[str] | None = None,
    highlight: Sequence[str] | None = None,
    line_width: float = 2.8,
    highlight_width: float = 4.5,
    member_width: float = 1.0,
    member_opacity: float = 0.18,
    band_opacity: float = 0.14,
    title: str | None = None,
    x_title: str = "Environment steps (millions)",
    y_title: str | None = None,
    y_range: Sequence[float] | None = (0, 100),
    x_in_millions: bool = True,
    budget_step: float | None = None,
    width: int = 1350,
    height: int | None = None,
    ncols: int = 2,
    shared_xaxes: bool = True,
    shared_yaxes: bool = True,
    template: str = "plotly_white",
    hovermode: str = "x unified",
    legend_orientation: str = "h",
    save_path: str | Path | None = None,
    show: bool = False,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, pd.DataFrame]:
    """Plot highly configurable per-run or aggregated survival comparisons.

    The most useful controls are:

    - ``filters`` / ``query`` / ``run_names`` for selection;
    - ``group_by`` and ``label_by`` for deciding which runs form one curve;
    - ``smooth`` and ``smooth_method`` for per-run smoothing;
    - ``uncertainty`` for ``std``, ``sem``, ``ci95``, ``minmax``, or ``iqr``;
    - ``show_members`` to show thin individual-seed curves behind each mean;
    - ``facet_by`` for small multiples;
    - ``comparison='difference'`` with ``baseline`` for a delta curve;
    - ``colors``, ``dashes``, and ``highlight`` for presentation.

    Set ``return_data=True`` to receive ``(figure, aggregated_dataframe)``.
    """

    summary, members = prepare_survival_comparison(
        curves,
        filters=filters,
        query=query,
        run_names=run_names,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        group_by=group_by,
        label_by=label_by,
        label_map=label_map,
        facet_by=facet_by,
        member_col=member_col,
        step_col=step_col,
        value_col=value_col,
        smooth=smooth,
        smooth_method=smooth_method,
        center=center,
        uncertainty=uncertainty,
        min_members=min_members,
        require_reached_step=require_reached_step,
        start_step=start_step,
        end_step=end_step,
        common_horizon=common_horizon,
        comparison=comparison,
        baseline=baseline,
    )

    if facet_by and not summary.empty:
        observed_facets = list(summary[facet_by].drop_duplicates())
        requested_order = list(facet_order or [])
        facets = [
            *[value for value in requested_order if value in observed_facets],
            *[value for value in observed_facets if value not in requested_order],
        ]
    else:
        facets = [None]
    ncols = 1 if not facet_by else max(1, min(int(ncols), len(facets)))
    nrows = int(np.ceil(len(facets) / ncols))
    if facet_by:
        figure = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[str(value) for value in facets],
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
        )
    else:
        figure = go.Figure()

    if summary.empty:
        figure.add_annotation(
            text="No survival data matched the requested selection.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    labels = list(summary.get("curve_label", pd.Series(dtype=str)).drop_duplicates())
    color_map = _resolve_style(labels, colors, DEFAULT_COLORS)
    dash_map = _resolve_style(labels, dashes, DEFAULT_DASHES)
    highlighted = set(str(item) for item in _as_list(highlight))
    legend_seen: set[str] = set()
    comparison_key = str(comparison).lower()

    for facet_index, facet_value in enumerate(facets):
        row = facet_index // ncols + 1
        col = facet_index % ncols + 1
        facet_summary = (
            summary[summary[facet_by].eq(facet_value)]
            if facet_by
            else summary
        )
        facet_members = (
            members[members[facet_by].eq(facet_value)]
            if facet_by and not members.empty
            else members
        )
        for label, curve in facet_summary.groupby("curve_label", sort=False):
            label = str(label)
            color = color_map[label]
            is_highlighted = not highlighted or label in highlighted
            opacity = 1.0 if is_highlighted else 0.24
            trace_kwargs = {"row": row, "col": col} if facet_by else {}
            if show_members and comparison_key == "absolute":
                group_members = facet_members[
                    facet_members["curve_label"].astype(str).eq(label)
                ]
                for _, member in group_members.groupby(member_col, sort=False):
                    x_values = (
                        member[step_col] / 1_000_000
                        if x_in_millions
                        else member[step_col]
                    )
                    figure.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=member["plot_value"],
                            mode="lines",
                            line={"color": color, "width": member_width},
                            opacity=member_opacity * opacity,
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=label,
                        ),
                        **trace_kwargs,
                    )

            x_values = (
                curve[step_col] / 1_000_000
                if x_in_millions
                else curve[step_col]
            )
            has_band = curve[["lower", "upper"]].notna().any().all()
            if has_band:
                figure.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=curve["lower"],
                        mode="lines",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=label,
                    ),
                    **trace_kwargs,
                )
                figure.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=curve["upper"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor=_hex_to_rgba(color, band_opacity * opacity),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=label,
                    ),
                    **trace_kwargs,
                )

            customdata = np.column_stack(
                [
                    curve["n_members"].to_numpy(),
                    curve["std"].fillna(0.0).to_numpy(),
                    curve["lower"].to_numpy(),
                    curve["upper"].to_numpy(),
                ]
            )
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=curve["center"],
                    mode="lines+markers" if show_points else "lines",
                    name=label,
                    legendgroup=label,
                    showlegend=label not in legend_seen,
                    line={
                        "color": color,
                        "dash": dash_map[label],
                        "width": highlight_width if label in highlighted else line_width,
                    },
                    marker={"size": 5},
                    opacity=opacity,
                    customdata=customdata,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        + (
                            "step: %{x:.3f}M<br>"
                            if x_in_millions
                            else "step: %{x:,.0f}<br>"
                        )
                        + "value: %{y:.2f}<br>"
                        + "members: %{customdata[0]:.0f}<br>"
                        + "std: %{customdata[1]:.2f}"
                        + "<extra></extra>"
                    ),
                ),
                **trace_kwargs,
            )
            legend_seen.add(label)

        if budget_step is not None:
            x_budget = (
                float(budget_step) / 1_000_000
                if x_in_millions
                else float(budget_step)
            )
            if facet_by:
                figure.add_vline(
                    x=x_budget,
                    line_dash="dot",
                    line_color="#555555",
                    row=row,
                    col=col,
                )
            else:
                figure.add_vline(
                    x=x_budget, line_dash="dot", line_color="#555555"
                )
        if comparison_key == "difference":
            if facet_by:
                figure.add_hline(
                    y=0, line_dash="dot", line_color="#555555", row=row, col=col
                )
            else:
                figure.add_hline(y=0, line_dash="dot", line_color="#555555")

    if y_title is None:
        y_title = (
            "Survival difference from baseline (percentage points)"
            if comparison_key == "difference"
            else "Test episodic survival (%)"
        )
    if height is None:
        height = 570 if not facet_by else max(520, 430 * nrows)
    figure.update_layout(
        title=title,
        template=template,
        width=width,
        height=height,
        hovermode=hovermode,
        legend={
            "orientation": legend_orientation,
            "yanchor": "bottom" if legend_orientation == "h" else "top",
            "y": 1.02 if legend_orientation == "h" else 1.0,
            "xanchor": "left",
            "x": 0,
        },
        margin={"l": 70, "r": 30, "t": 100, "b": 60},
    )
    figure.update_xaxes(title_text=x_title)
    figure.update_yaxes(title_text=y_title, range=list(y_range) if y_range else None)

    if save_path:
        output = Path(save_path).expanduser()
        if output.suffix.lower() != ".html":
            output = output.with_suffix(".html")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(output, include_plotlyjs="cdn")
        print(f"Saved survival comparison: {output.resolve()}")
    if show:
        figure.show()
    if return_data:
        return figure, summary
    return figure


__all__ = [
    "DEFAULT_SURVIVAL_METRICS",
    "extract_survival_curves",
    "filter_survival_curves",
    "prepare_survival_comparison",
    "plot_survival_comparison",
]
