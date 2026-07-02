"""Extract rerun episodic-survival Plotly outputs from the notebook as PNGs."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    REPO_ROOT
    / "Topology_Task"
    / "analysis"
    / "metrics"
    / "notebooks"
    / "episodic_survival"
    / "rerun_episodic_survival_plots.ipynb"
)
OUT_DIR = REPO_ROOT / "latex" / "figures"

FIGURES = {
    8: "rerun_gine_best00_vs_best10_14.png",
    11: "rerun_a0_core_pairwise_comparisons.png",
    12: "rerun_a0_core_all_curves.png",
    14: "rerun_a0_bias_comparison.png",
}

DTYPE_MAP = {
    "f8": np.float64,
    "f4": np.float32,
    "i8": np.int64,
    "i4": np.int32,
    "i2": np.int16,
    "u4": np.uint32,
    "u2": np.uint16,
    "u1": np.uint8,
}


def decode_array(value):
    if isinstance(value, dict) and "bdata" in value:
        dtype = DTYPE_MAP[value.get("dtype")]
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=dtype)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    return np.asarray([], dtype=float)


def axis_number(axis_name: str | None) -> int:
    if not axis_name:
        return 1
    match = re.search(r"(\d+)$", axis_name)
    return int(match.group(1)) if match else 1


def axis_layout_key(prefix: str, number: int) -> str:
    return prefix if number == 1 else f"{prefix}{number}"


def rgba_to_mpl(value: str | None, fallback: str = "#999999"):
    if not value:
        return fallback, 0.12
    if value.startswith("rgba"):
        nums = [float(part.strip()) for part in value[value.find("(") + 1 : value.rfind(")")].split(",")]
        return tuple(n / 255 for n in nums[:3]), nums[3]
    return value, 0.12


def line_style(dash: str | None) -> str:
    return {
        "dash": "--",
        "dot": ":",
        "dashdot": "-.",
    }.get(dash or "solid", "-")


def title_for_axis(fig, axis_num: int, order: list[int]) -> str:
    annotations = fig.get("layout", {}).get("annotations", [])
    if not annotations:
        return ""
    idx = order.index(axis_num)
    sorted_annotations = sorted(
        annotations,
        key=lambda item: (-float(item.get("y", 0)), float(item.get("x", 0))),
    )
    if idx < len(sorted_annotations):
        return sorted_annotations[idx].get("text", "")
    return ""


def subplot_order(fig, axes_with_data: set[int]) -> tuple[list[int], int, int]:
    layout = fig.get("layout", {})
    axis_info = []
    for axis_num in axes_with_data:
        xaxis = layout.get(axis_layout_key("xaxis", axis_num), {})
        yaxis = layout.get(axis_layout_key("yaxis", axis_num), {})
        x_domain = xaxis.get("domain", [0, 1])
        y_domain = yaxis.get("domain", [0, 1])
        axis_info.append((axis_num, float(y_domain[0]), float(x_domain[0])))

    ys = sorted({round(item[1], 6) for item in axis_info}, reverse=True)
    xs = sorted({round(item[2], 6) for item in axis_info})
    rows = max(1, len(ys))
    cols = max(1, len(xs))
    order = [
        item[0]
        for item in sorted(
            axis_info,
            key=lambda item: (ys.index(round(item[1], 6)), xs.index(round(item[2], 6))),
        )
    ]
    return order, rows, cols


def render_figure(fig, out_path: Path) -> None:
    axes_with_data = {
        axis_number(trace.get("xaxis"))
        for trace in fig.get("data", [])
        if len(decode_array(trace.get("x"))) and len(decode_array(trace.get("y")))
    }
    order, rows, cols = subplot_order(fig, axes_with_data)

    width = 11.5 if cols > 1 else 8.5
    height = 3.1 * rows if rows > 1 else 4.8
    figure, ax_grid = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    axis_lookup = {}
    for idx, axis_num in enumerate(order):
        row, col = divmod(idx, cols)
        axis_lookup[axis_num] = ax_grid[row][col]

    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(order):
                ax_grid[row][col].axis("off")

    for trace in fig.get("data", []):
        axis_num = axis_number(trace.get("xaxis"))
        ax = axis_lookup.get(axis_num)
        if ax is None:
            continue
        x = decode_array(trace.get("x"))
        y = decode_array(trace.get("y"))
        if len(x) == 0 or len(y) == 0:
            continue

        name = trace.get("name", "")
        line = trace.get("line", {})
        color = line.get("color", "#333333")
        dash = line_style(line.get("dash"))

        if trace.get("fill") == "toself":
            fill_color, alpha = rgba_to_mpl(trace.get("fillcolor"), fallback=color)
            ax.fill(x, y, color=fill_color, alpha=alpha, linewidth=0)
            continue

        if name.endswith(" member"):
            ax.plot(
                x,
                y,
                color=color,
                linestyle=dash,
                linewidth=0.8,
                alpha=float(trace.get("opacity", 0.16)),
            )
            continue

        ax.plot(
            x,
            y,
            color=color,
            linestyle=dash,
            linewidth=2.0 if trace.get("showlegend") else 1.0,
            alpha=float(trace.get("opacity", 1.0)),
            label=name,
        )

    for axis_num, ax in axis_lookup.items():
        title = title_for_axis(fig, axis_num, order)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training steps (M)")
        ax.set_ylabel("Episodic survival (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.8)
        ax.legend(loc="best", fontsize=7, frameon=True)

    figure.suptitle(fig.get("layout", {}).get("title", {}).get("text", ""), fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def collect_summary(fig):
    rows = []
    for trace in fig.get("data", []):
        if not trace.get("showlegend"):
            continue
        x = decode_array(trace.get("x"))
        y = decode_array(trace.get("y"))
        if len(x) == 0 or len(y) == 0:
            continue
        rows.append(
            {
                "name": trace.get("name", ""),
                "axis": axis_number(trace.get("xaxis")),
                "final_step_m": float(x[-1]),
                "final_survival": float(y[-1]),
                "peak_survival": float(np.nanmax(y)),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(NOTEBOOK.read_text())
    summaries = {}
    for cell_idx, filename in FIGURES.items():
        fig = None
        for output in notebook["cells"][cell_idx].get("outputs", []):
            fig = output.get("data", {}).get("application/vnd.plotly.v1+json") or fig
        if fig is None:
            raise RuntimeError(f"No Plotly figure found in notebook cell {cell_idx}")
        render_figure(fig, OUT_DIR / filename)
        summaries[filename] = collect_summary(fig)

    summary_path = OUT_DIR / "rerun_survival_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {len(FIGURES)} figures to {OUT_DIR}")
    print(f"Wrote summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
