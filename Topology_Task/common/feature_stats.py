"""Distribution summaries and two-sample shift metrics for feature streams.

Kept free of Grid2Op and torch imports so the statistics can be exercised on
their own.
"""

from typing import Any, Dict, List, Sequence

import numpy as np

QUANTILES: Sequence[float] = (0.01, 0.25, 0.5, 0.75, 0.99)


def describe(values: np.ndarray) -> Dict[str, float]:
    """Summarize one feature stream."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        empty = {f"p{int(q * 100):02d}": float("nan") for q in QUANTILES}
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), **empty, "max": float("nan")}
    quantiles = np.quantile(values, QUANTILES)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        **{f"p{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, quantiles)},
        "max": float(np.max(values)),
    }


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, without a scipy dependency."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein1(a: np.ndarray, b: np.ndarray, n_points: int = 1000) -> float:
    """1-D Wasserstein distance via matched quantiles."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = (np.arange(n_points) + 0.5) / n_points
    return float(np.mean(np.abs(np.quantile(a, grid) - np.quantile(b, grid))))


def compare_samples(
    samples_a: Dict[str, np.ndarray],
    samples_b: Dict[str, np.ndarray],
    label_a: str,
    label_b: str,
) -> List[Dict[str, Any]]:
    """Per-feature shift metrics between two sampled environments."""
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(samples_a) | set(samples_b)):
        a = np.asarray(samples_a.get(key, np.empty(0)), dtype=np.float64)
        b = np.asarray(samples_b.get(key, np.empty(0)), dtype=np.float64)
        stats_a = describe(a)
        stats_b = describe(b)

        # Pooled spread keeps the shift readable for features whose scale
        # differs by orders of magnitude between the two grids.
        pooled = float(np.sqrt(0.5 * (stats_a["std"] ** 2 + stats_b["std"] ** 2)))
        usable = np.isfinite(pooled) and pooled > 1e-12
        w1 = wasserstein1(a, b)
        rows.append(
            {
                "feature": key,
                "present_in_both": bool(a.size and b.size),
                f"mean_{label_a}": stats_a["mean"],
                f"mean_{label_b}": stats_b["mean"],
                f"std_{label_a}": stats_a["std"],
                f"std_{label_b}": stats_b["std"],
                f"p01_{label_a}": stats_a["p01"],
                f"p01_{label_b}": stats_b["p01"],
                f"p99_{label_a}": stats_a["p99"],
                f"p99_{label_b}": stats_b["p99"],
                "std_mean_shift": (
                    (stats_b["mean"] - stats_a["mean"]) / pooled
                    if usable
                    else float("nan")
                ),
                "std_ratio": (
                    stats_b["std"] / stats_a["std"]
                    if stats_a["std"] > 1e-12
                    else float("nan")
                ),
                "ks": ks_statistic(a, b),
                "w1_over_pooled_std": w1 / pooled if usable else float("nan"),
                f"n_{label_a}": stats_a["n"],
                f"n_{label_b}": stats_b["n"],
            }
        )
    return rows


def markdown_table(
    rows: List[Dict[str, Any]], label_a: str, label_b: str
) -> str:
    """Render comparison rows worst-shift-first."""
    columns = [
        "feature",
        f"mean_{label_a}",
        f"mean_{label_b}",
        f"std_{label_a}",
        f"std_{label_b}",
        "std_mean_shift",
        "std_ratio",
        "ks",
        "w1_over_pooled_std",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    ordered = sorted(
        rows,
        key=lambda row: -(row["ks"] if np.isfinite(row["ks"]) else -1.0),
    )
    for row in ordered:
        cells = [
            row[column] if isinstance(row[column], str) else f"{row[column]:.4g}"
            for column in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

def plot_distributions(
    samples_a: Dict[str, np.ndarray],
    samples_b: Dict[str, np.ndarray],
    label_a: str,
    label_b: str,
    output,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [key for key in sorted(set(samples_a) & set(samples_b))]
    if not keys:
        return
    n_cols = 3
    n_rows = int(np.ceil(len(keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.9 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        a = samples_a[key]
        b = samples_b[key]
        combined = np.concatenate([a, b])
        low, high = np.quantile(combined, [0.005, 0.995])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(combined)), float(np.max(combined)) + 1e-6
        bins = np.linspace(low, high, 60)
        ax.hist(a, bins=bins, density=True, alpha=0.55, label=label_a)
        ax.hist(b, bins=bins, density=True, alpha=0.55, label=label_b)
        ax.set_title(key, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(keys) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(
        f"Actor graph feature distributions: {label_a} vs {label_b}", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=140)
    plt.close(fig)


