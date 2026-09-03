"""Build the compact WCCI figures and appendix tables used by Chapters 8--9.

The main figures use verified 50-episode artifacts and the same easy/difficult
cohort split as ``wcci_transfer_data``.  WCCI-trained policies use final
checkpoints in the main comparisons; target-selected best-test checkpoints are
kept in the appendix tables.
"""

from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcci_transfer_data as W


sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
    }
)

data = W.load()
wsc = W.load_wsc(data)
wmlp = W.load_wmlp(data)
wmlp_all_runs = wmlp.runs.copy()
wmlp.runs = wmlp.runs.loc[wmlp.runs["mk"].isin([64, 256])].copy()
runs = data.runs.copy()

latex_dir = data.task_dir.parent / "latex"
figure_dir = latex_dir / "figures"
table_dir = latex_dir / "tables"
figure_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

DN_HARD = float(data.dn_hard)
DN_OVERALL = float(data.dn_overall)
N_HARD = int(data.n_hard)
N_EASY = int(data.n_easy)
MAIN_CAPS = [32, 64, 128, 256]

CAP_COLORS = {32: "#4C78A8", 64: "#F58518", 128: "#72B7B2", 256: "#E45756"}
GATE_COLORS = {"ungated": "#4C78A8", "gated": "#E45756"}
GATE_LABELS = {"ungated": "Ungated", "gated": r"Local $\rho=0.95$"}
ROUTE_COLORS = {
    "WMLP": "#9C755F",
    "Fixed-list GNN": "#B279A2",
    "Candidate scorer, scratch": "#54A24B",
    "Zero-shot": "#4C78A8",
    "Fine-tuned": "#E4A11B",
    "Greedy": "#59A14F",
}
POLICY_ARM_COLORS = {
    "Plain candidate scorer": "#4C78A8",
    "Gmax-delta": "#B279A2",
    "Adaptive budget": "#72B7B2",
}
POLICY_VARIANT_COLORS = {
    ("Gmax-delta", "ungated"): "#B279A2",
    ("Gmax-delta", "gated"): "#7A3E72",
    ("Adaptive budget", "ungated"): "#72B7B2",
    ("Adaptive budget", "gated"): "#2A7F7A",
}


def save(fig, name):
    fig.savefig(figure_dir / name, bbox_inches="tight")
    plt.close(fig)


def bootstrap_mean(values, n_boot=20_000, seed=20260827):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(n_boot, len(values)))
    ].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def tex_escape(value):
    return str(value).replace("_", r"\_")


def arm_frame(label):
    if label == "plain":
        return runs.loc[
            (runs["route"] == "bus14 zero-shot")
            & ~runs["group"].str.startswith("cas_hl_gmax_delta")
        ].copy()
    if label == "gmax":
        return runs.loc[
            (runs["route"] == "bus14 zero-shot")
            & runs["group"].str.startswith("cas_hl_gmax_delta")
        ].copy()
    if label == "AIB":
        return runs.loc[runs["route"] == "bus14 zero-shot + AIB"].copy()
    raise KeyError(label)


def gate_scope(frame, gate):
    if gate == "ungated":
        return frame.loc[frame["gate"] == "ungated"].copy()
    return frame.loc[(frame["gate"] == "gated") & np.isclose(frame["rho"], 0.95)].copy()


def paired_arm_delta(treatment, gate):
    left = gate_scope(arm_frame(treatment), gate)[["variant", "mk", "hard_pct"]]
    right = gate_scope(arm_frame("plain"), gate)[["variant", "mk", "hard_pct"]]
    paired = left.merge(right, on=["variant", "mk"], suffixes=("_treatment", "_plain"))
    paired["delta"] = paired["hard_pct_treatment"] - paired["hard_pct_plain"]
    return paired["delta"]


def paired_gate_delta(frame, index_columns):
    scoped = frame.loc[
        (frame["gate"] == "ungated")
        | ((frame["gate"] == "gated") & np.isclose(frame["rho"], 0.95))
    ].copy()
    wide = scoped.pivot_table(
        index=index_columns, columns="gate", values="hard_pct", aggfunc="first"
    ).dropna(subset=["ungated", "gated"])
    return wide["gated"] - wide["ungated"]


def grouped_gate_boxes(
    ax, frame, metric, ylabel, title, reference_lines=(), show_counts=True
):
    """Draw both gate conditions and report the available cells at each cap."""
    rng = np.random.default_rng(20260827)
    offsets = {"ungated": -0.18, "gated": 0.18}
    counts = {}
    for cap_index, cap in enumerate(MAIN_CAPS):
        for gate in ("ungated", "gated"):
            values = frame.loc[(frame["mk"] == cap) & (frame["gate"] == gate), metric].dropna().to_numpy()
            counts[(cap, gate)] = len(values)
            if not len(values):
                continue
            position = cap_index + offsets[gate]
            ax.boxplot(
                values,
                positions=[position],
                widths=0.29,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={"facecolor": GATE_COLORS[gate], "alpha": 0.20, "edgecolor": GATE_COLORS[gate], "linewidth": 1.3},
                whiskerprops={"color": GATE_COLORS[gate], "linewidth": 1.1},
                capprops={"color": GATE_COLORS[gate], "linewidth": 1.1},
                medianprops={"color": GATE_COLORS[gate], "linewidth": 2.1},
            )
            jitter = rng.uniform(-0.065, 0.065, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=25,
                color=GATE_COLORS[gate],
                alpha=0.76,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
    for value, label, style, colour in reference_lines:
        ax.axhline(value, color=colour, linestyle=style, linewidth=1.25, label=label)
    labels = [f"mk{cap}" for cap in MAIN_CAPS]
    if show_counts:
        labels = [
            f"mk{cap}\n{counts[(cap, 'ungated')]} / {counts[(cap, 'gated')]}"
            for cap in MAIN_CAPS
        ]
    ax.set_xticks(range(len(MAIN_CAPS)), labels)
    ax.set_xlim(-0.55, len(MAIN_CAPS) - 0.45)
    ax.set_xlabel(
        "Reduced action space\n(ungated / gated architectures available)"
        if show_counts
        else "Reduced action space"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return counts


# ---------------------------------------------------------------------------
# Figure 7.1: target-grid controls trained from scratch.
# ---------------------------------------------------------------------------
wmlp_final = wmlp_all_runs.loc[
    (wmlp_all_runs["selection"] == "last")
    & (wmlp_all_runs["gate"] == "ungated")
    & (wmlp_all_runs["mk"].isin([32, 64, 256]))
].copy()
wsc_final = wsc.runs.loc[wsc.runs["checkpoint_kind"] == "final"].copy()
candidate_scratch = runs.loc[
    (runs["route"] == "MAPPO scratch")
    & (runs["selection"] == "last")
    & (runs["gate"] == "ungated")
    & (runs["mk"].isin([32, 64, 256]))
].copy()

controls = pd.concat(
    [
        wmlp_final.assign(model_family="Flat MLP baseline"),
        wsc_final.assign(model_family="Selected GNN encoder + MLP head"),
        candidate_scratch.assign(model_family="GNN encoder + action scorer"),
    ],
    ignore_index=True,
)
family_order = [
    "Flat MLP baseline",
    "Selected GNN encoder + MLP head",
    "GNN encoder + action scorer",
]
family_colors = {
    "Flat MLP baseline": "#9C755F",
    "Selected GNN encoder + MLP head": "#B279A2",
    "GNN encoder + action scorer": "#54A24B",
}
scratch_caps = [64]
bar_width = 0.22
family_offsets = {
    "Flat MLP baseline": -bar_width,
    "Selected GNN encoder + MLP head": 0.0,
    "GNN encoder + action scorer": bar_width,
}

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1))
for ax, metric, title, floor in (
    (axes[0], "overall_pct", "All 50 chronics", DN_OVERALL),
    (axes[1], "hard_pct", f"The {N_HARD} difficult chronics", DN_HARD),
):
    for cap_index, cap in enumerate(scratch_caps):
        for family in family_order:
            values = controls.loc[
                (controls["mk"] == cap) & (controls["model_family"] == family),
                metric,
            ].dropna().to_numpy()
            if not len(values):
                continue
            center = cap_index + family_offsets[family]
            family_mean = float(np.mean(values))
            error_kw = None
            if len(values) > 1:
                error_kw = {
                    "yerr": np.array(
                        [
                            [family_mean - float(values.min())],
                            [float(values.max()) - family_mean],
                        ]
                    ),
                    "capsize": 3.5,
                    "error_kw": {"elinewidth": 1.25, "ecolor": "#222222"},
                }
            ax.bar(
                center,
                family_mean,
                width=bar_width * 0.88,
                color=family_colors[family],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.92,
                zorder=3,
                **(error_kw or {}),
            )
            label_y = family_mean - (1.6 if metric == "overall_pct" else 0.45)
            ax.text(
                center,
                label_y,
                f"n={len(values)}",
                ha="center",
                va="top",
                fontsize=8.2,
                color="white",
                fontweight="bold",
                zorder=5,
            )
    ax.axhline(floor, color="#222222", linestyle="--", linewidth=1.3)
    ax.set_xticks(range(len(scratch_caps)), [f"$k={cap}$" for cap in scratch_caps])
    ax.set_xlim(-0.55, len(scratch_caps) - 0.45)
    ax.set_xlabel("Reduced actions per agent")
    ax.set_ylabel("Mean survival (%)")
    ax.set_title(title)
axes[0].set_ylim(0, 70)
axes[1].set_ylim(0, 28)
handles = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor=family_colors[family],
        edgecolor="white",
        label=family,
    )
    for family in family_order
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)
fig.suptitle(
    "Matched WCCI controls trained from scratch at $k=64$",
    y=1.02,
)
fig.tight_layout(rect=(0, 0.10, 1, 1))
save(fig, "wcci_target_controls.png")


# ---------------------------------------------------------------------------
# Figure 9.1: gated and ungated transfer at each action-space size.
# ---------------------------------------------------------------------------
plain_all = arm_frame("plain")
plain_all = plain_all.loc[
    plain_all["mk"].isin(MAIN_CAPS)
    & (
        (plain_all["gate"] == "ungated")
        | ((plain_all["gate"] == "gated") & np.isclose(plain_all["rho"], 0.95))
    )
].copy()

fig = plt.figure(figsize=(13.8, 8.7))
grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.48, wspace=0.25)
ax_overall = fig.add_subplot(grid[0, :])
ax_hard = fig.add_subplot(grid[1, 0])
ax_relative = fig.add_subplot(grid[1, 1])

grouped_gate_boxes(
    ax_overall,
    plain_all,
    "overall_pct",
    "Mean survival over all 50 chronics (%)",
    "Absolute performance on the full test",
    reference_lines=[(DN_OVERALL, "Do nothing", "--", "#222222")],
    show_counts=False,
)
greedy = data.greedy_ceiling.set_index("mk").reindex(MAIN_CAPS)
ax_overall.plot(
    range(len(MAIN_CAPS)),
    greedy["greedy_overall_pct"],
    color="#59A14F",
    marker="*",
    markersize=10,
    linewidth=1.8,
    zorder=4,
)
for position, value in enumerate(greedy["greedy_overall_pct"]):
    ax_overall.annotate(
        f"{value:.0f}%",
        (position, value),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color="#3C7D36",
    )
ax_overall.set_ylim(0, 103)

grouped_gate_boxes(
    ax_hard,
    plain_all,
    "hard_pct",
    "Difficult-cohort survival (%)",
    "Absolute performance on the 24 difficult chronics",
    reference_lines=[(DN_HARD, "Do nothing", "--", "#222222")],
    show_counts=False,
)
ax_hard.set_ylim(0, 27)

grouped_gate_boxes(
    ax_relative,
    plain_all,
    "capture_hard_pct",
    "Do-nothing-to-greedy progress (%)",
    "Fraction of feasible difficult-cohort gain recovered",
    reference_lines=[
        (0, "Do nothing", "--", "#222222"),
        (100, "Greedy", ":", "#59A14F"),
    ],
    show_counts=False,
)
ax_relative.set_ylim(-42, 108)

gate_handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", color=GATE_COLORS[gate], label=GATE_LABELS[gate])
    for gate in ("ungated", "gated")
]
reference_handles = [
    plt.Line2D([0], [0], color="#222222", linestyle="--", label="Do nothing"),
    plt.Line2D([0], [0], color="#59A14F", linestyle=":", label="Greedy at the same $k$"),
]
fig.legend(
    handles=gate_handles + reference_handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.005),
)
fig.suptitle("Absolute zero-shot performance and the fraction of feasible gain", y=0.985)
fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.12)
save(fig, "wcci_zero_shot_determinants.png")


# ---------------------------------------------------------------------------
# Figure 9.2: architecture effects, reported separately by gate condition.
# ---------------------------------------------------------------------------

factor_specs = [
    ("scaling", ("raw inputs", "physical scaling"), "Physical scaling"),
    ("pooling", ("mean pool", "typed-mean pool"), "Typed-mean pooling"),
    ("descriptor", ("descriptor off", "descriptor on"), "Action descriptor"),
    ("idle", ("shared idle scorer", "dedicated idle head"), "Dedicated idle head"),
]
factor_rows = []
for gate in ("ungated", "gated"):
    scoped = plain_all.loc[plain_all["gate"] == gate]
    for factor, levels, label in factor_specs:
        within = [column for column in ["scaling", "pooling", "descriptor", "idle", "mk"] if column != factor]
        pairs = W.matched_pairs(scoped, factor, levels, within, value="hard_pct")
        mean, low, high = bootstrap_mean(pairs["delta"])
        factor_rows.append(
            {"gate": gate, "label": label, "mean": mean, "low": low, "high": high, "n": len(pairs)}
        )
factor_effects = pd.DataFrame(factor_rows)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharex=True)
for ax, gate in zip(axes, ("ungated", "gated")):
    frame = factor_effects.loc[factor_effects["gate"] == gate].reset_index(drop=True)
    y = np.arange(len(frame))
    ax.errorbar(
        frame["mean"],
        y,
        xerr=[frame["mean"] - frame["low"], frame["high"] - frame["mean"]],
        fmt="o",
        color=GATE_COLORS[gate],
        ecolor=GATE_COLORS[gate],
        capsize=4,
        linewidth=1.8,
    )
    ax.axvline(0, color="#222222", linewidth=1.1)
    for i, row in frame.iterrows():
        ax.text(row["high"] + 0.22, i, f"{row['mean']:+.1f} pp  ($n={int(row['n'])}$)", va="center", fontsize=8)
    ax.set_yticks(y, frame["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Matched change in difficult survival (pp)")
    ax.set_title(GATE_LABELS[gate])
axes[0].set_xlim(-5.5, 13.0)
fig.suptitle("Architecture effects depend on the gate condition", y=1.02)
fig.tight_layout()
save(fig, "wcci_zero_shot_architecture_effects.png")


# ---------------------------------------------------------------------------
# Figure 9.3: absolute performance of both policy changes in one view.
# ---------------------------------------------------------------------------
policy_arms = {"Gmax-delta": arm_frame("gmax"), "Adaptive budget": arm_frame("AIB")}
policy_variants = [
    ("Gmax-delta", "ungated"),
    ("Gmax-delta", "gated"),
    ("Adaptive budget", "ungated"),
    ("Adaptive budget", "gated"),
]
policy_offsets = np.linspace(-0.30, 0.30, len(policy_variants))


def grouped_policy_boxes(ax, metric, ylabel, title, reference_lines=()):
    """Draw two policy changes and two gate settings at every action cap."""
    rng = np.random.default_rng(20260828)
    for cap_index, cap in enumerate(MAIN_CAPS):
        for offset, (architecture, gate) in zip(policy_offsets, policy_variants):
            frame = gate_scope(policy_arms[architecture], gate)
            values = frame.loc[frame["mk"] == cap, metric].dropna().to_numpy()
            if not len(values):
                continue
            position = cap_index + offset
            colour = POLICY_VARIANT_COLORS[(architecture, gate)]
            ax.boxplot(
                values,
                positions=[position],
                widths=0.16,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={
                    "facecolor": colour,
                    "alpha": 0.22,
                    "edgecolor": colour,
                    "linewidth": 1.15,
                },
                whiskerprops={"color": colour, "linewidth": 1.0},
                capprops={"color": colour, "linewidth": 1.0},
                medianprops={"color": colour, "linewidth": 2.0},
            )
            jitter = rng.uniform(-0.035, 0.035, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=17,
                color=colour,
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
    for value, label, style, colour in reference_lines:
        ax.axhline(value, color=colour, linestyle=style, linewidth=1.25, label=label)
    ax.set_xticks(range(len(MAIN_CAPS)), [f"mk{cap}" for cap in MAIN_CAPS])
    ax.set_xlim(-0.55, len(MAIN_CAPS) - 0.45)
    ax.set_xlabel("Reduced action space")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


fig = plt.figure(figsize=(13.8, 8.7))
grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.48, wspace=0.25)
ax_overall = fig.add_subplot(grid[0, :])
ax_hard = fig.add_subplot(grid[1, 0])
ax_relative = fig.add_subplot(grid[1, 1])

grouped_policy_boxes(
    ax_overall,
    "overall_pct",
    "Mean survival over all 50 chronics (%)",
    "Absolute performance on the full test",
    reference_lines=[(DN_OVERALL, "Do nothing", "--", "#222222")],
)
greedy = data.greedy_ceiling.set_index("mk").reindex(MAIN_CAPS)
ax_overall.plot(
    range(len(MAIN_CAPS)),
    greedy["greedy_overall_pct"],
    color="#59A14F",
    marker="*",
    markersize=10,
    linewidth=1.8,
    zorder=4,
)
for position, value in enumerate(greedy["greedy_overall_pct"]):
    ax_overall.annotate(
        f"{value:.0f}%",
        (position, value),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color="#3C7D36",
    )
ax_overall.set_ylim(0, 103)

grouped_policy_boxes(
    ax_hard,
    "hard_pct",
    "Difficult-cohort survival (%)",
    f"Absolute performance on the {N_HARD} difficult chronics",
    reference_lines=[(DN_HARD, "Do nothing", "--", "#222222")],
)
ax_hard.set_ylim(0, 35)

grouped_policy_boxes(
    ax_relative,
    "capture_hard_pct",
    "Do-nothing-to-greedy progress (%)",
    "Fraction of feasible difficult-cohort gain recovered",
    reference_lines=[
        (0, "Do nothing", "--", "#222222"),
        (100, "Greedy", ":", "#59A14F"),
    ],
)
ax_relative.set_ylim(-25, 108)

handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color=POLICY_VARIANT_COLORS[(architecture, gate)],
        label=f"{architecture} - {GATE_LABELS[gate]}",
    )
    for architecture, gate in policy_variants
]
handles.extend(
    [
        plt.Line2D([0], [0], color="#222222", linestyle="--", label="Do nothing"),
        plt.Line2D([0], [0], color="#59A14F", linestyle=":", label="Greedy at the same $k$"),
    ]
)
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, 0.005),
)
fig.suptitle("Absolute performance of both policy changes", y=0.985)
fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.12)
save(fig, "wcci_policy_change_absolute_performance.png")


# ---------------------------------------------------------------------------
# Figure 10.4: matched modifications and gate dependence.
# ---------------------------------------------------------------------------
mod_rows = []
for treatment, display in (("gmax", "Gmax-delta"), ("AIB", "Adaptive budget")):
    for gate in ("ungated", "gated"):
        deltas = paired_arm_delta(treatment, gate)
        mean, low, high = bootstrap_mean(deltas)
        mod_rows.append(
            {"label": f"{display}, {gate}", "mean": mean, "low": low, "high": high, "n": len(deltas)}
        )
modifications = pd.DataFrame(mod_rows)

gate_rows = []
for arm, display in (("plain", "Plain zero-shot"), ("gmax", "Gmax-delta"), ("AIB", "Adaptive budget")):
    frame = arm_frame(arm).loc[lambda x: x["mk"].isin(MAIN_CAPS)]
    deltas = paired_gate_delta(frame, ["variant", "mk"])
    mean, low, high = bootstrap_mean(deltas)
    gate_rows.append({"label": display, "mean": mean, "low": low, "high": high, "n": len(deltas), "exact": False})

fine_tune = runs.loc[
    (runs["route"] == "MAPPO fine-tune (conservative)") & (runs["selection"] == "last")
]
scratch = runs.loc[
    (runs["route"] == "MAPPO scratch") & (runs["selection"] == "last")
]
for frame, display in ((fine_tune, "Fine-tuned scorer"), (scratch, "Scratch scorer")):
    deltas = paired_gate_delta(frame, ["variant", "mk"])
    mean, low, high = bootstrap_mean(deltas)
    gate_rows.append({"label": display, "mean": mean, "low": low, "high": high, "n": len(deltas), "exact": False})

wmlp_gate = wmlp.runs.loc[wmlp.runs["selection"] == "last"].pivot_table(
    index="mk", columns="gate", values="hard_pct", aggfunc="first"
).dropna(subset=["ungated", "gated"])
for cap, row in wmlp_gate.iterrows():
    delta = float(row["gated"] - row["ungated"])
    gate_rows.append({"label": f"WMLP, mk{int(cap)}", "mean": delta, "low": delta, "high": delta, "n": 1, "exact": True})
gate_effects = pd.DataFrame(gate_rows)

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
for ax, frame, title, xlabel in (
    (
        axes[0],
        modifications,
        "Variant minus plain zero-shot\n(same architecture, k, and gate)",
        r"$\Delta$ difficult survival = variant - plain zero-shot (pp)",
    ),
    (
        axes[1],
        gate_effects,
        r"Local-$\rho=0.95$ minus ungated" + "\n(same policy, architecture, and k)",
        r"$\Delta$ difficult survival = gated - ungated (pp)",
    ),
):
    positions = np.arange(len(frame))
    colours = ["#B279A2" if "Gmax" in label else "#72B7B2" for label in frame["label"]] if ax is axes[0] else ["#E45756" if "WMLP" in label else "#4C78A8" for label in frame["label"]]
    for position, (_, row), colour in zip(positions, frame.iterrows(), colours):
        ax.errorbar(
            row["mean"],
            position,
            xerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
            fmt="o",
            color=colour,
            ecolor=colour,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            capsize=4,
            linewidth=1.8,
            zorder=3,
        )
    ax.axvline(0, color="#222222", linewidth=1.1)
    ax.set_yticks(positions, frame["label"])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, row in frame.iterrows():
        ax.text(row["high"] + 0.25, i, f"{row['mean']:+.1f}", va="center", fontsize=8)
fig.suptitle("Useful changes are policy-dependent", y=1.02)
fig.tight_layout()
save(fig, "wcci_intervention_gate_effects.png")


# ---------------------------------------------------------------------------
# Figure 9.5: matched adaptation routes and the easy/difficult frontier.
# ---------------------------------------------------------------------------
zero = gate_scope(arm_frame("plain"), "ungated")
zero = zero.loc[(zero["family"] == "NLS") & (zero["mk"] == 64)]
scratch64 = scratch.loc[(scratch["family"] == "NLS") & (scratch["mk"] == 64) & (scratch["gate"] == "ungated")]
fine64 = fine_tune.loc[(fine_tune["family"] == "NLS") & (fine_tune["mk"] == 64) & (fine_tune["gate"] == "ungated")]
ladder = (
    zero[["variant", "hard_pct"]].rename(columns={"hard_pct": "Zero-shot"})
    .merge(scratch64[["variant", "hard_pct"]].rename(columns={"hard_pct": "Scratch"}), on="variant")
    .merge(fine64[["variant", "hard_pct"]].rename(columns={"hard_pct": "Fine-tuned"}), on="variant")
)

fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
route_order = ["Zero-shot", "Scratch", "Fine-tuned"]
for _, row in ladder.iterrows():
    values = row[route_order].to_numpy(dtype=float)
    axes[0].plot(range(3), values, color="#B8B8B8", linewidth=1.1, alpha=0.9, zorder=1)
    axes[0].scatter(range(3), values, color=["#4C78A8", "#54A24B", "#E4A11B"], s=36, zorder=2)
medians = ladder[route_order].median()
axes[0].plot(range(3), medians, color="#222222", linewidth=3.0, marker="o", markersize=6, zorder=4)
axes[0].axhline(DN_HARD, color="#222222", linestyle="--", linewidth=1.2)
axes[0].set_xticks(range(3), route_order)
axes[0].set(ylabel="Difficult-cohort survival (%)", title="Same eight architectures at mk64, no heuristic")

frontier_parts = []
frontier_parts.append(wmlp_final.assign(route="WMLP", label=lambda x: "WMLP mk" + x["mk"].astype(int).astype(str)))
frontier_parts.append(wsc_final.assign(route="Fixed-list GNN", label=lambda x: x["schema_label"] + " mk" + x["mk"].astype(int).astype(str)))
frontier_parts.append(candidate_scratch.assign(route="Candidate scorer, scratch", label=lambda x: x["variant"]))
frontier_parts.append(fine_tune.loc[fine_tune["gate"] == "ungated"].assign(route="Fine-tuned", label=lambda x: x["variant"] + " mk" + x["mk"].astype(int).astype(str)))
frontier_parts.append(zero.assign(route="Zero-shot", label=lambda x: x["variant"]))
frontier = pd.concat(frontier_parts, ignore_index=True)

for route, frame in frontier.groupby("route", observed=True):
    axes[1].scatter(
        frame["hard_pct"],
        frame["easy_kept"],
        s=44,
        label=route,
        color=ROUTE_COLORS[route],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.5,
    )
axes[1].scatter([DN_HARD], [N_EASY], marker="X", s=105, color="#222222", label="Do nothing", zorder=5)
greedy64 = data.greedy_ceiling.loc[data.greedy_ceiling["mk"] == 64].iloc[0]
axes[1].scatter([greedy64["greedy_hard_pct"]], [greedy64["greedy_easy_kept"]], marker="*", s=155, color="#59A14F", label="Greedy mk64", zorder=5)
axes[1].set(xlabel="Difficult-cohort survival (%)", ylabel=f"Easy chronics kept (out of {N_EASY})", title="Rescuing difficult cases without losing easy ones")
axes[1].set_ylim(-0.7, N_EASY + 1.4)
axes[1].legend(loc="lower right", frameon=True, ncol=2)
fig.suptitle("Transfer becomes reliable after target-grid fine-tuning", y=1.02)
fig.tight_layout()
save(fig, "wcci_adaptation_summary.png")


# ---------------------------------------------------------------------------
# Appendix tables: exact WMLP and WSC checkpoint results.
# ---------------------------------------------------------------------------
wmlp_table = wmlp.runs.copy()
wmlp_table["selection_label"] = wmlp_table["selection"].map({"last": "final", "best test": "best test"}).fillna(wmlp_table["selection"])
wmlp_table["gate_label"] = np.where(wmlp_table["gate"] == "gated", "local $\\rho=0.95$", "none")
wmlp_table = wmlp_table.sort_values(["mk", "selection_label", "gate"])
wmlp_lines = [
    f"    {int(row.mk)} & {row.selection_label} & {row.gate_label} & {row.overall_pct:.2f} & {row.hard_pct:.2f} & {int(row.hard_rescues)} & {int(row.easy_kept)} \\\\"
    for row in wmlp_table.itertuples()
]
(table_dir / "wcci_wmlp_results.tex").write_text(
    "\n".join(
        [
            "% Generated by build_wcci_report_assets.py -- do not edit by hand.",
            r"\begin{tabular}{rllrrrr}",
            r"  \toprule",
            r"  $k$ & Checkpoint & Heuristic & Overall (\%) & Difficult (\%) & Rescues & Easy kept \\",
            r"  \midrule",
            *wmlp_lines,
            r"  \bottomrule",
            r"\end{tabular}",
            "",
        ]
    ),
    encoding="utf-8",
)

wsc_table = wsc.runs.sort_values(["schema", "mk", "checkpoint_kind"])
wsc_lines = [
    f"    \\texttt{{{tex_escape(row.schema)}}} & {int(row.mk)} & {row.checkpoint_kind} & {row.overall_pct:.2f} & {row.hard_pct:.2f} & {int(row.hard_rescues)} & {int(row.easy_kept)} \\\\"
    for row in wsc_table.itertuples()
]
(table_dir / "wcci_wsc_results.tex").write_text(
    "\n".join(
        [
            "% Generated by build_wcci_report_assets.py -- do not edit by hand.",
            r"\begin{tabular}{lrlrrrr}",
            r"  \toprule",
            r"  Graph & $k$ & Checkpoint & Overall (\%) & Difficult (\%) & Rescues & Easy kept \\",
            r"  \midrule",
            *wsc_lines,
            r"  \bottomrule",
            r"\end{tabular}",
            "",
        ]
    ),
    encoding="utf-8",
)


# Detailed notebook figures are copied under stable report-facing names.
detail_sources = {
    "wcci_appendix_coverage.png": data.task_dir / "outputs/analysis/wcci_full_matrix_report/01_coverage_matrix.png",
    "wcci_appendix_zero_shot_matrix.png": data.task_dir / "outputs/analysis/wcci_full_matrix_report/02_core_zero_shot_hard.png",
    "wcci_appendix_aib_matrix.png": data.task_dir / "outputs/analysis/wcci_full_matrix_report/04_aib_matrix_hard.png",
    "wcci_appendix_gmax_matrix.png": data.task_dir / "outputs/analysis/wcci_full_matrix_report/06_gmax_matrix_hard.png",
    "wcci_appendix_target_board.png": data.task_dir / "outputs/analysis/wcci_full_matrix_report/08_target_trained_board.png",
    "wcci_appendix_chronics.png": data.task_dir / "outputs/analysis/wcci_full_matrix_report/11_showcase_hard_chronic_vs_greedy.png",
}
for destination, source in detail_sources.items():
    if source.is_file():
        shutil.copy2(source, figure_dir / destination)

print(f"Wrote 5 main figures, 2 appendix tables and {len(detail_sources)} appendix figures.")
print(f"WMLP rows: {len(wmlp.runs)} accepted, {len(wmlp.rejected)} rejected")
print(f"WSC rows: {len(wsc.runs)} accepted, {len(wsc.rejected)} rejected")
