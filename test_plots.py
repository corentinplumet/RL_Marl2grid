import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival")))
import wcci_transfer_data as W
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

data = W.load()
runs = data.runs
DN_HARD = data.dn_hard
POOL_COLORS = {"typed-mean pool": "#4C78A8", "mean pool": "#F58518"}

# GATED DATA
gated_runs = runs[(runs["route"] == "bus14 zero-shot") & (runs["gate"] == "gated")].copy()

# Sort variants by mean
GATED_HARD = gated_runs.pivot_table(index="variant", columns=["mk", "rho"], values="hard_pct")
order = GATED_HARD.mean(axis=1).sort_values(ascending=False).index.tolist()
GATED_HARD = GATED_HARD.loc[order]
# Flatten columns for heatmap
GATED_HARD.columns = [f"mk{mk}\n$\\rho$={rho}" for mk, rho in GATED_HARD.columns]

fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(2, 3, width_ratios=[0.24, 1, 0.34], height_ratios=[0.20, 1],
                      wspace=0.04, hspace=0.05)
ax_strip = fig.add_subplot(gs[1, 0])
ax_main = fig.add_subplot(gs[1, 1])
ax_right = fig.add_subplot(gs[1, 2])
ax_top = fig.add_subplot(gs[0, 1])

sns.heatmap(GATED_HARD, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.7, linecolor="white",
            cbar=False, ax=ax_main, vmin=0, vmax=GATED_HARD.to_numpy().max())
ax_main.set(xlabel="Action Cap and Local-$\\rho$ Threshold", ylabel="")
ax_main.set_yticklabels([])

SCALE_COLORS = {"NLS": "#B279A2", "NL": "#79706E"}
for y, variant in enumerate(order):
    family = "NLS" if variant.startswith("NLS") else "NL"
    ax_strip.add_patch(plt.Rectangle((0, y), 1, 1, color=SCALE_COLORS[family]))
    ax_strip.text(0.5, y + 0.5, variant, ha="center", va="center",
                  fontsize=7.5, color="white")
ax_strip.set(xlim=(0, 1), ylim=(len(order), 0), xticks=[], yticks=[])
ax_strip.grid(False)
for spine in ax_strip.spines.values():
    spine.set_visible(False)

ax_top.bar(np.arange(len(GATED_HARD.columns)) + 0.5, GATED_HARD.mean(), width=0.75, color="#4C78A8",
           edgecolor="white", linewidth=1.0)
for x, value in enumerate(GATED_HARD.mean()):
    ax_top.annotate(f"{value:.1f}", (x + 0.5, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
ax_top.axhline(DN_HARD, color="black", linestyle="--", linewidth=1.2)
ax_top.set(xlim=(0, len(GATED_HARD.columns)), xticks=[], ylabel="mean (%)")
ax_top.set_ylim(0, max(GATED_HARD.mean().max(), DN_HARD) * 1.35)
ax_top.set_title("Gated Screen: Survival by Architecture, Cap, and $\\rho$", fontsize=10, pad=8)

row_means = GATED_HARD.mean(axis=1)
pool_of = gated_runs.drop_duplicates("variant").set_index("variant")["pooling"]
ax_right.barh(np.arange(len(order)) + 0.5, row_means,
              color=[POOL_COLORS[pool_of[v]] for v in order],
              edgecolor="white", linewidth=1.0, height=0.75)
ax_right.axvline(DN_HARD, color="black", linestyle="--", linewidth=1.2)
ax_right.set(ylim=(len(order), 0), yticks=[], xlabel="mean by architecture (%)")
ax_right.set_xlim(0, max(row_means.max(), DN_HARD) * 1.2)

handles = (
    [Patch(color=c, label=f"row strip — {W.SCALING_LABELS[k]}") for k, c in SCALE_COLORS.items()]
    + [Patch(color=c, label=f"right bars — {k}") for k, c in POOL_COLORS.items()]
    + [plt.Line2D([], [], color="black", linestyle="--", label=f"do nothing ({DN_HARD:.1f}%)")]
)
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.05), frameon=False)
fig.savefig("test_gated_heatmap.png", bbox_inches="tight")


# COMPARISON PLOT
ungated = runs[(runs["route"] == "bus14 zero-shot") & (runs["gate"] == "ungated") & (runs["mk"].isin([64, 128, 256])) & (runs["variant"].isin(order))].copy()
ungated_pivot = ungated.pivot_table(index=["variant", "mk"], values="hard_pct").rename(columns={"hard_pct": "Ungated"})

gated_max = gated_runs.pivot_table(index=["variant", "mk"], values="hard_pct", aggfunc="max").rename(columns={"hard_pct": "Best Gated"})
gated_mean = gated_runs.pivot_table(index=["variant", "mk"], values="hard_pct", aggfunc="mean").rename(columns={"hard_pct": "Mean Gated"})

compare = ungated_pivot.join(gated_max).join(gated_mean).reset_index()

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=compare, x="Ungated", y="Best Gated", hue="variant", style="mk", s=100, ax=ax2)
max_val = max(compare["Ungated"].max(), compare["Best Gated"].max()) * 1.1
ax2.plot([0, max_val], [0, max_val], 'k--', zorder=0, alpha=0.5, label="y=x (No improvement)")
ax2.set_title("Ungated vs Best Gated Survival (per Variant and Cap)")
ax2.set_xlabel("Ungated Survival on Hard Chronics (%)")
ax2.set_ylabel("Best Gated Survival on Hard Chronics (%)")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig2.savefig("test_gated_compare.png", bbox_inches="tight")
