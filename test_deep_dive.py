import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival")))
import wcci_transfer_data as W
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = W.load()
runs = data.runs.dropna(subset=["overall_pct"])

cons_variants = runs[runs["route"] == "MAPPO fine-tune (conservative)"]["variant"].unique()

def get_best_perf(route, gate, mk=64):
    subset = runs[(runs["route"] == route) & (runs["gate"] == gate) & (runs["mk"] == mk) & (runs["variant"].isin(cons_variants))]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.groupby("variant")["overall_pct"].max()

df2x2 = pd.DataFrame({
    "Zero-Shot Ungated": get_best_perf("bus14 zero-shot", "ungated"),
    "Zero-Shot Gated": get_best_perf("bus14 zero-shot", "gated"),
    "MAPPO Ungated": get_best_perf("MAPPO fine-tune (conservative)", "ungated"),
    "MAPPO Gated": get_best_perf("MAPPO fine-tune (conservative)", "gated")
})
print("Full df2x2:")
print(df2x2)

df2x2 = df2x2.dropna()
print("Cleaned df2x2:")
print(df2x2)

mappo_bc = runs[(runs["route"].isin(["MAPPO fine-tune (conservative)", "BC on greedy labels"])) & 
                (runs["variant"] == "NLS_mean_f1_a0h0") & (runs["mk"] == 64)]
mappo_bc_pivot = mappo_bc.pivot_table(index="route", columns="gate", values="overall_pct", aggfunc="max")
print(mappo_bc_pivot)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

df2x2.index = df2x2.index.str.replace("NLS_", "")
df2x2_melted = df2x2.reset_index().melt(id_vars="variant", var_name="Condition", value_name="Overall Survival (%)")

sns.lineplot(data=df2x2_melted, x="Condition", y="Overall Survival (%)", hue="variant", marker="o", ax=axes[0], palette="tab10", linewidth=2.5, markersize=10)
axes[0].set_title("The 2x2 Transfer Matrix: Training vs. Gating (Conservative MAPPO, mk64)")
axes[0].set_xlabel("")
axes[0].axhline(data.dn_overall, color="black", linestyle="--", linewidth=1.2, label=f"do nothing ({data.dn_overall:.1f}%)")
axes[0].legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')

mappo_bc_pivot.T.plot(kind="bar", ax=axes[1], color=["#F58518", "#4C78A8"], edgecolor="white")
axes[1].set_title("MAPPO vs. Behavioural Cloning\n(Architecture: mean_f1_a0h0, mk64)")
axes[1].set_ylabel("Overall Survival (%)")
axes[1].set_xlabel("Evaluation Gate")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].axhline(data.dn_overall, color="black", linestyle="--", linewidth=1.2)
axes[1].legend(title="Algorithm", loc="lower right")

fig.tight_layout()
fig.savefig("test_deep_dive.png")
print("Done!")
