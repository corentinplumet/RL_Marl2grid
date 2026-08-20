import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival")))
import wcci_transfer_data as W
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = W.load()
runs = data.runs

# 1. Coverage Grid
ft = runs[runs["route"].str.contains("fine-tune") | runs["route"].str.contains("BC") | runs["route"].str.contains("scratch")].copy()
ft["config"] = ft["route"].astype(str) + "\n(" + ft["gate"].astype(str) + ", heuristic: " + ft["heuristic"].astype(str) + ")"

grid = ft.pivot_table(index="config", columns="mk", values="key", aggfunc="size", fill_value=0)
CAPS = sorted([c for c in grid.columns if c in [64, 128, 256, 512, 1024]])

fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.heatmap(grid[CAPS], annot=True, fmt="d", cmap="Blues", linewidths=0.6, linecolor="white", cbar=False, ax=ax1)
ax1.set(xlabel="Action Cap (mk)", ylabel="", title=f"Fine-Tuning Runs Coverage")
ax1.set_xticklabels([f"mk{c}" for c in CAPS])
fig1.tight_layout()
fig1.savefig("test_ft_grid.png")

# 2. Detailed analysis of heuristic vs no heuristic
ft_valid = ft.dropna(subset=["overall_pct"]).copy()

# Filter only conservative to do a paired comparison
conservative = ft_valid[ft_valid["route"] == "MAPPO fine-tune (conservative)"]
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(data=conservative, x="variant", y="overall_pct", hue="config", ax=ax2, palette="muted")
ax2.set_title("Impact of Heuristic Gate on Conservative Fine-Tuning")
ax2.set_xlabel("Architecture")
ax2.set_ylabel("Overall Survival (%)")
ax2.set_ylim(0, 100)
ax2.legend(title="Configuration")
fig2.tight_layout()
fig2.savefig("test_heuristic_impact.png")
