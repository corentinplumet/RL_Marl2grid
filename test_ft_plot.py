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

# Identify cells (variant, mk, gate) that have BOTH a zero-shot run and at least one other route
zero_shot_runs = runs[runs["route"] == "bus14 zero-shot"].copy()
other_runs = runs[runs["route"] != "bus14 zero-shot"].copy()

# For a clean plot, let's just get the "conservative" fine-tuning, "BC", and "scratch" routes.
ft_routes = [
    "MAPPO fine-tune (conservative)",
    "MAPPO fine-tune (aggressive)",
    "BC on greedy labels",
    "MAPPO scratch"
]

rows = []
for (variant, mk, gate), group in runs.groupby(["variant", "mk", "gate"]):
    routes_in_cell = set(group["route"])
    if "bus14 zero-shot" not in routes_in_cell:
        continue
    zero_val = group.loc[group["route"] == "bus14 zero-shot", "overall_pct"].max() # use max if multiple
    
    for route in ft_routes:
        if route in routes_in_cell:
            ft_val = group.loc[group["route"] == route, "overall_pct"].max()
            rows.append({
                "variant": variant,
                "mk": mk,
                "gate": gate,
                "route": route,
                "Zero-Shot": zero_val,
                "Trained": ft_val,
                "Gain": ft_val - zero_val
            })

df = pd.DataFrame(rows)
df["label"] = df["variant"] + " (mk" + df["mk"].astype(str) + ", " + df["gate"] + ")"

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=df.melt(id_vars=["label", "route"], value_vars=["Zero-Shot", "Trained"], var_name="Stage", value_name="Overall Survival (%)"), 
            x="label", y="Overall Survival (%)", hue="Stage", ax=ax, palette="Set2")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_title("Target-Grid Training vs Zero-Shot Transfer (Overall Survival %)")
fig.tight_layout()
fig.savefig("test_ft_plot.png")
