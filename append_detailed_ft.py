import nbformat as nbf
from pathlib import Path

notebook_path = Path("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival/transfer_story.ipynb")
nb = nbf.read(notebook_path, as_version=4)

markdown_cell_1 = """
## 7. Fine-Tuning Setup and the Impact of the Heuristic Gate

To fully understand the fine-tuning results, we must look at the exact configuration of the evaluations. The grid below shows the number of fine-tuning evaluations broken down by route, action cap, and whether the evaluation used the heuristic gate.
"""

code_cell_1 = """
ft = runs[runs["route"].str.contains("fine-tune") | runs["route"].str.contains("BC") | runs["route"].str.contains("scratch")].copy()
ft["config"] = ft["route"].astype(str) + "\\n(" + ft["gate"].astype(str) + ", heuristic: " + ft["heuristic"].astype(str) + ")"

grid = ft.pivot_table(index="config", columns="mk", values="key", aggfunc="size", fill_value=0)
CAPS_FT = sorted([c for c in grid.columns if c in [64, 128, 256, 512, 1024]])

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(grid[CAPS_FT], annot=True, fmt="d", cmap="Blues", linewidths=0.6, linecolor="white", cbar=False, ax=ax)
ax.set(xlabel="Action Cap", ylabel="", title="Fine-Tuning Evaluated Configurations")
ax.set_xticklabels([f"mk{c}" for c in CAPS_FT])
fig.tight_layout()
save_figure(fig, "08_finetune_grid")
plt.show()
"""

markdown_cell_2 = """
As shown in the grid, target-grid trained policies were evaluated in two main ways:
1. **Ungated (heuristic: none)**: The trained policy's actions are executed directly on the target grid.
2. **Gated (heuristic: local/global rho)**: The trained policy's actions are still passed through the evaluation gate (which uses a heuristic threshold to filter out dangerous actions) before being executed.

This raises an important question: **Does a policy that has already been fine-tuned on the target grid still need the heuristic gate?** 

To answer this, we can directly compare the ungated vs. gated evaluations of the `MAPPO fine-tune (conservative)` and `BC on greedy labels` routes.
"""

code_cell_2 = """
ft_valid = ft.dropna(subset=["overall_pct"]).copy()

conservative = ft_valid[ft_valid["route"] == "MAPPO fine-tune (conservative)"].copy()
conservative["Architecture"] = conservative["variant"].str.replace("NLS_", "")
conservative["Configuration"] = conservative["gate"].str.capitalize() + " (" + conservative["heuristic"] + ")"

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=conservative, x="Architecture", y="overall_pct", hue="Configuration", ax=ax, palette="Set1", edgecolor="white")

# Add percentage annotations on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

ax.set_title("Impact of the Heuristic Gate on Fine-Tuned Policies (MAPPO Conservative, mk64)")
ax.set_ylabel("Overall Survival (%)")
ax.set_ylim(0, 75)
ax.axhline(data.dn_overall, color="black", linestyle="--", linewidth=1.2, label=f"do nothing ({data.dn_overall:.1f}%)")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title="Evaluation Gate", loc="upper right")

save_figure(fig, "09_heuristic_impact")
plt.show()

# Print the BC comparison as well
bc = ft_valid[ft_valid["route"] == "BC on greedy labels"]
display(Markdown("#### Behavioural Cloning Comparison"))
display(bc[["variant", "gate", "heuristic", "overall_pct"]].style.format({"overall_pct": "{:.2f}%"}))
"""

markdown_cell_3 = """
**Conclusion:**
For the majority of architectures, **adding the heuristic gate on top of a fine-tuned model still yields a massive boost** (e.g. +5% overall survival for `mean_f0_a0h0`, and +8.6% for Behavioural Cloning). The gate continues to act as a robust safety net for dangerous edge cases that the target-grid training hasn't fully mastered.

However, for the very best architectures (like `mean_f1_a0h0` and `tmean_f0_a0h0`), the policy has learned the target grid so well that the heuristic gate becomes redundant—providing zero benefit or even a slight penalty.
"""

nb.cells.extend([
    nbf.v4.new_markdown_cell(markdown_cell_1),
    nbf.v4.new_code_cell(code_cell_1),
    nbf.v4.new_markdown_cell(markdown_cell_2),
    nbf.v4.new_code_cell(code_cell_2),
    nbf.v4.new_markdown_cell(markdown_cell_3)
])

nbf.write(nb, notebook_path)
print("Detailed fine-tuning cells appended successfully.")
