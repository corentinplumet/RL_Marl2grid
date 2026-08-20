import nbformat as nbf
from pathlib import Path

notebook_path = Path("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival/transfer_story.ipynb")
nb = nbf.read(notebook_path, as_version=4)

markdown_cell_1 = """
## 5. The Gated Screen: Why only a subset?

The ungated zero-shot sweep in the previous sections revealed that half of the architectures collapsed completely, failing to beat even a 0.5% survival rate on the difficult chronics. Because evaluating the local-$\\rho$ gate multiplies the required runs by four (for $\\rho \\in \\{0.85, 0.90, 0.95, 1.0\\}$), it was inefficient to waste compute on architectures that had already proven incapable of basic transfer. We therefore restricted the gated screen to the **8 salvageable architectures** and focused only on the most promising action caps (`mk64`, `mk128`, and `mk256`).

The plot below summarizes the performance of these 8 architectures across all action caps and $\\rho$ thresholds.
"""

code_cell_1 = """
# Filter for Gated Runs
gated_runs = runs[(runs["route"] == "bus14 zero-shot") & (runs["gate"] == "gated")].copy()

# Pivot for heatmap: Rows=Variant, Columns=(mk, rho)
GATED_HARD = gated_runs.pivot_table(index="variant", columns=["mk", "rho"], values="hard_pct")
# Sort variants by their overall mean survival across all gated evaluations
order = GATED_HARD.mean(axis=1).sort_values(ascending=False).index.tolist()
GATED_HARD = GATED_HARD.loc[order]

# Format columns for readability
GATED_HARD.columns = [f"mk{mk}\\n$\\rho$={rho}" for mk, rho in GATED_HARD.columns]

fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(2, 3, width_ratios=[0.24, 1, 0.34], height_ratios=[0.20, 1], wspace=0.04, hspace=0.05)
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
    ax_strip.text(0.5, y + 0.5, variant, ha="center", va="center", fontsize=7.5, color="white")
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
save_figure(fig, "05_gated_summary")
plt.show()
"""

markdown_cell_2 = """
### Does the Gate Actually Help? (Ungated vs. Gated Comparison)

To see whether evaluating actions through the local-$\\rho$ gate improves zero-shot transfer, we compare each architecture's performance *without* the gate against its *best* performance *with* the gate (taking the max across the four $\\rho$ thresholds).

The scatter plot below shows this comparison. Points above the dashed line ($y=x$) indicate that the gate improved performance for that specific architecture and action cap.
"""

code_cell_2 = """
ungated = runs[(runs["route"] == "bus14 zero-shot") & (runs["gate"] == "ungated") & (runs["mk"].isin([64, 128, 256])) & (runs["variant"].isin(order))].copy()
ungated_pivot = ungated.pivot_table(index=["variant", "mk"], values="hard_pct").rename(columns={"hard_pct": "Ungated"})

gated_max = gated_runs.pivot_table(index=["variant", "mk"], values="hard_pct", aggfunc="max").rename(columns={"hard_pct": "Best Gated"})
compare = ungated_pivot.join(gated_max).reset_index()

fig2, ax2 = plt.subplots(figsize=(8, 8))
sns.scatterplot(data=compare, x="Ungated", y="Best Gated", hue="variant", style="mk", s=150, ax=ax2, palette="tab10")

max_val = max(compare["Ungated"].max(), compare["Best Gated"].max()) * 1.05
ax2.plot([0, max_val], [0, max_val], 'k--', zorder=0, alpha=0.5, label="y=x (No improvement)")
ax2.fill_between([0, max_val], [0, max_val], max_val, color="green", alpha=0.05, label="Gate Improves Performance")
ax2.fill_between([0, max_val], 0, [0, max_val], color="red", alpha=0.05, label="Gate Harms Performance")

ax2.set_title("Ungated vs Best Gated Survival (per Variant and Cap)")
ax2.set_xlabel("Ungated Survival on Hard Chronics (%)")
ax2.set_ylabel("Best Gated Survival on Hard Chronics (%)")
ax2.set_xlim(0, max_val)
ax2.set_ylim(0, max_val)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Variant & Cap")

save_figure(fig2, "06_ungated_vs_gated")
plt.show()
"""

nb.cells.extend([
    nbf.v4.new_markdown_cell(markdown_cell_1),
    nbf.v4.new_code_cell(code_cell_1),
    nbf.v4.new_markdown_cell(markdown_cell_2),
    nbf.v4.new_code_cell(code_cell_2)
])
nbf.write(nb, notebook_path)
print("Cells appended successfully.")
