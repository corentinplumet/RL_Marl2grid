"""Add the exact-checkpoint bus14 versus WCCI plot to transfer_story.ipynb.

This update is intentionally separate from build_transfer_story.py because the
notebook contains several hand-added analysis sections.  It is idempotent: the
two tagged cells are replaced in place when the script is run again.
"""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "transfer_story.ipynb"
SECTION_ID = "matched_shared_bus14_wcci_section"
CODE_ID = "matched_shared_bus14_wcci_code"


MARKDOWN = r"""
### 5.1 Matched transferable checkpoints: does bus14 performance predict WCCI transfer?

The exact Screen F checkpoints in Section 6.7 have not been evaluated on WCCI,
so they cannot support a source-to-target plot yet. A valid comparison is
nevertheless available for the later transferable shared-checkpoint campaign.
For each of its eight architecture variants, the plot pairs the **same saved
checkpoint** evaluated on 201 bus14 test chronics and transferred without
target training to the WCCI test set.

The comparison is split by input condition because the raw-input and
physically scaled actors are separate training runs. WCCI is fixed to the
ungated `mk32` evaluation, and the target metric is survival on the difficult
chronics only. Fixing the action space and excluding the gate prevents those
choices from being mixed with the architecture comparison. The numbers inside
the markers identify the eight candidate-scorer variants using the key below
the plot.
"""


CODE = r'''
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

# Keep only the original shared candidate-scorer campaign.  The later
# Gmax-delta campaign deliberately reuses the same short variant labels and
# must not be averaged into these pairs.
base_wcci32 = runs.loc[
    (runs["route"] == "bus14 zero-shot")
    & (runs["gate"] == "ungated")
    & (runs["mk"] == 32)
    & runs["group"].str.match(ORIGINAL_ZERO_SHOT_GROUP, na=False)
].copy()

source_target = data.bus14.merge(
    base_wcci32[
        ["variant", "checkpoint_stem", "scaling", "pooling", "descriptor", "idle",
         "hard_pct", "overall_pct", "episode_csv"]
    ],
    left_on="bus14_checkpoint_stem",
    right_on="checkpoint_stem",
    how="inner",
    suffixes=("_bus14", "_wcci"),
    validate="one_to_one",
)
assert len(source_target) == 16, f"expected 16 exact checkpoint pairs, found {len(source_target)}"
assert source_target["variant_bus14"].equals(source_target["variant_wcci"])
assert source_target.groupby("scaling")["variant_wcci"].nunique().eq(8).all()
assert source_target["bus14_episodes"].eq(201).all()

design_order = [
    "mean_f0_a0h0", "mean_f0_a0h1", "mean_f1_a0h0", "mean_f1_a0h1",
    "tmean_f0_a0h0", "tmean_f0_a0h1", "tmean_f1_a0h0", "tmean_f1_a0h1",
]
design_number = {design: i + 1 for i, design in enumerate(design_order)}
source_target["design"] = source_target["variant_wcci"].str.replace(
    r"^(?:NL|NLS)_", "", regex=True
)
source_target["design_id"] = source_target["design"].map(design_number)

pool_colors = {"mean pool": "#4C78A8", "typed-mean pool": "#F58518"}
family_order = [("raw inputs", "Raw inputs"), ("physical scaling", "Physically scaled inputs")]

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), sharex=True, sharey=True)
correlation_rows = []
for ax, (scaling, title) in zip(axes, family_order):
    panel = source_target.loc[source_target["scaling"] == scaling].copy()
    rho, p_value = spearmanr(panel["bus14_pct"], panel["hard_pct"])
    pearson = panel["bus14_pct"].corr(panel["hard_pct"])
    correlation_rows.append({
        "input_condition": title,
        "n_variants": len(panel),
        "spearman_rho": rho,
        "spearman_p": p_value,
        "pearson_r": pearson,
    })

    # A dashed least-squares line is shown only as a descriptive guide.  With
    # eight single-seed variants it is not an estimate of a population effect.
    x_line = np.linspace(panel["bus14_pct"].min(), panel["bus14_pct"].max(), 100)
    slope, intercept = np.polyfit(panel["bus14_pct"], panel["hard_pct"], 1)
    ax.plot(x_line, slope * x_line + intercept, color="#555555", linestyle="--",
            linewidth=1.5, zorder=1)

    for row in panel.itertuples():
        color = pool_colors[row.pooling]
        ax.scatter(row.bus14_pct, row.hard_pct, s=230, color=color,
                   edgecolor="white", linewidth=1.4, zorder=3)
        ax.text(row.bus14_pct, row.hard_pct, str(int(row.design_id)),
                ha="center", va="center", color="white", fontsize=8.5,
                fontweight="bold", zorder=4)

    ax.axhline(DN_HARD, color="black", linestyle=":", linewidth=1.4, zorder=0)
    ax.set_title(f"{title}\nSpearman $\\rho$ = {rho:+.2f} (n = 8)", fontsize=11)
    ax.set_xlabel("bus14 full-test mean survival (%)")
    ax.set_xlim(60, 102)
    ax.set_ylim(-1, max(25.5, source_target["hard_pct"].max() + 1.5))

axes[0].set_ylabel(f"WCCI difficult-chronic survival at mk32 (%)\n({N_HARD} chronics, zero-shot, ungated)")
fig.suptitle("High bus14 survival is not a consistent predictor of zero-shot WCCI transfer",
             fontsize=13, y=0.995)

key_labels = {
    1: "mean, f0, shared idle", 2: "mean, f0, dedicated idle",
    3: "mean, f1, shared idle", 4: "mean, f1, dedicated idle",
    5: "typed mean, f0, shared idle", 6: "typed mean, f0, dedicated idle",
    7: "typed mean, f1, shared idle", 8: "typed mean, f1, dedicated idle",
}
key_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=9,
           markerfacecolor=pool_colors["typed-mean pool" if number >= 5 else "mean pool"],
           markeredgecolor="white", label=f"{number}: {label}")
    for number, label in key_labels.items()
]
key_handles += [
    Line2D([0], [0], color="#555555", linestyle="--", label="descriptive linear trend"),
    Line2D([0], [0], color="black", linestyle=":",
           label=f"do nothing on difficult chronics ({DN_HARD:.2f}%)"),
]
fig.legend(handles=key_handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
           ncol=3, frameon=False, fontsize=8.2, columnspacing=1.4, handletextpad=0.5)
fig.tight_layout(rect=(0, 0.16, 1, 0.96))
save_figure(fig, "05_bus14_vs_wcci_mk32_exact_checkpoints")
plt.show()

source_target_export = source_target[[
    "design_id", "design", "scaling", "bus14_pct", "hard_pct", "overall_pct",
    "bus14_checkpoint_stem", "bus14_episodes", "bus14_result_path", "episode_csv",
]].rename(columns={
    "scaling": "input_condition",
    "bus14_pct": "bus14_survival_pct",
    "hard_pct": "wcci_difficult_survival_pct",
    "overall_pct": "wcci_overall_survival_pct",
})
source_target_export.to_csv(EXPORT_DIR / "bus14_vs_wcci_mk32_exact_checkpoints.csv", index=False)
source_target_correlations = pd.DataFrame(correlation_rows)
source_target_correlations.to_csv(
    EXPORT_DIR / "bus14_vs_wcci_mk32_correlations.csv", index=False
)

display(Markdown(
    "The raw-input checkpoints show a weak negative rank association, while the "
    "physically scaled checkpoints show a moderate positive one. The signs disagree, "
    "and each panel contains only eight single-seed variants. The defensible conclusion "
    "is therefore that **bus14 survival alone does not consistently rank zero-shot WCCI "
    "transfer**. Architecture details and input handling matter beyond source-grid score."
))
display(source_target_correlations.round(3))
display(source_target_export.sort_values(["input_condition", "design_id"]).round(3))
'''


def cell(kind: str, source: str, analysis_id: str) -> dict:
    result = {
        "cell_type": kind,
        "metadata": {"analysis_id": analysis_id},
        "source": source.strip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]

# The Gmax-delta campaign was downloaded after the original notebook was
# written and reuses the same route and short variant names.  Keep the older
# zero-shot comparison cell tied to the original checkpoint groups as well, so
# a complete notebook execution remains one row per architecture.
for existing in cells:
    source = "".join(existing.get("source", []))
    if "zero32 = runs.loc[" not in source:
        continue
    zero32_block = source[source.index("zero32 = runs.loc["):source.index("order32 =")]
    if "ORIGINAL_ZERO_SHOT_GROUP" in zero32_block:
        continue
    source = source.replace(
        '    & runs["variant"].str.startswith("NLS_")\n].copy()\nassert len(zero32) == 8',
        '    & runs["variant"].str.startswith("NLS_")\n'
        '    & runs["group"].str.match(ORIGINAL_ZERO_SHOT_GROUP, na=False)\n'
        '].copy()\nassert len(zero32) == 8',
        1,
    )
    existing["source"] = source.splitlines(keepends=True)

# Replace an earlier version of this section rather than duplicating it.
cells[:] = [
    existing for existing in cells
    if existing.get("metadata", {}).get("analysis_id") not in {SECTION_ID, CODE_ID}
]

audit_index = next(
    i for i, existing in enumerate(cells)
    if "section67_audit = pd.DataFrame" in "".join(existing.get("source", []))
)
cells[audit_index + 1:audit_index + 1] = [
    cell("markdown", MARKDOWN, SECTION_ID),
    cell("code", CODE, CODE_ID),
]

NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"updated {NOTEBOOK} ({len(cells)} cells)")
