import json

notebook_path = "Topology_Task/analysis/metrics/notebooks/episodic_survival/baseline_episodic_survival_plots.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Code for new cell to create truncated history and re-plot
new_code = """# Create truncated history: max 15M steps for all runs except baselines
baseline_patterns = ['rerun_a0_opt', 'paper_mappo_discrete_baseline']
def is_baseline(name):
    return any(p in name for p in baseline_patterns)

history_df_cut = history_df.copy()
# Filter: keep if it's a baseline OR step <= 15e6
history_df_cut = history_df_cut[
    history_df_cut['run_name'].apply(is_baseline) | (history_df_cut['step'] <= 15000000)
]

# 1. GINE reruns cut
gine_rerun_result_cut = {
    "mean_groups": GINE_RERUN_MEAN_GROUPS,
    "fig": wm.plot_run_mean_groups(
        GINE_RERUN_MEAN_GROUPS,
        split="test",
        smooth=5,
        title="GINE reruns: editable comparisons with A0_baseline (Cut to 15M)",
        ncols=2,
        subplot_height=400,
        width=1500,
        y_range=[0, 105],
        show_members=True,
        show_std=True,
        save_name="rerun_gine_a0_baseline_editable_comparisons_cut",
        history=history_df_cut,
    ),
}
display(gine_rerun_result_cut["fig"])

# 2. A0 core pairwise cut
a0_core_pairwise_result_cut = {
    "mean_groups": A0_CORE_PAIRWISE_MEAN_GROUPS,
    "fig": wm.plot_run_mean_groups(
        A0_CORE_PAIRWISE_MEAN_GROUPS,
        split="test",
        smooth=5,
        title="A0 core reruns: editable pairwise comparisons (Cut to 15M)",
        ncols=3,
        subplot_height=340,
        width=1400,
        y_range=[0, 105],
        show_members=True,
        show_std=True,
        save_name="rerun_a0_core_pairwise_editable_comparisons_cut",
        history=history_df_cut,
    ),
}
display(a0_core_pairwise_result_cut["fig"])

# 3. A0 core all cut
a0_core_all_result_cut = wm.plot_named_run_means(
    A0_CORE_ALL_RUN_SPECS,
    title="A0 core reruns: editable single-panel comparison (Cut to 15M)",
    split="test",
    smooth=5,
    show_members=True,
    show_std=True,
    save_name="rerun_a0_core_all_curves_editable_cut",
    history=history_df_cut,
)
display(a0_core_all_result_cut["fig"])

# 4. A0 bias cut
a0_bias_result_cut = wm.plot_named_run_means(
    A0_BIAS_RUN_SPECS,
    title="A0 baseline vs init-bias controls (Cut to 15M)",
    split="test",
    smooth=5,
    show_members=True,
    show_std=True,
    save_name="rerun_a0_opt_vs_bias05_bias07_editable_cut",
    history=history_df_cut,
)
display(a0_bias_result_cut["fig"])

# 5. Paper vs A0 baselines cut
paper_vs_a0_baseline_result_cut = wm.plot_named_run_means(
    PAPER_BASELINE_RUN_SPECS,
    title="Paper MAPPO discrete baseline vs A0 baselines (Cut to 15M)",
    split="test",
    smooth=5,
    show_members=True,
    show_std=True,
    save_name="paper_mappo_discrete_baseline_vs_a0_baselines_cut",
    history=history_df_cut,
)
display(paper_vs_a0_baseline_result_cut["fig"])
"""

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in new_code.split("\n")]
}

nb['cells'].append(new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Added new cell to notebook.")
