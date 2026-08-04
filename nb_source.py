from pathlib import Path
import importlib
import sys

for candidate in [Path.cwd(), *Path.cwd().parents]:
    helpers = candidate / "helpers"
    repo_helpers = candidate / "Topology_Task" / "analysis" / "metrics" / "helpers"
    if helpers.exists() and (helpers / "wandb_metrics.py").exists():
        sys.path.insert(0, str(helpers))
        break
    if repo_helpers.exists() and (repo_helpers / "wandb_metrics.py").exists():
        sys.path.insert(0, str(repo_helpers))
        break
else:
    raise FileNotFoundError("Could not locate Topology_Task/analysis/metrics/helpers")

import wandb_metrics as wm
wm = importlib.reload(wm)
print("wandb_metrics:", wm.__file__)

data = wm.load_wandb_data()

runs_df = data.runs_df
history_df = data.history_df

GINE_RERUN_A0_BASELINE_SPEC = {
    "prefix": "rerun_a0_opt_s",
    "label": "Optimized MLP Baseline",
    "color": "#1f77b4",
    "width": 4,
}
GINE_RERUN_ANCIENT_BASELINE_SPEC = {
    "prefix": "best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s",
    "label": "Historical GINE Baseline",
    "color": "#7f7f7f",
    "dash": "dash",
    "width": 3,
}

GINE_RERUN_SUBPLOTS = {
    "Optimized MLP vs Historical GINE": [
        GINE_RERUN_A0_BASELINE_SPEC,
        GINE_RERUN_ANCIENT_BASELINE_SPEC,
    ],
    "Optimized MLP vs Light GINE with MLP critic": [
        GINE_RERUN_A0_BASELINE_SPEC,
        {
            "prefix": "best_10_shared_actor_gnn_light_gine_a4_concat_flat_critic_mlp_optcritic_s",
            "label": "Light GINE with MLP critic",
            "color": "#ff7f0e",
        },
    ],
    "Optimized MLP vs Historical GINE (Bias 0.0)": [
        GINE_RERUN_A0_BASELINE_SPEC,
        {
            "prefix": "best_11_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_initbias0_s",
            "label": "Historical GINE (Bias 0.0)",
            "color": "#ff7f0e",
        },
    ],
    "Optimized MLP vs Opt critic & Bias 0.0": [
        GINE_RERUN_A0_BASELINE_SPEC,
        {
            "prefix": "best_12_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_opt_initbias0_s",
            "label": "Optimized critic and Bias 0.0",
            "color": "#ff7f0e",
        },
    ],
    "Optimized MLP vs A0-style heavy GINE": [
        GINE_RERUN_A0_BASELINE_SPEC,
        {
            "prefix": "best_13_shared_actor_gnn_gine_a0_concat_flat_critic_gnn_opt_s",
            "label": "A0-style heavy GINE",
            "color": "#ff7f0e",
        },
    ],
    "Optimized MLP vs Light A0-style GINE (MLP critic)": [
        GINE_RERUN_A0_BASELINE_SPEC,
        {
            "prefix": "best_14_shared_actor_gnn_light_gine_a0_no_concat_flat_critic_mlp_optcritic_s",
            "label": "Light A0-style GINE with MLP critic",
            "color": "#ff7f0e",
        },
    ],
}

GINE_RERUN_MEAN_GROUPS = {
    title: wm.resolve_named_plot_specs(run_specs, history=history_df)
    for title, run_specs in GINE_RERUN_SUBPLOTS.items()
}
GINE_RERUN_MEAN_GROUPS = {
    title: specs
    for title, specs in GINE_RERUN_MEAN_GROUPS.items()
    if len(specs) >= 2
}

gine_rerun_result = {
    "mean_groups": GINE_RERUN_MEAN_GROUPS,
    "fig": wm.plot_run_mean_groups(
        GINE_RERUN_MEAN_GROUPS,
        split="test",
        smooth=5,
        title="GINE reruns: editable comparisons with A0_baseline",
        ncols=2,
        subplot_height=400,
        width=1500,
        y_range=[0, 105],
        show_members=True,
        show_std=True,
        save_name="rerun_gine_a0_baseline_editable_comparisons",
        history=history_df,
    ),
}
gine_rerun_result["fig"]


A0_CORE_PAIRWISE_BASELINE_SPEC = {
    "prefix": "rerun_a0_opt_s",
    "label": "Optimized MLP Baseline",
    "color": "#1f77b4",
    "width": 4,
}

A0_CORE_PAIRWISE_SUBPLOTS = {
    "Opt MLP vs Non-opt critic": [
        A0_CORE_PAIRWISE_BASELINE_SPEC,
        {
            "prefix": "rerun_a0_nonopt_s",
            "label": "Disable optimized critic update",
            "color": "#ff7f0e",
        },
    ],
    "Opt MLP vs Smaller MLP": [
        A0_CORE_PAIRWISE_BASELINE_SPEC,
        {
            "prefix": "rerun_a01_opt_s",
            "label": "Smaller MLP actor/critic",
            "color": "#2ca02c",
        },
    ],
    "Opt MLP vs No reward norm": [
        A0_CORE_PAIRWISE_BASELINE_SPEC,
        {
            "prefix": "rerun_a02_opt_s",
            "label": "Disable reward normalization",
            "color": "#d62728",
        },
    ],
    "Opt MLP vs Old baseline": [
        A0_CORE_PAIRWISE_BASELINE_SPEC,
        {
            "regex": r"^noval20_mlp_a1_no_entropy_decay_nonopt_s[0-2]_det$",
            "label": "Old baseline (no val)",
            "color": "#9467bd",
        },
    ],
}

A0_CORE_PAIRWISE_MEAN_GROUPS = {
    title: wm.resolve_named_plot_specs(run_specs, history=history_df)
    for title, run_specs in A0_CORE_PAIRWISE_SUBPLOTS.items()
}
A0_CORE_PAIRWISE_MEAN_GROUPS = {
    title: specs
    for title, specs in A0_CORE_PAIRWISE_MEAN_GROUPS.items()
    if len(specs) >= 2
}

a0_core_pairwise_result = {
    "mean_groups": A0_CORE_PAIRWISE_MEAN_GROUPS,
    "fig": wm.plot_run_mean_groups(
        A0_CORE_PAIRWISE_MEAN_GROUPS,
        split="test",
        smooth=5,
        title="A0 core reruns: editable pairwise comparisons",
        ncols=2,
        subplot_height=430,
        width=1500,
        y_range=[0, 105],
        show_members=True,
        show_std=True,
        save_name="rerun_a0_core_pairwise_editable_comparisons",
        history=history_df,
    ),
}
a0_core_pairwise_result["fig"]


A0_CORE_ALL_RUN_SPECS = [
    {
        "name": "paper_mappo_discrete_baseline_thread16",
        "label": "Paper MAPPO baseline",
        "color": "#111111",
        "width": 4,
    },
    {"prefix": "rerun_a0_opt_s", "label": "Optimized MLP Baseline", "color": "#1f77b4", "width": 4},
    {"prefix": "rerun_a0_nonopt_s", "label": "Disable optimized critic update", "color": "#ff7f0e"},
    {"prefix": "rerun_a01_opt_s", "label": "Smaller MLP actor/critic", "color": "#2ca02c"},
    {"prefix": "rerun_a02_opt_s", "label": "Disable reward normalization", "color": "#d62728"},
    {
        "regex": r"^noval20_mlp_a1_no_entropy_decay_nonopt_s[0-2]_det$",
        "label": "Old baseline (no val)",
        "color": "#9467bd",
    },
]

a0_core_all_result = wm.plot_named_run_means(
    A0_CORE_ALL_RUN_SPECS,
    title="A0 core reruns: editable single-panel comparison",
    split="test",
    smooth=5,
    show_members=True,
    show_std=True,
    save_name="rerun_a0_core_all_curves_editable",
)
a0_core_all_result["fig"]


A0_BIAS_RUN_SPECS = [
    {"prefix": "rerun_a0_opt_s", "label": "Optimized MLP Baseline", "color": "#1f77b4", "width": 4},
    {"name": "rerun_bias_05", "label": "Initial Bias 50%", "color": "#ff7f0e"},
    {"name": "rerun_bias_07", "label": "Initial Bias 70%", "color": "#d62728"},
]

a0_bias_result = wm.plot_named_run_means(
    A0_BIAS_RUN_SPECS,
    title="A0 baseline vs init-bias controls",
    split="test",
    smooth=5,
    show_members=True,
    show_std=True,
    save_name="rerun_a0_opt_vs_bias05_bias07_editable",
)
a0_bias_result["fig"]


PAPER_BASELINE_RUN_SPECS = [
    {
        "name": "paper_mappo_discrete_baseline_thread16",
        "label": "Paper MAPPO baseline",
        "color": "#111111",
        "width": 4,
    },
    {"prefix": "rerun_a0_opt_s", "label": "Optimized MLP Baseline", "color": "#1f77b4", "width": 4},
    {"name": "rerun_bias_05", "label": "Optimized MLP Baseline (Bias 50%)", "color": "#ff7f0e"},
    {"name": "rerun_bias_07", "label": "Optimized MLP Baseline (Bias 70%)", "color": "#d62728"},
]

paper_vs_a0_baseline_result = wm.plot_named_run_means(
    PAPER_BASELINE_RUN_SPECS,
    title="Paper MAPPO discrete baseline vs A0 baselines",
    split="test",
    smooth=5,
    show_members=True,
    show_std=True,
    save_name="paper_mappo_discrete_baseline_vs_a0_baselines",
)
paper_vs_a0_baseline_result["fig"]

# Create truncated history: max 15M steps for all runs except baselines
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
print(gine_rerun_result_cut["fig"])

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
print(a0_core_pairwise_result_cut["fig"])

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
print(a0_core_all_result_cut["fig"])

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
print(a0_bias_result_cut["fig"])

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
print(paper_vs_a0_baseline_result_cut["fig"])


