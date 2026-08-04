import json
import re

notebook_path = "Topology_Task/analysis/metrics/notebooks/episodic_survival/baseline_episodic_survival_plots.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = {
    '"label": "A0_baseline"': '"label": "Optimized MLP Baseline"',
    '"label": "ancient_GINE_baseline"': '"label": "Historical GINE Baseline"',
    '"label": "10_light_GINE_MLP_critic"': '"label": "Light GINE with MLP critic"',
    '"label": "11_legacy_init_bias_0.0"': '"label": "Historical GINE (Bias 0.0)"',
    '"label": "12_optcritic_init_bias_0.0"': '"label": "Optimized critic and Bias 0.0"',
    '"label": "13_GINE_A0_GNN_critic"': '"label": "A0-style heavy GINE"',
    '"label": "14_light_GINE_A0_MLP_critic"': '"label": "Light A0-style GINE with MLP critic"',
    '"label": "rerun_a0_opt"': '"label": "Optimized MLP Baseline"',
    '"label": "rerun_a0_nonopt"': '"label": "Disable optimized critic update"',
    '"label": "rerun_a01"': '"label": "Smaller MLP actor/critic"',
    '"label": "rerun_a02"': '"label": "Disable reward normalization"',
    '"label": "old_a0_nonopt_noval20"': '"label": "Old baseline (no val)"',
    '"label": "rerun_bias_05"': '"label": "Initial Bias 50%"',
    '"label": "rerun_bias_07"': '"label": "Initial Bias 70%"',
    '"label": "A0_baseline_bias05"': '"label": "Optimized MLP Baseline (Bias 50%)"',
    '"label": "A0_baseline_bias07"': '"label": "Optimized MLP Baseline (Bias 70%)"',
    '"label": "paper_mappo_discrete_baseline"': '"label": "Paper MAPPO baseline"',
}

# Also replace subplot titles
title_replacements = {
    '"A0 baseline vs ancient GINE baseline"': '"Optimized MLP vs Historical GINE"',
    '"A0 baseline vs 10 light GINE MLP critic"': '"Optimized MLP vs Light GINE with MLP critic"',
    '"A0 baseline vs 11 legacy init bias 0.0"': '"Optimized MLP vs Historical GINE (Bias 0.0)"',
    '"A0 baseline vs 12 optcritic init bias 0.0"': '"Optimized MLP vs Opt critic & Bias 0.0"',
    '"A0 baseline vs 13 GINE A0 GNN critic"': '"Optimized MLP vs A0-style heavy GINE"',
    '"A0 baseline vs 14 light GINE A0 MLP critic"': '"Optimized MLP vs Light A0-style GINE (MLP critic)"',
    '"rerun_a0_opt vs rerun_a0_nonopt"': '"Opt MLP vs Non-opt critic"',
    '"rerun_a0_opt vs rerun_a01"': '"Opt MLP vs Smaller MLP"',
    '"rerun_a0_opt vs rerun_a02"': '"Opt MLP vs No reward norm"',
    '"rerun_a0_opt vs old_a0_nonopt_noval20"': '"Opt MLP vs Old baseline"',
}

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            for old, new in replacements.items():
                line = line.replace(old, new)
            for old, new in title_replacements.items():
                line = line.replace(old, new)
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated labels in notebook.")
