import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival")))
import wcci_transfer_data as W
import pandas as pd
import json

data = W.load()
runs = data.runs
ft = runs[runs["route"].str.contains("fine-tune") | runs["route"].str.contains("BC") | runs["route"].str.contains("scratch")].copy()
ft = ft[ft["hard_pct"].isna()]
print("Missing hard_pct count:", len(ft))

# Let's inspect one specific JSON to see where the artifact path actually is
example_path = Path("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/outputs/full_test_eval/shared/wcci/ft64c_wcci36_mk64_localrho/ft64c_NLS_mean_f0_a0h0_s0_lr095.json")
with open(example_path, 'r') as f:
    payload = json.load(f)
    print("Action Artifacts inside JSON:", payload.get("action_artifacts", {}))
