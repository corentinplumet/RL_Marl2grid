import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival")))
import wcci_transfer_data as W
import pandas as pd
import numpy as np

data = W.load()
runs = data.runs.dropna(subset=["overall_pct"])

print("--- MAPPO vs BC ---")
# Focus on NLS_mean_f1_a0h0, mk64
mappo = runs[(runs["route"] == "MAPPO fine-tune (conservative)") & (runs["variant"] == "NLS_mean_f1_a0h0") & (runs["mk"] == 64)][["gate", "overall_pct"]]
bc = runs[(runs["route"] == "BC on greedy labels") & (runs["variant"] == "NLS_mean_f1_a0h0") & (runs["mk"] == 64)][["gate", "overall_pct"]]
print("MAPPO:\n", mappo)
print("BC:\n", bc)


print("\n--- 2x2 Factorial: Zero-Shot vs Fine-Tune AND Ungated vs Gated ---")
# Focus on the 8 conservative variants at mk64
cons_variants = runs[runs["route"] == "MAPPO fine-tune (conservative)"]["variant"].unique()

zero_ungated = runs[(runs["route"] == "bus14 zero-shot") & (runs["gate"] == "ungated") & (runs["mk"] == 64) & (runs["variant"].isin(cons_variants))]
zero_gated = runs[(runs["route"] == "bus14 zero-shot") & (runs["gate"] == "gated") & (runs["mk"] == 64) & (runs["variant"].isin(cons_variants))]
ft_ungated = runs[(runs["route"] == "MAPPO fine-tune (conservative)") & (runs["gate"] == "ungated") & (runs["mk"] == 64)]
ft_gated = runs[(runs["route"] == "MAPPO fine-tune (conservative)") & (runs["gate"] == "gated") & (runs["mk"] == 64)]

print(f"Zero Ungated: {len(zero_ungated)}")
print(f"Zero Gated: {len(zero_gated)}")
print(f"FT Ungated: {len(ft_ungated)}")
print(f"FT Gated: {len(ft_gated)}")

# Merge into a single dataframe for comparison
def get_perf(df):
    return df.set_index("variant")["overall_pct"]

df2x2 = pd.DataFrame({
    "Zero-Shot Ungated": get_perf(zero_ungated),
    "Zero-Shot Gated": get_perf(zero_gated),
    "Fine-Tuned Ungated": get_perf(ft_ungated),
    "Fine-Tuned Gated": get_perf(ft_gated)
})
print(df2x2)
print("Means:")
print(df2x2.mean())

