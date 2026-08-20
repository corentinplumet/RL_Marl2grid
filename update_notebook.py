import nbformat as nbf
from pathlib import Path

notebook_path = Path("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival/transfer_story.ipynb")
nb = nbf.read(notebook_path, as_version=4)

new_markdown = """
As shown in the grid above, the fine-tuned models were evaluated using different configurations. Let's clarify the difference between them:

1. **Ungated Fine-Tuning (`heuristic: none`)**: The agent's trained policy acts completely on its own. Whatever action the fine-tuned neural network outputs is executed directly on the target grid.
2. **Gated Fine-Tuning (`heuristic: local rho` or `global rho`)**: Even though the neural network has been fine-tuned on the target grid, we *still* don't trust it blindly. We pass its selected actions through the same safety evaluation gate used during zero-shot transfer. The gate uses a heuristic (like `local rho`) to estimate if the action will lead to immediate disaster; if it is deemed too dangerous, it selects the "do nothing" fallback.

This raises an important question: **Does a policy that has already been fine-tuned on the target grid actually need the heuristic gate anymore?** Did fine-tuning teach it to be perfectly safe, or does it still make mistakes that the gate needs to catch?

To answer this, we can directly compare the ungated vs. gated evaluations of the `MAPPO fine-tune (conservative)` runs (which we now have 8 architectures for).
"""

for cell in nb.cells:
    if "As shown in the grid, target-grid trained policies were evaluated in two main ways:" in cell.source:
        cell.source = new_markdown.strip()

nbf.write(nb, notebook_path)
print("Notebook markdown updated successfully.")
