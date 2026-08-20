import nbformat as nbf
nb = nbf.read("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival/transfer_story.ipynb", as_version=4)
for cell in nb.cells:
    if "grid_counts =" in cell.source or "sns.heatmap" in cell.source:
        print("==========")
        print(cell.source)
        break
