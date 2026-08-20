import nbformat as nbf
nb = nbf.read("/Users/corentinplumet/Documents/RL_Marl2grid/Topology_Task/analysis/metrics/notebooks/episodic_survival/wcci_four_questions.ipynb", as_version=4)
for cell in nb.cells:
    if "Is fine-tuning useful?" in cell.source or "ft_runs" in cell.source or "controlled contrast" in cell.source:
        print("==========")
        print(cell.source)
