import re

with open('latex/chapters/06_experiments.tex', 'r') as f:
    text = f.read()

replacements = [
    (
        r"seed-0 factorials trained on the corrected information boundary. They ask whether ",
        r"seed-0 factorials. They ask whether "
    ),
    (
        r"Evaluation is deterministic at a fixed checkpoint and seed. The corrected\nscreens",
        r"Evaluation is deterministic at a fixed checkpoint and seed. The\nscreens"
    ),
    (
        r"the corrected-boundary reference configuration of",
        r"the reference configuration of"
    ),
    (
        r"  survival of each run's best checkpoint. All cells use corrected graph inputs\n  and seed 0.",
        r"  survival of each run's best checkpoint at seed 0."
    ),
    (
        r"The corrected local graph may contain nodes that provide message-passing",
        r"The local graph may contain nodes that provide message-passing"
    ),
    (
        r"seed 0. All cells use GINE with $K=2$ and $d_h=128$ on the corrected busbar",
        r"seed 0. All cells use GINE with $K=2$ and $d_h=128$ on the busbar"
    ),
    (
        r"is fixed for the corrected reruns of Screens C and D, preventing readout from",
        r"is fixed for Screens C and D, preventing readout from"
    ),
    (
        r"the corrected reference from Screen B. Cells are named by their three switches, so",
        r"the reference from Screen B. Cells are named by their three switches, so"
    ),
    (
        r"\\section{Screen E: Candidate-Action Scoring on the Corrected Boundary}",
        r"\\section{Screen E: Candidate-Action Scoring}"
    ),
    (
        r"$99\\%$, so the candidate-action head is a viable actor on the corrected\nboundary. No factor separates.",
        r"$99\\%$, so the candidate-action head is a viable actor.\nNo factor separates."
    ),
    (
        r"\\subsection{Replication with Corrected Graph Inputs}",
        r"\\subsection{Replication with Improved Graph Features}"
    ),
    (
        r"At seed 0, the eight cells were repeated with corrected graph inputs \(NLS\):",
        r"At seed 0, the eight cells were repeated with improved graph features (NLS):"
    ),
    (
        r"  original \(NL\) and corrected \(NLS\) graph inputs. The final column is NLS minus",
        r"  baseline (NL) and improved (NLS) graph features. The final column is NLS minus"
    )
]

for target, replacement in replacements:
    target_exact = target.replace(r"\n", "\n").replace(r"\\", "\\").replace(r"\^", "^").replace(r"\$", "$").replace(r"\(", "(").replace(r"\)", ")").replace(r"\_", "_")
    replacement_exact = replacement.replace(r"\n", "\n").replace(r"\\", "\\").replace(r"\^", "^").replace(r"\$", "$").replace(r"\(", "(").replace(r"\)", ")").replace(r"\_", "_")
    
    if target_exact in text:
        text = text.replace(target_exact, replacement_exact)
        print("REPLACED:", target_exact[:30].replace("\n", " "), "...")
    else:
        print("COULD NOT FIND:", target_exact[:30].replace("\n", " "), "...")

with open('latex/chapters/06_experiments.tex', 'w') as f:
    f.write(text)

