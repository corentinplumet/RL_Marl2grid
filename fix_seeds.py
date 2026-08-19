import re

with open('latex/chapters/06_experiments.tex', 'r') as f:
    text = f.read()

replacements = [
    (
        r"completed seed-0 campaigns on the corrected information boundary; Screens C\nand D retain the earlier three-seed legacy campaigns; Screen E is a seed-0",
        r"completed seed-0 campaigns on the corrected information boundary across all screens; Screen E is a seed-0"
    ),
    (
        r"Seeds & Screens A, B, E: 0; Screens C, D: 0, 1, 2 \\\\",
        r"Seeds & Screens A, B, C, D, E: 0 \\\\"
    ),
    (
        r"screens do not report how far their runs got. Screens A and B report one seed\nper cell; Screens C and D aggregate three seeds per cell. Every seed-level",
        r"screens do not report how far their runs got. All screens report one seed\nper cell. Every seed-level"
    ),
    (
        r"Evaluation is deterministic at a fixed checkpoint and seed. The corrected\nScreens A and B nevertheless have only one training seed per cell, so their",
        r"Evaluation is deterministic at a fixed checkpoint and seed. The corrected\nscreens have only one training seed per cell, so their"
    ),
    (
        r"A complete \$2\^3\$ factorial with three seeds per cell, 24 runs. The graph type is\nthe corrupted \\texttt{bus14} baseline. The readout is the nominal \\texttt{mean},",
        r"A complete $2^3$ factorial at seed 0, 8 runs. The graph type is\nthe corrected \\texttt{bus14} baseline. The readout is \\texttt{energized\\_max},"
    ),
    (
        r"The detailed result tables, training curves, and per-seed heatmap are reported in\nAppendix~\\ref{app:screen-c-additional-results}.",
        r"The detailed result tables and training curves are reported in\nAppendix~\\ref{app:screen-c-additional-results}."
    ),
    (
        r"A complete \$3\\times3\$ factorial over the generator and load directions with\nthree seeds per cell: 27 runs.",
        r"A complete $3\\times3$ factorial over the generator and load directions at seed 0: 9 runs."
    ),
    (
        r"\\caption{Screen D. Full-test survival of the best checkpoint by generator\n  direction \(rows\) and load direction \(columns\), with the three individual seed\n  values printed under each cell mean. The baseline cell is",
        r"\\caption{Screen D. Full-test survival of the best checkpoint by generator\n  direction (rows) and load direction (columns). The baseline cell is"
    )
]

for target, replacement in replacements:
    # Use re.escape for exact matches unless we intentionally used regex.
    # Actually some targets contain regex chars, let's just do exact string replacement.
    target_exact = target.replace(r"\n", "\n").replace(r"\\", "\\").replace(r"\^", "^").replace(r"\$", "$").replace(r"\(", "(").replace(r"\)", ")").replace(r"\_", "_")
    replacement_exact = replacement.replace(r"\n", "\n").replace(r"\\", "\\").replace(r"\^", "^").replace(r"\$", "$").replace(r"\(", "(").replace(r"\)", ")").replace(r"\_", "_")
    
    if target_exact in text:
        text = text.replace(target_exact, replacement_exact)
        print("REPLACED:", target_exact[:30], "...")
    else:
        print("COULD NOT FIND:", target_exact[:30], "...")

with open('latex/chapters/06_experiments.tex', 'w') as f:
    f.write(text)

