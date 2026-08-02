#!/usr/bin/env python3
"""Render the distribution figure from a compare_graph_features output directory.

Split out from the collector so the plot can be produced on a machine that has
matplotlib but no Grid2Op: copy the directory over and run

    python tools/plot_feature_samples.py outputs/graph_feature_comparison/bus
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "marl2grid_mplconfig")
)

import numpy as np  # noqa: E402

from common.feature_stats import plot_distributions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory holding samples_<env>.npz written by compare_graph_features.",
    )
    parser.add_argument("--output", type=Path, default=None)
    cli = parser.parse_args()

    directory = cli.directory
    if not directory.is_absolute():
        directory = TASK_DIR / directory

    archives = sorted(directory.glob("samples_*.npz"))
    if len(archives) != 2:
        raise SystemExit(
            f"Expected exactly two samples_*.npz in {directory}, found {len(archives)}."
        )

    labels = [path.stem[len("samples_") :] for path in archives]
    samples = [dict(np.load(path)) for path in archives]

    output = cli.output or (directory / "distributions.png")
    plot_distributions(samples[0], samples[1], labels[0], labels[1], output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
