"""Extract the bus14 vs WCCI feature-shift figures from the analysis notebook.

Cells are located by a marker in their source rather than by index, so
inserting or reordering cells in the notebook does not silently pull the wrong
figure.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    REPO_ROOT
    / "Topology_Task"
    / "analysis"
    / "metrics"
    / "notebooks"
    / "graph_features"
    / "bus14_vs_wcci_feature_shift.ipynb"
)
OUT_DIR = REPO_ROOT / "latex" / "figures"

# marker in the cell source -> output filename
FIGURES = {
    'overlay(axes[0], "rho (all physical edges)"': "wcci_feature_dense.png",
    "(log_scale, clip) in enumerate": "wcci_feature_power.png",
    'for ax, key in zip(axes, ["gen_theta", "load_theta"])': "wcci_feature_angles.png",
    "KS is the largest vertical gap": "wcci_feature_ks_ecdf.png",
    "Green band + left of the dashed line": "wcci_feature_offset_scale.png",
}


def main() -> None:
    if not NOTEBOOK.exists():
        raise SystemExit(f"Notebook not found: {NOTEBOOK}")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        for marker, filename in FIGURES.items():
            if marker not in source or filename in written:
                continue
            for output in cell.get("outputs", []):
                png = output.get("data", {}).get("image/png")
                if not png:
                    continue
                path = OUT_DIR / filename
                path.write_bytes(base64.b64decode(png))
                written[filename] = path.stat().st_size
                break

    missing = sorted(set(FIGURES.values()) - set(written))
    for filename, size in sorted(written.items()):
        print(f"wrote {filename} ({size / 1024:.0f} KiB)")
    if missing:
        raise SystemExit(
            "No PNG output found for: "
            + ", ".join(missing)
            + ". Run the notebook first so its figures are stored."
        )


if __name__ == "__main__":
    main()
