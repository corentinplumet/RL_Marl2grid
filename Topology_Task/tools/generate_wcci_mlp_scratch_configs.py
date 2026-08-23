#!/usr/bin/env python3
"""Generate the flat-MLP-actor WCCI from-scratch baseline.

Chapter 3 asks whether a graph actor can replace the flat one without paying
for it. Every WCCI from-scratch run so far answers a narrower question: the
`wsc_*` screen varies the *graph* encoder, and `sc<cap>c_*` varies the candidate
head, but both keep a GNN. This campaign supplies the missing control — the same
environment, budget and optimizer with `actor_encoder = "mlp"`, i.e. no graph at
all, reading the flat observation directly.

It is derived from `wsc_bus_e0n0v0_<cap>_s0.toml` rather than from the bus14
baseline so that everything except the actor representation is held fixed:
same 15M-step budget, same chronic split, same reward, same PPO settings, same
`actor_action_head = "mlp"` over the reduced space. The difference between this
and the `wsc` line is then attributable to the encoder alone.

    python tools/generate_wcci_mlp_scratch_configs.py 32 64 256
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
CONFIG_ROOT = TASK_DIR / "configs" / "no_leakage_config"
SOURCE_DIR = CONFIG_ROOT / "X_wcci_scratch_arch"
OUTPUT_DIR = CONFIG_ROOT / "V_wcci_mlp_scratch"
SOURCE_CAP = 64
ACTION_SPACE = (
    "outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/"
    "reduced_action_space_wcci_full2048a_90_v3_mk{cap}.json"
)
# Every graph setting in the source config. With actor_encoder = "mlp" the actor
# builds no graph, so these are inert; they are dropped rather than left behind
# so the config cannot be misread as describing a GNN run.
GRAPH_KEY_RE = re.compile(r"^(gnn_|sparse_gt_|graphsage_|gcn_)")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one {old!r}, found {count}.")
    return text.replace(old, new, 1)


def strip_graph_settings(text: str) -> str:
    kept, dropped = [], 0
    for line in text.splitlines(keepends=True):
        if GRAPH_KEY_RE.match(line):
            dropped += 1
            continue
        kept.append(line)
    if dropped == 0:
        raise ValueError("No graph settings found to strip; source config changed.")
    return "".join(kept)


def generate(cap: int) -> tuple[str, str]:
    source_path = SOURCE_DIR / f"wsc_bus_e0n0v0_mk{SOURCE_CAP}_s0.toml"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source config: {source_path}")
    target = f"wmlp_mk{cap}_s0"
    text = source_path.read_text(encoding="utf-8")

    text = replace_once(text, ACTION_SPACE.format(cap=SOURCE_CAP), ACTION_SPACE.format(cap=cap))
    text = replace_once(text, f'name = "wsc_bus_e0n0v0_mk{SOURCE_CAP}_s0"', f'name = "{target}"')
    text = replace_once(text, f'exp_tag = "wsc_bus_e0n0v0_mk{SOURCE_CAP}_s0"',
                        f'exp_tag = "{target}"')
    text = replace_once(text, 'actor_encoder = "gnn"', 'actor_encoder = "mlp"')
    text = replace_once(text, 'run_dir = "outputs/izar-{config_name}-{job_id}"',
                        'run_dir = "outputs/{config_name}-{job_id}"')
    text = strip_graph_settings(text)

    header = (
        f"# WCCI flat-MLP-actor from-scratch baseline, mk{cap}, seed 0.\n"
        "# actor_encoder = mlp: no graph encoder at all, reading the flat observation.\n"
        f"# Derived from wsc_bus_e0n0v0_mk{SOURCE_CAP}_s0.toml with every graph setting\n"
        "# removed and the action space re-capped; the environment, the 15M-step budget,\n"
        "# the chronic split and every PPO setting are byte-identical, so the difference\n"
        "# against the wsc screen is attributable to the encoder alone.\n\n"
    )
    body_start = text.find("\n\n")
    if body_start < 0:
        raise ValueError("Source config lacks a header paragraph")
    return target, header + text[body_start + 2:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caps", type=int, nargs="+", help="per-agent action caps, e.g. 32 64 256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    written = []
    for cap in args.caps:
        action_space = TASK_DIR / ACTION_SPACE.format(cap=cap)
        if not action_space.is_file():
            parser.error(f"Missing reduced action space {action_space}")
        target, text = generate(cap)
        if not args.dry_run:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / f"{target}.toml").write_text(text, encoding="utf-8")
        written.append(target)

    verb = "Would write" if args.dry_run else "Wrote"
    print(f"{verb} {len(written)} config(s) to {OUTPUT_DIR.relative_to(TASK_DIR)}")
    for name in written:
        print(f"  {name}.toml")


if __name__ == "__main__":
    main()
