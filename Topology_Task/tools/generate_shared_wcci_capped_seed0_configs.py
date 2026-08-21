#!/usr/bin/env python3
"""Re-cap the seed-0 NLS WCCI scratch/fine-tuning campaign to another action space.

The mk64 campaign in `W_wcci_cas_hl_shared_mk64_seed0` already carries the two
optimization recipes that matter — the conservative 5M fine-tune and the matched
15M from-scratch control — so a new cap is derived from it rather than rebuilt
from the mk256 originals. Only the reduced-action-space artifact and the names
change; every optimizer, encoder and budget setting is copied verbatim, which is
what keeps the caps comparable to each other.

This works at all because the transferred actor has no per-action parameters:
`actor_action_head = "candidate_pool"` with `share_candidate_scorer = true`
scores a variable-length candidate set with shared weights, so the checkpoint
does not constrain the cap. `reduced_action_space` is a property of the target
environment, not of the weights being loaded.

    python tools/generate_shared_wcci_capped_seed0_configs.py 32
    python tools/generate_shared_wcci_capped_seed0_configs.py 16 --regime finetune
"""

from __future__ import annotations

import argparse
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
CONFIG_ROOT = TASK_DIR / "configs" / "no_leakage_config"
SOURCE_CAP = 64
ACTION_SPACE = (
    "outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/"
    "reduced_action_space_wcci_full2048a_90_v3_mk{cap}.json"
)
VARIANTS = (
    "mean_f0_a0h0", "mean_f0_a0h1", "mean_f1_a0h0", "mean_f1_a0h1",
    "tmean_f0_a0h0", "tmean_f0_a0h1", "tmean_f1_a0h0", "tmean_f1_a0h1",
)
PREFIXES = {"finetune": "ft{cap}c", "scratch": "sc{cap}c"}

# The two input conditions differ in exactly four things. NL configs are derived
# from the NLS ones rather than from the mk256 originals so that both families
# inherit the same conservative recipe, verbatim.
NL_SUBSTITUTIONS = (
    ('gnn_angle_representation = "edge_diff"', 'gnn_angle_representation = "node"'),
    ("gnn_physical_scaling = true", "gnn_physical_scaling = false"),
    ("# Input preprocessing: corrected encoder inputs", "# Input preprocessing factors"),
)
NLS_CHECKPOINT_DIR = "checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar"
NL_CHECKPOINT_DIR = "checkpoint/no_leak/shared/NL_cas_hl_shared"


def replace_all(text: str, old: str, new: str, expected: int | None = None) -> str:
    count = text.count(old)
    if count == 0 or (expected is not None and count != expected):
        raise ValueError(f"Expected {expected or '>=1'} of {old!r}, found {count}.")
    return text.replace(old, new)


def to_nl(text: str, variant: str) -> str:
    """Turn a scaled (NLS) config into its raw-input (NL) twin."""
    for old, new in NL_SUBSTITUTIONS:
        text = replace_all(text, old, new, expected=1)
    text = replace_all(
        text,
        f"{NLS_CHECKPOINT_DIR}/best_test_cas_hl_NLS_izar_shared_{variant}_s0.tar",
        f"{NL_CHECKPOINT_DIR}/best_test_cas_hl_NL_shared_{variant}_s0.tar",
    ) if NLS_CHECKPOINT_DIR in text else text
    return replace_all(text, "_NLS_", "_NL_")


def recap(text: str, variant: str, regime: str, cap: int, family: str = "NLS") -> str:
    source = f"{PREFIXES[regime].format(cap=SOURCE_CAP)}_NLS_{variant}_s0"
    target = f"{PREFIXES[regime].format(cap=cap)}_NLS_{variant}_s0"

    text = replace_all(text, ACTION_SPACE.format(cap=SOURCE_CAP),
                       ACTION_SPACE.format(cap=cap), expected=1)
    text = replace_all(text, f'name = "{source}"', f'name = "{target}"', expected=1)
    text = replace_all(text, f'exp_tag = "{source}"', f'exp_tag = "{target}"', expected=1)
    # Header and section comments carry the cap; rewrite them so a config never
    # describes a campaign it is not part of.
    # Inherited from the Izar mk64 campaign; these configs run on either cluster,
    # so the run directory should not claim one.
    text = replace_all(text, 'run_dir = "outputs/izar-{config_name}-{job_id}"',
                       'run_dir = "outputs/{config_name}-{job_id}"', expected=1)
    text = replace_all(text, f"mk{SOURCE_CAP}", f"mk{cap}")
    if f"mk{SOURCE_CAP}" in text:
        raise ValueError(f"{target}: stale mk{SOURCE_CAP} reference left behind")
    if family == "NL":
        text = to_nl(text, variant)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cap", type=int, help="per-agent action cap, e.g. 32")
    parser.add_argument("--regime", choices=("finetune", "scratch"), action="append",
                        help="restrict to one regime (repeatable); default is both")
    parser.add_argument("--family", choices=("NLS", "NL", "both"), default="NLS",
                        help="input condition: NLS (physical scaling, default), NL (raw "
                             "inputs), or both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.cap == SOURCE_CAP:
        parser.error(f"mk{SOURCE_CAP} is the source campaign; pick another cap")
    action_space = TASK_DIR / ACTION_SPACE.format(cap=args.cap)
    if not action_space.is_file():
        parser.error(
            f"Missing reduced action space {action_space}.\n"
            f"Generate it first with: TOP_KS={args.cap} "
            "Topology_Task/teacher_student/reduce_wcci_action_space_sizes.sh"
        )

    source_dir = CONFIG_ROOT / f"W_wcci_cas_hl_shared_mk{SOURCE_CAP}_seed0"
    output_dir = CONFIG_ROOT / f"W_wcci_cas_hl_shared_mk{args.cap}_seed0"
    regimes = args.regime or list(PREFIXES)

    families = ["NLS", "NL"] if args.family == "both" else [args.family]
    written = []
    for family in families:
        for regime in regimes:
            for variant in VARIANTS:
                source_name = f"{PREFIXES[regime].format(cap=SOURCE_CAP)}_NLS_{variant}_s0"
                source_path = source_dir / f"{source_name}.toml"
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing source config: {source_path}")
                target = f"{PREFIXES[regime].format(cap=args.cap)}_{family}_{variant}_s0"
                text = recap(source_path.read_text(encoding="utf-8"), variant, regime,
                             args.cap, family)
                if regime == "finetune":
                    marker = 'transfer_encoder_checkpoint = "'
                    start = text.index(marker) + len(marker)
                    checkpoint = TASK_DIR / text[start:text.index('"', start)]
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"{target}: missing source {checkpoint}")
                if not args.dry_run:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / f"{target}.toml").write_text(text, encoding="utf-8")
                written.append(target)

    verb = "Would write" if args.dry_run else "Wrote"
    print(f"{verb} {len(written)} config(s) to {output_dir.relative_to(TASK_DIR)}")
    for name in written:
        print(f"  {name}.toml")


if __name__ == "__main__":
    main()
