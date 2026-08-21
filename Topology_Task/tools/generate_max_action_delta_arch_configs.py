#!/usr/bin/env python3
"""Generate paired NL/NLS bus14 MAPPO architecture ablations."""

from __future__ import annotations

import tomllib
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = TASK_DIR / "configs/no_leakage_config/Z_nl_cas_hl_shared"
OUTPUT_ROOT = TASK_DIR / "configs/no_leakage_config/F_nl_cas_hl_max_arch"

FAMILIES = {
    "NL": SOURCE_DIR / "cas_hl_NL_shared_mean_f1_a0h0_s0.toml",
    "NLS": SOURCE_DIR / "cas_hl_NLS_shared_mean_f1_a0h0_s0.toml",
}

VARIANTS = {
    "base": {
        "delta": False,
        "residual": False,
        "jumping_knowledge": "none",
    },
    "delta": {
        "delta": True,
        "residual": False,
        "jumping_knowledge": "none",
    },
    "resjk": {
        "delta": False,
        "residual": True,
        "jumping_knowledge": "concat",
    },
    "delta_resjk": {
        "delta": True,
        "residual": True,
        "jumping_knowledge": "concat",
    },
}


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(
            f"Expected exactly one occurrence of {old!r}; found {text.count(old)}."
        )
    return text.replace(old, new, 1)


def generate_config(family: str, variant: str) -> tuple[str, str]:
    settings = VARIANTS[variant]
    source_path = FAMILIES[family]
    source_name = tomllib.loads(source_path.read_text())["run"]["name"]
    name = f"cas_hl_{family}_shared_max_{variant}_f1_a0h0_s0"
    text = source_path.read_text()

    text = replace_once(
        text,
        "# Shared-candidate-scorer bus14 run.\n"
        "# NL and NLS differ only in their input preprocessing.",
        "# Paired max-readout/action-delta architecture ablation, seed 0.\n"
        f"# Variant: {variant}; NL and NLS differ only in input preprocessing.",
    )
    text = replace_once(text, f'name = "{source_name}"', f'name = "{name}"')
    text = replace_once(
        text,
        f'exp_tag = "{source_name}"',
        f'exp_tag = "{name}"',
    )
    text = replace_once(
        text,
        'candidate_action_pool = "mean"',
        'candidate_action_pool = "max"',
    )
    text = replace_once(
        text,
        "candidate_action_do_nothing_head = false",
        "candidate_action_do_nothing_head = false\n"
        f"candidate_action_delta_encoder = {str(settings['delta']).lower()}\n"
        "candidate_action_delta_dim = 0",
    )
    text = replace_once(
        text,
        "gnn_layers = 2",
        "gnn_layers = 2\n"
        f"gnn_residual = {str(settings['residual']).lower()}\n"
        f'gnn_jumping_knowledge = "{settings["jumping_knowledge"]}"',
    )
    text = replace_once(
        text,
        'gnn_readout_aggr = "mean"',
        'gnn_readout_aggr = "energized_max"',
    )
    return name, text


def validate_config(
    path: Path,
    *,
    family: str,
    variant: str,
) -> None:
    config = tomllib.loads(path.read_text())
    run = config["run"]
    args = config["args"]
    expected = VARIANTS[variant]

    if run["name"] != path.stem or args["exp_tag"] != path.stem:
        raise ValueError(f"Run identity does not match filename: {path}")
    checks = {
        "seed": args["seed"] == 0,
        "candidate pool": args["candidate_action_pool"] == "max",
        "action features": args["candidate_action_use_features"] is True,
        "action-0 head": args["candidate_action_do_nothing_head"] is False,
        "shared graph encoder": args["share_actor_gnn"] is True,
        "shared scorer": args["share_candidate_scorer"] is True,
        "mp2": args["gnn_layers"] == 2,
        "max readout": args["gnn_readout_aggr"] == "energized_max",
        "delta": args["candidate_action_delta_encoder"] is expected["delta"],
        "residual": args["gnn_residual"] is expected["residual"],
        "JK": args["gnn_jumping_knowledge"] == expected["jumping_knowledge"],
        "NL scaling": family != "NL" or args["gnn_physical_scaling"] is False,
        "NLS scaling": family != "NLS" or args["gnn_physical_scaling"] is True,
        "NLS angle": family != "NLS" or args["gnn_angle_representation"] == "edge_diff",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid generated config {path}: {', '.join(failed)}")


def main() -> None:
    generated: list[Path] = []
    for family in FAMILIES:
        output_dir = OUTPUT_ROOT / family
        output_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            name, text = generate_config(family, variant)
            path = output_dir / f"{name}.toml"
            path.write_text(text)
            validate_config(path, family=family, variant=variant)
            generated.append(path)

    print(f"Generated and validated {len(generated)} configs:")
    for path in generated:
        print(path.relative_to(TASK_DIR))


if __name__ == "__main__":
    main()
