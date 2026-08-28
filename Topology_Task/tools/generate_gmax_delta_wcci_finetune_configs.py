#!/usr/bin/env python3
"""Generate conservative NLS Gmax-delta WCCI fine-tuning runs."""

from __future__ import annotations

from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    TASK_DIR
    / "configs"
    / "no_leakage_config"
    / "F_nl_cas_hl_max_arch"
    / "NLS"
)
OUTPUT_DIR = (
    TASK_DIR
    / "configs"
    / "no_leakage_config"
    / "Y_wcci_gmax_delta_finetune"
)
CHECKPOINT_DIR = "checkpoint/no_leak/shared/cas_hl_gmax_delta"
ACTION_SPACE_DIR = (
    "outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata"
)

# The eighth NLS cell, tmean_f1_a0h1, has no completed source checkpoint.
VARIANTS = (
    "mean_f0_a0h0",
    "mean_f0_a0h1",
    "mean_f1_a0h0",
    "mean_f1_a0h1",
    "tmean_f0_a0h0",
    "tmean_f0_a0h1",
    "tmean_f1_a0h0",
)
ACTION_CAPS = (32, 64)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one {old!r}, found {count}.")
    return text.replace(old, new, 1)


def source_name(variant: str) -> str:
    return f"cas_hl_NLS_shared_gmax_delta_{variant}_s0"


def target_name(action_cap: int, variant: str) -> str:
    return f"ftgd{action_cap}_NLS_{variant}_s0"


def checkpoint_path(variant: str) -> str:
    return f"{CHECKPOINT_DIR}/best_test_{source_name(variant)}.tar"


def action_space_path(action_cap: int) -> str:
    return (
        f"{ACTION_SPACE_DIR}/"
        f"reduced_action_space_wcci_full2048a_90_v3_mk{action_cap}.json"
    )


def generate_config(action_cap: int, variant: str) -> str:
    source = source_name(variant)
    source_path = SOURCE_DIR / f"{source}.toml"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source config: {source_path}")

    target = target_name(action_cap, variant)
    text = source_path.read_text(encoding="utf-8")
    text = replace_once(text, f'name = "{source}"', f'name = "{target}"')
    text = replace_once(
        text,
        'run_dir = "outputs/izar-{config_name}-{job_id}"',
        'run_dir = "outputs/wcci-{config_name}-{job_id}"',
    )
    text = replace_once(text, f'exp_tag = "{source}"', f'exp_tag = "{target}"')
    text = text.replace(
        "# Paired bus14 screening protocol",
        f"# Conservative WCCI mk{action_cap} Gmax-delta fine-tuning",
    )
    text = text.replace(
        "# bus14 environment and fixed MLP critic",
        "# WCCI without maintenance and a fresh target-specific MLP critic",
    )
    text = text.replace(
        "# 15M-step bus14 reference budget",
        f"# 5M-step conservative WCCI mk{action_cap} fine-tuning budget",
    )
    text = replace_once(text, 'env_id = "bus14"', 'env_id = "bus36_wcci_nomaint"')
    text = replace_once(
        text,
        'action_type = "topology"',
        'action_type = "topology"\n'
        f'reduced_action_space = "{action_space_path(action_cap)}"',
    )

    text = replace_once(text, "total_timesteps = 15000000", "total_timesteps = 5000000")
    text = replace_once(text, "actor_lr = 0.0001", "actor_lr = 0.00003")
    text = replace_once(
        text,
        "anneal_lr = true",
        "anneal_lr = true\n"
        "# Retain 10% of both learning rates at the end of fine-tuning.\n"
        "lr_final_frac = 0.1",
    )
    text = replace_once(text, "update_epochs = 10", "update_epochs = 5")
    text = replace_once(text, "target_kl = 0.02", "target_kl = 0.01")
    text = replace_once(text, "clip_coef = 0.2", "clip_coef = 0.1")
    text = replace_once(text, "entropy_coef = 0.01", "entropy_coef = 0.001")
    text = replace_once(
        text,
        "entropy_coef_final = 0.01",
        "entropy_coef_final = 0.0001",
    )

    transfer_block = (
        "# Transfer the bus14 GNN, delta encoder and shared candidate scorer.\n"
        "# The WCCI critic, optimizer and normalization statistics start fresh.\n"
        f'transfer_encoder_checkpoint = "{checkpoint_path(variant)}"\n'
        "transfer_freeze_encoder = false\n"
        "transfer_action_head = true\n"
        'transfer_action_head_source_agent = "agent_0"\n'
        "transfer_freeze_action_head = false\n\n"
    )
    text = replace_once(
        text,
        "# No intervention mechanism is mixed into the representation screen",
        transfer_block
        + "# No intervention mechanism is mixed into the fine-tuning campaign",
    )

    header = (
        f"# Conservative WCCI mk{action_cap} Gmax-delta fine-tuning, seed 0.\n"
        f"# architecture=NLS_{variant}\n"
        f"# Source checkpoint: {checkpoint_path(variant)}\n\n"
    )
    first_comment_end = text.find("\n\n")
    if first_comment_end < 0:
        raise ValueError(f"Source config lacks a header paragraph: {source_path}")
    return header + text[first_comment_end + 2 :]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for action_cap in ACTION_CAPS:
        for variant in VARIANTS:
            name = target_name(action_cap, variant)
            (OUTPUT_DIR / f"{name}.toml").write_text(
                generate_config(action_cap, variant),
                encoding="utf-8",
            )
            generated.append(name)
    print(f"Generated {len(generated)} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
