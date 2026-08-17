#!/usr/bin/env python3
"""Generate the matched seed-0 NLS WCCI mk64 training campaign."""

from __future__ import annotations

from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    TASK_DIR
    / "configs"
    / "no_leakage_config"
    / "W_wcci_cas_hl_shared_transfer"
)
OUTPUT_DIR = (
    TASK_DIR
    / "configs"
    / "no_leakage_config"
    / "W_wcci_cas_hl_shared_mk64_seed0"
)

VARIANTS = (
    "mean_f0_a0h0",
    "mean_f0_a0h1",
    "mean_f1_a0h0",
    "mean_f1_a0h1",
    "tmean_f0_a0h0",
    "tmean_f0_a0h1",
    "tmean_f1_a0h0",
    "tmean_f1_a0h1",
)
REGIMES = {
    "finetune": "ft64c",
    "scratch": "sc64c",
}
MK256_ACTION_SPACE = (
    "outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/"
    "reduced_action_space_wcci_full2048a_90_v3_mk256.json"
)
MK64_ACTION_SPACE = (
    "outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/"
    "reduced_action_space_wcci_full2048a_90_v3_mk64.json"
)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one {old!r}, found {count}.")
    return text.replace(old, new, 1)


def target_name(variant: str, regime: str) -> str:
    return f"{REGIMES[regime]}_NLS_{variant}_s0"


def generate_config(variant: str, regime: str) -> str:
    source_name = f"trcas_shared_NLS_{variant}_{regime}_s0"
    source_path = SOURCE_DIR / f"{source_name}.toml"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source config: {source_path}")

    target = target_name(variant, regime)
    text = source_path.read_text(encoding="utf-8")
    text = replace_once(text, f'name = "{source_name}"', f'name = "{target}"')
    text = replace_once(
        text,
        f'exp_tag = "{source_name}"',
        f'exp_tag = "{target}"',
    )
    text = replace_once(text, MK256_ACTION_SPACE, MK64_ACTION_SPACE)
    text = replace_once(
        text,
        "# Paired WCCI scratch/fine-tuning protocol",
        "# Matched conservative WCCI mk64 protocol",
    )
    text = replace_once(
        text,
        "# 15M-step WCCI target-training budget",
        "# 5M-step conservative WCCI mk64 target-training budget",
    )
    text = replace_once(text, "total_timesteps = 15000000", "total_timesteps = 5000000")
    text = replace_once(text, "actor_lr = 0.0001", "actor_lr = 0.00003")
    text = replace_once(
        text,
        "anneal_lr = true\n",
        "anneal_lr = true\n"
        "# Retain 10% of both learning rates at the end of the 5M-step budget.\n"
        "lr_final_frac = 0.1\n",
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

    source_checkpoint = "random initialization"
    if regime == "finetune":
        marker = 'transfer_encoder_checkpoint = "'
        start = text.index(marker) + len(marker)
        end = text.index('"', start)
        source_checkpoint = text[start:end]

    header = (
        "# Conservative WCCI mk64 shared-candidate campaign, seed 0.\n"
        f"# architecture=NLS_{variant}; initialization={regime}\n"
        "# The scratch/fine-tune pair differs only in transferred actor "
        "initialization.\n"
        f"# Actor source: {source_checkpoint}\n\n"
    )
    first_paragraph_end = text.find("\n\n")
    if first_paragraph_end < 0:
        raise ValueError(f"Source config lacks a header paragraph: {source_path}")
    return header + text[first_paragraph_end + 2 :]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for regime in REGIMES:
        for variant in VARIANTS:
            name = target_name(variant, regime)
            (OUTPUT_DIR / f"{name}.toml").write_text(
                generate_config(variant, regime),
                encoding="utf-8",
            )
            generated.append(name)
    print(f"Generated {len(generated)} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
