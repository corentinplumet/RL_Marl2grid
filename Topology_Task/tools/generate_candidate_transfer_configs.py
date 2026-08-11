#!/usr/bin/env python3
"""Generate the paired NL/NLS candidate-actor transfer study configs."""

from __future__ import annotations

from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = TASK_DIR / "configs" / "transfer_candidate_preprocessing"

VARIANTS = (
    "mean_f1_a0h0",
    "mean_f0_a0h1",
    "mean_f0_a0h0",
)
PREPROCESSING = {
    "NL": {
        "source_dir": TASK_DIR / "configs" / "gnn_action_scoring" / "no_leak",
        "checkpoint_dir": "checkpoint/no_leak/cas_hl_NL",
    },
    "NLS": {
        "source_dir": TASK_DIR
        / "configs"
        / "gnn_action_scoring"
        / "no_leak_scaled",
        "checkpoint_dir": "checkpoint/no_leak/NL_cas_hl_scaled",
    },
}
REGIMES = ("frozen", "scratch", "finetune")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one {old!r}, found {count}.")
    return text.replace(old, new, 1)


def generate_config(preprocessing: str, variant: str, regime: str) -> str:
    settings = PREPROCESSING[preprocessing]
    source_name = f"cas_hl_{preprocessing}_{variant}_s0"
    source_path = settings["source_dir"] / f"{source_name}.toml"
    text = source_path.read_text(encoding="utf-8")
    target_name = f"trcas_{preprocessing}_{variant}_{regime}_s0"

    text = replace_once(text, f'name = "{source_name}"', f'name = "{target_name}"')
    text = replace_once(
        text,
        f'exp_tag = "{source_name}"',
        f'exp_tag = "{target_name}"',
    )
    text = text.replace(
        'run_dir = "outputs/jed-{config_name}-{job_id}"',
        'run_dir = "outputs/izar-{config_name}-{job_id}"',
    )
    text = text.replace(
        "# Paired bus14 screening protocol",
        "# Paired WCCI transfer protocol",
    )
    text = text.replace(
        "# bus14 environment and fixed MLP critic",
        "# WCCI without maintenance and a target-specific MLP critic",
    )
    text = text.replace(
        "# 15M-step bus14 reference budget",
        "# Transfer budget (the frozen arm stops after its first evaluation)",
    )
    text = text.replace("time_limit = 10080", "time_limit = 4260")
    text = text.replace(
        'MAX_TIME_LIMIT_MINUTES = "10080"',
        'MAX_TIME_LIMIT_MINUTES = "4260"',
    )
    text = replace_once(text, 'env_id = "bus14"', 'env_id = "bus36_wcci_nomaint"')
    text = replace_once(
        text,
        'action_type = "topology"',
        'action_type = "topology"\n'
        'reduced_action_space = '
        '"outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/'
        'reduced_action_space_wcci_full2048a_90_v3_mk256.json"',
    )
    text = replace_once(
        text,
        "gnn_context_requires_connection = true",
        "gnn_context_requires_connection = true\n"
        "# Preserve the 13-column schema used by the source checkpoints.\n"
        "gnn_include_legacy_connected_feature = true",
    )

    total_timesteps = 82944 if regime == "frozen" else 15000000
    text = replace_once(
        text,
        "total_timesteps = 15000000",
        f"total_timesteps = {total_timesteps}",
    )
    if regime == "frozen":
        text = replace_once(
            text,
            "actor_lr = 0.0001",
            "# A zero actor learning rate keeps the transferred GNN and head exact.\n"
            "actor_lr = 0.0",
        )

    if regime in {"frozen", "finetune"}:
        checkpoint = (
            f'{settings["checkpoint_dir"]}/best_test_{source_name}.tar'
        )
        transfer_block = (
            "# Complete candidate actor transfer. The learned scorer from source\n"
            "# agent_0 is broadcast; every WCCI actor retains its own target metadata.\n"
            f'transfer_encoder_checkpoint = "{checkpoint}"\n'
            "transfer_freeze_encoder = false\n"
            "transfer_action_head = true\n"
            'transfer_action_head_source_agent = "agent_0"\n'
            "transfer_freeze_action_head = false\n\n"
        )
        text = replace_once(
            text,
            "# No intervention mechanism is mixed into the representation screen",
            transfer_block
            + "# No intervention mechanism is mixed into the representation screen",
        )

    header = (
        "# WCCI no-maintenance candidate-actor transfer study.\n"
        f"# preprocessing={preprocessing}; candidate={variant}; regime={regime}; seed=0\n"
        "# NL and NLS are matched source/target preprocessing conditions.\n\n"
    )
    first_comment_end = text.find("\n\n")
    if first_comment_end >= 0:
        text = header + text[first_comment_end + 2 :]
    else:
        text = header + text
    return text


def launch_script(names: list[str]) -> str:
    checkpoints = []
    for preprocessing, settings in PREPROCESSING.items():
        for variant in VARIANTS:
            source_name = f"cas_hl_{preprocessing}_{variant}_s0"
            checkpoints.append(
                f'{settings["checkpoint_dir"]}/best_test_{source_name}.tar'
            )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        '# Launch from Topology_Task: bash configs/transfer_candidate_preprocessing/launch_izar.sh',
        "",
        "required_checkpoints=(",
    ]
    lines.extend(f'  "{checkpoint}"' for checkpoint in checkpoints)
    lines.extend(
        [
            ")",
            "",
            'for checkpoint in "${required_checkpoints[@]}"; do',
            '  if [[ ! -f "$checkpoint" ]]; then',
            '    echo "Missing source checkpoint: $checkpoint" >&2',
            "    exit 1",
            "  fi",
            "done",
            "",
        ]
    )
    for name in names:
        lines.append(
            "sbatch job_izar.sh "
            f"configs/transfer_candidate_preprocessing/{name}.toml"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for preprocessing in PREPROCESSING:
        for variant in VARIANTS:
            for regime in REGIMES:
                name = f"trcas_{preprocessing}_{variant}_{regime}_s0"
                path = OUTPUT_DIR / f"{name}.toml"
                path.write_text(
                    generate_config(preprocessing, variant, regime),
                    encoding="utf-8",
                )
                names.append(name)
    (OUTPUT_DIR / "launch_izar.sh").write_text(
        launch_script(names), encoding="utf-8"
    )
    print(f"Generated {len(names)} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
