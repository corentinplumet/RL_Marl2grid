#!/usr/bin/env python3
"""Generate the scaled/unscaled shared-candidate WCCI transfer campaign."""

from __future__ import annotations

from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    TASK_DIR / "configs" / "no_leakage_config" / "Z_nl_cas_hl_shared"
)
OUTPUT_DIR = (
    TASK_DIR
    / "configs"
    / "no_leakage_config"
    / "W_wcci_cas_hl_shared_transfer"
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
REGIMES = ("scratch", "finetune")
PREPROCESSING = {
    "NL": {
        "source_prefix": "cas_hl_NL_shared",
        "checkpoint_dir": "checkpoint/no_leak/shared/NL_cas_hl_shared",
    },
    # Use the Izar source family because these are the checkpoints used by the
    # existing NLS zero-shot WCCI evaluation.
    "NLS": {
        "source_prefix": "cas_hl_NLS_izar_shared",
        "checkpoint_dir": (
            "checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar"
        ),
    },
}
REDUCED_ACTION_SPACE = (
    "outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/"
    "reduced_action_space_wcci_full2048a_90_v3_mk256.json"
)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one {old!r}, found {count}.")
    return text.replace(old, new, 1)


def source_name(preprocessing: str, variant: str) -> str:
    return f"{PREPROCESSING[preprocessing]['source_prefix']}_{variant}_s0"


def target_name(preprocessing: str, variant: str, regime: str) -> str:
    return f"trcas_shared_{preprocessing}_{variant}_{regime}_s0"


def checkpoint_path(preprocessing: str, variant: str) -> str:
    return (
        f"{PREPROCESSING[preprocessing]['checkpoint_dir']}/"
        f"best_test_{source_name(preprocessing, variant)}.tar"
    )


def generate_config(preprocessing: str, variant: str, regime: str) -> str:
    source = source_name(preprocessing, variant)
    source_path = SOURCE_DIR / f"{source}.toml"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source config: {source_path}")

    target = target_name(preprocessing, variant, regime)
    text = source_path.read_text(encoding="utf-8")
    text = replace_once(text, f'name = "{source}"', f'name = "{target}"')
    text = replace_once(
        text,
        f'exp_tag = "{source}"',
        f'exp_tag = "{target}"',
    )
    text = text.replace(
        "# Paired bus14 screening protocol",
        "# Paired WCCI scratch/fine-tuning protocol",
    )
    text = text.replace(
        "# bus14 environment and fixed MLP critic",
        "# WCCI without maintenance and a target-specific MLP critic",
    )
    text = text.replace(
        "# 15M-step bus14 reference budget",
        "# 15M-step WCCI target-training budget",
    )
    text = replace_once(text, 'env_id = "bus14"', 'env_id = "bus36_wcci_nomaint"')
    text = replace_once(
        text,
        'action_type = "topology"',
        'action_type = "topology"\n'
        f'reduced_action_space = "{REDUCED_ACTION_SPACE}"',
    )

    # The unscaled source relied on the parser default. Make it explicit here
    # so the NL/NLS target preprocessing distinction is visible in every TOML.
    if preprocessing == "NL" and "gnn_angle_representation" not in text:
        text = replace_once(
            text,
            "# Input preprocessing factors\n",
            "# Input preprocessing factors\n"
            'gnn_angle_representation = "node"\n',
        )

    if regime == "finetune":
        transfer = (
            "# Fine-tune the exact candidate actor used by the zero-shot arm.\n"
            "# The target critic and WCCI observation statistics still start fresh.\n"
            f'transfer_encoder_checkpoint = "{checkpoint_path(preprocessing, variant)}"\n'
            "transfer_freeze_encoder = false\n"
            "transfer_action_head = true\n"
            'transfer_action_head_source_agent = "agent_0"\n'
            "transfer_freeze_action_head = false\n\n"
        )
        text = replace_once(
            text,
            "# No intervention mechanism is mixed into the representation screen",
            transfer
            + "# No intervention mechanism is mixed into the representation screen",
        )

    header = (
        "# WCCI no-maintenance shared-candidate transfer comparison.\n"
        f"# preprocessing={preprocessing}; candidate={variant}; "
        f"regime={regime}; seed=0\n"
        f"# Source run: {source}\n"
    )
    if regime == "scratch":
        header += "# Random actor initialization; no source checkpoint is loaded.\n\n"
    else:
        header += f"# Source checkpoint: {checkpoint_path(preprocessing, variant)}\n\n"

    first_comment_end = text.find("\n\n")
    if first_comment_end < 0:
        raise ValueError(f"Source config lacks a header paragraph: {source_path}")
    return header + text[first_comment_end + 2 :]


def launch_script() -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run from anywhere. Optional arguments are ANDed filename substrings, e.g.:
#   DRY_RUN=true bash launch_izar.sh NLS finetune
#   bash launch_izar.sh NL mean_f0_a0h0

script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
repo_dir="$(cd "$script_dir/../../../.." && pwd)"
task_dir="$repo_dir/Topology_Task"
config_dir="$task_dir/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer"
task_config_dir="configs/no_leakage_config/W_wcci_cas_hl_shared_transfer"
reduced_action_space="$task_dir/{REDUCED_ACTION_SPACE}"
dry_run="${{DRY_RUN:-false}}"

cd "$repo_dir"

selected=()
for config_path in "$config_dir"/trcas_shared_*.toml; do
  config_name="$(basename "$config_path" .toml)"
  matches=true
  for filter in "$@"; do
    if [[ "$config_name" != *"$filter"* ]]; then
      matches=false
      break
    fi
  done
  if [[ "$matches" == true ]]; then
    selected+=("$config_path")
  fi
done

if [[ ${{#selected[@]}} -eq 0 ]]; then
  echo "No configs matched filters: $*" >&2
  exit 1
fi

if [[ "$dry_run" != true && ! -f "$reduced_action_space" ]]; then
  echo "Missing WCCI reduced action space: $reduced_action_space" >&2
  exit 1
fi

echo "Selected ${{#selected[@]}} config(s). DRY_RUN=$dry_run"
for config_path in "${{selected[@]}}"; do
  config_name="$(basename "$config_path" .toml)"
  if [[ "$config_name" == *"_finetune_"* ]]; then
    checkpoint="$(sed -n 's/^transfer_encoder_checkpoint = "\\(.*\\)"/\\1/p' "$config_path")"
    if [[ -z "$checkpoint" ]]; then
      echo "Missing transfer checkpoint field: $config_path" >&2
      exit 1
    fi
    if [[ "$dry_run" != true && ! -f "$task_dir/$checkpoint" ]]; then
      echo "Missing source checkpoint: $task_dir/$checkpoint" >&2
      exit 1
    fi
  fi

  task_config="$task_config_dir/$config_name.toml"
  if [[ "$dry_run" == true ]]; then
    echo "sbatch --job-name=$config_name job_izar.sh $task_config"
  else
    sbatch --job-name="$config_name" job_izar.sh "$task_config"
  fi
done
'''


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for preprocessing in PREPROCESSING:
        for variant in VARIANTS:
            for regime in REGIMES:
                name = target_name(preprocessing, variant, regime)
                (OUTPUT_DIR / f"{name}.toml").write_text(
                    generate_config(preprocessing, variant, regime),
                    encoding="utf-8",
                )
                names.append(name)

    launcher = OUTPUT_DIR / "launch_izar.sh"
    launcher.write_text(launch_script(), encoding="utf-8")
    launcher.chmod(0o755)
    print(f"Generated {len(names)} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
