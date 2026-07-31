#!/usr/bin/env python3
"""Resume one training run by giving its TOML config name."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 guard
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Python < 3.11 needs the 'tomli' package to read TOML configs. "
            "Install it with: pip install tomli"
        ) from exc

from prepare_resume_runs import collect_rows, load_config_index, relative_to_task, resolve_path


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def config_keys(config_path: Path, config: dict[str, Any]) -> set[str]:
    keys = {config_path.stem}
    run = config.get("run", {})
    args = config.get("args", {})
    for value in (run.get("name"), args.get("exp_tag")):
        if value:
            keys.add(str(value))
    return keys


def resolve_config(config_name: str, task_dir: Path) -> Path:
    candidate = resolve_path(config_name, task_dir)
    if candidate.is_file():
        return candidate.resolve()

    if not config_name.endswith(".toml"):
        candidate = resolve_path(config_name + ".toml", task_dir)
        if candidate.is_file():
            return candidate.resolve()

    config_index = load_config_index(task_dir / "configs")
    matches = config_index.get(config_name, [])
    if not matches:
        raise SystemExit(
            f"Could not find config '{config_name}'. Pass a config stem like "
            "'hvg_00_baseline_s0' or a path like "
            "'configs/heuristic_vs_gate_s0_s1_s2/hvg_00_baseline_s0.toml'."
        )

    exact_stem = [path for path in matches if path.stem == config_name]
    matches = exact_stem or matches
    if len(matches) > 1:
        formatted = "\n".join(f"  - {relative_to_task(path, task_dir)}" for path in matches)
        raise SystemExit(f"Config name '{config_name}' is ambiguous:\n{formatted}")
    return matches[0].resolve()


def checkpoint_parent_is_default(row: dict[str, Any], task_dir: Path) -> bool:
    checkpoint_path = Path(row["checkpoint_path"]).expanduser().resolve()
    return checkpoint_path.parent == (task_dir / "checkpoint").resolve()


def resume_run_name_for_command(row: dict[str, Any], task_dir: Path) -> str:
    if checkpoint_parent_is_default(row, task_dir):
        return str(row["checkpoint_name"])
    return str(Path(row["checkpoint_path"]).expanduser().resolve())


def checkpoint_score(row: dict[str, Any]) -> tuple[int, int, float]:
    name = str(row.get("checkpoint_name", ""))
    global_step = int(row.get("global_step") or 0)
    prefix_rank = 0 if name.startswith("best_test_") else 1
    if name.startswith("final_"):
        prefix_rank = 2
    try:
        mtime = Path(row["checkpoint_path"]).stat().st_mtime
    except OSError:
        mtime = 0.0
    return global_step, prefix_rank, mtime


def matching_checkpoints(
    rows: list[dict[str, Any]],
    wanted_keys: set[str],
    config_rel_path: str,
) -> list[dict[str, Any]]:
    matches = []
    for row in rows:
        if row.get("error"):
            continue
        row_keys = {
            str(row.get("exp_tag", "")),
            str(row.get("config_path", "")),
            Path(str(row.get("config_path", ""))).stem,
        }
        if wanted_keys & row_keys or row.get("config_path") == config_rel_path:
            matches.append(row)
    return matches


def build_command(
    row: dict[str, Any],
    config_rel_path: str,
    task_dir: Path,
    launcher: str,
    target_timesteps: int,
    time_limit: float,
    wandb_run_name: str,
    wandb_run_id: str,
) -> list[str]:
    command = [
        "sbatch",
        launcher,
        config_rel_path,
        "--resume-run-name",
        resume_run_name_for_command(row, task_dir),
        "--resume-total-timesteps",
        str(target_timesteps),
    ]
    if time_limit > 0:
        command.extend(["--resume-time-limit", str(time_limit)])
    if wandb_run_name:
        command.extend(["--resume-wandb-run-name", wandb_run_name])
    if wandb_run_id:
        command.extend(["--resume-wandb-run-id", wandb_run_id])
    return command


def main() -> int:
    default_task_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        help="Config stem/name or path, e.g. hvg_00_baseline_s0.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoint",
        help="Directory containing checkpoint .tar files. Default: Topology_Task/checkpoint.",
    )
    parser.add_argument(
        "--target-timesteps",
        type=int,
        default=16_000_000,
        help="Total timesteps to continue to.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1300.0,
        help="Resume job time limit in minutes. Use 0 to keep the checkpoint value.",
    )
    parser.add_argument(
        "--launcher",
        default="job_jed.sh",
        help="Cluster launcher to use.",
    )
    parser.add_argument(
        "--task-dir",
        default=str(default_task_dir),
        help="Path to Topology_Task.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default="",
        help="Optional explicit WandB run id/name if the checkpoint file was renamed.",
    )
    parser.add_argument(
        "--wandb-run-id",
        default="",
        help="Optional immutable W&B run ID to append resumed metrics to.",
    )
    parser.add_argument(
        "--include-complete",
        action="store_true",
        help="Allow submitting even if the checkpoint is already at target timesteps.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the sbatch command. By default the command is only printed.",
    )
    ns = parser.parse_args()

    task_dir = Path(ns.task_dir).expanduser().resolve()
    repo_dir = task_dir.parent
    config_path = resolve_config(ns.config, task_dir)
    config = load_toml(config_path)
    wanted_keys = config_keys(config_path, config)
    config_rel_path = relative_to_task(config_path, task_dir)

    config_index = load_config_index(task_dir / "configs")
    checkpoint_target = resolve_path(ns.checkpoint_dir, task_dir)
    rows = collect_rows(checkpoint_target, "*.tar", config_index, task_dir)
    matches = matching_checkpoints(rows, wanted_keys, config_rel_path)

    if not matches:
        scanned = len([row for row in rows if not row.get("error")])
        raise SystemExit(
            f"No checkpoint in {checkpoint_target} matched config "
            f"'{relative_to_task(config_path, task_dir)}'. Scanned {scanned} readable "
            "checkpoint(s)."
        )

    eligible = matches
    if not ns.include_complete:
        eligible = [
            row
            for row in matches
            if int(row.get("global_step") or 0) < ns.target_timesteps
        ]
    if not eligible:
        best = max(matches, key=checkpoint_score)
        raise SystemExit(
            "Matching checkpoint is already at or beyond the requested target: "
            f"{best['checkpoint_name']} at {int(best.get('global_step') or 0):,} "
            f"steps. Use --include-complete to submit anyway."
        )

    selected = max(eligible, key=checkpoint_score)
    command = build_command(
        selected,
        config_rel_path,
        task_dir,
        ns.launcher,
        ns.target_timesteps,
        ns.time_limit,
        ns.wandb_run_name,
        ns.wandb_run_id,
    )
    printable = " ".join(shlex.quote(part) for part in command)

    print(f"Config: {config_rel_path}")
    print(f"Checkpoint: {selected['checkpoint_name']}.tar")
    print(f"Checkpoint step: {int(selected.get('global_step') or 0):,}")
    print(f"Command: {printable}")

    if not ns.submit:
        print("\nDry run only. Add --submit to launch it on the cluster.")
        return 0

    return subprocess.run(command, cwd=repo_dir, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
