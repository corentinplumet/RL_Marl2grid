#!/usr/bin/env python3
"""Inspect MAPPO checkpoints and generate cluster resume commands."""

from __future__ import annotations

import argparse
import csv
import shlex
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

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This helper needs torch so it can read checkpoint .tar files. "
        "Run it inside the same conda environment used for training."
    ) from exc


CHECKPOINT_SAVE_PREFIXES = ("final_", "best_test_")


def checkpoint_stem(path_or_name: str | Path) -> str:
    path = Path(path_or_name)
    return path.stem if path.suffix == ".tar" else path.name


def strip_checkpoint_save_prefixes(path_or_name: str | Path) -> str:
    stem = checkpoint_stem(path_or_name)
    changed = True
    while changed:
        changed = False
        for prefix in CHECKPOINT_SAVE_PREFIXES:
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                changed = True
    return stem


def namespace_get(namespace: Any, key: str, default: Any = None) -> Any:
    if isinstance(namespace, dict):
        return namespace.get(key, default)
    return getattr(namespace, key, default)


def task_dir_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str, task_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (Path.cwd(), task_dir, task_dir.parent):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return Path.cwd() / candidate


def resolve_output_path(path: str, task_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == task_dir.name:
        return task_dir.parent / candidate
    if candidate.parts and candidate.parts[0] in {"checkpoint", "configs", "outputs"}:
        return task_dir / candidate
    return Path.cwd() / candidate


def load_config_index(config_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not config_root.exists():
        return index

    for path in sorted(config_root.rglob("*.toml")):
        try:
            with path.open("rb") as file:
                config = tomllib.load(file)
        except Exception:
            continue

        keys = {path.stem}
        run = config.get("run", {})
        args = config.get("args", {})
        for value in (run.get("name"), args.get("exp_tag")):
            if value:
                keys.add(str(value))
        for key in keys:
            index.setdefault(key, []).append(path)
    return index


def relative_to_task(path: Path, task_dir: Path) -> str:
    try:
        return str(path.relative_to(task_dir))
    except ValueError:
        return str(path)


def find_config_path(
    exp_tag: str,
    checkpoint_name: str,
    config_index: dict[str, list[Path]],
    task_dir: Path,
) -> str:
    candidates = config_index.get(exp_tag) or config_index.get(checkpoint_name) or []
    if not candidates:
        return ""
    exact_stem = [path for path in candidates if path.stem in {exp_tag, checkpoint_name}]
    path = exact_stem[0] if exact_stem else candidates[0]
    return relative_to_task(path, task_dir)


def load_checkpoint_summary(path: Path, config_index: dict[str, list[Path]], task_dir: Path) -> dict[str, Any]:
    record = torch.load(path, map_location="cpu", weights_only=False)
    args = record.get("args")
    exp_tag = str(namespace_get(args, "exp_tag", "") or "")
    checkpoint_name = checkpoint_stem(path)
    base_run_name = strip_checkpoint_save_prefixes(path)
    config_path = find_config_path(exp_tag, checkpoint_name, config_index, task_dir)
    return {
        "checkpoint_path": str(path),
        "checkpoint_name": checkpoint_name,
        "resume_run_name": checkpoint_name,
        "base_run_name": base_run_name,
        "exp_tag": exp_tag,
        "seed": namespace_get(args, "seed", ""),
        "global_step": int(record.get("global_step", 0) or 0),
        "saved_total_timesteps": namespace_get(args, "total_timesteps", ""),
        "last_rollout": record.get("last_rollout", ""),
        "config_path": config_path,
    }


def build_resume_command(
    row: dict[str, Any],
    launcher: str,
    target_timesteps: int,
    time_limit: float,
    allow_device_migration: bool = False,
    reset_environments: bool = False,
) -> str:
    args = [
        "sbatch",
        launcher,
        row["config_path"],
        "--resume-run-name",
        row["resume_run_name"],
        "--resume-total-timesteps",
        str(target_timesteps),
    ]
    if time_limit > 0:
        args.extend(["--resume-time-limit", str(time_limit)])
    if allow_device_migration:
        args.extend(["--resume-allow-device-migration", "true"])
    if reset_environments:
        args.extend(["--resume-reset-environments", "true"])
    return " ".join(shlex.quote(str(part)) for part in args)


def collect_rows(
    checkpoint_target: Path,
    checkpoint_glob: str,
    config_index: dict[str, list[Path]],
    task_dir: Path,
) -> list[dict[str, Any]]:
    if checkpoint_target.is_file():
        checkpoint_files = [checkpoint_target]
    else:
        checkpoint_files = sorted(checkpoint_target.glob(checkpoint_glob))

    rows = []
    for path in checkpoint_files:
        if path.suffix != ".tar":
            continue
        try:
            rows.append(load_checkpoint_summary(path, config_index, task_dir))
        except Exception as exc:
            rows.append(
                {
                    "checkpoint_path": str(path),
                    "checkpoint_name": checkpoint_stem(path),
                    "resume_run_name": checkpoint_stem(path),
                    "base_run_name": strip_checkpoint_save_prefixes(path),
                    "exp_tag": "",
                    "seed": "",
                    "global_step": "",
                    "saved_total_timesteps": "",
                    "last_rollout": "",
                    "config_path": "",
                    "error": str(exc),
                }
            )
    return rows


def print_rows(rows: list[dict[str, Any]]) -> None:
    columns = [
        "checkpoint_name",
        "exp_tag",
        "seed",
        "global_step",
        "saved_total_timesteps",
        "last_rollout",
        "config_path",
        "status",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(column, "")) for column in columns))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_launch_script(commands: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run this from the repository root on the cluster.",
        *commands,
        "",
    ]
    path.write_text("\n".join(body))
    path.chmod(path.stat().st_mode | 0o111)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        default="checkpoint",
        help="Checkpoint directory or a single .tar checkpoint. Default: Topology_Task/checkpoint",
    )
    parser.add_argument(
        "--target-timesteps",
        type=int,
        default=16_000_000,
        help="Total timesteps each resumed run should target.",
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
        help="Cluster launcher to use in generated sbatch commands.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="*.tar",
        help="Glob used when checkpoint_dir is a directory.",
    )
    parser.add_argument(
        "--task-dir",
        default=str(task_dir_from_script()),
        help="Path to Topology_Task.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV path for the checkpoint summary.",
    )
    parser.add_argument(
        "--write-launch-script",
        default="",
        help="Optional path for a generated bash script with sbatch commands.",
    )
    parser.add_argument(
        "--include-complete",
        action="store_true",
        help="Include checkpoints already at or beyond --target-timesteps.",
    )
    parser.add_argument(
        "--allow-device-migration",
        action="store_true",
        help=(
            "Permit checkpoints saved on an unavailable accelerator to resume "
            "on the launcher's device."
        ),
    )
    parser.add_argument(
        "--reset-environments",
        action="store_true",
        help=(
            "Reset environment workers instead of replaying saved trajectories. "
            "This is a faster, non-exact continuation."
        ),
    )
    ns = parser.parse_args()

    task_dir = Path(ns.task_dir).expanduser().resolve()
    checkpoint_target = resolve_path(ns.checkpoint_dir, task_dir)
    config_index = load_config_index(task_dir / "configs")
    rows = collect_rows(checkpoint_target, ns.checkpoint_glob, config_index, task_dir)
    commands: list[str] = []

    for row in rows:
        if row.get("error"):
            row["status"] = "error"
            continue
        if not row.get("config_path"):
            row["status"] = "missing_config"
            continue
        global_step = int(row.get("global_step") or 0)
        if global_step >= ns.target_timesteps and not ns.include_complete:
            row["status"] = "already_at_target"
            continue
        row["status"] = "ready"
        command = build_resume_command(
            row,
            ns.launcher,
            ns.target_timesteps,
            ns.time_limit,
            ns.allow_device_migration,
            ns.reset_environments,
        )
        row["command"] = command
        commands.append(command)

    if not rows:
        print(f"No .tar checkpoints found in {checkpoint_target}", file=sys.stderr)
        return 1

    print_rows(rows)

    if ns.csv:
        write_csv(rows, resolve_output_path(ns.csv, task_dir))
    if ns.write_launch_script:
        write_launch_script(
            commands, resolve_output_path(ns.write_launch_script, task_dir)
        )
        print(f"\nWrote {len(commands)} commands to {ns.write_launch_script}")

    if commands and not ns.write_launch_script:
        print("\nResume commands:")
        for command in commands:
            print(command)

    return 0 if commands or not ns.write_launch_script else 1


if __name__ == "__main__":
    raise SystemExit(main())
