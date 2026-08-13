#!/usr/bin/env python3
"""Report, for every config of a screen, whether its run can be resumed.

``prepare_resume_runs.py`` inventories the checkpoints that exist. After a run
dies without ever writing one -- a full disk or an exceeded quota kills the
first save as readily as the hundredth -- the run it belonged to simply does not
appear, which is the case that matters most when deciding what to relaunch.

This walks the configs instead, so every config lands in exactly one bucket:

  resumable    a checkpoint loads and sits below the config's total_timesteps
  complete     a checkpoint loads and already reached it
  unreadable   a checkpoint exists but will not load (truncated write)
  missing      no checkpoint at all, so the run has to start over

Target timesteps and the job time limit are read from each TOML, so a resumed
run continues to the budget its own config asked for.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 guard
    import tomli as tomllib  # type: ignore[no-redef]

from prepare_resume_runs import (  # noqa: E402
    build_resume_command,
    load_checkpoint_summary,
    load_config_index,
    relative_to_task,
)

STATUS_ORDER = ("resumable", "unreadable", "missing", "complete")


def read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def inspect(
    config_path: Path,
    checkpoint_dir: Path,
    config_index: dict[str, list[Path]],
    task_dir: Path,
) -> dict[str, Any]:
    config = read_config(config_path)
    args = config.get("args", {})
    exp_tag = str(args.get("exp_tag") or config.get("run", {}).get("name") or config_path.stem)
    row: dict[str, Any] = {
        "config_path": relative_to_task(config_path, task_dir),
        "exp_tag": exp_tag,
        "target_timesteps": int(args.get("total_timesteps") or 0),
        "time_limit": float(args.get("time_limit") or 0.0),
        "global_step": 0,
        "detail": "",
    }

    checkpoint = checkpoint_dir / f"{exp_tag}.tar"
    if not checkpoint.is_file():
        row["status"] = "missing"
        row["detail"] = "no checkpoint written"
        return row

    try:
        summary = load_checkpoint_summary(checkpoint, config_index, task_dir)
    except Exception as exc:  # a partial write is exactly what we are looking for
        row["status"] = "unreadable"
        row["detail"] = f"{type(exc).__name__}: {exc}"
        return row

    row["global_step"] = int(summary.get("global_step") or 0)
    # The config on disk is authoritative: it is what a relaunch would use, and
    # a checkpoint may have been written under an earlier budget.
    summary["config_path"] = row["config_path"]
    row["summary"] = summary
    row["status"] = "complete" if row["global_step"] >= row["target_timesteps"] else "resumable"
    return row


def fresh_command(row: dict[str, Any], launcher: str) -> str:
    return " ".join(
        shlex.quote(part)
        for part in ("sbatch", "--job-name", Path(row["config_path"]).stem, launcher, row["config_path"])
    )


def main() -> int:
    task_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_dir", help="Screen directory holding the .toml configs.")
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    parser.add_argument("--launcher", default="job_izar.sh")
    parser.add_argument("--glob", default="*.toml")
    parser.add_argument(
        "--write-launch-script",
        default="",
        help="Write the resume and relaunch commands to a bash script.",
    )
    parser.add_argument(
        "--reset-environments",
        action="store_true",
        help="Continue from fresh env workers instead of replaying saved trajectories.",
    )
    parser.add_argument(
        "--allow-device-migration",
        action="store_true",
        help="Permit a checkpoint saved on an unavailable accelerator to resume here.",
    )
    ns = parser.parse_args()

    config_dir = Path(ns.config_dir).expanduser()
    if not config_dir.is_absolute():
        config_dir = (task_dir / config_dir) if (task_dir / config_dir).exists() else config_dir.resolve()
    if not config_dir.is_dir():
        raise SystemExit(f"Not a directory: {config_dir}")

    checkpoint_dir = Path(ns.checkpoint_dir).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = task_dir / checkpoint_dir

    config_index = load_config_index(task_dir / "configs")
    rows = [
        inspect(path, checkpoint_dir, config_index, task_dir)
        for path in sorted(config_dir.glob(ns.glob))
    ]
    if not rows:
        raise SystemExit(f"No configs matched {ns.glob} in {config_dir}")

    width = max(len(r["exp_tag"]) for r in rows)
    print(f"{'config':<{width}}  {'status':<10}  {'global_step':>12}  {'target':>12}  detail")
    for status in STATUS_ORDER:
        for row in rows:
            if row["status"] != status:
                continue
            step = f"{row['global_step']:,}" if row["global_step"] else "-"
            print(
                f"{row['exp_tag']:<{width}}  {row['status']:<10}  {step:>12}  "
                f"{row['target_timesteps']:>12,}  {row['detail']}"
            )

    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_ORDER}
    print("\n" + ", ".join(f"{n} {s}" for s, n in counts.items() if n))

    commands: list[str] = []
    for row in rows:
        if row["status"] == "resumable":
            commands.append(
                build_resume_command(
                    row["summary"],
                    ns.launcher,
                    row["target_timesteps"],
                    row["time_limit"],
                    ns.allow_device_migration,
                    ns.reset_environments,
                )
            )
        elif row["status"] in ("missing", "unreadable"):
            commands.append(fresh_command(row, ns.launcher))

    if not commands:
        print("Nothing to submit.")
        return 0

    if ns.write_launch_script:
        out = Path(ns.write_launch_script).expanduser()
        if not out.is_absolute():
            out = task_dir / out
        out.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n")
        out.chmod(0o755)
        print(f"\nWrote {len(commands)} commands to {out}")
    else:
        print("\nCommands:")
        for command in commands:
            print(f"  {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
