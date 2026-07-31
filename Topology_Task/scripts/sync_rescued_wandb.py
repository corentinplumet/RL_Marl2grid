#!/usr/bin/env python3
"""Safely append rescued offline W&B files to their original cloud runs."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import wandb


RUN_FILE_RE = re.compile(r"^run-(.+)\.wandb(?:\.synced)?$")


def offline_run_name(path: Path) -> str:
    match = RUN_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported W&B run filename: {path.name}")
    return match.group(1)


def exact_name_filter(names: list[str]) -> dict:
    return {
        "$or": [
            {"display_name": name}
            for name in names
        ]
    }


def chunks(values: list[str], size: int = 20):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def cloud_runs_by_name(
    api: wandb.Api,
    entity: str,
    project: str,
    names: list[str],
) -> dict[str, list]:
    matches = defaultdict(list)
    for batch in chunks(names):
        for run in api.runs(
            f"{entity}/{project}",
            filters=exact_name_filter(batch),
        ):
            matches[str(run.name)].append(run)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rescue_root", type=Path)
    parser.add_argument("--entity", default="corentin-plumet-epfl")
    parser.add_argument("--project", default="Grid2Op")
    parser.add_argument("--api-timeout", type=int, default=60)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the uploads. Without this flag, only print the commands.",
    )
    args = parser.parse_args()

    rescue_root = args.rescue_root.expanduser().resolve()
    run_files = sorted(
        path
        for path in rescue_root.rglob("run-*.wandb*")
        if path.is_file()
    )
    if not run_files:
        raise SystemExit(f"No rescued run-*.wandb files found in {rescue_root}")

    files_by_name = defaultdict(list)
    for path in run_files:
        files_by_name[offline_run_name(path)].append(path)

    api = wandb.Api(timeout=args.api_timeout)
    cloud_matches = cloud_runs_by_name(
        api,
        args.entity,
        args.project,
        sorted(files_by_name),
    )

    failures = 0
    for run_name, paths in sorted(files_by_name.items()):
        matches = cloud_matches.get(run_name, [])
        if len(matches) != 1:
            print(
                f"SKIP {run_name}: expected exactly one cloud run, "
                f"found {len(matches)}."
            )
            failures += 1
            continue

        target_id = str(matches[0].id)
        for path in paths:
            command = [
                "wandb",
                "sync",
                "--append",
                "--include-synced",
                "--no-mark-synced",
                "--entity",
                args.entity,
                "--project",
                args.project,
                "--id",
                target_id,
                str(path),
            ]
            print(
                f"{'RUN' if args.execute else 'DRY-RUN'} "
                f"{run_name} -> {target_id}\n  {' '.join(command)}"
            )
            if args.execute:
                result = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                output = result.stdout or ""
                if output:
                    print(
                        output,
                        end="" if output.endswith("\n") else "\n",
                    )
                semantic_error = any(
                    marker in output.lower()
                    for marker in (
                        "wandb: error",
                        "error while calling w&b api",
                        "previously created and deleted",
                    )
                )
                if result.returncode != 0 or semantic_error:
                    print(f"FAILED {run_name}: rescued file was retained at {path}")
                    failures += 1

    if not args.execute:
        print("\nDry run only. Add --execute after checking the mappings.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
