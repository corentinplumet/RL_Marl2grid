#!/usr/bin/env python3
"""Split checkpoints by observation stats and rename usable checkpoints.

This script performs the same cleanup used for local checkpoint downloads:

1. Move checkpoint ``.tar`` files into:
   - ``with_obs_stats`` when ``training_state.obs_stats`` is present and complete.
   - ``without_obs_stats`` otherwise, including unreadable/corrupted files.
2. Rename files inside ``with_obs_stats`` using the experiment tag stored in the
   checkpoint:
   - regular checkpoint: ``{exp_tag}.tar``
   - best test checkpoint: ``best_test_{exp_tag}.tar``
   - final checkpoint: ``final_{exp_tag}.tar``

Old ``final_final_`` files are preserved as ``final_final_{exp_tag}.tar`` so the
script never overwrites a distinct checkpoint by accident.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script needs PyTorch to read checkpoint metadata. "
        "Run it inside the same environment used for training/evaluation."
    ) from exc


TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = TASK_DIR / "checkpoint"
OBS_STAT_KEYS = ("count", "mean", "var")


@dataclass(frozen=True)
class Classification:
    has_obs_stats: bool
    reason: str


@dataclass(frozen=True)
class RenamePlan:
    src: Path
    dst: Path
    exp_tag: str


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _has_complete_obs_stats(state: Any) -> Classification:
    if not isinstance(state, dict):
        return Classification(False, "checkpoint is not a dict")

    training_state = state.get("training_state")
    if not isinstance(training_state, dict):
        return Classification(False, "missing training_state")

    obs_stats = training_state.get("obs_stats")
    if not isinstance(obs_stats, dict) or not obs_stats:
        return Classification(False, "missing obs_stats")

    complete_agents = []
    incomplete_agents = []
    for agent_id, stats in obs_stats.items():
        if isinstance(stats, dict) and all(stats.get(key) is not None for key in OBS_STAT_KEYS):
            complete_agents.append(str(agent_id))
        else:
            incomplete_agents.append(str(agent_id))

    if complete_agents and not incomplete_agents:
        return Classification(True, f"complete obs_stats for {len(complete_agents)} agents")
    return Classification(
        False,
        "incomplete obs_stats: "
        f"complete={complete_agents}, incomplete={incomplete_agents}",
    )


def _classify_checkpoint(path: Path) -> Classification:
    try:
        state = _load_checkpoint(path)
    except Exception as exc:  # noqa: BLE001 - keep organizing despite corrupt files
        return Classification(False, f"load_error: {type(exc).__name__}: {exc}")
    return _has_complete_obs_stats(state)


def _checkpoint_sources(root: Path, with_dir: Path, without_dir: Path) -> list[Path]:
    sources: list[Path] = []
    seen: set[Path] = set()
    for directory in (root, with_dir, without_dir):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.tar")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            sources.append(path)
    return sources


def _move_checkpoint(path: Path, dst_dir: Path, dry_run: bool) -> str:
    dst = dst_dir / path.name
    if path.resolve() == dst.resolve():
        return "already"
    if dst.exists():
        return "skipped_existing_destination"
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dst))
    return "moved"


def split_checkpoints(root: Path, dry_run: bool) -> dict[str, Any]:
    with_dir = root / "with_obs_stats"
    without_dir = root / "without_obs_stats"
    if not dry_run:
        with_dir.mkdir(parents=True, exist_ok=True)
        without_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {
        "moved_to_with_obs_stats": [],
        "moved_to_without_obs_stats": [],
        "skipped_existing_destination": [],
    }

    for path in _checkpoint_sources(root, with_dir, without_dir):
        counts["scanned"] += 1
        classification = _classify_checkpoint(path)
        reason_counts[classification.reason] += 1
        dst_dir = with_dir if classification.has_obs_stats else without_dir
        result = _move_checkpoint(path, dst_dir, dry_run=dry_run)

        if result == "already":
            key = "already_with_obs_stats" if classification.has_obs_stats else "already_without_obs_stats"
            counts[key] += 1
        elif result == "moved":
            key = "moved_to_with_obs_stats" if classification.has_obs_stats else "moved_to_without_obs_stats"
            counts[key] += 1
            if len(examples[key]) < 8:
                examples[key].append(path.name)
        else:
            counts[result] += 1
            if len(examples[result]) < 8:
                examples[result].append(path.name)

    return {
        "counts": dict(counts),
        "reasons": dict(reason_counts),
        "examples": examples,
    }


def _args_exp_tag(args: Any) -> str:
    if isinstance(args, dict):
        value = args.get("exp_tag", "")
    else:
        value = getattr(args, "exp_tag", "")
    return str(value or "").strip()


def _safe_checkpoint_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _checkpoint_kind(name: str) -> str:
    if name.startswith("best_test_"):
        return "best_test"
    if name.startswith("final_final_"):
        return "final_final"
    if name.startswith("final_"):
        return "final"
    return "regular"


def _target_name(path: Path, exp_tag: str) -> str:
    kind = _checkpoint_kind(path.name)
    if kind == "best_test":
        return f"best_test_{exp_tag}.tar"
    if kind == "final":
        return f"final_{exp_tag}.tar"
    if kind == "final_final":
        return f"final_final_{exp_tag}.tar"
    return f"{exp_tag}.tar"


def _rename_plan(path: Path, with_dir: Path) -> tuple[RenamePlan | None, str]:
    try:
        state = _load_checkpoint(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"load_error: {type(exc).__name__}: {exc}"

    exp_tag = _args_exp_tag(state.get("args") if isinstance(state, dict) else None)
    if not exp_tag:
        return None, "missing args.exp_tag"

    safe_exp_tag = _safe_checkpoint_stem(exp_tag)
    if not safe_exp_tag:
        return None, f"invalid args.exp_tag: {exp_tag!r}"

    return RenamePlan(path, with_dir / _target_name(path, safe_exp_tag), safe_exp_tag), "planned"


def rename_with_obs_stats(root: Path, dry_run: bool) -> dict[str, Any]:
    with_dir = root / "with_obs_stats"
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    plans: list[RenamePlan] = []
    skipped: dict[str, list[str]] = {}

    for path in sorted(with_dir.glob("*.tar")):
        counts["scanned"] += 1
        plan, reason = _rename_plan(path, with_dir)
        if plan is None:
            counts["skipped"] += 1
            reason_counts[reason] += 1
            skipped.setdefault(reason, []).append(path.name)
            continue
        plans.append(plan)

    source_set = {plan.src.resolve() for plan in plans}
    destination_counts = Counter(plan.dst.resolve() for plan in plans)
    ready: list[RenamePlan] = []
    unchanged: list[RenamePlan] = []

    for plan in plans:
        src_resolved = plan.src.resolve()
        dst_resolved = plan.dst.resolve()
        if src_resolved == dst_resolved:
            counts["unchanged"] += 1
            unchanged.append(plan)
            continue
        if destination_counts[dst_resolved] > 1:
            reason = f"target collision: {plan.dst.name}"
            counts["skipped_collision"] += 1
            skipped.setdefault(reason, []).append(plan.src.name)
            continue
        if plan.dst.exists() and dst_resolved not in source_set:
            reason = f"target already exists: {plan.dst.name}"
            counts["skipped_existing_destination"] += 1
            skipped.setdefault(reason, []).append(plan.src.name)
            continue
        ready.append(plan)

    if dry_run:
        counts["would_rename"] = len(ready)
    else:
        temp_plans: list[tuple[Path, Path]] = []
        for index, plan in enumerate(ready):
            tmp = plan.src.with_name(f".organize_tmp_{index}_{plan.src.name}")
            if tmp.exists():
                raise FileExistsError(f"Temporary rename path already exists: {tmp}")
            plan.src.rename(tmp)
            temp_plans.append((tmp, plan.dst))
        for tmp, dst in temp_plans:
            tmp.rename(dst)
        counts["renamed"] = len(temp_plans)

    return {
        "counts": dict(counts),
        "reasons": dict(reason_counts),
        "skipped": {key: values[:12] for key, values in skipped.items()},
        "examples": {
            "planned": [
                {"from": plan.src.name, "to": plan.dst.name}
                for plan in ready[:12]
            ],
            "unchanged": [plan.src.name for plan in unchanged[:12]],
        },
    }


def _print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in sorted(summary.get("counts", {}).items()):
        print(f"{key}: {value}")
    if summary.get("reasons"):
        print("reasons:")
        for reason, count in sorted(summary["reasons"].items(), key=lambda item: (-item[1], item[0]))[:20]:
            print(f"  {count}: {reason}")
    for section in ("examples", "skipped"):
        if summary.get(section):
            print(f"{section}:")
            for key, values in summary[section].items():
                if values:
                    print(f"  {key}: {values}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split checkpoint .tar files into with_obs_stats / without_obs_stats "
            "and rename obs-stat checkpoints by args.exp_tag."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint root directory. Default: {DEFAULT_CHECKPOINT_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without moving or renaming files.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.checkpoint_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {root}")

    print(f"Checkpoint directory: {root}")
    print(f"Mode: {'dry-run' if args.dry_run else 'apply'}")

    split_summary = split_checkpoints(root, dry_run=args.dry_run)
    rename_summary = rename_with_obs_stats(root, dry_run=args.dry_run)
    report = {
        "checkpoint_dir": str(root),
        "dry_run": bool(args.dry_run),
        "split": split_summary,
        "rename": rename_summary,
    }

    _print_summary("Split summary", split_summary)
    _print_summary("Rename summary", rename_summary)

    if args.report_json:
        report_path = args.report_json.expanduser().resolve()
        if not args.dry_run:
            report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
