#!/usr/bin/env python3
"""Move teacher-student datasets into split shard/metadata layout.

This is intentionally an in-place organizer, not a copier. It moves:

    DATASET/shard_*.npz      -> DATASET/shards/shard_*.npz
    DATASET/metadata.json    -> DATASET/metadata/metadata.json
    DATASET/summary.json     -> DATASET/metadata/summary.json

The behavior-cloning loader remains backward compatible with the old flat
layout, but the split layout makes it cheap to copy only metadata locally.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from teacher_student.dataset import (
    METADATA_DIR_NAME,
    METADATA_EXPORT_DIR_NAME,
    SHARDS_DIR_NAME,
    export_metadata_file,
    metadata_dir,
    metadata_export_dir,
    shards_dir,
    task_relative,
)


DEFAULT_ROOT = TASK_DIR / "outputs" / "teacher_student_datasets"


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()
    return path


def _dataset_dirs(root: Path, datasets: Iterable[str]) -> List[Path]:
    names = [name for name in datasets if str(name).strip()]
    if names:
        dirs = [_resolve_path(root / name) for name in names]
    else:
        dirs = sorted(path for path in root.iterdir() if path.is_dir())
    missing = [path for path in dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset directories: "
            + ", ".join(task_relative(path) for path in missing)
        )
    return dirs


def _move_file(src: Path, dst: Path, *, dry_run: bool, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        if src.resolve() == dst.resolve():
            return False
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {task_relative(dst)}. "
                "Use --overwrite true if you want to replace it."
            )
        if not dry_run:
            dst.unlink()
    print(f"{'would move' if dry_run else 'move'} {task_relative(src)} -> {task_relative(dst)}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return True


def organize_dataset(dataset_dir: Path, *, dry_run: bool, overwrite: bool) -> dict:
    shard_root = shards_dir(dataset_dir)
    meta_root = metadata_dir(dataset_dir)
    if not dry_run:
        shard_root.mkdir(parents=True, exist_ok=True)
        meta_root.mkdir(parents=True, exist_ok=True)

    moved_shards = 0
    for shard in sorted(dataset_dir.glob("shard_*.npz")):
        moved_shards += int(
            _move_file(
                shard,
                shard_root / shard.name,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        )
    moved_metadata = int(
        _move_file(
            dataset_dir / "metadata.json",
            meta_root / "metadata.json",
            dry_run=dry_run,
            overwrite=overwrite,
        )
    )
    moved_summary = int(
        _move_file(
            dataset_dir / "summary.json",
            meta_root / "summary.json",
            dry_run=dry_run,
            overwrite=overwrite,
        )
    )
    metadata_export = None
    summary_export = None
    if not dry_run:
        metadata_export = export_metadata_file(
            dataset_dir,
            meta_root / "metadata.json",
            "metadata",
        )
        summary_export = export_metadata_file(
            dataset_dir,
            meta_root / "summary.json",
            "summary",
        )
        if metadata_export is not None:
            print(f"export {task_relative(metadata_export)}")
        if summary_export is not None:
            print(f"export {task_relative(summary_export)}")
    return {
        "dataset": task_relative(dataset_dir),
        "moved_shards": moved_shards,
        "moved_metadata": moved_metadata,
        "moved_summary": moved_summary,
        "shards_dir": task_relative(shard_root),
        "metadata_dir": task_relative(meta_root),
        "metadata_export": task_relative(metadata_export) if metadata_export else "",
        "summary_export": task_relative(summary_export) if summary_export else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root containing teacher-student dataset directories.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset folder name under --root. Repeat to organize only selected datasets.",
    )
    parser.add_argument("--dry-run", type=str2bool, default=False)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _resolve_path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {task_relative(root)}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {task_relative(root)}")

    print("========== Organize teacher-student datasets ==========")
    print(f"Root: {task_relative(root)}")
    print(f"Layout: {SHARDS_DIR_NAME}/ for shards, {METADATA_DIR_NAME}/ for metadata")
    print(f"Export: {METADATA_EXPORT_DIR_NAME}/ for uniquely named JSON files")
    print(f"Dry run: {args.dry_run}")
    print("=======================================================")

    summaries = []
    for dataset_dir in _dataset_dirs(root, args.dataset):
        summaries.append(
            organize_dataset(
                dataset_dir,
                dry_run=bool(args.dry_run),
                overwrite=bool(args.overwrite),
            )
        )

    print("========== Summary ==========")
    for item in summaries:
        print(
            f"{item['dataset']}: "
            f"shards={item['moved_shards']} "
            f"metadata={item['moved_metadata']} "
            f"summary={item['moved_summary']}"
        )
    print("=============================")


if __name__ == "__main__":
    main()
