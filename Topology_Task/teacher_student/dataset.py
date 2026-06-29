"""Dataset helpers for teacher-student behavior cloning."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
SHARDS_DIR_NAME = "shards"
METADATA_DIR_NAME = "metadata"
METADATA_EXPORT_DIR_NAME = "metadata_exports"


def resolve_dataset_dir(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {path}")
    return path


def task_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(TASK_DIR))
    except ValueError:
        return str(path.resolve())


def metadata_dir(dataset_dir: Path) -> Path:
    return dataset_dir / METADATA_DIR_NAME


def metadata_export_dir(dataset_dir: Path) -> Path:
    return dataset_dir.parent / METADATA_EXPORT_DIR_NAME


def metadata_export_path(dataset_dir: Path, kind: str) -> Path:
    if kind not in {"metadata", "summary"}:
        raise ValueError("kind must be 'metadata' or 'summary'.")
    return metadata_export_dir(dataset_dir) / f"{dataset_dir.name}_{kind}.json"


def export_metadata_file(dataset_dir: Path, source_path: Path, kind: str) -> Optional[Path]:
    if not source_path.exists():
        return None
    export_path = metadata_export_path(dataset_dir, kind)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, export_path)
    return export_path


def shards_dir(dataset_dir: Path) -> Path:
    return dataset_dir / SHARDS_DIR_NAME


def metadata_path(dataset_dir: Path) -> Path:
    new_path = metadata_dir(dataset_dir) / "metadata.json"
    if new_path.exists():
        return new_path
    old_path = dataset_dir / "metadata.json"
    if old_path.exists():
        return old_path
    if metadata_dir(dataset_dir).exists():
        return new_path
    return old_path


def summary_path(dataset_dir: Path) -> Path:
    new_dir = metadata_dir(dataset_dir)
    if new_dir.exists():
        return new_dir / "summary.json"
    return dataset_dir / "summary.json"


def load_metadata(dataset_dir: Path) -> Dict[str, Any]:
    path = metadata_path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing metadata.json in {task_relative(dataset_dir)} "
            f"or {task_relative(metadata_dir(dataset_dir))}"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_shards(dataset_dir: Path, max_shards: Optional[int] = None) -> List[Path]:
    shard_root = shards_dir(dataset_dir)
    shards = sorted(shard_root.glob("shard_*.npz")) if shard_root.exists() else []
    if not shards:
        shards = sorted(dataset_dir.glob("shard_*.npz"))
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.npz files found in {task_relative(shard_root)} "
            f"or {task_relative(dataset_dir)}"
        )
    return shards


def load_agent_arrays(shard: Path, agent_id: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(shard) as data:
        obs = np.asarray(data[f"obs_{agent_id}"], dtype=np.float32)
        target = np.asarray(data[f"teacher_action_{agent_id}"], dtype=np.int64)
    return obs, target


def make_minibatches(
    targets: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    balanced_nonidle_frac: float = 0.0,
) -> Iterator[np.ndarray]:
    n = int(targets.shape[0])
    batch_size = int(batch_size)
    if n == 0:
        return

    if balanced_nonidle_frac <= 0.0:
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            yield indices[start : start + batch_size]
        return

    nonidle = np.flatnonzero(targets != 0)
    idle = np.flatnonzero(targets == 0)
    if nonidle.size == 0 or idle.size == 0:
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            yield indices[start : start + batch_size]
        return

    frac = float(np.clip(balanced_nonidle_frac, 0.0, 1.0))
    n_nonidle = int(round(batch_size * frac))
    n_nonidle = min(max(n_nonidle, 1), batch_size - 1)
    n_idle = batch_size - n_nonidle
    n_batches = int(np.ceil(n / batch_size))

    for _ in range(n_batches):
        batch_nonidle = rng.choice(nonidle, size=n_nonidle, replace=nonidle.size < n_nonidle)
        batch_idle = rng.choice(idle, size=n_idle, replace=idle.size < n_idle)
        batch = np.concatenate([batch_nonidle, batch_idle])
        rng.shuffle(batch)
        yield batch
