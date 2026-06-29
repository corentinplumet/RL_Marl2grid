"""Dataset helpers for teacher-student behavior cloning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]


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


def load_metadata(dataset_dir: Path) -> Dict[str, Any]:
    path = dataset_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {task_relative(dataset_dir)}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_shards(dataset_dir: Path, max_shards: Optional[int] = None) -> List[Path]:
    shards = sorted(dataset_dir.glob("shard_*.npz"))
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz files found in {task_relative(dataset_dir)}")
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
