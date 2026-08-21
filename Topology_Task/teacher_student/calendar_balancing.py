"""Calendar-aware ordering helpers for WCCI chronic collection."""

from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, TypeVar

import numpy as np


T = TypeVar("T")

MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
MONTH_BY_NAME = {
    name: f"{index:02d}" for index, name in enumerate(MONTH_NAMES, start=1)
}
MONTH_BY_NAME.update(
    {name[:3]: f"{index:02d}" for index, name in enumerate(MONTH_NAMES, start=1)}
)


def calendar_month_key(*values: object) -> str:
    """Return ``01``..``12`` from a timestamp or scenario name."""
    for value in values:
        text = str(value or "").strip().lower()
        if not text or text == "unknown":
            continue
        iso_match = re.search(
            r"(?<!\d)(?:19|20)\d{2}[-_/](0[1-9]|1[0-2])(?:\D|$)", text
        )
        if iso_match:
            return iso_match.group(1)
        tokens = re.findall(r"[a-z]+", Path(text).name)
        for token in tokens:
            if token in MONTH_BY_NAME:
                return MONTH_BY_NAME[token]
    return "unknown"


def calendar_round_robin(
    values: Sequence[T],
    *,
    key: Callable[[T], str],
    seed: int,
) -> Tuple[List[T], Dict[str, int]]:
    """Shuffle within calendar groups, then interleave one chronic per group."""
    grouped: Dict[str, List[T]] = defaultdict(list)
    for value in values:
        grouped[str(key(value))].append(value)
    if not grouped:
        return [], {}

    rng = np.random.default_rng(int(seed))
    queues: Dict[str, deque[T]] = {}
    for group, items in grouped.items():
        permutation = rng.permutation(len(items))
        queues[group] = deque(items[int(index)] for index in permutation)

    groups = sorted(queues)
    if len(groups) > 1:
        groups = [groups[int(index)] for index in rng.permutation(len(groups))]
    ordered: List[T] = []
    while queues:
        for group in list(groups):
            queue = queues.get(group)
            if queue is None:
                continue
            ordered.append(queue.popleft())
            if not queue:
                del queues[group]
        groups = [group for group in groups if group in queues]
    return ordered, dict(sorted(Counter(map(key, values)).items()))


def format_month_counts(counts: Iterable[Tuple[str, int]] | Dict[str, int]) -> str:
    """Format month counters compactly for progress logs."""
    items = counts.items() if isinstance(counts, dict) else counts
    return ",".join(f"{month}:{int(count)}" for month, count in sorted(items))
