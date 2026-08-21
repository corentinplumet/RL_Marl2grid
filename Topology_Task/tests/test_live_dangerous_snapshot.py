from pathlib import Path

import numpy as np

from teacher_student.snapshot_live_dangerous_graph_bc_dataset import (
    _inspect_shards,
    _stable_prefix,
)


def _write_shard(path: Path, start: int, months: list[str]) -> None:
    n_rows = len(months)
    arrays = {
        "state_id": np.arange(start, start + n_rows),
        "episode_id": np.arange(start, start + n_rows),
        "chronic_name": np.asarray(["Scenario_april_0"] * n_rows),
        "chronic_datetime": np.asarray(["2012-04-01"] * n_rows),
        "chronic_fingerprint": np.asarray(
            [f"fp_{index}" for index in range(start, start + n_rows)]
        ),
        "calendar_month": np.asarray(months),
        "policy_logits_agent_0": np.zeros((n_rows, 32), dtype=np.float32),
        "concerned_agent_0": np.ones(n_rows, dtype=bool),
        "target_is_nonidle_agent_0": np.ones(n_rows, dtype=bool),
        "candidate_agent_0_rho_after": np.zeros(
            (n_rows, 32), dtype=np.float32
        ),
    }
    np.savez(path, **arrays)


def test_inspects_a_stable_contiguous_snapshot_prefix(tmp_path: Path):
    source = tmp_path / "live"
    shard_root = source / "shards"
    shard_root.mkdir(parents=True)
    first = shard_root / "shard_00000.npz"
    second = shard_root / "shard_00001.npz"
    _write_shard(first, 0, ["04", "04"])
    _write_shard(second, 2, ["08"])

    paths = _stable_prefix(source, min_age_seconds=0)
    summary = _inspect_shards(paths)

    assert paths == [first, second]
    assert summary["total_rows"] == 3
    assert summary["action_sizes"] == {"agent_0": 32}
    assert summary["dangerous_states_by_month"] == {"04": 2, "08": 1}
    assert summary["has_candidate_outcomes"] is True
