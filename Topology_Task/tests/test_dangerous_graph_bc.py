from pathlib import Path

import numpy as np

from teacher_student.dangerous_graph_bc import (
    DangerousGraphBCWriter,
    best_action_labels,
    choose_concerned_agents,
    copy_actor_observation,
    load_agent_batch,
)


def _outcome(agent, action, rho, *, done=False, valid=True):
    return {
        "agent_id": agent,
        "action_id": action,
        "rho_after": rho,
        "sim_done": done,
        "action_is_valid": valid,
        "action_is_legal": valid,
        "action_is_ambiguous": False,
        "simulation_exception": False,
        "sim_reward": 0.0,
    }


def _obs(offset=0.0):
    return {
        "flat": np.asarray([offset, offset + 1], dtype=np.float32),
        "graph": {
            "node_features": np.full((3, 2), offset, dtype=np.float32),
            "edge_features": np.full((4, 2), offset, dtype=np.float32),
            "node_mask": np.ones(3, dtype=np.float32),
            "energized_node_mask": np.ones(3, dtype=np.float32),
            "edge_mask": np.ones(4, dtype=np.float32),
            "edge_type": np.arange(4, dtype=np.int64),
            "controlled_node_mask": np.ones(3, dtype=np.float32),
        },
    }


def test_choose_concerned_agents_and_fallback():
    concerned, fallback = choose_concerned_agents(
        {"agent_0": 0.89, "agent_1": 0.94, "agent_2": 0.95}, 0.90
    )
    assert concerned == ["agent_1", "agent_2"]
    assert fallback is False

    concerned, fallback = choose_concerned_agents(
        {"agent_0": 0.70, "agent_1": 0.80}, 0.90
    )
    assert concerned == ["agent_1"]
    assert fallback is True


def test_best_action_requires_improvement_and_avoids_terminal_actions():
    labels = best_action_labels(
        agent_ids=["agent_0", "agent_1"],
        concerned_agents=["agent_0"],
        outcomes=[
            _outcome("agent_0", 1, 0.70, done=True),
            _outcome("agent_0", 2, 0.85),
            _outcome("agent_0", 3, 0.80),
            _outcome("agent_1", 1, 0.20),
        ],
        do_nothing_outcome={"rho_after": 0.90, "sim_done": False},
        min_improvement=0.05,
    )
    assert labels["agent_0"]["target_action"] == 3
    assert labels["agent_0"]["target_is_nonidle"] is True
    assert labels["agent_1"]["target_action"] == 0

    no_gain = best_action_labels(
        agent_ids=["agent_0"],
        concerned_agents=["agent_0"],
        outcomes=[_outcome("agent_0", 2, 0.88)],
        do_nothing_outcome={"rho_after": 0.90, "sim_done": False},
        min_improvement=0.05,
    )
    assert no_gain["agent_0"]["target_action"] == 0


def test_nonterminal_action_rescues_terminal_do_nothing():
    labels = best_action_labels(
        agent_ids=["agent_0"],
        concerned_agents=["agent_0"],
        outcomes=[_outcome("agent_0", 4, 1.05)],
        do_nothing_outcome={"rho_after": 0.0, "sim_done": True},
        min_improvement=0.05,
    )
    assert labels["agent_0"]["target_action"] == 4
    assert labels["agent_0"]["target_improves_do_nothing"] is True


def test_writer_round_trip(tmp_path: Path):
    writer = DangerousGraphBCWriter(
        tmp_path, ["agent_0", "agent_1"], shard_size=1, compress=True
    )
    values = {}
    for index, agent in enumerate(["agent_0", "agent_1"]):
        values[agent] = {
            "obs": copy_actor_observation(_obs(float(index))),
            "policy_action": 0,
            "policy_logits": np.arange(4, dtype=np.float32),
            "target_action": index,
            "concerned": bool(index),
            "target_is_nonidle": bool(index),
            "target_improves_do_nothing": bool(index),
            "local_max_rho": 0.9 + index,
            "best_rho_after": 0.8,
            "rho_after_do_nothing": 0.95,
            "do_nothing_sim_done": False,
            "best_action_sim_done": False,
            "improvement_vs_do_nothing": 0.15,
        }
    path = writer.append(
        row={
            "state_id": 0,
            "episode_id": 0,
            "episode_step": 2,
            "dataset_step": 2,
            "global_max_rho": 0.95,
            "chronic_name": "c0",
            "chronic_fingerprint": "fp0",
            "chronic_datetime": "d0",
            "concerned_fallback": False,
        },
        agent_values=values,
    )
    assert path is not None
    obs, target, logits = load_agent_batch(path, "agent_1")
    assert target.tolist() == [1]
    assert logits.shape == (1, 4)
    assert obs["graph"]["node_features"].shape == (1, 3, 2)
