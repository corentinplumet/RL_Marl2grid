import math

import torch as th

from common.solver_lookahead import (
    build_joint_action_candidates,
    choose_best_simulated_candidate,
)


def test_joint_candidates_use_factorized_policy_score_and_keep_noop():
    candidates = build_joint_action_candidates(
        {
            "agent_0": th.tensor([3.0, 2.0, 0.0]),
            "agent_1": th.tensor([2.0, 1.5, 0.0]),
        },
        ["agent_0", "agent_1"],
        top_k=3,
        include_noop=True,
    )

    assert [candidate["action_tuple"] for candidate in candidates[:3]] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert sum(candidate["is_noop"] for candidate in candidates) == 1
    assert all(math.isfinite(candidate["policy_score"]) for candidate in candidates)


def test_forced_noop_removes_an_agents_nonidle_candidates():
    candidates = build_joint_action_candidates(
        {
            "agent_0": th.tensor([0.0, 5.0]),
            "agent_1": th.tensor([0.0, 4.0, 3.0]),
        },
        ["agent_0", "agent_1"],
        top_k=3,
        force_noop={"agent_0": True},
        include_noop=False,
    )

    assert all(candidate["actions"]["agent_0"] == 0 for candidate in candidates)
    assert [candidate["actions"]["agent_1"] for candidate in candidates] == [1, 2, 0]


def test_solver_selection_prefers_physics_then_fewer_interventions():
    candidates = [
        {
            "action_tuple": (1, 0),
            "actions": {"agent_0": 1, "agent_1": 0},
            "policy_score": -0.1,
            "policy_rank": 1,
            "is_noop": False,
            "nonidle_agents": 1,
            "outcome": {
                "action_is_valid": True,
                "sim_done": False,
                "rho_after": 1.01,
                "n_overloaded_after": 1,
                "overload_amount_after": 0.01,
            },
        },
        {
            "action_tuple": (0, 2),
            "actions": {"agent_0": 0, "agent_1": 2},
            "policy_score": -2.0,
            "policy_rank": 2,
            "is_noop": False,
            "nonidle_agents": 1,
            "outcome": {
                "action_is_valid": True,
                "sim_done": False,
                "rho_after": 0.99,
                "n_overloaded_after": 0,
                "overload_amount_after": 0.0,
            },
        },
        {
            "action_tuple": (1, 2),
            "actions": {"agent_0": 1, "agent_1": 2},
            "policy_score": -0.05,
            "policy_rank": 3,
            "is_noop": False,
            "nonidle_agents": 2,
            "outcome": {
                "action_is_valid": True,
                "sim_done": False,
                "rho_after": 0.99,
                "n_overloaded_after": 0,
                "overload_amount_after": 0.0,
            },
        },
    ]

    selected = choose_best_simulated_candidate(candidates)
    assert selected is not None
    assert selected["action_tuple"] == (0, 2)


def test_solver_selection_rejects_game_over_and_invalid_candidates():
    candidates = [
        {
            "action_tuple": (1,),
            "policy_score": 0.0,
            "nonidle_agents": 1,
            "outcome": {
                "action_is_valid": True,
                "sim_done": True,
                "rho_after": 0.0,
            },
        },
        {
            "action_tuple": (0,),
            "policy_score": -1.0,
            "nonidle_agents": 0,
            "outcome": {
                "action_is_valid": False,
                "sim_done": False,
                "rho_after": 0.8,
            },
        },
    ]

    assert choose_best_simulated_candidate(candidates) is None
