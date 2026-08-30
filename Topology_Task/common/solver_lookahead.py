"""Utilities for policy-guided one-step solver look-ahead evaluation."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th


def _finite_log_probs(scores: th.Tensor) -> th.Tensor:
    values = scores.detach().to(dtype=th.float32, device="cpu").reshape(-1)
    if values.numel() == 0:
        raise ValueError("Action-score vectors must not be empty.")
    values = th.nan_to_num(values, nan=-th.inf, posinf=1e6, neginf=-th.inf)
    if not bool(th.isfinite(values).any()):
        values = th.full_like(values, -th.inf)
        values[0] = 0.0
    return th.log_softmax(values, dim=0)


def build_joint_action_candidates(
    scores_by_agent: Dict[str, th.Tensor],
    agent_ids: List[str],
    *,
    top_k: int,
    force_noop: Optional[Dict[str, bool]] = None,
    include_noop: bool = True,
) -> List[Dict[str, Any]]:
    """Return the highest-scoring factorized joint actions.

    Each agent contributes at most ``top_k`` local actions.  The Cartesian
    product is then ranked by the sum of per-agent log probabilities, which is
    the joint score of the factorized multi-agent policy.
    """
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    force_noop = force_noop or {}

    local_choices: List[List[tuple[int, float]]] = []
    noop_score = 0.0
    for agent_id in agent_ids:
        if agent_id not in scores_by_agent:
            raise KeyError(f"Missing action scores for {agent_id}.")
        log_probs = _finite_log_probs(scores_by_agent[agent_id])
        noop_score += float(log_probs[0])
        if force_noop.get(agent_id, False):
            local_choices.append([(0, float(log_probs[0]))])
            continue
        count = min(top_k, int(log_probs.numel()))
        values, indices = th.topk(log_probs, k=count, largest=True, sorted=True)
        local_choices.append(
            [(int(index), float(value)) for value, index in zip(values, indices)]
        )

    ranked: List[Dict[str, Any]] = []
    for combination in product(*local_choices):
        action_ids = tuple(action_id for action_id, _ in combination)
        ranked.append(
            {
                "actions": dict(zip(agent_ids, action_ids)),
                "policy_score": float(sum(score for _, score in combination)),
                "action_tuple": action_ids,
            }
        )
    ranked.sort(key=lambda item: (-item["policy_score"], item["action_tuple"]))

    selected = ranked[:top_k]
    if include_noop:
        noop_tuple = tuple(0 for _ in agent_ids)
        if not any(item["action_tuple"] == noop_tuple for item in selected):
            selected.append(
                {
                    "actions": {agent_id: 0 for agent_id in agent_ids},
                    "policy_score": float(noop_score),
                    "action_tuple": noop_tuple,
                }
            )

    for rank, candidate in enumerate(selected, start=1):
        candidate["policy_rank"] = rank if rank <= top_k else None
        candidate["is_noop"] = not any(candidate["action_tuple"])
        candidate["nonidle_agents"] = int(
            sum(action_id != 0 for action_id in candidate["action_tuple"])
        )
    return selected


def choose_best_simulated_candidate(
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Choose the safest valid candidate using a physical lexicographic key."""
    valid: List[Dict[str, Any]] = []
    for candidate in candidates:
        outcome = candidate.get("outcome", {})
        rho_after = float(outcome.get("rho_after", float("nan")))
        if not bool(outcome.get("action_is_valid", False)):
            continue
        if bool(outcome.get("sim_done", False)) or not np.isfinite(rho_after):
            continue
        n_overloaded = int(outcome.get("n_overloaded_after", rho_after > 1.0))
        overload_amount = float(
            outcome.get("overload_amount_after", max(rho_after - 1.0, 0.0))
        )
        candidate["selection_key"] = (
            n_overloaded,
            rho_after,
            overload_amount,
            int(candidate.get("nonidle_agents", 0)),
            -float(candidate.get("policy_score", -np.inf)),
            tuple(candidate.get("action_tuple", ())),
        )
        valid.append(candidate)
    if not valid:
        return None
    return min(valid, key=lambda item: item["selection_key"])
