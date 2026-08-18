"""Shared helpers for dangerous-state graph behavior cloning."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


GRAPH_KEYS = (
    "node_features",
    "edge_features",
    "node_mask",
    "energized_node_mask",
    "edge_mask",
    "node_type",
    "edge_type",
    "controlled_node_mask",
)


def copy_actor_observation(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy one unbatched graph actor observation into NumPy storage form."""
    if not isinstance(value, Mapping) or "graph" not in value:
        raise TypeError(
            "Dangerous-state graph BC requires nested actor observations with "
            "a 'graph' entry."
        )
    graph = value["graph"]
    if not isinstance(graph, Mapping):
        raise TypeError("The actor observation 'graph' entry must be a mapping.")
    missing = [key for key in ("node_features", "edge_features") if key not in graph]
    if missing:
        raise KeyError("Graph observation is missing: " + ", ".join(missing))

    copied_graph = {
        key: np.asarray(graph[key]).copy()
        for key in GRAPH_KEYS
        if key in graph
    }
    flat = np.asarray(value.get("flat", np.empty((0,), dtype=np.float32))).copy()
    return {"flat": flat, "graph": copied_graph}


def choose_concerned_agents(
    local_max_rhos: Mapping[str, float],
    threshold: float,
) -> Tuple[List[str], bool]:
    """Select agents whose observation domain contains dangerous loading.

    A global dangerous state should normally have at least one local match. If
    domain bookkeeping produces none, the finite maximum-local-rho agent is
    retained as an auditable fallback instead of silently dropping the state.
    """
    concerned = sorted(
        agent
        for agent, value in local_max_rhos.items()
        if np.isfinite(value) and float(value) >= float(threshold)
    )
    if concerned:
        return concerned, False

    finite = [
        (float(value), agent)
        for agent, value in local_max_rhos.items()
        if np.isfinite(value)
    ]
    if not finite:
        return [], False
    _, fallback = max(finite)
    return [fallback], True


def _valid_nonterminal_outcome(outcome: Mapping[str, Any]) -> bool:
    return bool(
        outcome.get("action_is_valid", False)
        and outcome.get("action_is_legal", True)
        and not outcome.get("action_is_ambiguous", False)
        and not outcome.get("simulation_exception", False)
        and not outcome.get("sim_done", False)
        and np.isfinite(float(outcome.get("rho_after", np.nan)))
    )


def best_action_labels(
    *,
    agent_ids: Iterable[str],
    concerned_agents: Iterable[str],
    outcomes: Iterable[Mapping[str, Any]],
    do_nothing_outcome: Mapping[str, Any],
    min_improvement: float,
) -> Dict[str, Dict[str, Any]]:
    """Choose one safe unilateral target action for each concerned agent.

    Non-concerned agents receive action 0. A concerned agent receives a
    non-idle label only if the best valid non-terminal action either rescues a
    terminal do-nothing transition or reduces next-step max rho by at least
    ``min_improvement``. This prevents the supervised stage from imitating a
    merely least-bad but still harmful intervention.
    """
    concerned = set(concerned_agents)
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        grouped[str(outcome["agent_id"])].append(outcome)

    noop_rho = float(do_nothing_outcome.get("rho_after", np.nan))
    noop_done = bool(do_nothing_outcome.get("sim_done", False))
    labels: Dict[str, Dict[str, Any]] = {}
    for agent in agent_ids:
        base = {
            "target_action": 0,
            "target_is_nonidle": False,
            "target_improves_do_nothing": False,
            "best_rho_after": noop_rho,
            "rho_after_do_nothing": noop_rho,
            "do_nothing_sim_done": noop_done,
            "best_action_sim_done": noop_done,
            "improvement_vs_do_nothing": 0.0,
        }
        if agent not in concerned:
            labels[agent] = base
            continue

        candidates = [
            outcome
            for outcome in grouped.get(agent, [])
            if int(outcome.get("action_id", 0)) != 0
            and _valid_nonterminal_outcome(outcome)
        ]
        if not candidates:
            labels[agent] = base
            continue

        best = min(
            candidates,
            key=lambda outcome: (
                float(outcome["rho_after"]),
                -float(outcome.get("sim_reward", 0.0)),
                int(outcome["action_id"]),
            ),
        )
        best_rho = float(best["rho_after"])
        improvement = noop_rho - best_rho if np.isfinite(noop_rho) else float("nan")
        improves = bool(
            noop_done
            or (
                np.isfinite(improvement)
                and improvement >= float(min_improvement)
            )
        )
        if improves:
            base.update(
                target_action=int(best["action_id"]),
                target_is_nonidle=True,
                target_improves_do_nothing=True,
                best_rho_after=best_rho,
                best_action_sim_done=False,
                improvement_vs_do_nothing=improvement,
            )
        labels[agent] = base
    return labels


class DangerousGraphBCWriter:
    """Write one graph observation and one target per agent/dangerous state."""

    def __init__(
        self,
        output_dir: Path,
        agent_ids: List[str],
        shard_size: int,
        compress: bool,
    ) -> None:
        self.output_dir = output_dir
        self.agent_ids = list(agent_ids)
        self.shard_size = int(shard_size)
        self.compress = bool(compress)
        self.rows: List[Dict[str, Any]] = []
        self.agent_rows: Dict[str, List[Dict[str, Any]]] = {
            agent: [] for agent in self.agent_ids
        }
        self.paths: List[Path] = []
        self.total_states = 0

    def append(
        self,
        *,
        row: Dict[str, Any],
        agent_values: Dict[str, Dict[str, Any]],
    ) -> Optional[Path]:
        missing = sorted(set(self.agent_ids).difference(agent_values))
        if missing:
            raise KeyError("Missing agent rows: " + ", ".join(missing))
        self.rows.append(row)
        for agent in self.agent_ids:
            self.agent_rows[agent].append(agent_values[agent])
        self.total_states += 1
        if len(self.rows) >= self.shard_size:
            return self.flush()
        return None

    def flush(self) -> Optional[Path]:
        if not self.rows:
            return None

        arrays: Dict[str, Any] = {
            "state_id": np.asarray([row["state_id"] for row in self.rows], dtype=np.int64),
            "episode_id": np.asarray([row["episode_id"] for row in self.rows], dtype=np.int64),
            "episode_step": np.asarray([row["episode_step"] for row in self.rows], dtype=np.int32),
            "dataset_step": np.asarray([row["dataset_step"] for row in self.rows], dtype=np.int64),
            "global_max_rho": np.asarray(
                [row["global_max_rho"] for row in self.rows], dtype=np.float32
            ),
            "chronic_name": np.asarray([row["chronic_name"] for row in self.rows], dtype=str),
            "chronic_fingerprint": np.asarray(
                [row["chronic_fingerprint"] for row in self.rows], dtype=str
            ),
            "chronic_datetime": np.asarray(
                [row["chronic_datetime"] for row in self.rows], dtype=str
            ),
            "concerned_fallback": np.asarray(
                [row["concerned_fallback"] for row in self.rows], dtype=bool
            ),
        }

        scalar_dtypes = {
            "policy_action": np.int32,
            "target_action": np.int32,
            "concerned": bool,
            "target_is_nonidle": bool,
            "target_improves_do_nothing": bool,
            "local_max_rho": np.float32,
            "best_rho_after": np.float32,
            "rho_after_do_nothing": np.float32,
            "do_nothing_sim_done": bool,
            "best_action_sim_done": bool,
            "improvement_vs_do_nothing": np.float32,
        }
        for agent in self.agent_ids:
            values = self.agent_rows[agent]
            observations = [value["obs"] for value in values]
            arrays[f"flat_{agent}"] = np.stack(
                [obs["flat"] for obs in observations]
            )
            graph_keys = list(observations[0]["graph"])
            for key in graph_keys:
                arrays[f"graph_{agent}_{key}"] = np.stack(
                    [obs["graph"][key] for obs in observations]
                )
            arrays[f"policy_logits_{agent}"] = np.stack(
                [value["policy_logits"] for value in values]
            ).astype(np.float32)
            for key, dtype in scalar_dtypes.items():
                arrays[f"{key}_{agent}"] = np.asarray(
                    [value[key] for value in values], dtype=dtype
                )

        path = self.output_dir / f"shard_{len(self.paths):05d}.npz"
        if self.compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)
        self.paths.append(path)
        self.rows = []
        self.agent_rows = {agent: [] for agent in self.agent_ids}
        return path


def load_agent_batch(
    shard: Path,
    agent_id: str,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Load one agent's graph observations, hard labels, and reference logits."""
    with np.load(shard) as data:
        prefix = f"graph_{agent_id}_"
        graph = {
            key[len(prefix) :]: np.asarray(data[key])
            for key in data.files
            if key.startswith(prefix)
        }
        obs = {
            "flat": np.asarray(data[f"flat_{agent_id}"], dtype=np.float32),
            "graph": graph,
        }
        target = np.asarray(data[f"target_action_{agent_id}"], dtype=np.int64)
        logits = np.asarray(data[f"policy_logits_{agent_id}"], dtype=np.float32)
    return obs, target, logits


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
