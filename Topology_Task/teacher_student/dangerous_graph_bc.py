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

CANDIDATE_OUTCOME_DTYPES = {
    "rho_after": np.float32,
    "utility_vs_noop": np.float32,
    "sim_reward": np.float32,
    "observed_mask": bool,
    "action_valid_mask": bool,
    "legal_mask": bool,
    "ambiguous_mask": bool,
    "exception_mask": bool,
    "terminal_mask": bool,
    "finite_rho_mask": bool,
    "trainable_mask": bool,
    "rank": np.int32,
}


def candidate_outcome_schema() -> Dict[str, Any]:
    """Describe the optional dense candidate-outcome vectors saved per agent."""
    return {
        "version": 1,
        "action_axis": (
            "target reduced-action index; index 0 is the shared do-nothing outcome"
        ),
        "fields": {
            "rho_after": "simulated next-step maximum rho",
            "utility_vs_noop": ("rho_after_do_nothing - rho_after; larger is better"),
            "sim_reward": "reward returned by the one-step simulation",
            "observed_mask": "the candidate was simulated at this state",
            "action_valid_mask": "Grid2Op action_is_valid flag",
            "legal_mask": "Grid2Op action_is_legal flag",
            "ambiguous_mask": "Grid2Op action_is_ambiguous flag",
            "exception_mask": "the candidate simulation raised an exception",
            "terminal_mask": "the candidate simulation ended the episode",
            "finite_rho_mask": "rho_after is finite",
            "trainable_mask": (
                "observed, valid, legal, unambiguous, exception-free, "
                "non-terminal candidate with finite rho"
            ),
            "rank": (
                "zero-based rank by rho_after, then reward, then action index, "
                "among trainable candidates; -1 otherwise"
            ),
        },
        "missing_float_value": "NaN",
        "missing_rank_value": -1,
        "notes": (
            "Only locally concerned agents have non-idle candidates simulated. "
            "Do-nothing is observed for every agent. Use trainable_mask for "
            "continuous or listwise losses and terminal/validity masks for an "
            "explicit unsafe-action penalty."
        ),
    }


def candidate_outcome_vectors(
    *,
    action_size: int,
    outcomes: Iterable[Mapping[str, Any]],
    do_nothing_outcome: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Align simulated outcomes to one reduced action space and rank survivors.

    The raw continuous outcomes are retained instead of reducing the state to a
    single one-hot target. Missing non-idle simulations (normally for agents
    that are not locally concerned) remain masked rather than being fabricated
    as do-nothing labels.
    """
    action_size = int(action_size)
    if action_size <= 0:
        raise ValueError("action_size must be positive.")

    values: Dict[str, np.ndarray] = {
        "rho_after": np.full(action_size, np.nan, dtype=np.float32),
        "utility_vs_noop": np.full(action_size, np.nan, dtype=np.float32),
        "sim_reward": np.full(action_size, np.nan, dtype=np.float32),
        "observed_mask": np.zeros(action_size, dtype=bool),
        "action_valid_mask": np.zeros(action_size, dtype=bool),
        "legal_mask": np.zeros(action_size, dtype=bool),
        "ambiguous_mask": np.zeros(action_size, dtype=bool),
        "exception_mask": np.zeros(action_size, dtype=bool),
        "terminal_mask": np.zeros(action_size, dtype=bool),
        "finite_rho_mask": np.zeros(action_size, dtype=bool),
        "trainable_mask": np.zeros(action_size, dtype=bool),
        "rank": np.full(action_size, -1, dtype=np.int32),
    }

    def record(action_id: int, outcome: Mapping[str, Any], *, is_noop: bool) -> None:
        if not 0 <= action_id < action_size:
            raise ValueError(
                f"Candidate action index {action_id} is outside [0, {action_size})."
            )
        if values["observed_mask"][action_id]:
            raise ValueError(f"Duplicate simulated candidate action {action_id}.")

        rho_after = float(outcome.get("rho_after", np.nan))
        sim_reward = float(outcome.get("sim_reward", np.nan))
        action_valid = bool(outcome.get("action_is_valid", is_noop))
        legal = bool(outcome.get("action_is_legal", True))
        ambiguous = bool(outcome.get("action_is_ambiguous", False))
        exception = bool(outcome.get("simulation_exception", False))
        terminal = bool(outcome.get("sim_done", False))
        finite_rho = bool(np.isfinite(rho_after))
        trainable = bool(
            action_valid
            and legal
            and not ambiguous
            and not exception
            and not terminal
            and finite_rho
        )

        values["rho_after"][action_id] = rho_after
        values["sim_reward"][action_id] = sim_reward
        values["observed_mask"][action_id] = True
        values["action_valid_mask"][action_id] = action_valid
        values["legal_mask"][action_id] = legal
        values["ambiguous_mask"][action_id] = ambiguous
        values["exception_mask"][action_id] = exception
        values["terminal_mask"][action_id] = terminal
        values["finite_rho_mask"][action_id] = finite_rho
        values["trainable_mask"][action_id] = trainable

    record(0, do_nothing_outcome, is_noop=True)
    for outcome in outcomes:
        action_id = int(outcome.get("action_id", -1))
        if action_id == 0:
            raise ValueError("Non-idle candidate outcomes must not repeat action 0.")
        record(action_id, outcome, is_noop=False)

    noop_rho = float(values["rho_after"][0])
    if np.isfinite(noop_rho):
        finite = values["finite_rho_mask"]
        values["utility_vs_noop"][finite] = noop_rho - values["rho_after"][finite]

    def rank_key(action_id: int) -> Tuple[float, float, int]:
        reward = float(values["sim_reward"][action_id])
        reward = reward if np.isfinite(reward) else 0.0
        return (
            float(values["rho_after"][action_id]),
            -reward,
            int(action_id),
        )

    ranked = sorted(np.flatnonzero(values["trainable_mask"]).tolist(), key=rank_key)
    for rank, action_id in enumerate(ranked):
        values["rank"][action_id] = rank
    return values


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
        key: np.asarray(graph[key]).copy() for key in GRAPH_KEYS if key in graph
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
            or (np.isfinite(improvement) and improvement >= float(min_improvement))
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
            "state_id": np.asarray(
                [row["state_id"] for row in self.rows], dtype=np.int64
            ),
            "episode_id": np.asarray(
                [row["episode_id"] for row in self.rows], dtype=np.int64
            ),
            "episode_step": np.asarray(
                [row["episode_step"] for row in self.rows], dtype=np.int32
            ),
            "dataset_step": np.asarray(
                [row["dataset_step"] for row in self.rows], dtype=np.int64
            ),
            "global_max_rho": np.asarray(
                [row["global_max_rho"] for row in self.rows], dtype=np.float32
            ),
            "chronic_name": np.asarray(
                [row["chronic_name"] for row in self.rows], dtype=str
            ),
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
        has_calendar_month = ["calendar_month" in row for row in self.rows]
        if any(has_calendar_month) and not all(has_calendar_month):
            raise ValueError("Calendar month is missing from some dataset rows.")
        if all(has_calendar_month):
            arrays["calendar_month"] = np.asarray(
                [row["calendar_month"] for row in self.rows], dtype=str
            )

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
            arrays[f"flat_{agent}"] = np.stack([obs["flat"] for obs in observations])
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

            has_candidate_outcomes = ["candidate_outcomes" in value for value in values]
            if any(has_candidate_outcomes) and not all(has_candidate_outcomes):
                raise ValueError(
                    f"Candidate outcomes are missing from some rows for {agent}."
                )
            if all(has_candidate_outcomes):
                for key, dtype in CANDIDATE_OUTCOME_DTYPES.items():
                    try:
                        stacked = np.stack(
                            [value["candidate_outcomes"][key] for value in values]
                        )
                    except KeyError as exc:
                        raise KeyError(
                            f"Candidate outcomes for {agent} are missing {key!r}."
                        ) from exc
                    arrays[f"candidate_{agent}_{key}"] = stacked.astype(
                        dtype, copy=False
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


def load_candidate_outcome_batch(
    shard: Path,
    agent_id: str,
) -> Dict[str, np.ndarray]:
    """Load optional dense candidate outcomes for one agent from a shard."""
    prefix = f"candidate_{agent_id}_"
    with np.load(shard) as data:
        available = {
            key[len(prefix) :]: np.asarray(data[key])
            for key in data.files
            if key.startswith(prefix)
        }
    missing = sorted(set(CANDIDATE_OUTCOME_DTYPES).difference(available))
    if missing:
        raise KeyError(
            f"Shard {shard} has no complete candidate-outcome block for "
            f"{agent_id}; missing: {', '.join(missing)}"
        )
    return available


def chronic_row_split(
    shards: List[Path], validation_fraction: float, seed: int
) -> Tuple[Dict[Path, np.ndarray], Dict[Path, np.ndarray], int, int]:
    """Split rows by chronic, stratified by the chronic calendar month."""
    fingerprints_by_shard: Dict[Path, np.ndarray] = {}
    all_fingerprints: set[str] = set()
    month_by_fingerprint: Dict[str, str] = {}
    for shard in shards:
        with np.load(shard) as data:
            fingerprints = np.asarray(data["chronic_fingerprint"], dtype=str)
            datetimes = (
                np.asarray(data["chronic_datetime"], dtype=str)
                if "chronic_datetime" in data.files
                else np.full(len(fingerprints), "unknown", dtype=str)
            )
        fingerprints_by_shard[shard] = fingerprints
        all_fingerprints.update(fingerprints.tolist())
        for fingerprint, datetime_value in zip(fingerprints, datetimes):
            month = (
                str(datetime_value)[:7] if len(str(datetime_value)) >= 7 else "unknown"
            )
            previous = month_by_fingerprint.setdefault(str(fingerprint), month)
            if previous != month:
                raise ValueError(
                    f"Chronic fingerprint {fingerprint!r} spans months "
                    f"{previous!r} and {month!r}."
                )

    if validation_fraction <= 0.0:
        validation_fingerprints: set[str] = set()
    else:
        rng = np.random.default_rng(seed)
        by_month: Dict[str, List[str]] = defaultdict(list)
        for fingerprint in sorted(all_fingerprints):
            by_month[month_by_fingerprint.get(fingerprint, "unknown")].append(
                fingerprint
            )
        validation_fingerprints = set()
        for month in sorted(by_month):
            ordered = np.asarray(by_month[month], dtype=str)
            rng.shuffle(ordered)
            if len(ordered) <= 1:
                n_validation = 0
            else:
                n_validation = int(round(len(ordered) * validation_fraction))
                n_validation = min(max(n_validation, 1), len(ordered) - 1)
            validation_fingerprints.update(ordered[:n_validation].tolist())

    train_rows = {}
    validation_rows = {}
    for shard, fingerprints in fingerprints_by_shard.items():
        is_validation = np.isin(fingerprints, list(validation_fingerprints))
        validation_rows[shard] = np.flatnonzero(is_validation)
        train_rows[shard] = np.flatnonzero(~is_validation)
    if not validation_fingerprints:
        validation_rows = {shard: rows.copy() for shard, rows in train_rows.items()}
    return (
        train_rows,
        validation_rows,
        len(all_fingerprints) - len(validation_fingerprints),
        len(validation_fingerprints),
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
