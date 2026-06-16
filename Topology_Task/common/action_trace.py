from common.imports import *


def tensor_scalar_to_int(value: Any) -> int:
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().numpy()
    return int(np.asarray(value).reshape(-1)[0])


def tensor_scalar_to_float(value: Any) -> float:
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def build_action_trace_table(
    records: List[Dict[str, Any]],
    agent_ids: List[str],
    decoded_actions: Optional[Dict[str, Dict[int, str]]] = None,
) -> Any:
    """Build a WandB table containing exact per-agent action choices."""
    decoded_actions = decoded_actions or {}
    include_decoded = bool(decoded_actions)
    columns = [
        "source",
        "rollout",
        "global_step",
        "step",
        "env_idx",
        "episode",
        "episode_step",
        "non_idle_agents",
        "reward_agent_0",
        "done",
    ]
    columns.extend([f"action_id_{agent}" for agent in agent_ids])
    if include_decoded:
        columns.extend([f"action_decoded_{agent}" for agent in agent_ids])

    rows = []
    for record in records:
        action_ids = record["actions"]
        row = [
            record["source"],
            int(record["rollout"]),
            int(record["global_step"]),
            int(record["step"]),
            int(record["env_idx"]),
            int(record["episode"]),
            int(record["episode_step"]),
            int(record["non_idle_agents"]),
            float(record["reward_agent_0"]),
            bool(record["done"]),
        ]
        row.extend([int(action_ids[agent]) for agent in agent_ids])
        if include_decoded:
            row.extend(
                [
                    decoded_actions.get(agent, {}).get(int(action_ids[agent]), "")
                    for agent in agent_ids
                ]
            )
        rows.append(row)

    return wb.Table(data=rows, columns=columns)


def collect_unique_action_ids(
    records: List[Dict[str, Any]], agent_ids: List[str]
) -> Dict[str, List[int]]:
    unique_ids = {agent: set() for agent in agent_ids}
    for record in records:
        for agent in agent_ids:
            unique_ids[agent].add(int(record["actions"][agent]))
    return {agent: sorted(ids) for agent, ids in unique_ids.items()}


def decode_action_ids_safely(
    decode_fn: Callable[[Dict[str, List[int]]], Dict[str, Dict[int, str]]],
    action_ids_by_agent: Dict[str, List[int]],
) -> Dict[str, Dict[int, str]]:
    try:
        return decode_fn(action_ids_by_agent)
    except Exception as exc:
        message = f"decode error: {type(exc).__name__}: {exc}"
        return {
            agent: {int(action_id): message for action_id in action_ids}
            for agent, action_ids in action_ids_by_agent.items()
        }
