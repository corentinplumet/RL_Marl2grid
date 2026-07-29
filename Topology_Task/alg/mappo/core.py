from collections import Counter
from time import time

from .agent import Actor, Critic
from .config import get_alg_args
from common.action_trace import (
    build_action_trace_table,
    collect_unique_action_ids,
    decode_action_ids_safely,
    tensor_scalar_to_float,
)
from common.checkpoint import (
    EXACT_BOUNDARY_PHASE,
    CheckpointSaver,
    capture_rng_state,
    exact_resume_state,
    restore_rng_state,
)
from common.explainability import (
    EXPLAIN_LINE_KEYS,
    EXPLAIN_SCALAR_KEYS,
    explain_arrays_from_infos,
    summarize_explain_arrays,
)
from common.gnn import build_graph_encoder
from common.imports import *
from common.logger import Logger
from common.utils import (
    ReturnNormalizer,
    cast_np_to_tensors,
    clone_nested,
    flatten_rollout_obs,
    get_joint_obs,
    index_nested,
    set_nested_at_step,
    set_nested_env_index,
    strip_state_graph,
    zeros_like_with_leading,
)
from env.eval import Evaluator


def _linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return start + progress * (end - start)


def _scheduled_entropy_coef(args: Namespace, global_step: int) -> float:
    final_coef = getattr(args, "entropy_coef_final", None)
    if final_coef is None:
        return args.entropy_coef
    progress = global_step / max(args.total_timesteps, 1)
    return _linear_schedule(args.entropy_coef, final_coef, progress)


def _scheduled_action0_bonus(args: Namespace, global_step: int) -> float:
    init_bonus = getattr(args, "action0_logit_bonus_init", 0.0)
    final_bonus = getattr(args, "action0_logit_bonus_final", 0.0)
    schedule_fraction = getattr(args, "action0_logit_bonus_fraction", 1.0)
    if init_bonus == final_bonus:
        return init_bonus
    if schedule_fraction <= 0.0:
        return final_bonus
    schedule_steps = max(int(args.total_timesteps * schedule_fraction), 1)
    progress = global_step / schedule_steps
    return _linear_schedule(init_bonus, final_bonus, progress)


def _lr_schedule_fraction(
    args: Namespace,
    *,
    global_step: int,
    iteration: int,
    n_rollouts: int,
) -> float:
    final_frac = float(getattr(args, "lr_final_frac", 0.0))
    if final_frac < 0.0 or final_frac > 1.0:
        raise ValueError(f"--lr-final-frac must be in [0, 1]. Got {final_frac}.")

    anneal_timesteps = getattr(args, "lr_anneal_timesteps", None)
    if anneal_timesteps is None:
        progress = (iteration - 1.0) / max(n_rollouts, 1)
    else:
        anneal_timesteps = int(anneal_timesteps)
        if anneal_timesteps <= 0:
            raise ValueError(
                f"--lr-anneal-timesteps must be positive. Got {anneal_timesteps}."
            )
        progress = global_step / anneal_timesteps
    return _linear_schedule(1.0, final_frac, progress)


def _truthy_info_value(value: Any) -> bool:
    if isinstance(value, th.Tensor):
        return bool(value.detach().cpu().any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    if isinstance(value, (list, tuple, set)):
        return any(_truthy_info_value(item) for item in value)
    return bool(value)


def _transition_info(info: Any) -> Any:
    if isinstance(info, dict) and "final_info" in info:
        return info["final_info"]
    return info


def _info_chronic_field(info: Any, field: str) -> str:
    transition = _transition_info(info)
    for candidate in (transition, info):
        if isinstance(candidate, dict) and candidate.get(field) is not None:
            return str(candidate[field])
    return "unknown"


def _info_chronic_name(info: Any) -> str:
    return _info_chronic_field(info, "chronic_name")


def _info_chronic_fingerprint(info: Any) -> str:
    return _info_chronic_field(info, "chronic_fingerprint")


def _format_chronic_counter(labels: List[str], max_items: int = 4) -> str:
    if not labels:
        return "-"
    counts = Counter(labels)
    items = counts.most_common(max_items)
    text = ", ".join(f"{label}x{count}" for label, count in items)
    if len(counts) > max_items:
        text += f", ... ({len(counts)} unique)"
    return text


def _agent_took_illegal_action(info: Any, agent_id: str) -> bool:
    info = _transition_info(info)
    if not isinstance(info, dict):
        return False

    agent_info = info.get(agent_id)
    if isinstance(agent_info, dict):
        if "is_illegal" in agent_info:
            return _truthy_info_value(agent_info["is_illegal"])
        exception = agent_info.get("exception")
        if exception is not None and "IllegalAction" in str(exception):
            return True

    if "is_illegal" in info:
        return _truthy_info_value(info["is_illegal"])
    exception = info.get("exception")
    return exception is not None and "IllegalAction" in str(exception)


def _unique_parameters(modules: List[nn.Module]) -> List[nn.Parameter]:
    params = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params


def _joint_non_idle_action_counts(
    actions: Dict[str, th.Tensor],
    agent_ids: List[str],
) -> np.ndarray:
    """Count how many agents chose non-idle actions for each env step."""
    non_idle = th.stack(
        [(actions[agent_id].long() != 0) for agent_id in agent_ids],
        dim=0,
    )
    return non_idle.sum(dim=0).reshape(-1).detach().cpu().numpy()


def _sparse_intervention_penalty_enabled(args: Namespace) -> bool:
    return (
        float(getattr(args, "intervention_penalty", 0.0)) > 0.0
        or float(getattr(args, "safe_intervention_penalty", 0.0)) > 0.0
    )


def _adaptive_intervention_budget_enabled(args: Namespace) -> bool:
    return bool(getattr(args, "adaptive_intervention_budget", False))


def _intervention_budget_requires_global_rho(args: Namespace) -> bool:
    return (
        _adaptive_intervention_budget_enabled(args)
        and str(getattr(args, "intervention_budget_cost_mode", "local_safe"))
        == "global_safe"
    )


def _intervention_budget_requires_local_rho(args: Namespace) -> bool:
    return (
        _adaptive_intervention_budget_enabled(args)
        and str(getattr(args, "intervention_budget_cost_mode", "local_safe"))
        == "local_safe"
    )


def _as_reward_shape(values: Any, reference: np.ndarray) -> np.ndarray:
    """Convert scalar/vector rho values to the current vector-env reward shape."""
    reference = np.asarray(reference)
    values_np = np.asarray(values, dtype=np.float32)
    if values_np.shape == ():
        return np.full(reference.shape, float(values_np), dtype=np.float32)
    return np.broadcast_to(values_np, reference.shape).astype(np.float32, copy=False)


def _smooth_safe_weight(
    max_rho: Any, reference: np.ndarray, args: Namespace
) -> np.ndarray:
    """High when the grid is comfortably safe, low when lines approach overload."""
    rho = _as_reward_shape(max_rho, reference)
    threshold = float(getattr(args, "intervention_budget_rho_threshold", 0.90))
    sharpness = float(getattr(args, "intervention_budget_rho_sharpness", 25.0))
    logits = np.clip(sharpness * (threshold - rho), -60.0, 60.0)
    weights = 1.0 / (1.0 + np.exp(-logits))
    return np.where(np.isfinite(rho), weights, 0.0).astype(np.float32)


def _apply_sparse_intervention_penalty(
    reward: Dict[str, np.ndarray],
    action: Dict[str, th.Tensor],
    agent_ids: List[str],
    args: Namespace,
    pre_action_max_rho: Optional[np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """Subtract sparse-control penalties from per-agent training rewards."""
    intervention_penalty = float(getattr(args, "intervention_penalty", 0.0))
    safe_intervention_penalty = float(
        getattr(args, "safe_intervention_penalty", 0.0)
    )
    if intervention_penalty <= 0.0 and safe_intervention_penalty <= 0.0:
        empty_penalties = {
            agent: np.zeros_like(np.asarray(reward[agent]), dtype=np.float32)
            for agent in agent_ids
        }
        return reward, empty_penalties, np.zeros_like(
            np.asarray(reward[agent_ids[0]]), dtype=bool
        )

    if safe_intervention_penalty > 0.0:
        if pre_action_max_rho is None:
            raise ValueError(
                "--safe-intervention-penalty requires pre-action max rho values."
            )
        max_rho = np.asarray(pre_action_max_rho, dtype=np.float32)
        threshold = float(getattr(args, "safe_intervention_rho_threshold", 0.90))
        safe_state = np.isfinite(max_rho) & (max_rho < threshold)
    else:
        safe_state = np.zeros_like(np.asarray(reward[agent_ids[0]]), dtype=bool)

    penalties = {}
    for agent in agent_ids:
        non_idle = action[agent].detach().cpu().numpy().astype(np.int64) != 0
        penalty = intervention_penalty * non_idle.astype(np.float32)
        if safe_intervention_penalty > 0.0:
            penalty = penalty + safe_intervention_penalty * (
                non_idle & safe_state
            ).astype(np.float32)
        penalties[agent] = penalty.astype(np.float32)
        reward[agent] = np.asarray(reward[agent], dtype=np.float32) - penalties[agent]
    return reward, penalties, safe_state


def _apply_adaptive_intervention_budget(
    reward: Dict[str, np.ndarray],
    action: Dict[str, th.Tensor],
    agent_ids: List[str],
    args: Namespace,
    intervention_lambdas: Dict[str, float],
    pre_action_max_rho: Optional[np.ndarray],
    pre_action_agent_max_rho: Optional[Dict[str, np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Subtract adaptive Lagrangian intervention costs from per-agent rewards."""
    mode = str(getattr(args, "intervention_budget_cost_mode", "local_safe"))
    penalties, costs, weights = {}, {}, {}
    for agent in agent_ids:
        reward_np = np.asarray(reward[agent], dtype=np.float32)
        non_idle = action[agent].detach().cpu().numpy().astype(np.int64) != 0
        if mode == "nonidle":
            safety_weight = np.ones_like(reward_np, dtype=np.float32)
        elif mode == "global_safe":
            if pre_action_max_rho is None:
                raise ValueError(
                    "--adaptive-intervention-budget with "
                    "--intervention-budget-cost-mode=global_safe requires "
                    "pre-action max rho values."
                )
            safety_weight = _smooth_safe_weight(pre_action_max_rho, reward_np, args)
        elif mode == "local_safe":
            if pre_action_agent_max_rho is None:
                raise ValueError(
                    "--adaptive-intervention-budget with "
                    "--intervention-budget-cost-mode=local_safe requires "
                    "pre-action local max rho values."
                )
            safety_weight = _smooth_safe_weight(
                pre_action_agent_max_rho.get(agent, float("nan")), reward_np, args
            )
        else:
            raise ValueError(f"Unsupported intervention budget cost mode: {mode!r}")

        cost = non_idle.astype(np.float32) * safety_weight
        penalty = float(intervention_lambdas[agent]) * cost
        penalties[agent] = penalty.astype(np.float32)
        costs[agent] = cost.astype(np.float32)
        weights[agent] = safety_weight.astype(np.float32)
        reward[agent] = reward_np - penalties[agent]
    return reward, penalties, costs, weights


def _update_adaptive_intervention_lambdas(
    intervention_lambdas: Dict[str, float],
    intervention_budget_costs: Dict[str, th.Tensor],
    agent_ids: List[str],
    args: Namespace,
    global_step: int,
) -> Dict[str, Dict[str, float]]:
    """Update Lagrange multipliers from rollout-mean intervention cost."""
    target = float(getattr(args, "intervention_budget_target", 0.25))
    lr = float(getattr(args, "intervention_budget_lr", 0.01))
    max_lambda = max(float(getattr(args, "intervention_budget_max_lambda", 10.0)), 0.0)
    warmup_steps = int(getattr(args, "intervention_budget_warmup_steps", 0))
    should_update = global_step >= warmup_steps and lr > 0.0
    stats = {}
    for agent in agent_ids:
        observed = float(intervention_budget_costs[agent].mean().item())
        old_lambda = float(intervention_lambdas[agent])
        violation = observed - target
        new_lambda = old_lambda
        if should_update:
            new_lambda = float(np.clip(old_lambda + lr * violation, 0.0, max_lambda))
            intervention_lambdas[agent] = new_lambda
        stats[agent] = {
            "cost": observed,
            "target": target,
            "violation": violation,
            "lambda_before": old_lambda,
            "lambda_after": new_lambda,
            "updated": float(should_update),
        }
    return stats


def _should_log_training_action_trace(
    args: Namespace, iteration: int, init_rollout: int
) -> bool:
    if not bool(getattr(args, "trace_rollout_actions", False)):
        return False
    every = int(getattr(args, "trace_rollout_every", 10))
    if every <= 0:
        return False
    return iteration == init_rollout or iteration % every == 0


def _build_training_action_trace_records(
    args: Namespace,
    actions: Dict[str, th.Tensor],
    rewards: Dict[str, th.Tensor],
    dones: th.Tensor,
    agent_ids: List[str],
    iteration: int,
    global_step: int,
) -> List[Dict[str, Any]]:
    max_steps = min(
        max(int(getattr(args, "trace_rollout_max_steps", 0)), 0),
        int(args.n_steps),
    )
    if max_steps <= 0:
        return []
    env_idx = int(
        np.clip(getattr(args, "trace_rollout_env_idx", 0), 0, args.n_envs - 1)
    )
    rollout_start_step = global_step - int(args.n_steps * args.n_envs)
    records = []
    episode = 0
    episode_step = 0
    for step in range(max_steps):
        action_ids = {
            agent: int(actions[agent][step, env_idx].long().detach().cpu().item())
            for agent in agent_ids
        }
        done = bool(dones[step, env_idx].detach().cpu().item())
        records.append(
            {
                "source": "train",
                "rollout": iteration,
                "global_step": rollout_start_step + (step + 1) * int(args.n_envs),
                "step": step,
                "env_idx": env_idx,
                "episode": episode,
                "episode_step": episode_step,
                "non_idle_agents": sum(
                    action_id != 0 for action_id in action_ids.values()
                ),
                "reward_agent_0": tensor_scalar_to_float(
                    rewards[agent_ids[0]][step, env_idx]
                ),
                "done": done,
                "actions": action_ids,
            }
        )
        if done:
            episode += 1
            episode_step = 0
        else:
            episode_step += 1
    return records


def _build_shared_actor_graph_encoder(envs: gym.Env, args: Namespace, agent_ids: List[str]):
    if not getattr(args, "share_actor_gnn", False):
        return None
    if getattr(args, "actor_encoder", "mlp") != "gnn":
        raise ValueError("share_actor_gnn=True requires actor_encoder='gnn'.")
    if getattr(envs, "graph_specs", None) is None:
        raise ValueError("share_actor_gnn=True requires graph observations.")

    first_spec = envs.graph_specs[agent_ids[0]]
    node_dim = int(first_spec["node_dim"])
    edge_dim = int(first_spec["edge_dim"])
    for agent_id in agent_ids[1:]:
        spec = envs.graph_specs[agent_id]
        if int(spec["node_dim"]) != node_dim or int(spec["edge_dim"]) != edge_dim:
            raise ValueError(
                "Shared actor GNN requires all actor graph specs to have the same "
                "node_dim and edge_dim."
            )
    return build_graph_encoder(first_spec, args)


def _evaluate_preserving_training_rng(
    evaluator: Evaluator, global_step: int, actors: Dict
) -> float:
    """Run eval without letting stochastic eval sampling change training RNG state."""
    rng_state = capture_rng_state()
    try:
        return evaluator.evaluate(global_step, actors)
    finally:
        restore_rng_state(rng_state)


def _checkpoint_cpu_copy(obj: Any) -> Any:
    """Copy nested rollout state without retaining accelerator storage."""
    if isinstance(obj, th.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _checkpoint_cpu_copy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_checkpoint_cpu_copy(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_checkpoint_cpu_copy(value) for value in obj)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    return obj


def _checkpoint_to_device(obj: Any, device: th.device) -> Any:
    if isinstance(obj, th.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _checkpoint_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_checkpoint_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_checkpoint_to_device(value, device) for value in obj)
    return obj


class MAPPO:
    """Multi-agent Proximal Policy Optimization (PPO) implementation for training an agent in a given environment: https://arxiv.org/abs/2103.01955."""

    def __init__(
        self,
        envs: gym.Env,
        run_name: str,
        start_time: float,
        args: Dict[str, Any],
        ckpt: CheckpointSaver,
    ):
        """Init method for PPO

        Args:
            envs (gym.Env): The environments used for training.
            run_name (str): The name of the current training run.
            start_time (float): The time when training started.
            args (Dict[str, Any]): The command line arguments for configuration.
            ckpt (CheckpointSaver): The checkpoint handler for saving and loading training state.
        """
        # Load algorithm-specific arguments if not resuming from a checkpoint
        if not ckpt.resumed:
            # ``main`` already composes these arguments before creating the
            # environments. Keep this merge for direct MAPPO callers while
            # allowing the already-present keys to be refreshed safely.
            args = ap.Namespace(**{**vars(args), **vars(get_alg_args())})

        assert args.n_steps % args.n_envs == 0, (
            f"Invalid train frequency (n_steps): {args.n_steps}. Must be multiple of n_envs {args.n_envs}"
        )

        if args.cuda and th.cuda.is_available():
            device = th.device("cuda")
        elif args.cuda and th.backends.mps.is_available():
            device = th.device("mps")
        else:
            device = th.device("cpu")

        # We don't take directly the env keys because agents are not ordered (0, 1, 2, ...) and causes problem with indexing elsewhere
        agent_ids = [
            f"agent_{idx}" for idx in range(len(envs.observation_space.keys()))
        ]

        # Initialize the rollout, actor, critic, optimizer, and buffer
        batch_size = int(args.n_envs * args.n_steps)
        minibatch_size = int(batch_size // args.n_minibatches)
        n_rollouts = args.total_timesteps // batch_size
        loaded_exact_state = (
            exact_resume_state(ckpt.loaded_run) if ckpt.resumed else None
        )
        if loaded_exact_state is not None:
            init_rollout = int(loaded_exact_state["next_rollout"])
        elif ckpt.resumed:
            init_rollout = int(ckpt.loaded_run["last_rollout"])
            if getattr(args, "resume_start_next_rollout", False):
                init_rollout += 1
            print(
                "Warning: this is a legacy or mid-rollout checkpoint. Model and "
                "optimizer state will be restored, but the environment trajectory, "
                "RNG streams, and reward-normalizer state cannot be continued exactly."
            )
        else:
            init_rollout = 1

        # Determine action space type
        continuous_actions = True if args.action_type == "redispatch" else False
        shared_actor_graph_encoder = _build_shared_actor_graph_encoder(
            envs, args, agent_ids
        )
        actors = {
            f"agent_{idx}": Actor(
                idx,
                envs,
                args,
                continuous_actions,
                shared_graph_encoder=shared_actor_graph_encoder,
            ).to(device)
            for idx in range(len(agent_ids))
        }

        critic = Critic(envs, args).to(device)

        if ckpt.resumed:
            for agent in actors.keys():
                actors[agent].load_state_dict(ckpt.loaded_run[agent])
            critic.load_state_dict(ckpt.loaded_run["critic"])

        actor_params = _unique_parameters(list(actors.values()))
        actor_param_count = int(sum(param.numel() for param in actor_params))
        critic_param_count = int(sum(param.numel() for param in critic.parameters()))
        token_specs = getattr(envs, "token_specs", None) or {}
        token_count_metrics = {
            f"model/token_count_{key}": int(spec["n_tokens"])
            for key, spec in token_specs.items()
            if isinstance(spec, dict) and "n_tokens" in spec
        }
        if continuous_actions:
            raise ("Redispatching actions are not yet implemented")
        actor_optim = optim.Adam(actor_params, lr=args.actor_lr, eps=1e-5)
        critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)

        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run["actor_optim"])
            critic_optim.load_state_dict(ckpt.loaded_run["critic_optim"])

        loaded_training_state = (
            ckpt.loaded_run.get("training_state", {}) if ckpt.resumed else {}
        )
        normalization_enabled = bool(
            getattr(args, "norm_obs", False)
            or getattr(args, "gnn_running_norm", False)
        )
        if ckpt.resumed and normalization_enabled and loaded_exact_state is None:
            saved_obs_stats = loaded_training_state.get("obs_stats", {})
            if saved_obs_stats:
                envs.set_obs_stats(saved_obs_stats)

        if loaded_exact_state is not None:
            saved_global_step = int(loaded_exact_state["global_step"])
            if saved_global_step != int(ckpt.loaded_run["global_step"]):
                raise ValueError(
                    "Checkpoint boundary global_step does not match the model record: "
                    f"{saved_global_step} != {ckpt.loaded_run['global_step']}."
                )
            envs.set_checkpoint_state(loaded_exact_state["environment_states"])
            next_obs = _checkpoint_to_device(
                loaded_exact_state["next_obs"], device
            )
        else:
            next_obs, _ = envs.reset()
            next_obs = cast_np_to_tensors(next_obs, device)
        joint_obs_template = get_joint_obs(
            next_obs, args.critic_encoder, args.decentralized
        )
        joint_observations = zeros_like_with_leading(
            joint_obs_template, (args.n_steps,), device=device
        )
        values = th.zeros((args.n_steps, args.n_envs)).to(device)
        dones = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        terminations = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        observations, actions, logprobs, rewards = [{} for _ in range(4)]
        intervention_gate_metric_names = (
            "prob_do_nothing",
            "prob_intervene",
            "gate_entropy",
            "nonidle_action_entropy",
        )
        collect_intervention_gate_metrics = bool(getattr(args, "track", False))
        intervention_gate_metrics = {}
        collect_explain_metrics = bool(getattr(args, "track", False))
        explain_metrics = (
            {
                key: th.zeros((args.n_steps, args.n_envs), device=device)
                for key in EXPLAIN_SCALAR_KEYS + EXPLAIN_LINE_KEYS
            }
            if collect_explain_metrics
            else {}
        )
        for id in agent_ids:
            observations[id] = zeros_like_with_leading(
                strip_state_graph(next_obs[id]), (args.n_steps,), device=device
            )
            actions[id] = th.zeros((args.n_steps, args.n_envs)).to(
                device
            )  # Assuming discrete actions
            logprobs[id] = th.zeros((args.n_steps, args.n_envs)).to(device)
            rewards[id] = th.zeros((args.n_steps, args.n_envs)).to(device)
            if collect_intervention_gate_metrics and getattr(
                actors[id], "intervention_gate", False
            ):
                intervention_gate_metrics[id] = {
                    name: th.zeros((args.n_steps, args.n_envs), device=device)
                    for name in intervention_gate_metric_names
                }

        assert args.eval_freq % args.n_envs == 0, (
            f"Invalid eval frequency: {args.eval_freq}. Must be multiple of n_envs {args.n_envs}"
        )
        logger = Logger(run_name, args) if args.track else None
        split_chronics = getattr(args, "split_chronics", False)
        evaluator = Evaluator(
            args,
            logger,
            device,
            chronic_split="test" if split_chronics else None,
            metric_prefix="test" if split_chronics else None,
        )
        train_evaluator = (
            Evaluator(
                args,
                logger,
                device,
                chronic_split="train",
                metric_prefix="train_eval",
            )
            if split_chronics and getattr(args, "eval_train_chronics", False)
            else None
        )
        best_checkpoint_score = (
            float(loaded_exact_state.get("best_checkpoint_score", -np.inf))
            if loaded_exact_state is not None
            else -np.inf
        )

        global_step = 0 if not ckpt.resumed else ckpt.loaded_run["global_step"]
        start_time = start_time
        last_ckpt_time = start_time  # <-- track last checkpoint timestamp

        reward_normalizer = (
            {
                agent: ReturnNormalizer(args.n_envs, args.gamma)
                for agent in agent_ids
            }
            if getattr(args, "norm_reward", False)
            else None
        )
        if reward_normalizer is not None and loaded_exact_state is not None:
            saved_normalizers = loaded_exact_state.get("reward_normalizers")
            if not isinstance(saved_normalizers, dict):
                raise ValueError(
                    "Exact checkpoint is missing reward-normalizer state."
                )
            for agent in agent_ids:
                reward_normalizer[agent].load_state_dict(saved_normalizers[agent])
        sparse_penalty_enabled = _sparse_intervention_penalty_enabled(args)
        safe_penalty_enabled = (
            float(getattr(args, "safe_intervention_penalty", 0.0)) > 0.0
        )
        adaptive_budget_enabled = _adaptive_intervention_budget_enabled(args)
        loaded_lambdas = loaded_training_state.get("intervention_lambdas", {})
        intervention_lambdas = {
            agent: float(
                loaded_lambdas.get(
                    agent, getattr(args, "intervention_budget_init_lambda", 0.0)
                )
            )
            for agent in agent_ids
        }
        sparse_penalties = (
            {
                agent: th.zeros((args.n_steps, args.n_envs), device=device)
                for agent in agent_ids
            }
            if sparse_penalty_enabled
            else {}
        )
        safe_intervention_state = (
            th.zeros((args.n_steps, args.n_envs), device=device)
            if sparse_penalty_enabled
            else None
        )
        intervention_budget_penalties = (
            {
                agent: th.zeros((args.n_steps, args.n_envs), device=device)
                for agent in agent_ids
            }
            if adaptive_budget_enabled
            else {}
        )
        intervention_budget_costs = (
            {
                agent: th.zeros((args.n_steps, args.n_envs), device=device)
                for agent in agent_ids
            }
            if adaptive_budget_enabled
            else {}
        )
        intervention_budget_weights = (
            {
                agent: th.zeros((args.n_steps, args.n_envs), device=device)
                for agent in agent_ids
            }
            if adaptive_budget_enabled
            else {}
        )

        def _current_training_state() -> Dict[str, Any]:
            state = {"intervention_lambdas": dict(intervention_lambdas)}
            if normalization_enabled:
                state["obs_stats"] = envs.get_obs_stats()
            return state

        if loaded_exact_state is not None:
            evaluator._eval_window_index = int(
                loaded_exact_state.get("eval_window_index", 0)
            )
            if train_evaluator is not None:
                train_evaluator._eval_window_index = int(
                    loaded_exact_state.get("train_eval_window_index", 0)
                )

        at_exact_boundary = False
        saved_boundary_this_process = False

        def _save_exact_boundary(completed_rollout: int) -> None:
            """Save a checkpoint that resumes at the next rollout exactly."""
            boundary_rng = capture_rng_state()
            try:
                resume_state = {
                    "global_step": int(global_step),
                    "next_rollout": int(completed_rollout) + 1,
                    "rng_state": boundary_rng,
                    "environment_states": envs.get_checkpoint_state(),
                    "next_obs": _checkpoint_cpu_copy(next_obs),
                    "reward_normalizers": (
                        {
                            agent: reward_normalizer[agent].state_dict()
                            for agent in agent_ids
                        }
                        if reward_normalizer is not None
                        else None
                    ),
                    "best_checkpoint_score": float(best_checkpoint_score),
                    "eval_window_index": int(evaluator._eval_window_index),
                    "train_eval_window_index": (
                        int(train_evaluator._eval_window_index)
                        if train_evaluator is not None
                        else None
                    ),
                }
                ckpt.set_record(
                    args,
                    actors,
                    critic,
                    global_step,
                    actor_optim,
                    critic_optim,
                    "" if not logger else logger.wb_path,
                    completed_rollout,
                    training_state=_current_training_state(),
                    resume_state=resume_state,
                    checkpoint_phase=EXACT_BOUNDARY_PHASE,
                )
                ckpt.save()
            finally:
                # Checkpointing itself must not advance the training RNG streams.
                restore_rng_state(boundary_rng)

        if loaded_exact_state is not None:
            # Model/env/evaluator construction consumes randomness. The next draw
            # must instead be the one immediately following the saved boundary.
            restore_rng_state(loaded_exact_state["rng_state"])

        sps_start_step = int(global_step)
        sps_start_time = time()
        iteration = init_rollout - 1
        try:
            for iteration in range(init_rollout, n_rollouts + 1):
                at_exact_boundary = False
                # Annealing the rate if instructed to do so
                if args.anneal_lr:
                    frac = _lr_schedule_fraction(
                        args,
                        global_step=global_step,
                        iteration=iteration,
                        n_rollouts=n_rollouts,
                    )
                    actor_optim.param_groups[0]["lr"] = frac * args.actor_lr
                    critic_optim.param_groups[0]["lr"] = frac * args.critic_lr
                entropy_coef = _scheduled_entropy_coef(args, global_step)
                action0_bonus = _scheduled_action0_bonus(args, global_step)
                illegal_action_counts = {agent: 0 for agent in agent_ids}
                illegal_action_totals = {agent: 0 for agent in agent_ids}
                rollout_chronic_labels_by_env = [
                    [] for _ in range(args.n_envs)
                ]
                rollout_chronic_fingerprints_by_env = [
                    [] for _ in range(args.n_envs)
                ]

                for step in range(0, args.n_steps):
                    global_step += args.n_envs

                    action, logprob = {}, {}
                    for agent in agent_ids:
                        set_nested_at_step(
                            observations[agent],
                            step,
                            strip_state_graph(next_obs[agent]),
                        )

                        with th.no_grad():
                            action[agent], logprob[agent], _ = actors[agent].get_action(
                                next_obs[agent], action0_bonus=action0_bonus
                            )
                            if agent in intervention_gate_metrics:
                                gate_diagnostics = actors[
                                    agent
                                ].get_intervention_gate_diagnostics(
                                    next_obs[agent], action0_bonus=action0_bonus
                                )
                                for name, value in gate_diagnostics.items():
                                    intervention_gate_metrics[agent][name][step] = value

                        actions[agent][step] = action[agent]  # .unsqueeze(-1)
                        logprobs[agent][step] = logprob[agent]  # .unsqueeze(-1)

                    # get joint obs for this
                    with th.no_grad():
                        joint_obs = get_joint_obs(
                            next_obs, args.critic_encoder, args.decentralized
                        )
                        value = critic.get_value(joint_obs)
                        set_nested_at_step(joint_observations, step, joint_obs)
                        values[step] = value.flatten()

                    pre_action_max_rho = (
                        envs.get_current_max_rho()
                        if (
                            (sparse_penalty_enabled and safe_penalty_enabled)
                            or _intervention_budget_requires_global_rho(args)
                        )
                        else None
                    )
                    pre_action_agent_max_rho = (
                        envs.get_current_agent_max_rho()
                        if _intervention_budget_requires_local_rho(args)
                        else None
                    )

                    next_obs, reward, next_terminations, next_truncations, infos = (
                        envs.step(action)
                    )
                    step_infos = (
                        list(infos)
                        if isinstance(infos, (list, tuple))
                        else [infos]
                    )
                    done_np = np.logical_or(
                        next_terminations[agent_ids[0]],
                        next_truncations[agent_ids[0]],
                    )
                    for env_idx, done in enumerate(done_np):
                        if done and env_idx < len(step_infos):
                            rollout_chronic_labels_by_env[env_idx].append(
                                _info_chronic_name(step_infos[env_idx])
                            )
                            rollout_chronic_fingerprints_by_env[env_idx].append(
                                _info_chronic_fingerprint(step_infos[env_idx])
                            )
                    if collect_explain_metrics:
                        explain_arrays = explain_arrays_from_infos(step_infos)
                        for name, values_np in explain_arrays.items():
                            explain_metrics[name][step] = th.as_tensor(
                                values_np, dtype=th.float32, device=device
                            )
                    for agent in agent_ids:
                        illegal_action_totals[agent] += len(step_infos)
                        illegal_action_counts[agent] += sum(
                            int(_agent_took_illegal_action(info, agent))
                            for info in step_infos
                        )

                    if sparse_penalty_enabled:
                        reward, penalty, safe_state = _apply_sparse_intervention_penalty(
                            reward,
                            action,
                            agent_ids,
                            args,
                            pre_action_max_rho,
                        )
                        for agent in agent_ids:
                            sparse_penalties[agent][step] = th.as_tensor(
                                penalty[agent], dtype=th.float32, device=device
                            )
                        safe_intervention_state[step] = th.as_tensor(
                            safe_state, dtype=th.float32, device=device
                        )

                    if adaptive_budget_enabled:
                        (
                            reward,
                            budget_penalty,
                            budget_cost,
                            budget_weight,
                        ) = _apply_adaptive_intervention_budget(
                            reward,
                            action,
                            agent_ids,
                            args,
                            intervention_lambdas,
                            pre_action_max_rho,
                            pre_action_agent_max_rho,
                        )
                        for agent in agent_ids:
                            intervention_budget_penalties[agent][step] = th.as_tensor(
                                budget_penalty[agent],
                                dtype=th.float32,
                                device=device,
                            )
                            intervention_budget_costs[agent][step] = th.as_tensor(
                                budget_cost[agent], dtype=th.float32, device=device
                            )
                            intervention_budget_weights[agent][step] = th.as_tensor(
                                budget_weight[agent], dtype=th.float32, device=device
                            )

                    if reward_normalizer is not None:
                        for agent in agent_ids:
                            reward[agent] = reward_normalizer[agent](
                                np.asarray(reward[agent]), done_np
                            )

                    reward = cast_np_to_tensors(reward, device)
                    for agent in agent_ids:
                        rewards[agent][step] = reward[agent]

                    dones[step] = th.tensor(
                        done_np
                    ).to(device)
                    terminations[step] = th.tensor(next_terminations[agent_ids[0]]).to(
                        device
                    )

                    next_obs = cast_np_to_tensors(next_obs, device)
                    real_next_obs = clone_nested(next_obs)
                    for idx, done in enumerate(dones[step]):
                        if done:
                            final_obs = cast_np_to_tensors(
                                infos[idx]["final_observation"], device
                            )
                            for agent in agent_ids:
                                set_nested_env_index(
                                    real_next_obs[agent], idx, final_obs[agent]
                                )

                    if global_step % args.eval_freq == 0:
                        obs_stats = envs.get_obs_stats()
                        train_eval_survival = None
                        if train_evaluator is not None:
                            train_evaluator.env.env.set_obs_stats(obs_stats)
                            train_eval_survival = _evaluate_preserving_training_rng(
                                train_evaluator, global_step, actors
                            )
                        evaluator.env.env.set_obs_stats(obs_stats)
                        eval_survival = _evaluate_preserving_training_rng(
                            evaluator, global_step, actors
                        )
                        checkpoint_score = eval_survival
                        if train_eval_survival is not None:
                            checkpoint_score += train_eval_survival
                        # On equal score, prefer the later checkpoint: a late policy
                        # with the same train+test survival is usually more stable
                        # than an early lucky one. The filename keeps best_test_ for
                        # backward compatibility with existing scripts.
                        if split_chronics and checkpoint_score >= best_checkpoint_score:
                            best_checkpoint_score = checkpoint_score
                            if args.checkpoint:
                                ckpt.set_record(
                                    args,
                                    actors,
                                    critic,
                                    global_step,
                                    actor_optim,
                                    critic_optim,
                                    "" if not logger else logger.wb_path,
                                    iteration,
                                    mark_final=False,
                                    training_state=_current_training_state(),
                                    checkpoint_phase="mid_rollout_evaluation",
                                )
                                ckpt.save_as("best_test_" + ckpt.checkpoint_base_name)
                        if args.verbose:
                            session_steps = int(global_step) - sps_start_step
                            session_elapsed = max(time() - sps_start_time, 1e-9)
                            print(f"SPS={int(session_steps / session_elapsed)}")

                # Bootstrap value if not done
                with th.no_grad():
                    advantages, returns = {}, {}
                    joint_real_next_obs = get_joint_obs(
                        real_next_obs, args.critic_encoder, args.decentralized
                    )
                    for agent in agent_ids:
                        advantages[agent] = th.zeros_like(rewards[agent]).to(device)
                        lastgaelam = 0
                        for t in reversed(range(args.n_steps)):
                            if t == args.n_steps - 1:
                                nextvalues = critic.get_value(
                                    joint_real_next_obs
                                ).reshape(1, -1)
                            else:
                                nextvalues = values[t + 1]
                            delta = (
                                rewards[agent][t]
                                + args.gamma * nextvalues * (1 - terminations[t])
                                - values[t]
                            )
                            advantages[agent][t] = lastgaelam = (
                                delta
                                + args.gamma
                                * args.gae_lambda
                                * (1 - dones[t])
                                * lastgaelam
                            )
                        returns[agent] = advantages[agent] + values

                intervention_budget_stats = {}
                if adaptive_budget_enabled:
                    intervention_budget_stats = _update_adaptive_intervention_lambdas(
                        intervention_lambdas,
                        intervention_budget_costs,
                        agent_ids,
                        args,
                        global_step,
                    )

                b_values = values.reshape(-1)
                b_joint_obs = flatten_rollout_obs(joint_observations)
                optimize_critic_updates = getattr(args, "optimize_critic_updates", True)
                b_critic_returns = (
                    th.stack([returns[agent].reshape(-1) for agent in agent_ids]).mean(dim=0)
                    if optimize_critic_updates
                    else None
                )

                # Per-rollout training metric accumulators
                train_metrics = {
                    ag: {"entropy": [], "pg_loss": [], "approx_kl": [], "clipfrac": []}
                    for ag in agent_ids
                }
                v_loss_history = []

                for agent in agent_ids:
                    # Flatten the batch
                    b_obs = flatten_rollout_obs(observations[agent])
                    b_logprobs = logprobs[agent].reshape(-1)
                    b_actions = actions[agent].reshape(
                        -1,
                    )
                    b_advantages = advantages[agent].reshape(-1)
                    b_returns = (
                        returns[agent].reshape(-1)
                        if not optimize_critic_updates
                        else None
                    )

                    # Optimizing the policy, and optionally the legacy value update.
                    b_inds = np.arange(batch_size)
                    clipfracs = []
                    for _ in range(args.update_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, batch_size, minibatch_size):
                            end = start + minibatch_size
                            mb_inds = b_inds[start:end]
                            action, newlogprob, entropy = actors[agent].get_action(
                                index_nested(b_obs, mb_inds),
                                b_actions.long()[mb_inds],
                                action0_bonus=action0_bonus,
                            )
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            with th.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                # old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [
                                    ((ratio - 1.0).abs() > args.clip_coef)
                                    .float()
                                    .mean()
                                    .item()
                                ]

                            mb_advantages = b_advantages[mb_inds]
                            if args.norm_adv:
                                mb_advantages = (
                                    mb_advantages - mb_advantages.mean()
                                ) / (mb_advantages.std() + 1e-8)

                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * th.clamp(
                                ratio, 1 - args.clip_coef, 1 + args.clip_coef
                            )
                            pg_loss = th.max(pg_loss1, pg_loss2).mean()

                            entropy_loss = entropy.mean()
                            pg_loss = pg_loss - entropy_coef * entropy_loss

                            actor_optim.zero_grad()
                            pg_loss.backward()
                            nn.utils.clip_grad_norm_(
                                actors[agent].parameters(), args.max_grad_norm
                            )
                            actor_optim.step()

                            # Accumulate per-minibatch training metrics
                            train_metrics[agent]["entropy"].append(float(entropy_loss.detach()))
                            train_metrics[agent]["pg_loss"].append(float(pg_loss.detach()))
                            train_metrics[agent]["approx_kl"].append(float(approx_kl.detach()))
                            train_metrics[agent]["clipfrac"].append(float(clipfracs[-1]))

                            if optimize_critic_updates:
                                continue

                            # Legacy value loss: fit the shared critic once inside
                            # each actor update loop. This preserves old runs.
                            newvalue = critic.get_value(
                                index_nested(b_joint_obs, mb_inds)
                            ).view(-1)
                            if args.clip_vfloss:
                                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                                v_clipped = b_values[mb_inds] + th.clamp(
                                    newvalue - b_values[mb_inds],
                                    -args.clip_coef,
                                    args.clip_coef,
                                )
                                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                                v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = (
                                    0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                                )

                            v_loss *= args.vf_coef

                            critic_optim.zero_grad()
                            v_loss.backward()
                            nn.utils.clip_grad_norm_(
                                critic.parameters(), args.max_grad_norm
                            )
                            critic_optim.step()
                            v_loss_history.append(float(v_loss.detach()))

                        if args.target_kl is not None and approx_kl > args.target_kl:
                            break

                if optimize_critic_updates:
                    # Optimized value loss: the centralized critic and shared joint
                    # reward make the per-agent critic updates redundant. Fit it
                    # once per rollout minibatch with the mean value target.
                    b_inds = np.arange(batch_size)
                    for _ in range(args.update_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, batch_size, minibatch_size):
                            end = start + minibatch_size
                            mb_inds = b_inds[start:end]
                            newvalue = critic.get_value(
                                index_nested(b_joint_obs, mb_inds)
                            ).view(-1)
                            if args.clip_vfloss:
                                v_loss_unclipped = (
                                    newvalue - b_critic_returns[mb_inds]
                                ) ** 2
                                v_clipped = b_values[mb_inds] + th.clamp(
                                    newvalue - b_values[mb_inds],
                                    -args.clip_coef,
                                    args.clip_coef,
                                )
                                v_loss_clipped = (
                                    v_clipped - b_critic_returns[mb_inds]
                                ) ** 2
                                v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = 0.5 * (
                                    (newvalue - b_critic_returns[mb_inds]) ** 2
                                ).mean()

                            v_loss *= args.vf_coef

                            critic_optim.zero_grad()
                            v_loss.backward()
                            nn.utils.clip_grad_norm_(
                                critic.parameters(), args.max_grad_norm
                            )
                            critic_optim.step()
                            v_loss_history.append(float(v_loss.detach()))

                # Log per-rollout training metrics to wandb
                if logger is not None:
                    joint_non_idle_counts = _joint_non_idle_action_counts(
                        actions, agent_ids
                    )
                    joint_non_idle_hist = np.bincount(
                        joint_non_idle_counts,
                        minlength=len(agent_ids) + 1,
                    )
                    joint_non_idle_frac = (
                        joint_non_idle_hist / max(joint_non_idle_counts.size, 1)
                    )
                    metrics_to_log: Dict[str, Any] = {
                        "model/actor_params": float(actor_param_count),
                        "model/critic_params": float(critic_param_count),
                        "train/lr_actor": float(actor_optim.param_groups[0]["lr"]),
                        "train/lr_critic": float(critic_optim.param_groups[0]["lr"]),
                        "train/entropy_coef": float(entropy_coef),
                        "train/action0_logit_bonus": float(action0_bonus),
                        "train/optimize_critic_updates": float(optimize_critic_updates),
                        "train/v_loss": float(np.mean(v_loss_history)) if v_loss_history else 0.0,
                        "train/non_idle_agents_mean": float(np.mean(joint_non_idle_counts)),
                        "train/non_idle_agents_std": float(np.std(joint_non_idle_counts)),
                        "train/non_idle_agents_max": float(np.max(joint_non_idle_counts)),
                        "train/frac_any_non_idle": float(np.mean(joint_non_idle_counts > 0)),
                        "train/frac_multi_agent_non_idle": float(np.mean(joint_non_idle_counts > 1)),
                    }
                    metrics_to_log.update(token_count_metrics)
                    for count, frac in enumerate(joint_non_idle_frac):
                        metrics_to_log[
                            f"train/non_idle_agents_count_{count}_frac"
                        ] = float(frac)

                    if any(getattr(actor, "intervention_gate", False) for actor in actors.values()):
                        metrics_to_log[
                            "train/intervention_gate_entropy_mode_separate"
                        ] = float(
                            getattr(args, "intervention_gate_entropy_mode", "coupled")
                            == "separate"
                        )
                        metrics_to_log["train/intervention_gate_entropy_mult"] = float(
                            getattr(args, "intervention_gate_entropy_mult", 1.0)
                        )
                        metrics_to_log[
                            "train/intervention_nonidle_entropy_mult"
                        ] = float(
                            getattr(args, "intervention_nonidle_entropy_mult", 1.0)
                        )

                    if sparse_penalty_enabled:
                        all_penalties = th.stack(
                            [sparse_penalties[ag] for ag in agent_ids], dim=0
                        )
                        metrics_to_log["train/intervention_penalty_coef"] = float(
                            getattr(args, "intervention_penalty", 0.0)
                        )
                        metrics_to_log[
                            "train/safe_intervention_penalty_coef"
                        ] = float(getattr(args, "safe_intervention_penalty", 0.0))
                        metrics_to_log[
                            "train/safe_intervention_rho_threshold"
                        ] = float(
                            getattr(args, "safe_intervention_rho_threshold", 0.90)
                        )
                        metrics_to_log[
                            "train/intervention_penalty_mean"
                        ] = float(all_penalties.mean().item())
                        metrics_to_log[
                            "train/intervention_penalty_total"
                        ] = float(all_penalties.sum().item())
                        if safe_penalty_enabled:
                            metrics_to_log["train/safe_state_frac"] = float(
                                safe_intervention_state.mean().item()
                            )

                    if adaptive_budget_enabled:
                        all_budget_costs = th.stack(
                            [intervention_budget_costs[ag] for ag in agent_ids], dim=0
                        )
                        all_budget_penalties = th.stack(
                            [
                                intervention_budget_penalties[ag]
                                for ag in agent_ids
                            ],
                            dim=0,
                        )
                        all_budget_weights = th.stack(
                            [
                                intervention_budget_weights[ag]
                                for ag in agent_ids
                            ],
                            dim=0,
                        )
                        metrics_to_log["train/intervention_budget_enabled"] = 1.0
                        metrics_to_log["train/intervention_budget_target"] = float(
                            getattr(args, "intervention_budget_target", 0.25)
                        )
                        metrics_to_log["train/intervention_budget_lr"] = float(
                            getattr(args, "intervention_budget_lr", 0.01)
                        )
                        metrics_to_log[
                            "train/intervention_budget_rho_threshold"
                        ] = float(
                            getattr(args, "intervention_budget_rho_threshold", 0.90)
                        )
                        metrics_to_log[
                            "train/intervention_budget_rho_sharpness"
                        ] = float(
                            getattr(args, "intervention_budget_rho_sharpness", 25.0)
                        )
                        metrics_to_log[
                            "train/intervention_budget_cost_mean"
                        ] = float(all_budget_costs.mean().item())
                        metrics_to_log[
                            "train/intervention_budget_cost_violation_mean"
                        ] = float(
                            all_budget_costs.mean().item()
                            - float(getattr(args, "intervention_budget_target", 0.25))
                        )
                        metrics_to_log[
                            "train/intervention_budget_penalty_mean"
                        ] = float(all_budget_penalties.mean().item())
                        metrics_to_log[
                            "train/intervention_budget_safety_weight_mean"
                        ] = float(all_budget_weights.mean().item())
                        metrics_to_log[
                            "train/intervention_budget_lambda_mean"
                        ] = float(
                            np.mean([intervention_lambdas[ag] for ag in agent_ids])
                        )

                    if collect_explain_metrics:
                        metrics_to_log.update(
                            summarize_explain_arrays(
                                {
                                    key: value.detach().cpu().numpy()
                                    for key, value in explain_metrics.items()
                                },
                                prefix="train/explain",
                            )
                        )

                    non_idle_table = wb.Table(
                        data=[
                            [int(count), float(frac)]
                            for count, frac in enumerate(joint_non_idle_frac)
                        ],
                        columns=["non_idle_agents", "fraction"],
                    )
                    metrics_to_log["train/non_idle_agents_distribution"] = wb.plot.bar(
                        non_idle_table,
                        "non_idle_agents",
                        "fraction",
                        title="Non-idle agents per environment step",
                    )
                    chronic_rows = []
                    all_chronic_labels = []
                    all_chronic_fingerprints = []
                    for env_idx, labels in enumerate(rollout_chronic_labels_by_env):
                        fingerprints = rollout_chronic_fingerprints_by_env[env_idx]
                        all_chronic_labels.extend(labels)
                        all_chronic_fingerprints.extend(fingerprints)
                        chronic_rows.append(
                            [
                                int(env_idx),
                                int(len(labels)),
                                int(len(set(labels))),
                                int(len(set(fingerprints))),
                                _format_chronic_counter(labels, max_items=8),
                                _format_chronic_counter(fingerprints, max_items=8),
                            ]
                        )
                    if all_chronic_labels:
                        metrics_to_log["train/chronic_episode_count"] = float(
                            len(all_chronic_labels)
                        )
                        metrics_to_log["train/chronic_label_unique_count"] = float(
                            len(set(all_chronic_labels))
                        )
                        metrics_to_log["train/chronic_unique_count"] = float(
                            len(set(all_chronic_labels))
                        )
                        metrics_to_log["train/chronic_fingerprint_unique_count"] = float(
                            len(set(all_chronic_fingerprints))
                        )
                        metrics_to_log["train/chronic_worker_table"] = wb.Table(
                            data=chronic_rows,
                            columns=[
                                "env_idx",
                                "episodes",
                                "unique_labels",
                                "unique_fingerprints",
                                "label_counts",
                                "fingerprint_counts",
                            ],
                        )
                        if args.verbose:
                            summary = " | ".join(
                                (
                                    f"env{env_idx}:"
                                    f"labels={_format_chronic_counter(labels)}; "
                                    "fingerprints="
                                    f"{_format_chronic_counter(fingerprints)}"
                                )
                                for env_idx, (labels, fingerprints) in enumerate(
                                    zip(
                                        rollout_chronic_labels_by_env,
                                        rollout_chronic_fingerprints_by_env,
                                    )
                                )
                            )
                            print(
                                f"train rollout chronics at step {global_step}: "
                                f"{summary}"
                            )
                    for ag in agent_ids:
                        m = train_metrics[ag]
                        if m["entropy"]:
                            metrics_to_log[f"train/entropy_{ag}"] = float(np.mean(m["entropy"]))
                            metrics_to_log[f"train/approx_kl_{ag}"] = float(np.mean(m["approx_kl"]))
                            metrics_to_log[f"train/pg_loss_{ag}"] = float(np.mean(m["pg_loss"]))
                            metrics_to_log[f"train/clipfrac_{ag}"] = float(np.mean(m["clipfrac"]))
                        actions_flat = actions[ag].long().reshape(-1).cpu().numpy()
                        non_idle_flat = actions_flat != 0
                        metrics_to_log[f"train/frac_action_0_{ag}"] = float(np.mean(actions_flat == 0))
                        metrics_to_log[
                            f"train/explain/action_nonidle_{ag}"
                        ] = float(np.mean(non_idle_flat))
                        if sparse_penalty_enabled:
                            penalty_flat = (
                                sparse_penalties[ag].reshape(-1).detach().cpu().numpy()
                            )
                            safe_flat = (
                                safe_intervention_state.reshape(-1)
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(bool)
                            )
                            metrics_to_log[
                                f"train/intervention_penalty_mean_{ag}"
                            ] = float(np.mean(penalty_flat))
                            if safe_penalty_enabled:
                                if np.any(safe_flat):
                                    metrics_to_log[
                                        f"train/intervention_rate_when_safe_{ag}"
                                    ] = float(np.mean(non_idle_flat[safe_flat]))
                                if np.any(~safe_flat):
                                    metrics_to_log[
                                        f"train/intervention_rate_when_hazard_{ag}"
                                    ] = float(np.mean(non_idle_flat[~safe_flat]))
                        if adaptive_budget_enabled:
                            budget_cost_flat = (
                                intervention_budget_costs[ag]
                                .reshape(-1)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            budget_penalty_flat = (
                                intervention_budget_penalties[ag]
                                .reshape(-1)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            budget_weight_flat = (
                                intervention_budget_weights[ag]
                                .reshape(-1)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            stats = intervention_budget_stats.get(ag, {})
                            metrics_to_log[
                                f"train/intervention_budget_lambda_{ag}"
                            ] = float(stats.get("lambda_after", intervention_lambdas[ag]))
                            metrics_to_log[
                                f"train/intervention_budget_lambda_before_{ag}"
                            ] = float(stats.get("lambda_before", intervention_lambdas[ag]))
                            metrics_to_log[
                                f"train/intervention_budget_lambda_updated_{ag}"
                            ] = float(stats.get("updated", 0.0))
                            metrics_to_log[
                                f"train/intervention_budget_cost_{ag}"
                            ] = float(stats.get("cost", np.mean(budget_cost_flat)))
                            metrics_to_log[
                                f"train/intervention_budget_cost_violation_{ag}"
                            ] = float(
                                stats.get(
                                    "violation",
                                    np.mean(budget_cost_flat)
                                    - float(
                                        getattr(
                                            args, "intervention_budget_target", 0.25
                                        )
                                    ),
                                )
                            )
                            metrics_to_log[
                                f"train/intervention_budget_penalty_mean_{ag}"
                            ] = float(np.mean(budget_penalty_flat))
                            metrics_to_log[
                                f"train/intervention_budget_safety_weight_{ag}"
                            ] = float(np.mean(budget_weight_flat))
                            costly = budget_weight_flat > 0.5
                            if np.any(costly):
                                metrics_to_log[
                                    f"train/intervention_rate_when_budget_costly_{ag}"
                                ] = float(np.mean(non_idle_flat[costly]))
                            if np.any(~costly):
                                metrics_to_log[
                                    f"train/intervention_rate_when_budget_free_{ag}"
                                ] = float(np.mean(non_idle_flat[~costly]))
                        if ag in intervention_gate_metrics:
                            gate_buffers = intervention_gate_metrics[ag]
                            metrics_to_log[
                                f"train/explain/gate_intervened_{ag}"
                            ] = float(np.mean(actions_flat != 0))
                            metrics_to_log[
                                f"train/intervention_gate_do_nothing_frac_{ag}"
                            ] = float(np.mean(actions_flat == 0))
                            metrics_to_log[
                                f"train/intervention_gate_intervene_frac_{ag}"
                            ] = float(np.mean(actions_flat != 0))
                            metrics_to_log[
                                f"train/intervention_gate_prob_do_nothing_{ag}"
                            ] = float(gate_buffers["prob_do_nothing"].mean().item())
                            metrics_to_log[
                                f"train/intervention_gate_prob_intervene_{ag}"
                            ] = float(gate_buffers["prob_intervene"].mean().item())
                            metrics_to_log[
                                f"train/intervention_gate_entropy_{ag}"
                            ] = float(gate_buffers["gate_entropy"].mean().item())
                            metrics_to_log[
                                f"train/nonidle_action_entropy_{ag}"
                            ] = float(
                                gate_buffers["nonidle_action_entropy"].mean().item()
                            )
                        illegal_total = max(illegal_action_totals[ag], 1)
                        metrics_to_log[f"train/illegal_action_rate_{ag}"] = (
                            illegal_action_counts[ag] / illegal_total
                        )
                        metrics_to_log[f"train/illegal_action_count_{ag}"] = float(
                            illegal_action_counts[ag]
                        )

                    # Explained variance of the shared critic against the selected
                    # value target.
                    y_pred = values.reshape(-1).cpu().numpy()
                    y_true_t = (
                        b_critic_returns
                        if optimize_critic_updates
                        else returns[agent_ids[0]].reshape(-1)
                    )
                    y_true = y_true_t.cpu().numpy()
                    var_y = float(np.var(y_true))
                    metrics_to_log["train/explained_variance"] = (
                        float("nan") if var_y == 0.0 else 1.0 - float(np.var(y_true - y_pred)) / var_y
                    )

                    if _should_log_training_action_trace(
                        args, iteration, init_rollout
                    ):
                        trace_records = _build_training_action_trace_records(
                            args,
                            actions,
                            rewards,
                            dones,
                            agent_ids,
                            iteration,
                            global_step,
                        )
                        if trace_records:
                            decoded_actions = {}
                            if getattr(args, "trace_rollout_decode_actions", True):
                                action_ids_by_agent = collect_unique_action_ids(
                                    trace_records, agent_ids
                                )
                                decoded_actions = decode_action_ids_safely(
                                    lambda ids: envs.decode_action_ids(
                                        ids,
                                        env_idx=getattr(
                                            args, "trace_rollout_env_idx", 0
                                        ),
                                    ),
                                    action_ids_by_agent,
                                )
                            metrics_to_log[
                                "train/rollout_action_trace"
                            ] = build_action_trace_table(
                                trace_records, agent_ids, decoded_actions
                            )

                    logger.log_train_metrics(global_step, metrics_to_log)

                # The environment, optimizers, schedules, and normalizers now all
                # represent the same unambiguous post-update boundary.
                at_exact_boundary = True
                if args.checkpoint and (
                    not saved_boundary_this_process
                    or time() - last_ckpt_time >= 3600
                ):
                    _save_exact_boundary(iteration)
                    saved_boundary_this_process = True
                    last_ckpt_time = time()

                # If we reach the node's time limit, we just exit the training loop, save metrics and ckpt
                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            if args.checkpoint:
                if at_exact_boundary:
                    _save_exact_boundary(iteration)
                else:
                    print(
                        "Training stopped inside a rollout/update. The last exact "
                        "post-update checkpoint was kept; no ambiguous checkpoint "
                        "was written over it."
                    )
            if logger:
                logger.close()
            envs.close()
