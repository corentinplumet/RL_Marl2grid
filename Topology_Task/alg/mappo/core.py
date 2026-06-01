from time import time

from .agent import Actor, Critic
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
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
    python_state = rnd.getstate()
    numpy_state = np.random.get_state()
    torch_state = th.get_rng_state()
    cuda_states = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
    try:
        return evaluator.evaluate(global_step, actors)
    finally:
        rnd.setstate(python_state)
        np.random.set_state(numpy_state)
        th.set_rng_state(torch_state)
        if cuda_states is not None:
            th.cuda.set_rng_state_all(cuda_states)


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
            args = ap.Namespace(**vars(args), **vars(get_alg_args()))

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
        init_rollout = 1 if not ckpt.resumed else ckpt.loaded_run["last_rollout"]

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
        if continuous_actions:
            raise ("Redispatching actions are not yet implemented")
        actor_optim = optim.Adam(actor_params, lr=args.actor_lr, eps=1e-5)
        critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)

        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run["actor_optim"])
            critic_optim.load_state_dict(ckpt.loaded_run["critic_optim"])

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
        for id in agent_ids:
            observations[id] = zeros_like_with_leading(
                strip_state_graph(next_obs[id]), (args.n_steps,), device=device
            )
            actions[id] = th.zeros((args.n_steps, args.n_envs)).to(
                device
            )  # Assuming discrete actions
            logprobs[id] = th.zeros((args.n_steps, args.n_envs)).to(device)
            rewards[id] = th.zeros((args.n_steps, args.n_envs)).to(device)

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
        best_test_survival = -np.inf

        global_step = 0 if not ckpt.resumed else ckpt.loaded_run["global_step"]
        start_time = start_time
        last_ckpt_time = start_time  # <-- track last checkpoint timestamp

        reward_normalizer = (
            ReturnNormalizer(args.n_envs, args.gamma) if getattr(args, "norm_reward", False) else None
        )
        try:
            for iteration in range(init_rollout, n_rollouts + 1):
                # Annealing the rate if instructed to do so
                if args.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / n_rollouts
                    actor_optim.param_groups[0]["lr"] = frac * args.actor_lr
                    critic_optim.param_groups[0]["lr"] = frac * args.critic_lr
                entropy_coef = _scheduled_entropy_coef(args, global_step)
                action0_bonus = _scheduled_action0_bonus(args, global_step)
                illegal_action_counts = {agent: 0 for agent in agent_ids}
                illegal_action_totals = {agent: 0 for agent in agent_ids}

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

                    next_obs, reward, next_terminations, next_truncations, infos = (
                        envs.step(action)
                    )
                    step_infos = (
                        list(infos)
                        if isinstance(infos, (list, tuple))
                        else [infos]
                    )
                    for agent in agent_ids:
                        illegal_action_totals[agent] += len(step_infos)
                        illegal_action_counts[agent] += sum(
                            int(_agent_took_illegal_action(info, agent))
                            for info in step_infos
                        )

                    if reward_normalizer is not None:
                        done_np = np.logical_or(
                            next_terminations[agent_ids[0]],
                            next_truncations[agent_ids[0]],
                        )
                        # Reward is identical across agents in this env (joint reward),
                        # so normalize once and broadcast to every agent's stream.
                        normed = reward_normalizer(
                            np.asarray(reward[agent_ids[0]]), done_np
                        )
                        for agent in agent_ids:
                            reward[agent] = normed

                    reward = cast_np_to_tensors(reward, device)
                    for agent in agent_ids:
                        rewards[agent][step] = reward[agent]

                    dones[step] = th.tensor(
                        np.logical_or(
                            next_terminations[agent_ids[0]],
                            next_truncations[agent_ids[0]],
                        )
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
                        if train_evaluator is not None:
                            train_evaluator.env.env.set_obs_stats(obs_stats)
                            _evaluate_preserving_training_rng(
                                train_evaluator, global_step, actors
                            )
                        evaluator.env.env.set_obs_stats(obs_stats)
                        eval_survival = _evaluate_preserving_training_rng(
                            evaluator, global_step, actors
                        )
                        if split_chronics and eval_survival > best_test_survival:
                            best_test_survival = eval_survival
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
                                )
                                ckpt.save_as("best_test_" + run_name)
                        if args.verbose:
                            print(f"SPS={int(global_step / (time() - start_time))}")

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

                # --- HOURLY CHECKPOINT (wall clock) ---
                if time() - last_ckpt_time >= 3600:
                    ckpt.set_record(
                        args,
                        actors,
                        critic,
                        global_step,
                        actor_optim,
                        critic_optim,
                        "" if not logger else logger.wb_path,
                        iteration,
                    )
                    ckpt.save()  # overwrites the previous checkpoint
                    last_ckpt_time = time()
                # --------------------------------------

                b_values = values.reshape(-1)
                b_joint_obs = flatten_rollout_obs(joint_observations)

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
                    b_returns = returns[agent].reshape(-1)

                    # Optimizing the policy and value network
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

                            # Value loss
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

                            # Accumulate per-minibatch training metrics
                            train_metrics[agent]["entropy"].append(float(entropy_loss.detach()))
                            train_metrics[agent]["pg_loss"].append(float(pg_loss.detach()))
                            train_metrics[agent]["approx_kl"].append(float(approx_kl.detach()))
                            train_metrics[agent]["clipfrac"].append(float(clipfracs[-1]))
                            v_loss_history.append(float(v_loss.detach()))

                        if args.target_kl is not None and approx_kl > args.target_kl:
                            break

                # Log per-rollout training metrics to wandb
                if logger is not None:
                    metrics_to_log: Dict[str, float] = {
                        "train/lr_actor": float(actor_optim.param_groups[0]["lr"]),
                        "train/lr_critic": float(critic_optim.param_groups[0]["lr"]),
                        "train/entropy_coef": float(entropy_coef),
                        "train/action0_logit_bonus": float(action0_bonus),
                        "train/v_loss": float(np.mean(v_loss_history)) if v_loss_history else 0.0,
                    }
                    for ag in agent_ids:
                        m = train_metrics[ag]
                        if m["entropy"]:
                            metrics_to_log[f"train/entropy_{ag}"] = float(np.mean(m["entropy"]))
                            metrics_to_log[f"train/approx_kl_{ag}"] = float(np.mean(m["approx_kl"]))
                            metrics_to_log[f"train/pg_loss_{ag}"] = float(np.mean(m["pg_loss"]))
                            metrics_to_log[f"train/clipfrac_{ag}"] = float(np.mean(m["clipfrac"]))
                        actions_flat = actions[ag].long().reshape(-1).cpu().numpy()
                        metrics_to_log[f"train/frac_action_0_{ag}"] = float(np.mean(actions_flat == 0))
                        illegal_total = max(illegal_action_totals[ag], 1)
                        metrics_to_log[f"train/illegal_action_rate_{ag}"] = (
                            illegal_action_counts[ag] / illegal_total
                        )
                        metrics_to_log[f"train/illegal_action_count_{ag}"] = float(
                            illegal_action_counts[ag]
                        )

                    # Explained variance of the (shared) critic against agent_0's returns
                    # (all agents have identical returns under shared joint reward)
                    y_pred = values.reshape(-1).cpu().numpy()
                    y_true = returns[agent_ids[0]].reshape(-1).cpu().numpy()
                    var_y = float(np.var(y_true))
                    metrics_to_log["train/explained_variance"] = (
                        float("nan") if var_y == 0.0 else 1.0 - float(np.var(y_true - y_pred)) / var_y
                    )

                    logger.log_train_metrics(global_step, metrics_to_log)

                # If we reach the node's time limit, we just exit the training loop, save metrics and ckpt
                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            if args.checkpoint:
                # Save the checkpoint and logger data
                ckpt.set_record(
                    args,
                    actors,
                    critic,
                    global_step,
                    actor_optim,
                    critic_optim,
                    "" if not logger else logger.wb_path,
                    iteration,
                )
                ckpt.save()
            if logger:
                logger.close()
            envs.close()
