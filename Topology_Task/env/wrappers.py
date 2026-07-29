import multiprocessing as mp
from collections import deque
import copy
import time

from common.imports import *
from common.utils import split_action_tensor_dict


_ACTION_REPLAY_CHECKPOINT = "action_replay_v1"
_PICKLED_ENV_CHECKPOINT = "pickled_env_v1"


def _capture_worker_rng_state() -> Dict[str, Any]:
    """Capture the RNG streams owned by one environment worker."""
    return {
        "python": rnd.getstate(),
        "numpy": np.random.get_state(),
        "torch": th.get_rng_state().cpu().numpy().copy(),
    }


def _restore_worker_rng_state(state: Dict[str, Any]) -> None:
    """Restore the RNG streams owned by one environment worker."""
    rnd.setstate(state["python"])
    np.random.set_state(state["numpy"])
    th.set_rng_state(th.from_numpy(np.asarray(state["torch"]).copy()))


def _checkpoint_action_component(value: Any) -> np.ndarray:
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        if array.size:
            int32 = np.iinfo(np.int32)
            minimum = int(array.min())
            maximum = int(array.max())
            if minimum < int32.min or maximum > int32.max:
                raise ValueError(
                    "An integer action cannot be represented as int32: "
                    f"[{minimum}, {maximum}]."
                )
        array = array.astype(np.int32, copy=False)
    return array.copy(order="C")


def _new_action_log() -> Dict[str, Any]:
    return {"count": 0, "components": {}}


def _append_checkpoint_action(
    action_log: Dict[str, Any], action: Dict[str, Any]
) -> None:
    """Append one multi-agent action using compact, fixed-width byte buffers."""
    components = {
        key: _checkpoint_action_component(value) for key, value in action.items()
    }
    expected_keys = set(action_log["components"])
    actual_keys = set(components)
    if action_log["count"] and actual_keys != expected_keys:
        raise ValueError(
            "Action keys changed during an exact-checkpoint replay log: "
            f"{sorted(actual_keys)} != {sorted(expected_keys)}."
        )

    for key, component in components.items():
        if key not in action_log["components"]:
            action_log["components"][key] = {
                "shape": tuple(component.shape),
                "dtype": component.dtype.str,
                "data": bytearray(),
            }
        target = action_log["components"][key]
        if tuple(component.shape) != tuple(target["shape"]):
            raise ValueError(
                f"Action shape for {key} changed from {target['shape']} "
                f"to {component.shape}."
            )
        if component.dtype.str != target["dtype"]:
            component = component.astype(np.dtype(target["dtype"]), copy=False)
        target["data"].extend(component.tobytes(order="C"))
    action_log["count"] += 1


def _pack_action_log(action_log: Dict[str, Any]) -> Dict[str, Any]:
    """Make the mutable worker log safe and efficient to send through a pipe."""
    return {
        "count": int(action_log["count"]),
        "components": {
            key: {
                "shape": tuple(component["shape"]),
                "dtype": component["dtype"],
                "data": bytes(component["data"]),
            }
            for key, component in action_log["components"].items()
        },
    }


def _unpack_action_log(packed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "count": int(packed["count"]),
        "components": {
            key: {
                "shape": tuple(component["shape"]),
                "dtype": component["dtype"],
                "data": bytearray(component["data"]),
            }
            for key, component in packed["components"].items()
        },
    }


def _iter_checkpoint_actions(packed: Dict[str, Any]):
    count = int(packed["count"])
    arrays = {}
    for key, component in packed["components"].items():
        shape = tuple(component["shape"])
        dtype = np.dtype(component["dtype"])
        expected_values = count * int(np.prod(shape, dtype=np.int64))
        if not shape:
            expected_values = count
        values = np.frombuffer(component["data"], dtype=dtype)
        if values.size != expected_values:
            raise ValueError(
                f"Corrupt action replay data for {key}: got {values.size} values, "
                f"expected {expected_values}."
            )
        arrays[key] = values.reshape((count,) + shape)

    for index in range(count):
        action = {}
        for key, values in arrays.items():
            value = values[index]
            action[key] = value.item() if value.ndim == 0 else value.copy()
        yield action


def _checkpoint_values_equal(actual: Any, expected: Any) -> bool:
    """Compare nested observations without tolerating trajectory drift."""
    if isinstance(actual, dict) or isinstance(expected, dict):
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return False
        if actual.keys() != expected.keys():
            return False
        return all(
            _checkpoint_values_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or not isinstance(
            expected, (list, tuple)
        ):
            return False
        return len(actual) == len(expected) and all(
            _checkpoint_values_equal(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(actual, th.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(expected, th.Tensor):
        expected = expected.detach().cpu().numpy()
    try:
        return bool(
            np.array_equal(
                np.asarray(actual),
                np.asarray(expected),
                equal_nan=True,
            )
        )
    except (TypeError, ValueError):
        return actual == expected


class AsyncMultiAgentVecEnv:
    def __init__(self, env_fns: List[Callable], context: str = "spawn"):
        self.num_envs = len(env_fns)
        ctx = mp.get_context(context)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [
            ctx.Process(target=self._worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.waiting = False
        self.closed = False

        # Probe observation and action spaces from first environment
        self.remotes[0].send(("get_spaces", None))
        (
            self.observation_space,
            self.action_space,
            self.graph_specs,
            self.token_specs,
        ) = (
            self.remotes[0].recv()
        )

    def reset(self, seed: int = None):
        for remote in self.remotes:
            remote.send(("reset", {"seed": seed}))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return self._stack_dicts(obs), list(infos)

    def step_async(self, actions: List[Dict[str, Any]]):
        # Have to split the actions dict into a list of dicts (one for each env).
        # Move any tensors to CPU first — multiprocessing pipes can't share non-CPU storage.
        actions = {
            k: v.detach().cpu() if isinstance(v, th.Tensor) else v
            for k, v in actions.items()
        }
        for remote, action in zip(self.remotes, split_action_tensor_dict(actions)):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, truncs, infos = zip(*results)
        self.waiting = False
        return (
            self._stack_dicts(obs),
            self._stack_dicts(rews),
            self._stack_dicts(dones),
            self._stack_dicts(truncs),
            list(infos),
        )

    def step(self, actions: List[Dict[str, Any]]):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            self.step_wait()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_obs_stats(self):
        """Collect normalization stats and merge them with parallel Welford.

        Entries include per-agent flat-observation statistics and, when graph
        running normalization is enabled, shared graph node/edge statistics.
        ``var`` is the running sum of squared deviations (M2), matching the
        representation used inside ``MAEnvWrapper``.
        """
        for remote in self.remotes:
            remote.send(("get_obs_stats", None))
        per_worker = [remote.recv() for remote in self.remotes]

        merged: Dict[str, Dict[str, Any]] = {}
        for stats in per_worker:
            for agent_id, s in stats.items():
                if s["mean"] is None:
                    continue
                if agent_id not in merged:
                    merged[agent_id] = {
                        "count": float(s["count"]),
                        "mean": s["mean"].copy(),
                        "var": s["var"].copy(),
                    }
                    continue
                a = merged[agent_id]
                n_a = a["count"]
                n_b = float(s["count"])
                n = n_a + n_b
                delta = s["mean"] - a["mean"]
                a["mean"] = a["mean"] + delta * (n_b / n)
                a["var"] = a["var"] + s["var"] + (delta ** 2) * (n_a * n_b / n)
                a["count"] = n
        return merged

    def set_obs_stats(self, stats):
        """Broadcast flat and graph normalization statistics to every worker."""
        for remote in self.remotes:
            remote.send(("set_obs_stats", stats))
        for remote in self.remotes:
            remote.recv()

    def get_checkpoint_state(self) -> List[Dict[str, Any]]:
        """Snapshot every worker at a reproducible environment boundary."""
        if self.waiting:
            raise RuntimeError(
                "Cannot checkpoint while an asynchronous environment step is pending."
            )
        for remote in self.remotes:
            remote.send(("get_checkpoint_state", None))
        results = [remote.recv() for remote in self.remotes]
        errors = [
            result.get("error", "unknown worker error")
            for result in results
            if not result.get("ok", False)
        ]
        if errors:
            raise RuntimeError(
                "Could not capture the live environment state required for an exact "
                "continuation:\n- " + "\n- ".join(errors)
            )
        return [result["state"] for result in results]

    def set_checkpoint_state(self, states: List[Dict[str, Any]]) -> None:
        """Reconstruct every live worker from a previously captured state."""
        if self.waiting:
            raise RuntimeError(
                "Cannot restore while an asynchronous environment step is pending."
            )
        if len(states) != self.num_envs:
            raise ValueError(
                "Checkpoint environment count does not match this run: "
                f"{len(states)} != {self.num_envs}."
            )
        for remote, state in zip(self.remotes, states):
            remote.send(("set_checkpoint_state", state))
        results = [remote.recv() for remote in self.remotes]
        errors = [
            result.get("error", "unknown worker error")
            for result in results
            if not result.get("ok", False)
        ]
        if errors:
            raise RuntimeError(
                "Could not restore the exact environment state:\n- "
                + "\n- ".join(errors)
            )

    def get_current_max_rho(self) -> np.ndarray:
        """Collect each worker's pre-action max rho."""
        if self.waiting:
            raise RuntimeError("Cannot query max rho while an async env step is pending.")
        for remote in self.remotes:
            remote.send(("get_current_max_rho", None))
        return np.asarray([remote.recv() for remote in self.remotes], dtype=np.float32)

    def get_current_agent_max_rho(self) -> Dict[str, np.ndarray]:
        """Collect each worker's per-agent local pre-action max rho."""
        if self.waiting:
            raise RuntimeError(
                "Cannot query local max rho while an async env step is pending."
            )
        for remote in self.remotes:
            remote.send(("get_current_agent_max_rho", None))
        per_worker = [remote.recv() for remote in self.remotes]
        agent_ids = per_worker[0].keys() if per_worker else []
        return {
            agent_id: np.asarray(
                [worker.get(agent_id, float("nan")) for worker in per_worker],
                dtype=np.float32,
            )
            for agent_id in agent_ids
        }

    def get_current_rho_summary(self) -> List[Dict[str, Any]]:
        """Collect each worker's global pre-action max rho line summary."""
        if self.waiting:
            raise RuntimeError(
                "Cannot query rho summaries while an async env step is pending."
            )
        for remote in self.remotes:
            remote.send(("get_current_rho_summary", None))
        return [remote.recv() for remote in self.remotes]

    def get_current_agent_rho_summary(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect each worker's per-agent local max rho line summary."""
        if self.waiting:
            raise RuntimeError(
                "Cannot query local rho summaries while an async env step is pending."
            )
        for remote in self.remotes:
            remote.send(("get_current_agent_rho_summary", None))
        per_worker = [remote.recv() for remote in self.remotes]
        agent_ids = per_worker[0].keys() if per_worker else []
        return {
            agent_id: [worker.get(agent_id, {}) for worker in per_worker]
            for agent_id in agent_ids
        }

    def decode_action_ids(
        self, action_ids_by_agent: Dict[str, List[int]], env_idx: int = 0
    ) -> Dict[str, Dict[int, str]]:
        """Decode discrete action ids in one worker environment."""
        if self.waiting:
            raise RuntimeError("Cannot decode actions while an async env step is pending.")
        env_idx = int(np.clip(env_idx, 0, self.num_envs - 1))
        self.remotes[env_idx].send(("decode_action_ids", action_ids_by_agent))
        return self.remotes[env_idx].recv()

    def simulate_action_outcomes(
        self,
        requests: List[Dict[str, Any]],
        env_idx: int = 0,
        time_step: int = 1,
        num_workers: int = 1,
    ) -> List[Dict[str, Any]]:
        """Simulate unilateral action requests in one worker environment."""
        if self.waiting:
            raise RuntimeError(
                "Cannot simulate actions while an async env step is pending."
            )
        env_idx = int(np.clip(env_idx, 0, self.num_envs - 1))
        self.remotes[env_idx].send(
            (
                "simulate_action_outcomes",
                {
                    "requests": requests,
                    "time_step": time_step,
                    "num_workers": num_workers,
                },
            )
        )
        return self.remotes[env_idx].recv()

    def _stack_dicts(self, dicts: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        # Transpose list of dicts into dict of lists, then stack
        stacked = {}
        keys = dicts[0].keys()
        for k in keys:
            stacked[k] = self._stack_values([d[k] for d in dicts])
        return stacked

    def _stack_values(self, values: List[Any]):
        if isinstance(values[0], dict):
            return {
                key: self._stack_values([value[key] for value in values])
                for key in values[0].keys()
            }
        return np.stack(values)

    @staticmethod
    def _worker(remote, parent_remote, env_fn_wrapper):
        parent_remote.close()
        constructor_rng_state = _capture_worker_rng_state()
        env = env_fn_wrapper.fn()
        replay_reset_rng_state = None
        replay_reset_kwargs = None
        replay_initial_obs_stats = None
        replay_action_log = None
        last_observation = None
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == "reset":
                    reset_rng_state = _capture_worker_rng_state()
                    initial_obs_stats = (
                        copy.deepcopy(env.get_obs_stats())
                        if hasattr(env, "get_obs_stats")
                        else None
                    )
                    obs, info = env.reset(**data)
                    replay_reset_rng_state = reset_rng_state
                    replay_reset_kwargs = copy.deepcopy(data)
                    replay_initial_obs_stats = initial_obs_stats
                    replay_action_log = _new_action_log()
                    last_observation = obs
                    remote.send((obs, info))
                elif cmd == "step":
                    if replay_action_log is not None:
                        _append_checkpoint_action(replay_action_log, data)
                    observation, reward, terminated, truncated, info = env.step(data)
                    if terminated["agent_0"] or truncated["agent_0"]:
                        old_observation, old_info = observation, info
                        observation, info = env.reset()
                        info["final_observation"] = old_observation
                        info["final_info"] = old_info
                    last_observation = observation
                    remote.send((observation, reward, terminated, truncated, info))
                elif cmd == "close":
                    remote.close()
                    break
                elif cmd == "get_spaces":
                    remote.send(
                        (
                            env.observation_space,
                            env.action_space,
                            getattr(env, "graph_specs", None),
                            getattr(env, "token_specs", None),
                        )
                    )
                elif cmd == "get_obs_stats":
                    remote.send(env.get_obs_stats())
                elif cmd == "set_obs_stats":
                    env.set_obs_stats(data)
                    remote.send(None)
                elif cmd == "get_checkpoint_state":
                    try:
                        worker_rng_state = _capture_worker_rng_state()
                        if replay_action_log is not None:
                            state = {
                                "mode": _ACTION_REPLAY_CHECKPOINT,
                                "constructor_rng_state": constructor_rng_state,
                                "reset_rng_state": replay_reset_rng_state,
                                "reset_kwargs": replay_reset_kwargs,
                                "initial_obs_stats": replay_initial_obs_stats,
                                "actions": _pack_action_log(replay_action_log),
                                "expected_observation": copy.deepcopy(last_observation),
                                "worker_rng_state": worker_rng_state,
                            }
                        else:
                            # Compatibility for exact checkpoints written before
                            # action replay was introduced. New checkpoints always
                            # use replay after the first explicit reset.
                            import cloudpickle

                            state = {
                                "mode": _PICKLED_ENV_CHECKPOINT,
                                "env": cloudpickle.dumps(env),
                                "python_rng": worker_rng_state["python"],
                                "numpy_rng": worker_rng_state["numpy"],
                                "torch_rng": worker_rng_state["torch"],
                            }
                        remote.send({"ok": True, "state": state})
                    except Exception as exc:
                        remote.send(
                            {
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                elif cmd == "set_checkpoint_state":
                    try:
                        mode = data.get("mode", _PICKLED_ENV_CHECKPOINT)
                        if mode == _ACTION_REPLAY_CHECKPOINT:
                            restored_constructor_rng = data["constructor_rng_state"]
                            _restore_worker_rng_state(restored_constructor_rng)
                            restored_env = env_fn_wrapper.fn()
                            try:
                                initial_obs_stats = data.get("initial_obs_stats")
                                if initial_obs_stats is not None and hasattr(
                                    restored_env, "set_obs_stats"
                                ):
                                    restored_env.set_obs_stats(initial_obs_stats)
                                _restore_worker_rng_state(data["reset_rng_state"])
                                observation, _ = restored_env.reset(
                                    **data["reset_kwargs"]
                                )
                                for action in _iter_checkpoint_actions(data["actions"]):
                                    (
                                        observation,
                                        _,
                                        terminated,
                                        truncated,
                                        _,
                                    ) = restored_env.step(action)
                                    if terminated["agent_0"] or truncated["agent_0"]:
                                        observation, _ = restored_env.reset()

                                if not _checkpoint_values_equal(
                                    observation,
                                    data["expected_observation"],
                                ):
                                    raise RuntimeError(
                                        "Action replay did not reconstruct the "
                                        "saved observation exactly."
                                    )
                            except Exception:
                                restored_env.close()
                                raise

                            old_env = env
                            env = restored_env
                            old_env.close()
                            constructor_rng_state = restored_constructor_rng
                            replay_reset_rng_state = data["reset_rng_state"]
                            replay_reset_kwargs = copy.deepcopy(data["reset_kwargs"])
                            replay_initial_obs_stats = copy.deepcopy(
                                data.get("initial_obs_stats")
                            )
                            replay_action_log = _unpack_action_log(data["actions"])
                            last_observation = copy.deepcopy(observation)
                            _restore_worker_rng_state(data["worker_rng_state"])
                        elif mode == _PICKLED_ENV_CHECKPOINT:
                            import cloudpickle

                            restored_env = cloudpickle.loads(data["env"])
                            old_env = env
                            env = restored_env
                            old_env.close()
                            rnd.setstate(data["python_rng"])
                            np.random.set_state(data["numpy_rng"])
                            th.set_rng_state(
                                th.from_numpy(np.asarray(data["torch_rng"]).copy())
                            )
                            replay_reset_rng_state = None
                            replay_reset_kwargs = None
                            replay_initial_obs_stats = None
                            replay_action_log = None
                            last_observation = None
                        else:
                            raise ValueError(
                                f"Unknown environment checkpoint mode: {mode!r}."
                            )
                        remote.send({"ok": True})
                    except Exception as exc:
                        remote.send(
                            {
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                elif cmd == "get_current_max_rho":
                    remote.send(env.get_current_max_rho())
                elif cmd == "get_current_agent_max_rho":
                    remote.send(env.get_current_agent_max_rho())
                elif cmd == "get_current_rho_summary":
                    remote.send(env.get_current_rho_summary())
                elif cmd == "get_current_agent_rho_summary":
                    remote.send(env.get_current_agent_rho_summary())
                elif cmd == "decode_action_ids":
                    remote.send(env.decode_action_ids(data))
                elif cmd == "simulate_action_outcomes":
                    remote.send(
                        env.simulate_action_outcomes(
                            data["requests"],
                            time_step=data.get("time_step", 1),
                            num_workers=data.get("num_workers", 1),
                        )
                    )
                else:
                    raise NotImplementedError(f"Unknown command: {cmd}")
        except KeyboardInterrupt:
            print("Worker interrupted")
        finally:
            env.close()


class CloudpickleWrapper:
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        import pickle
        self.fn = pickle.loads(ob)


class RecordEpisodeStatistics(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:      # This is our case since we Record Episode Statistics only in the eval environment
            self.num_envs = 1
            self.is_vector_env = False

        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def decode_action_ids(
        self, action_ids_by_agent: Dict[str, List[int]]
    ) -> Dict[str, Dict[int, str]]:
        return self.env.decode_action_ids(action_ids_by_agent)

    def get_current_max_rho(self) -> float:
        return self.env.get_current_max_rho()

    def get_current_agent_max_rho(self) -> Dict[str, float]:
        return self.env.get_current_agent_max_rho()

    def get_current_rho_summary(self) -> Dict[str, Any]:
        return self.env.get_current_rho_summary()

    def get_current_agent_rho_summary(self) -> Dict[str, Dict[str, Any]]:
        return self.env.get_current_agent_rho_summary()

    def reshuffle_chronics(self, seed: Optional[int] = None) -> None:
        return self.env.reshuffle_chronics(seed=seed)

    def set_chronic_window(self, start: int, count: int) -> Dict[str, int]:
        return self.env.set_chronic_window(start=start, count=count)

    def simulate_action_outcomes(
        self,
        requests: List[Dict[str, Any]],
        *,
        time_step: int = 1,
        num_workers: int = 1,
    ) -> List[Dict[str, Any]]:
        return self.env.simulate_action_outcomes(
            requests,
            time_step=time_step,
            num_workers=num_workers,
        )

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards['agent_0']
        self.episode_lengths += 1
        dones = np.logical_or(terminations['agent_0'], truncations['agent_0'])
        num_dones = np.sum(dones)

        if not num_dones and ("episode" in infos):
            del infos["episode"]
            
        if num_dones:
            infos["episode"] = {
                "r": np.where(dones, self.episode_returns, 0.0),
                "l": np.where(dones, self.episode_lengths, 0),
                "t": np.where(
                    dones,
                    np.round(time.perf_counter() - self.episode_start_times, 6),
                    0.0,
                ),
            }
            if self.is_vector_env:
                infos["_episode"] = np.where(dones, True, False)

            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
