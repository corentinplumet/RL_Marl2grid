import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch as th

from common.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    EXACT_BOUNDARY_PHASE,
    MAPPOCheckpoint,
    capture_rng_state,
    exact_resume_state,
    initialize_resume_observation,
    restore_rng_state,
)
from common.utils import ReturnNormalizer
from env.wrappers import AsyncMultiAgentVecEnv


class DeterministicCheckpointEnv:
    """Small deterministic env that deliberately cannot be pickled."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "agent_0": gym.spaces.Box(
                    low=-1e6, high=1e6, shape=(2,), dtype=np.float32
                )
            }
        )
        self.action_space = gym.spaces.Dict({"agent_0": gym.spaces.Discrete(4)})
        self.graph_specs = None
        self.token_specs = None
        self.value = 0
        self.episode_steps = 0
        self.obs_bias = 0.0
        self.local_rng = np.random.default_rng(0)
        self.unpickleable = (item for item in range(1))

    def _observation(self):
        return {
            "agent_0": np.asarray(
                [self.value + self.obs_bias, self.local_rng.random()],
                dtype=np.float32,
            )
        }

    def reset(self, seed=None):
        self.value = 0
        self.episode_steps = 0
        if seed is not None:
            self.local_rng = np.random.default_rng(seed)
        return self._observation(), {}

    def step(self, action):
        # Exercise both an RNG stored on the env and the worker-global NumPy RNG.
        self.episode_steps += 1
        self.value += (
            int(action["agent_0"])
            + int(self.local_rng.integers(0, 5))
            + int(np.random.randint(0, 5))
            + int(random.randint(0, 4))
            + int(th.randint(0, 5, ()).item())
        )
        observation = self._observation()
        reward = {"agent_0": float(self.value)}
        done = {"agent_0": self.episode_steps == 2}
        return observation, reward, done, done.copy(), {}

    def get_obs_stats(self):
        return {"bias": self.obs_bias}

    def set_obs_stats(self, stats):
        self.obs_bias = float(stats["bias"])

    def close(self):
        pass


def make_checkpoint_env():
    return DeterministicCheckpointEnv()


class ExactCheckpointTests(unittest.TestCase):
    def test_fast_resume_resets_workers_after_restoring_observation_stats(self):
        class FakeVectorEnv:
            def __init__(self):
                self.calls = []

            def set_obs_stats(self, stats):
                self.calls.append(("set_obs_stats", stats))

            def set_checkpoint_state(self, state):
                self.calls.append(("set_checkpoint_state", state))

            def reset(self):
                self.calls.append(("reset", None))
                return {"agent_0": np.asarray([[3.0]], dtype=np.float32)}, {}

        envs = FakeVectorEnv()
        next_obs, restored_exact = initialize_resume_observation(
            envs,
            device=th.device("cpu"),
            loaded_exact_state={
                "environment_states": ["saved-worker"],
                "next_obs": {"agent_0": th.tensor([[99.0]])},
            },
            reset_environments=True,
            normalization_enabled=True,
            saved_obs_stats={"agent_0": {"mean": 7.0}},
        )

        self.assertFalse(restored_exact)
        self.assertEqual(
            envs.calls,
            [
                ("set_obs_stats", {"agent_0": {"mean": 7.0}}),
                ("reset", None),
            ],
        )
        th.testing.assert_close(next_obs["agent_0"], th.tensor([[3.0]]))

    def test_default_resume_still_restores_exact_environment_state(self):
        class FakeVectorEnv:
            def __init__(self):
                self.calls = []

            def set_obs_stats(self, stats):
                self.calls.append(("set_obs_stats", stats))

            def set_checkpoint_state(self, state):
                self.calls.append(("set_checkpoint_state", state))

            def reset(self):
                raise AssertionError("exact resume must not reset workers")

        envs = FakeVectorEnv()
        next_obs, restored_exact = initialize_resume_observation(
            envs,
            device=th.device("cpu"),
            loaded_exact_state={
                "environment_states": ["saved-worker"],
                "next_obs": {"agent_0": th.tensor([[99.0]])},
            },
            reset_environments=False,
            normalization_enabled=True,
            saved_obs_stats={"agent_0": {"mean": 7.0}},
        )

        self.assertTrue(restored_exact)
        self.assertEqual(
            envs.calls,
            [("set_checkpoint_state", ["saved-worker"])],
        )
        th.testing.assert_close(next_obs["agent_0"], th.tensor([[99.0]]))

    def test_global_rng_state_round_trip(self):
        original_state = capture_rng_state()
        try:
            random.seed(17)
            np.random.seed(23)
            th.manual_seed(29)
            state = capture_rng_state()

            expected = (
                random.random(),
                np.random.random(),
                th.rand(4),
            )
            for _ in range(10):
                random.random()
                np.random.random()
                th.rand(1)

            restore_rng_state(state)
            actual = (
                random.random(),
                np.random.random(),
                th.rand(4),
            )
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            th.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        finally:
            restore_rng_state(original_state)

    def test_claimed_exact_checkpoint_is_validated_strictly(self):
        record = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint_boundary": {
                "phase": EXACT_BOUNDARY_PHASE,
                "exact": True,
                "next_rollout": 8,
            },
            "resume_state": {
                "global_step": 100,
                "next_rollout": 8,
                "rng_state": {},
                "environment_states": [],
                "next_obs": {},
                "reward_normalizers": None,
            },
        }
        self.assertIs(exact_resume_state(record), record["resume_state"])

        del record["resume_state"]["next_obs"]
        with self.assertRaisesRegex(ValueError, "missing: next_obs"):
            exact_resume_state(record)

    def test_device_migration_skips_only_unavailable_accelerator_rng(self):
        state = capture_rng_state()
        state["torch_cuda"] = [th.arange(8, dtype=th.uint8)]
        with patch(
            "common.checkpoint.th.cuda.is_available", return_value=False
        ):
            with self.assertRaisesRegex(
                RuntimeError, "resume-allow-device-migration"
            ):
                restore_rng_state(state)
            with self.assertWarnsRegex(
                RuntimeWarning, "Skipping saved CUDA RNG state"
            ):
                restore_rng_state(state, allow_device_migration=True)

    def test_device_migration_loads_checkpoint_storage_through_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gpu_checkpoint.tar"
            path.touch()
            args = SimpleNamespace(
                resume_run_name=str(path),
                resume_allow_device_migration=True,
                resume_delete_checkpoint_after_load=False,
            )
            with patch(
                "common.checkpoint.th.load", return_value={"loaded": True}
            ) as mocked_load:
                checkpoint = MAPPOCheckpoint("resume", args)

            mocked_load.assert_called_once_with(
                str(path), map_location="cpu", weights_only=False
            )
            self.assertTrue(checkpoint.resumed)

    def test_reward_normalizer_round_trip(self):
        first = ReturnNormalizer(n_envs=3, gamma=0.97)
        first(
            np.asarray([1.0, -2.0, 3.0]),
            np.asarray([False, True, False]),
        )
        state = first.state_dict()

        restored = ReturnNormalizer(n_envs=3, gamma=0.97)
        restored.load_state_dict(state)
        np.testing.assert_array_equal(restored.returns, first.returns)
        self.assertEqual(restored.ret_rms.mean, first.ret_rms.mean)
        self.assertEqual(restored.ret_rms.var, first.ret_rms.var)
        self.assertEqual(restored.ret_rms.count, first.ret_rms.count)

        reward = np.asarray([0.5, 1.5, -0.5])
        done = np.asarray([False, False, True])
        np.testing.assert_array_equal(
            restored(reward, done),
            first(reward, done),
        )

    def test_vector_environment_continues_from_identical_boundary(self):
        envs = AsyncMultiAgentVecEnv(
            [make_checkpoint_env, make_checkpoint_env], context="spawn"
        )
        try:
            envs.set_obs_stats({"bias": 17.0})
            envs.reset(seed=101)
            envs.step({"agent_0": th.tensor([1, 2])})
            state = envs.get_checkpoint_state()
            self.assertTrue(
                all(worker["mode"] == "action_replay_v1" for worker in state)
            )
            self.assertTrue(all("env" not in worker for worker in state))

            expected_first = envs.step({"agent_0": th.tensor([3, 0])})
            expected_second = envs.step({"agent_0": th.tensor([2, 1])})

            envs.set_checkpoint_state(state)
            actual_first = envs.step({"agent_0": th.tensor([3, 0])})
            actual_second = envs.step({"agent_0": th.tensor([2, 1])})

            for expected, actual in (
                (expected_first, actual_first),
                (expected_second, actual_second),
            ):
                for expected_part, actual_part in zip(expected[:4], actual[:4]):
                    for key in expected_part:
                        np.testing.assert_array_equal(
                            expected_part[key], actual_part[key]
                        )

            state[0]["expected_observation"]["agent_0"][0] += 1.0
            with self.assertRaisesRegex(
                RuntimeError,
                "Action replay did not reconstruct the saved observation exactly",
            ):
                envs.set_checkpoint_state(state)
        finally:
            envs.close()


if __name__ == "__main__":
    unittest.main()
