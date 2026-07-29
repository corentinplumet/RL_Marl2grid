import random
import unittest

import gymnasium as gym
import numpy as np
import torch as th

from common.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    EXACT_BOUNDARY_PHASE,
    capture_rng_state,
    exact_resume_state,
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
