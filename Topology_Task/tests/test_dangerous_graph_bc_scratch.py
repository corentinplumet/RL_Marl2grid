from argparse import Namespace
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch as th

from teacher_student.scratch_initialization import (
    apply_scratch_architecture_overrides,
    build_scratch_actors,
    scratch_checkpoint_base,
)


class DangerousGraphBCScratchTest(unittest.TestCase):
    def test_overrides_scratch_message_passing_depth(self):
        args = Namespace(gnn_layers=2)

        overrides = apply_scratch_architecture_overrides(
            args,
            initialization="scratch",
            gnn_layers=4,
        )

        self.assertEqual(args.gnn_layers, 4)
        self.assertEqual(
            overrides,
            {"gnn_layers": {"template": 2, "target": 4}},
        )

    def test_overrides_all_new_scratch_architecture_components(self):
        args = Namespace(
            gnn_layers=2,
            candidate_action_delta_encoder=False,
            gnn_residual=False,
            gnn_jumping_knowledge="none",
            gnn_readout_aggr="energized_mean",
        )

        overrides = apply_scratch_architecture_overrides(
            args,
            initialization="scratch",
            gnn_layers=4,
            action_delta_encoder=True,
            gnn_residual=True,
            gnn_jumping_knowledge="concat",
            gnn_readout_aggr="energized_mean_max",
        )

        self.assertEqual(args.gnn_layers, 4)
        self.assertTrue(args.candidate_action_delta_encoder)
        self.assertTrue(args.gnn_residual)
        self.assertEqual(args.gnn_jumping_knowledge, "concat")
        self.assertEqual(args.gnn_readout_aggr, "energized_mean_max")
        self.assertEqual(
            set(overrides),
            {
                "gnn_layers",
                "candidate_action_delta_encoder",
                "gnn_residual",
                "gnn_jumping_knowledge",
                "gnn_readout_aggr",
            },
        )

    def test_rejects_depth_override_for_warm_start(self):
        args = Namespace(gnn_layers=2)

        with self.assertRaisesRegex(ValueError, "only with --initialization scratch"):
            apply_scratch_architecture_overrides(
                args,
                initialization="warm_start",
                gnn_layers=3,
            )

        self.assertEqual(args.gnn_layers, 2)

    def test_rejects_nonpositive_scratch_depth(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            apply_scratch_architecture_overrides(
                Namespace(gnn_layers=2),
                initialization="scratch",
                gnn_layers=0,
            )

    def test_builds_fresh_target_actors_without_checkpoint_state(self):
        actor_env = SimpleNamespace(
            observation_space={"agent_0": object(), "agent_1": object()}
        )
        evaluator = SimpleNamespace(env=SimpleNamespace(env=actor_env))
        args = Namespace(action_type="topology")
        actors = {
            "agent_0": th.nn.Linear(2, 2),
            "agent_1": th.nn.Linear(2, 2),
        }
        for actor in actors.values():
            actor.eval()

        build = Mock(return_value=actors)
        fake_core = types.ModuleType("alg.mappo.core")
        fake_core.build_actor_modules = build
        with patch.dict(sys.modules, {"alg.mappo.core": fake_core}):
            result = build_scratch_actors(args, evaluator, th.device("cpu"))

        self.assertIs(result, actors)
        build.assert_called_once_with(
            actor_env,
            args,
            ["agent_0", "agent_1"],
            False,
            device=th.device("cpu"),
        )
        self.assertTrue(all(actor.training for actor in result.values()))

    def test_scratch_checkpoint_does_not_copy_mappo_training_state(self):
        args = Namespace(env_id="bus36_wcci_nomaint")
        record = scratch_checkpoint_base(args)

        self.assertEqual(record["global_step"], 0)
        self.assertEqual(record["last_rollout"], 0)
        self.assertEqual(record["training_state"], {})
        for forbidden in ("critic", "critic_optim", "actor_optim", "agent_0"):
            self.assertNotIn(forbidden, record)


if __name__ == "__main__":
    unittest.main()
