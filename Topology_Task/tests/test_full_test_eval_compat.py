from argparse import Namespace
import unittest

import torch as th

from full_test_eval.evaluate_checkpoint import (
    _apply_transfer_target_overrides,
    _configure_legacy_connected_feature,
)


class FullTestEvalCompatibilityTest(unittest.TestCase):
    @staticmethod
    def args():
        return Namespace(
            gnn_graph_type="heterogeneous_line",
            gnn_angle_representation="node",
            gnn_node_id_embeddings=False,
            gnn_node_id_emb_dim=8,
            env_id="bus14",
            env_config_path="scenario.json",
        )

    @staticmethod
    def record(input_width):
        return {
            "agent_0": {
                "encoder.graph_encoder.node_pre_encoder.0.weight": th.zeros(
                    128, input_width
                )
            }
        }

    def test_enables_legacy_column_for_thirteen_input_checkpoint(self):
        args = _configure_legacy_connected_feature(self.args(), self.record(13))
        self.assertTrue(args.gnn_include_legacy_connected_feature)

    def test_keeps_current_schema_for_twelve_input_checkpoint(self):
        args = _configure_legacy_connected_feature(self.args(), self.record(12))
        self.assertFalse(
            getattr(args, "gnn_include_legacy_connected_feature", False)
        )

    def test_shared_candidate_actor_can_be_retargeted_without_flat_inputs(self):
        args = Namespace(
            env_id="bus14",
            actor_encoder="gnn",
            actor_action_head="candidate_pool",
            share_actor_gnn=True,
            share_candidate_scorer=True,
            gnn_concat_flat=False,
        )
        cli = Namespace(
            target_env_id="bus36_wcci_nomaint",
            target_reduced_action_space="outputs/reduced.json",
        )

        args, enabled, source_env_id = _apply_transfer_target_overrides(args, cli)

        self.assertTrue(enabled)
        self.assertEqual(source_env_id, "bus14")
        self.assertEqual(args.env_id, "bus36_wcci_nomaint")
        self.assertEqual(args.reduced_action_space, "outputs/reduced.json")

    def test_cross_grid_eval_rejects_nonshared_candidate_scorer(self):
        args = Namespace(
            env_id="bus14",
            actor_encoder="gnn",
            actor_action_head="candidate_pool",
            share_actor_gnn=True,
            share_candidate_scorer=False,
            gnn_concat_flat=False,
        )
        cli = Namespace(
            target_env_id="bus36_wcci_nomaint",
            target_reduced_action_space="outputs/reduced.json",
        )

        with self.assertRaisesRegex(ValueError, "share_candidate_scorer=true"):
            _apply_transfer_target_overrides(args, cli)


if __name__ == "__main__":
    unittest.main()
