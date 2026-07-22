import unittest
from argparse import Namespace

from common.runtime_config import (
    merge_runtime_args,
    sparse_transformer_self_edges_enabled,
)


class RuntimeGraphConfigTests(unittest.TestCase):
    def test_algorithm_graph_arguments_are_available_before_env_creation(self):
        combined = merge_runtime_args(
            Namespace(seed=3, actor_encoder="mlp"),
            Namespace(
                env_id="bus14",
                actor_encoder="gnn",
                gnn_include_neighbors=True,
            ),
            Namespace(
                gnn_graph_type="heterogeneous_line",
                gnn_add_substation_edges=True,
                gnn_physical_scaling=True,
                gnn_running_norm=True,
            ),
        )

        self.assertEqual(combined.actor_encoder, "gnn")
        self.assertEqual(combined.gnn_graph_type, "heterogeneous_line")
        self.assertTrue(combined.gnn_add_substation_edges)
        self.assertTrue(combined.gnn_physical_scaling)
        self.assertTrue(combined.gnn_running_norm)

    def test_sparse_self_edges_do_not_leak_into_ordinary_gnns(self):
        self.assertFalse(sparse_transformer_self_edges_enabled("gine", True))
        self.assertFalse(sparse_transformer_self_edges_enabled("gat", None))
        self.assertTrue(
            sparse_transformer_self_edges_enabled("sparse_transformer", None)
        )
        self.assertFalse(
            sparse_transformer_self_edges_enabled("sparse_transformer", False)
        )


if __name__ == "__main__":
    unittest.main()
