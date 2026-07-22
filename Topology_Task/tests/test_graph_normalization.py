from types import SimpleNamespace
import unittest

import numpy as np

from common.graph_normalization import GraphFeatureProcessor


class MockGridEnv:
    gen_pmax = np.asarray([20.0], dtype=np.float32)
    parameters = SimpleNamespace(
        NB_TIMESTEP_COOLDOWN_SUB=4,
        NB_TIMESTEP_COOLDOWN_LINE=3,
        NB_TIMESTEP_OVERFLOW_ALLOWED=2,
    )

    @staticmethod
    def max_episode_duration():
        return 100


def clone_graph(graph):
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in graph.items()
    }


class GraphPhysicalScalingTest(unittest.TestCase):
    def test_continuous_features_are_scaled_and_structural_values_are_preserved(self):
        spec = {
            "node_feature_names": [
                "gen_p",
                "gen_theta",
                "time_before_cooldown_sub",
                "domain_mask",
            ],
            "edge_feature_names": [
                "line_status",
                "rho",
                "timestep_overflow",
                "time_before_cooldown_line",
            ],
            "edge_type_names": {
                "physical_line": 1,
                "same_substation_busbar": 2,
            },
        }
        graph = {
            "node_features": np.asarray(
                [[10.0, 90.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "edge_features": np.asarray(
                [[1.0, 0.75, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "node_mask": np.ones(2, dtype=np.float32),
            "edge_mask": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            "edge_type": np.asarray([1, 1, 2], dtype=np.int64),
        }
        processor = GraphFeatureProcessor(
            MockGridEnv(),
            spec,
            physical_scaling=True,
            running_normalization=False,
        )
        result = processor.process({"state": clone_graph(graph)}, update=True)[
            "state"
        ]

        np.testing.assert_allclose(
            result["node_features"][0],
            [0.5, 0.5, 0.5, 1.0],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            result["edge_features"][0],
            [1.0, 0.75, 0.5, 1.0],
            rtol=1e-6,
        )
        self.assertEqual(result["edge_features"][2, 1], 1.0)
        self.assertEqual(processor.scale_factors["power_mw"], 20.0)
        self.assertEqual(processor.scale_factors["maintenance_horizon"], 100.0)


class GraphRunningNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "node_feature_names": ["p", "connected"],
            "edge_feature_names": ["rho", "timestep_overflow", "relation_physical_line"],
            "n_node_types": 2,
            "edge_type_names": {
                "physical_line": 1,
                "same_substation_busbar": 2,
            },
        }
        self.state = {
            "node_features": np.asarray(
                [[1.0, 1.0], [3.0, 1.0], [100.0, 1.0], [200.0, 0.0]],
                dtype=np.float32,
            ),
            "node_type": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "node_mask": np.ones(4, dtype=np.float32),
            "edge_features": np.asarray(
                [[0.5, 1.0, 1.0], [1.5, 3.0, 1.0], [1.0, 0.0, 0.0], [0.0, 99.0, 0.0]],
                dtype=np.float32,
            ),
            "edge_type": np.asarray([1, 1, 2, 1], dtype=np.int64),
            "edge_mask": np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        }
        self.local = {
            "node_features": np.asarray(
                [[1.0, 1.0], [100.0, 1.0]], dtype=np.float32
            ),
            "node_type": np.asarray([0, 1], dtype=np.int64),
            "node_mask": np.ones(2, dtype=np.float32),
            "edge_features": np.asarray([[0.5, 1.0, 1.0]], dtype=np.float32),
            "edge_type": np.asarray([1], dtype=np.int64),
            "edge_mask": np.asarray([1.0], dtype=np.float32),
        }

    def test_stats_are_type_aware_shared_and_mask_aware(self):
        processor = GraphFeatureProcessor(
            MockGridEnv(),
            self.spec,
            physical_scaling=False,
            running_normalization=True,
            clip=10.0,
        )
        result = processor.process(
            {"state": clone_graph(self.state), "agent_0": clone_graph(self.local)},
            update=True,
        )

        np.testing.assert_allclose(
            result["state"]["node_features"][:, 0],
            [-1.0, 1.0, -1.0, 1.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            result["agent_0"]["node_features"][:, 0],
            [-1.0, -1.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            result["state"]["edge_features"][:2, 1],
            [-1.0, 1.0],
            rtol=1e-5,
        )
        # Rho, categorical/binary columns, structural edges, and masked
        # candidates are not standardized.
        np.testing.assert_allclose(
            result["state"]["edge_features"][:, 0], [0.5, 1.5, 1.0, 0.0]
        )
        self.assertEqual(result["state"]["edge_features"][2, 1], 0.0)
        self.assertEqual(result["state"]["edge_features"][3, 1], 99.0)
        np.testing.assert_allclose(
            result["state"]["node_features"][:, 1], [1.0, 1.0, 1.0, 0.0]
        )

        stats = processor.get_stats()
        self.assertEqual(stats["__graph_node_type_0__"]["count"], 2.0)
        self.assertEqual(stats["__graph_node_type_1__"]["count"], 2.0)
        self.assertEqual(stats["__graph_physical_edge__"]["count"], 2.0)

        restored = GraphFeatureProcessor(
            MockGridEnv(),
            self.spec,
            running_normalization=True,
        )
        restored.set_stats(stats)
        restored_result = restored.process(
            {"state": clone_graph(self.state), "agent_0": clone_graph(self.local)},
            update=False,
        )
        np.testing.assert_allclose(
            restored_result["agent_0"]["node_features"][:, 0],
            result["agent_0"]["node_features"][:, 0],
        )


if __name__ == "__main__":
    unittest.main()
