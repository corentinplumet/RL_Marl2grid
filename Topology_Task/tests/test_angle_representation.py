from types import SimpleNamespace
import unittest

import numpy as np

from common.graph import (
    GridGraphBuilder,
    HeterogeneousGridGraphBuilder,
    HeterogeneousLineGraphBuilder,
    make_grid_graph_builder,
)


class MockGridEnv:
    n_sub = 2
    n_line = 1
    n_gen = 1
    n_load = 1
    n_busbar_per_sub = np.asarray([2, 2], dtype=np.int64)

    line_or_to_subid = np.asarray([0], dtype=np.int64)
    line_ex_to_subid = np.asarray([1], dtype=np.int64)
    gen_to_subid = np.asarray([0], dtype=np.int64)
    load_to_subid = np.asarray([1], dtype=np.int64)

    line_or_pos_topo_vect = np.asarray([0], dtype=np.int64)
    line_ex_pos_topo_vect = np.asarray([1], dtype=np.int64)
    gen_pos_topo_vect = np.asarray([2], dtype=np.int64)
    load_pos_topo_vect = np.asarray([3], dtype=np.int64)


def make_obs(**overrides):
    values = {
        "topo_vect": np.asarray([1, 2, 2, 1], dtype=np.float32),
        "line_status": np.asarray([1], dtype=np.float32),
        "rho": np.asarray([0.75], dtype=np.float32),
        "timestep_overflow": np.asarray([2], dtype=np.float32),
        "time_before_cooldown_line": np.asarray([3], dtype=np.float32),
        "time_before_cooldown_sub": np.asarray([4, 5], dtype=np.float32),
        "gen_p": np.asarray([10.0], dtype=np.float32),
        "gen_theta": np.asarray([0.1], dtype=np.float32),
        "load_p": np.asarray([8.0], dtype=np.float32),
        "load_theta": np.asarray([-0.2], dtype=np.float32),
        "theta_or": np.asarray([12.5], dtype=np.float32),
        "theta_ex": np.asarray([4.5], dtype=np.float32),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class BusGraphAngleRepresentationTest(unittest.TestCase):
    def build(self, representation):
        return GridGraphBuilder(
            MockGridEnv(), {"agent_0": [0]}, angle_representation=representation
        )

    def test_node_mode_is_the_unchanged_default(self):
        builder = GridGraphBuilder(MockGridEnv(), {"agent_0": [0]})
        self.assertEqual(builder.angle_representation, "node")
        self.assertIn("gen_theta", builder.node_features)
        self.assertIn("load_theta", builder.node_features)
        self.assertNotIn("theta_diff", builder.edge_features)
        self.assertEqual((builder.node_dim, builder.edge_dim), (6, 4))

    def test_edge_mode_moves_angles_from_nodes_to_edges(self):
        builder = self.build("edge_diff")
        self.assertNotIn("gen_theta", builder.node_features)
        self.assertNotIn("load_theta", builder.node_features)
        self.assertEqual(builder.edge_features[-1], "theta_diff")
        self.assertEqual((builder.node_dim, builder.edge_dim), (4, 5))

    def test_theta_diff_is_the_drop_across_the_line(self):
        builder = self.build("edge_diff")
        graph = builder.build(make_obs())["state"]
        column = builder.edge_features.index("theta_diff")
        values = graph["edge_features"][:, column]
        active = np.asarray(graph["edge_mask"], dtype=np.float32) > 0

        self.assertTrue(np.any(active))
        # 12.5 - 4.5 on every active physical edge.
        self.assertTrue(np.allclose(values[active], 8.0))

    def test_theta_diff_is_gauge_invariant(self):
        """Shifting every angle by a constant must not change the feature."""
        builder = self.build("edge_diff")
        column = builder.edge_features.index("theta_diff")

        base = builder.build(make_obs())["state"]["edge_features"][:, column]
        shifted = builder.build(
            make_obs(
                theta_or=np.asarray([12.5 + 30.0], dtype=np.float32),
                theta_ex=np.asarray([4.5 + 30.0], dtype=np.float32),
            )
        )["state"]["edge_features"][:, column]

        self.assertTrue(np.allclose(base, shifted))

    def test_sign_follows_origin_minus_extremity(self):
        builder = self.build("edge_diff")
        column = builder.edge_features.index("theta_diff")
        graph = builder.build(
            make_obs(
                theta_or=np.asarray([1.0], dtype=np.float32),
                theta_ex=np.asarray([6.0], dtype=np.float32),
            )
        )["state"]
        active = np.asarray(graph["edge_mask"], dtype=np.float32) > 0
        self.assertTrue(np.allclose(graph["edge_features"][active, column], -5.0))

    def test_missing_angles_do_not_break_the_build(self):
        builder = self.build("edge_diff")
        obs = make_obs()
        del obs.theta_or
        graph = builder.build(obs)["state"]
        column = builder.edge_features.index("theta_diff")
        self.assertTrue(np.allclose(graph["edge_features"][:, column], 0.0))

    def test_node_features_still_carry_power(self):
        builder = self.build("edge_diff")
        graph = builder.build(make_obs())["state"]
        column = builder.node_features.index("gen_p")
        self.assertAlmostEqual(float(graph["node_features"][:, column].sum()), 10.0)

    def test_invalid_representation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build("edge_difference")


class HeterogeneousGraphAngleRepresentationTest(unittest.TestCase):
    def build(self, representation):
        return HeterogeneousGridGraphBuilder(
            MockGridEnv(), {"agent_0": [0]}, angle_representation=representation
        )

    def test_node_mode_is_the_unchanged_default(self):
        builder = HeterogeneousGridGraphBuilder(MockGridEnv(), {"agent_0": [0]})
        self.assertIn("theta", builder.node_features)
        self.assertNotIn("theta_diff", builder.physical_edge_features)

    def test_edge_mode_drops_the_typed_angle_channel(self):
        builder = self.build("edge_diff")
        self.assertNotIn("theta", builder.node_features)
        self.assertIn("theta_diff", builder.physical_edge_features)
        self.assertIn("theta_diff", builder.edge_features)

    def test_theta_diff_reaches_the_built_graph(self):
        builder = self.build("edge_diff")
        graph = builder.build(make_obs())["state"]
        column = builder.edge_features.index("theta_diff")
        values = graph["edge_features"][:, column]
        self.assertTrue(np.any(np.isclose(values, 8.0)))

    def test_relation_channels_are_still_last(self):
        """Relation one-hots must stay after the physical block."""
        builder = self.build("edge_diff")
        first_relation = min(
            index
            for index, name in enumerate(builder.edge_features)
            if name.startswith("relation_")
        )
        self.assertLess(
            builder.edge_features.index("theta_diff"), first_relation
        )


class LineNodeGraphAngleRepresentationTest(unittest.TestCase):
    def test_edge_mode_is_refused_rather_than_ignored(self):
        with self.assertRaises(ValueError) as ctx:
            HeterogeneousLineGraphBuilder(
                MockGridEnv(), {"agent_0": [0]}, angle_representation="edge_diff"
            )
        self.assertIn("line-node graph", str(ctx.exception))

    def test_node_mode_still_works(self):
        builder = HeterogeneousLineGraphBuilder(
            MockGridEnv(), {"agent_0": [0]}, angle_representation="node"
        )
        self.assertIn("theta", builder.node_features)


class FactoryTest(unittest.TestCase):
    def test_option_reaches_each_supported_builder(self):
        for graph_type, expected in (("bus", 5), ("heterogeneous", 12)):
            builder = make_grid_graph_builder(
                graph_type,
                MockGridEnv(),
                {"agent_0": [0]},
                angle_representation="edge_diff",
            )
            self.assertIn("theta_diff", builder.edge_features)
            self.assertEqual(builder.edge_dim, expected)


if __name__ == "__main__":
    unittest.main()
