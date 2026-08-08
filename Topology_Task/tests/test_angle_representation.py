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
    """The line is a node here, so the drop replaces the angle on that node.

    Attachment edges in this schema carry no physical channel at all, so there
    is no edge for ``theta_diff`` to move to as there is in the other two
    representations.
    """

    def build(self, representation, **kwargs):
        builder = HeterogeneousLineGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            angle_representation=representation,
            **kwargs,
        )
        return builder, builder.specs["state"]

    def rows_of_type(self, builder, spec, node_type):
        return np.nonzero(
            np.asarray(spec["node_type"])
            == builder.NODE_TYPE_NAMES[node_type]
        )[0]

    def column(self, builder, name):
        return builder.node_features.index(name)

    def test_node_mode_still_works(self):
        builder, _ = self.build("node")
        self.assertIn("theta", builder.node_features)
        self.assertNotIn("theta_diff", builder.node_features)

    def test_the_drop_replaces_the_absolute_angle_in_place(self):
        node_builder, _ = self.build("node")
        diff_builder, _ = self.build("edge_diff")
        self.assertNotIn("theta", diff_builder.node_features)
        self.assertIn("theta_diff", diff_builder.node_features)
        # Same width and same position, so the encoder input does not move.
        self.assertEqual(node_builder.node_dim, diff_builder.node_dim)
        self.assertEqual(
            self.column(node_builder, "theta"),
            self.column(diff_builder, "theta_diff"),
        )

    def test_edges_stay_free_of_physical_channels(self):
        node_builder, _ = self.build("node")
        diff_builder, _ = self.build("edge_diff")
        self.assertEqual(diff_builder.physical_edge_features, [])
        self.assertNotIn("theta_diff", diff_builder.edge_features)
        self.assertEqual(node_builder.edge_features, diff_builder.edge_features)

    def test_line_nodes_carry_the_drop(self):
        builder, spec = self.build("edge_diff")
        graph = builder.build(make_obs())["state"]
        line_rows = self.rows_of_type(builder, spec, "transmission_line")
        values = graph["node_features"][
            line_rows, self.column(builder, "theta_diff")
        ]
        # theta_or - theta_ex = 12.5 - 4.5
        np.testing.assert_allclose(values, [8.0], rtol=1e-6)

    def test_absolute_asset_angles_are_gone(self):
        builder, spec = self.build("edge_diff")
        graph = builder.build(make_obs())["state"]
        column = self.column(builder, "theta_diff")
        for node_type in ("generator", "load", "busbar"):
            rows = self.rows_of_type(builder, spec, node_type)
            np.testing.assert_allclose(
                graph["node_features"][rows, column],
                np.zeros(len(rows)),
                atol=0.0,
            )

    def test_the_drop_is_gauge_invariant(self):
        builder, spec = self.build("edge_diff")
        column = self.column(builder, "theta_diff")
        line_rows = self.rows_of_type(builder, spec, "transmission_line")

        def drop(shift):
            obs = make_obs(
                theta_or=np.asarray([12.5 + shift], dtype=np.float32),
                theta_ex=np.asarray([4.5 + shift], dtype=np.float32),
                gen_theta=np.asarray([0.1 + shift], dtype=np.float32),
                load_theta=np.asarray([-0.2 + shift], dtype=np.float32),
            )
            return builder.build(obs)["state"]["node_features"][line_rows, column]

        np.testing.assert_allclose(drop(0.0), drop(30.0), rtol=1e-6)

    def test_a_line_that_is_out_reports_no_drop(self):
        # The simulator zeroes rho on a dead line but keeps its last solved
        # endpoint angles, so an unmasked drop would look like live physics.
        builder, spec = self.build("edge_diff")
        obs = make_obs(line_status=np.asarray([0], dtype=np.float32))
        graph = builder.build(obs)["state"]
        line_rows = self.rows_of_type(builder, spec, "transmission_line")
        values = graph["node_features"][
            line_rows, self.column(builder, "theta_diff")
        ]
        np.testing.assert_allclose(values, np.zeros(len(line_rows)), atol=0.0)

    def test_node_mode_output_is_unchanged(self):
        builder, spec = self.build("node")
        graph = builder.build(make_obs())["state"]
        column = self.column(builder, "theta")
        gen_rows = self.rows_of_type(builder, spec, "generator")
        line_rows = self.rows_of_type(builder, spec, "transmission_line")
        np.testing.assert_allclose(
            graph["node_features"][gen_rows, column], [0.1], rtol=1e-6
        )
        np.testing.assert_allclose(
            graph["node_features"][line_rows, column],
            np.zeros(len(line_rows)),
            atol=0.0,
        )


class FactoryTest(unittest.TestCase):
    def test_option_reaches_each_edge_carrying_builder(self):
        for graph_type, expected in (("bus", 5), ("heterogeneous", 11)):
            builder = make_grid_graph_builder(
                graph_type,
                MockGridEnv(),
                {"agent_0": [0]},
                angle_representation="edge_diff",
            )
            self.assertIn("theta_diff", builder.edge_features)
            self.assertEqual(builder.edge_dim, expected)

    def test_option_reaches_the_line_node_builder_as_a_node_feature(self):
        builder = make_grid_graph_builder(
            "heterogeneous_line",
            MockGridEnv(),
            {"agent_0": [0]},
            angle_representation="edge_diff",
        )
        self.assertIn("theta_diff", builder.node_features)
        self.assertNotIn("theta_diff", builder.edge_features)


if __name__ == "__main__":
    unittest.main()
