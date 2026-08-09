from types import SimpleNamespace
import unittest

import numpy as np
import torch as th

from common.gnn import GraphEncoder, SparseGraphTransformerEncoder
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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class HeterogeneousGridGraphBuilderTest(unittest.TestCase):
    def setUp(self):
        self.builder = HeterogeneousGridGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
        )

    def test_full_graph_schema_and_dynamic_connections(self):
        spec = self.builder.specs["state"]
        graph = self.builder.build(make_obs())["state"]

        self.assertEqual(spec["graph_type"], "heterogeneous")
        self.assertEqual(len(spec["node_ids"]), 6)
        self.assertEqual(spec["edge_index"].shape, (2, 16))
        self.assertEqual(spec["node_dim"], 8)
        self.assertEqual(spec["edge_dim"], 10)
        self.assertNotIn("relation_self", spec["edge_feature_names"])
        np.testing.assert_array_equal(
            np.bincount(spec["node_type"], minlength=3), [4, 1, 1]
        )

        active_types = spec["edge_type"][graph["edge_mask"].astype(bool)]
        expected_counts = {
            self.builder.EDGE_TYPE_PHYSICAL_LINE: 2,
            self.builder.EDGE_TYPE_GENERATOR_TO_BUSBAR: 1,
            self.builder.EDGE_TYPE_BUSBAR_TO_GENERATOR: 1,
            self.builder.EDGE_TYPE_LOAD_TO_BUSBAR: 1,
            self.builder.EDGE_TYPE_BUSBAR_TO_LOAD: 1,
        }
        for edge_type, count in expected_counts.items():
            self.assertEqual(int(np.sum(active_types == edge_type)), count)
        self.assertEqual(int(graph["edge_mask"].sum()), 6)
        self.assertEqual(int(graph["energized_node_mask"].sum()), 6)

        node_cols = {
            name: idx for idx, name in enumerate(spec["node_feature_names"])
        }
        gen_row = int(
            np.nonzero(spec["node_type"] == self.builder.NODE_TYPE_GENERATOR)[0][0]
        )
        load_row = int(
            np.nonzero(spec["node_type"] == self.builder.NODE_TYPE_LOAD)[0][0]
        )
        self.assertEqual(graph["node_features"][gen_row, node_cols["p"]], 10.0)
        self.assertEqual(graph["node_features"][load_row, node_cols["p"]], 8.0)
        self.assertEqual(
            graph["node_features"][gen_row, node_cols["connected"]], 1.0
        )

        relation_offset = len(self.builder.physical_edge_features)
        active_rows = np.nonzero(graph["edge_mask"])[0]
        for row in active_rows:
            edge_type = int(spec["edge_type"][row])
            relation_column = self.builder.relation_edge_columns[edge_type]
            self.assertEqual(
                graph["edge_features"][row, relation_offset + relation_column],
                1.0,
            )
        self.assertTrue(
            np.all(graph["edge_features"][graph["edge_mask"] == 0] == 0.0)
        )

    def test_legacy_schema_only_restores_the_old_checkpoint_width(self):
        builder = HeterogeneousGridGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
            include_legacy_self_relation_feature=True,
        )
        spec = builder.specs["state"]
        self.assertEqual(spec["edge_dim"], 11)
        self.assertIn("relation_self", spec["edge_feature_names"])

    def test_topology_change_updates_masks_without_changing_shapes(self):
        original = self.builder.build(make_obs())["state"]
        changed = self.builder.build(
            make_obs(
                topo_vect=np.asarray([1, 2, 0, 2], dtype=np.float32),
                line_status=np.asarray([0], dtype=np.float32),
            )
        )["state"]

        self.assertEqual(original["node_features"].shape, changed["node_features"].shape)
        self.assertEqual(original["edge_features"].shape, changed["edge_features"].shape)
        self.assertEqual(int(changed["edge_mask"].sum()), 2)

        spec = self.builder.specs["state"]
        node_cols = {
            name: idx for idx, name in enumerate(spec["node_feature_names"])
        }
        gen_row = int(
            np.nonzero(spec["node_type"] == self.builder.NODE_TYPE_GENERATOR)[0][0]
        )
        self.assertEqual(
            changed["node_features"][gen_row, node_cols["connected"]], 0.0
        )
        self.assertEqual(changed["node_features"][gen_row, node_cols["p"]], 0.0)

    def test_generator_and_load_directions_are_configurable_independently(self):
        builder = HeterogeneousGridGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
            generator_edge_direction="busbar_to_asset",
            load_edge_direction="asset_to_busbar",
        )
        spec = builder.specs["state"]
        graph = builder.build(make_obs())["state"]

        self.assertEqual(spec["edge_index"].shape, (2, 12))
        self.assertEqual(int(graph["edge_mask"].sum()), 4)
        present_types = set(spec["edge_type"].tolist())
        self.assertNotIn(builder.EDGE_TYPE_GENERATOR_TO_BUSBAR, present_types)
        self.assertIn(builder.EDGE_TYPE_BUSBAR_TO_GENERATOR, present_types)
        self.assertIn(builder.EDGE_TYPE_LOAD_TO_BUSBAR, present_types)
        self.assertNotIn(builder.EDGE_TYPE_BUSBAR_TO_LOAD, present_types)
        self.assertEqual(spec["generator_edge_direction"], "busbar_to_asset")
        self.assertEqual(spec["load_edge_direction"], "asset_to_busbar")

    def test_local_graph_contains_assets_from_controlled_substations(self):
        builder = HeterogeneousGridGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=False,
        )
        spec = builder.specs["agent_0"]
        graph = builder.build(make_obs())["agent_0"]

        self.assertEqual(len(spec["node_ids"]), 3)
        np.testing.assert_array_equal(
            np.bincount(spec["node_type"], minlength=3), [2, 1, 0]
        )
        self.assertEqual(len(spec["line_ids"]), 0)
        self.assertEqual(spec["edge_index"].shape, (2, 4))
        self.assertEqual(int(graph["edge_mask"].sum()), 2)

    def test_local_graph_keeps_neighbor_busbar_but_excludes_neighbor_assets(self):
        builder = HeterogeneousGridGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
        )
        spec = builder.specs["agent_0"]
        graph = builder.build(make_obs())["agent_0"]

        np.testing.assert_array_equal(
            np.bincount(spec["node_type"], minlength=3), [4, 1, 0]
        )
        self.assertEqual(spec["gen_ids"].tolist(), [0])
        self.assertEqual(spec["load_ids"].tolist(), [])
        non_busbar = spec["node_type"] != builder.NODE_TYPE_BUSBAR
        self.assertTrue(
            bool(np.all(spec["node_substation_ids"][non_busbar] == 0))
        )
        self.assertEqual(spec["edge_index"].shape, (2, 12))
        self.assertEqual(int(graph["edge_mask"].sum()), 4)

    def test_same_substation_busbar_edges_are_typed_and_always_active(self):
        builder = HeterogeneousGridGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
            add_substation_edges=True,
        )
        spec = builder.specs["state"]
        graph = builder.build(make_obs())["state"]
        relation_rows = np.nonzero(
            spec["edge_type"] == builder.EDGE_TYPE_SAME_SUBSTATION
        )[0]

        self.assertEqual(spec["edge_index"].shape, (2, 20))
        self.assertEqual(len(relation_rows), 4)
        self.assertEqual(int(graph["edge_mask"].sum()), 10)
        self.assertTrue(bool(np.all(graph["edge_mask"][relation_rows] == 1.0)))
        for src, dst in spec["edge_index"][:, relation_rows].T:
            self.assertNotEqual(int(src), int(dst))
            self.assertEqual(
                int(spec["node_substation_ids"][src]),
                int(spec["node_substation_ids"][dst]),
            )
            self.assertEqual(
                int(spec["node_type"][src]), builder.NODE_TYPE_BUSBAR
            )
            self.assertEqual(
                int(spec["node_type"][dst]), builder.NODE_TYPE_BUSBAR
            )

        rho_idx = spec["edge_feature_names"].index("rho")
        self.assertTrue(
            bool(np.all(graph["edge_features"][relation_rows, rho_idx] == 1.0))
        )
        disconnected = builder.build(
            make_obs(
                topo_vect=np.asarray([1, 2, 0, 0], dtype=np.float32),
                line_status=np.asarray([0], dtype=np.float32),
            )
        )["state"]
        self.assertEqual(int(disconnected["edge_mask"].sum()), 4)
        self.assertTrue(
            bool(np.all(disconnected["edge_mask"][relation_rows] == 1.0))
        )
        self.assertEqual(int(disconnected["energized_node_mask"].sum()), 0)

        bus_builder = GridGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
            add_substation_edges=True,
        )
        bus_spec = bus_builder.specs["state"]
        bus_graph = bus_builder.build(make_obs())["state"]
        bus_relation_rows = np.nonzero(
            bus_spec["edge_type"] == bus_builder.EDGE_TYPE_SAME_SUBSTATION
        )[0]
        self.assertEqual(bus_spec["edge_index"].shape, (2, 12))
        self.assertEqual(len(bus_relation_rows), 4)
        self.assertEqual(int(bus_graph["edge_mask"].sum()), 6)
        bus_rho_idx = bus_spec["edge_feature_names"].index("rho")
        self.assertTrue(
            bool(
                np.all(
                    bus_graph["edge_features"][bus_relation_rows, bus_rho_idx]
                    == 1.0
                )
            )
        )

    def test_optional_relation_edges_and_encoder_node_ids(self):
        builder = HeterogeneousGridGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
            add_self_edges=True,
            add_substation_edges=True,
        )
        spec = builder.specs["state"]
        graph = builder.build(make_obs())["state"]
        self.assertEqual(spec["edge_index"].shape, (2, 26))
        self.assertEqual(int(graph["edge_mask"].sum()), 16)

        encoder = GraphEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=1,
            conv_type="gine",
            node_pre_encoder=True,
            node_id_embeddings=True,
            node_id_emb_dim=2,
        )
        tensor_graph = {
            key: th.tensor(value)
            for key, value in graph.items()
            if key in {"node_features", "edge_features", "node_mask", "edge_mask"}
        }
        embedding = encoder(tensor_graph)
        self.assertEqual(tuple(embedding.shape), (4,))
        self.assertTrue(bool(th.isfinite(embedding).all()))

        substation_only_encoder = GraphEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=2,
            conv_type="gine",
            readout_aggr="mean",
            node_pre_encoder=True,
            add_substation_nodes=True,
        )
        self.assertFalse(substation_only_encoder.use_virtual_node)
        substation_only_embedding = substation_only_encoder(tensor_graph)
        self.assertEqual(tuple(substation_only_embedding.shape), (4,))
        self.assertTrue(bool(th.isfinite(substation_only_embedding).all()))
        substation_only_embedding.sum().backward()
        self.assertIsNotNone(
            substation_only_encoder.substation_node_encoder.weight.grad
        )

        sparse_encoder = SparseGraphTransformerEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=1,
            heads=2,
            node_id_embeddings=True,
            node_id_emb_dim=2,
        )
        sparse_tensor_graph = {
            key: th.tensor(value)
            for key, value in graph.items()
            if key
            in {
                "node_features",
                "edge_features",
                "node_mask",
                "edge_mask",
                "edge_type",
                "controlled_node_mask",
                "energized_node_mask",
            }
        }
        sparse_embedding = sparse_encoder(sparse_tensor_graph)
        self.assertEqual(tuple(sparse_embedding.shape), (4,))
        self.assertTrue(bool(th.isfinite(sparse_embedding).all()))

    def test_factory_keeps_bus_representation_available(self):
        bus_builder = make_grid_graph_builder(
            "bus", MockGridEnv(), {"agent_0": [0]}, include_neighbors=False
        )
        hetero_builder = make_grid_graph_builder(
            "heterogeneous",
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=False,
        )
        line_builder = make_grid_graph_builder(
            "heterogeneous_line",
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=False,
        )
        self.assertIsInstance(bus_builder, GridGraphBuilder)
        self.assertNotIsInstance(bus_builder, HeterogeneousGridGraphBuilder)
        self.assertIsInstance(hetero_builder, HeterogeneousGridGraphBuilder)
        self.assertIsInstance(line_builder, HeterogeneousLineGraphBuilder)
        for builder in (bus_builder, hetero_builder, line_builder):
            self.assertNotIn(
                "relation_self", builder.specs["state"]["edge_feature_names"]
            )


class HeterogeneousLineGraphBuilderTest(unittest.TestCase):
    def setUp(self):
        self.builder = HeterogeneousLineGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
        )

    def test_line_nodes_schema_and_dynamic_connections(self):
        spec = self.builder.specs["state"]
        graph = self.builder.build(make_obs())["state"]

        self.assertEqual(spec["graph_type"], "heterogeneous_line")
        self.assertEqual(len(spec["node_ids"]), 7)
        self.assertEqual(spec["edge_index"].shape, (2, 16))
        self.assertEqual(spec["node_dim"], 13)
        self.assertEqual(spec["edge_dim"], 9)
        self.assertNotIn("rho", spec["edge_feature_names"])
        self.assertNotIn("relation_self", spec["edge_feature_names"])
        np.testing.assert_array_equal(
            np.bincount(spec["node_type"], minlength=4), [4, 1, 1, 1]
        )

        active_types = spec["edge_type"][graph["edge_mask"].astype(bool)]
        expected_counts = {
            self.builder.EDGE_TYPE_BUSBAR_TO_LINE_ORIGIN: 1,
            self.builder.EDGE_TYPE_LINE_ORIGIN_TO_BUSBAR: 1,
            self.builder.EDGE_TYPE_BUSBAR_TO_LINE_EXTREMITY: 1,
            self.builder.EDGE_TYPE_LINE_EXTREMITY_TO_BUSBAR: 1,
            self.builder.EDGE_TYPE_GENERATOR_TO_BUSBAR: 1,
            self.builder.EDGE_TYPE_BUSBAR_TO_GENERATOR: 1,
            self.builder.EDGE_TYPE_LOAD_TO_BUSBAR: 1,
            self.builder.EDGE_TYPE_BUSBAR_TO_LOAD: 1,
        }
        for edge_type, count in expected_counts.items():
            self.assertEqual(int(np.sum(active_types == edge_type)), count)
        self.assertEqual(int(graph["edge_mask"].sum()), 8)
        self.assertEqual(int(graph["energized_node_mask"].sum()), 7)

        node_cols = {
            name: idx for idx, name in enumerate(spec["node_feature_names"])
        }
        line_row = int(
            np.nonzero(
                spec["node_type"] == self.builder.NODE_TYPE_TRANSMISSION_LINE
            )[0][0]
        )
        expected_line_features = {
            "is_transmission_line": 1.0,
            "line_status": 1.0,
            "rho": 0.75,
            "timestep_overflow": 2.0,
            "time_before_cooldown_line": 3.0,
            "connected": 1.0,
        }
        for feature, value in expected_line_features.items():
            self.assertEqual(
                graph["node_features"][line_row, node_cols[feature]], value
            )

        relation_offset = len(self.builder.physical_edge_features)
        for row in np.nonzero(graph["edge_mask"])[0]:
            edge_type = int(spec["edge_type"][row])
            relation_column = self.builder.relation_edge_columns[edge_type]
            self.assertEqual(
                graph["edge_features"][row, relation_offset + relation_column],
                1.0,
            )
            self.assertEqual(float(graph["edge_features"][row].sum()), 1.0)
        self.assertTrue(
            np.all(graph["edge_features"][graph["edge_mask"] == 0] == 0.0)
        )

    def test_local_graph_uses_shared_line_node_without_neighbor_equipment(self):
        for include_neighbors in (False, True):
            with self.subTest(include_neighbors=include_neighbors):
                builder = HeterogeneousLineGraphBuilder(
                    MockGridEnv(),
                    {"agent_0": [0]},
                    include_neighbors=include_neighbors,
                )
                spec = builder.specs["agent_0"]
                graph = builder.build(make_obs())["agent_0"]

                np.testing.assert_array_equal(
                    np.bincount(spec["node_type"], minlength=4), [2, 1, 0, 1]
                )
                self.assertEqual(spec["line_ids"].tolist(), [0])
                self.assertEqual(spec["load_ids"].tolist(), [])
                np.testing.assert_array_equal(
                    spec["busbar_id_to_node_row"], [0, 1, -1, -1]
                )
                np.testing.assert_array_equal(spec["line_id_to_node_row"], [3])
                self.assertEqual(spec["edge_index"].shape, (2, 8))
                self.assertEqual(int(graph["edge_mask"].sum()), 4)
                self.assertTrue(bool(np.all(spec["controlled_node_mask"] > 0)))

                disconnected = builder.build(
                    make_obs(line_status=np.asarray([0], dtype=np.float32))
                )["agent_0"]
                line_row = int(spec["line_id_to_node_row"][0])
                line_edges = spec["edge_line_ids"] == 0
                node_cols = {
                    name: index
                    for index, name in enumerate(spec["node_feature_names"])
                }
                self.assertEqual(disconnected["node_mask"][line_row], 1.0)
                self.assertEqual(
                    disconnected["node_features"][
                        line_row, node_cols["line_status"]
                    ],
                    0.0,
                )
                self.assertTrue(
                    bool(np.all(disconnected["edge_mask"][line_edges] == 0.0))
                )

    def test_disconnected_line_remains_a_node_and_masks_attachments(self):
        original = self.builder.build(make_obs())["state"]
        changed = self.builder.build(
            make_obs(
                topo_vect=np.asarray([1, 2, 0, 2], dtype=np.float32),
                line_status=np.asarray([0], dtype=np.float32),
            )
        )["state"]

        self.assertEqual(original["node_features"].shape, changed["node_features"].shape)
        self.assertEqual(original["edge_features"].shape, changed["edge_features"].shape)
        self.assertEqual(int(changed["edge_mask"].sum()), 2)

        spec = self.builder.specs["state"]
        node_cols = {
            name: idx for idx, name in enumerate(spec["node_feature_names"])
        }
        line_row = int(
            np.nonzero(
                spec["node_type"] == self.builder.NODE_TYPE_TRANSMISSION_LINE
            )[0][0]
        )
        gen_row = int(
            np.nonzero(spec["node_type"] == self.builder.NODE_TYPE_GENERATOR)[0][0]
        )
        self.assertEqual(
            changed["node_features"][line_row, node_cols["connected"]], 0.0
        )
        self.assertEqual(
            changed["node_features"][line_row, node_cols["line_status"]], 0.0
        )
        self.assertEqual(
            changed["node_features"][line_row, node_cols["time_before_cooldown_line"]],
            3.0,
        )
        self.assertEqual(
            changed["node_features"][gen_row, node_cols["connected"]], 0.0
        )
        self.assertEqual(changed["energized_node_mask"][line_row], 0.0)
        self.assertEqual(int(changed["energized_node_mask"].sum()), 2)

    def test_one_way_relations_can_make_busbars_the_aggregators(self):
        builder = HeterogeneousLineGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
            generator_edge_direction="asset_to_busbar",
            load_edge_direction="asset_to_busbar",
            line_node_edge_direction="line_to_busbar",
        )
        spec = builder.specs["state"]
        graph = builder.build(make_obs())["state"]

        self.assertEqual(spec["edge_index"].shape, (2, 8))
        self.assertEqual(int(graph["edge_mask"].sum()), 4)
        present_types = set(spec["edge_type"].tolist())
        self.assertNotIn(builder.EDGE_TYPE_BUSBAR_TO_GENERATOR, present_types)
        self.assertNotIn(builder.EDGE_TYPE_BUSBAR_TO_LOAD, present_types)
        self.assertNotIn(builder.EDGE_TYPE_BUSBAR_TO_LINE_ORIGIN, present_types)
        self.assertNotIn(
            builder.EDGE_TYPE_BUSBAR_TO_LINE_EXTREMITY, present_types
        )
        self.assertIn(builder.EDGE_TYPE_GENERATOR_TO_BUSBAR, present_types)
        self.assertIn(builder.EDGE_TYPE_LOAD_TO_BUSBAR, present_types)
        self.assertIn(builder.EDGE_TYPE_LINE_ORIGIN_TO_BUSBAR, present_types)
        self.assertIn(builder.EDGE_TYPE_LINE_EXTREMITY_TO_BUSBAR, present_types)

    def test_optional_relations_and_encoders(self):
        builder = HeterogeneousLineGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=True,
            add_self_edges=True,
            add_substation_edges=True,
        )
        spec = builder.specs["state"]
        graph = builder.build(make_obs())["state"]
        self.assertEqual(spec["edge_index"].shape, (2, 27))
        self.assertEqual(int(graph["edge_mask"].sum()), 19)

        tensor_graph = {
            key: th.tensor(value)
            for key, value in graph.items()
            if key in {"node_features", "edge_features", "node_mask", "edge_mask"}
        }
        encoder = GraphEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=1,
            conv_type="gine",
            node_pre_encoder=True,
            node_id_embeddings=True,
            node_id_emb_dim=2,
        )
        embedding = encoder(tensor_graph)
        self.assertEqual(tuple(embedding.shape), (4,))
        self.assertTrue(bool(th.isfinite(embedding).all()))

        sparse_tensor_graph = {
            key: th.tensor(value)
            for key, value in graph.items()
            if key
            in {
                "node_features",
                "edge_features",
                "node_mask",
                "edge_mask",
                "edge_type",
                "controlled_node_mask",
                "energized_node_mask",
            }
        }
        sparse_encoder = SparseGraphTransformerEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=1,
            heads=2,
            node_id_embeddings=True,
            node_id_emb_dim=2,
        )
        sparse_embedding = sparse_encoder(sparse_tensor_graph)
        self.assertEqual(tuple(sparse_embedding.shape), (4,))
        self.assertTrue(bool(th.isfinite(sparse_embedding).all()))

        energized_sparse_encoder = SparseGraphTransformerEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=1,
            heads=2,
            readout_aggr="energized_mean",
        )
        energized_embedding = energized_sparse_encoder(sparse_tensor_graph)
        self.assertEqual(tuple(energized_embedding.shape), (4,))
        self.assertTrue(bool(th.isfinite(energized_embedding).all()))

    def test_virtual_node_readout_across_encoders(self):
        spec = self.builder.specs["state"]
        graph = self.builder.build(make_obs())["state"]
        tensor_graph = {
            key: th.stack([th.tensor(value), th.tensor(value)])
            for key, value in graph.items()
            if key in {"node_features", "edge_features", "node_mask", "edge_mask"}
        }

        for conv_type in ["gcn", "gat", "gine", "graphsage"]:
            with self.subTest(conv_type=conv_type):
                encoder = GraphEncoder(
                    spec,
                    hidden_dim=8,
                    out_dim=4,
                    n_layers=2,
                    conv_type=conv_type,
                    heads=2,
                    readout_aggr="virtual_node",
                    node_pre_encoder=True,
                    edge_pre_encoder=True,
                    node_id_embeddings=True,
                    node_id_emb_dim=2,
                    add_substation_nodes=True,
                    gcn_edge_weight_feature="none",
                )
                self.assertEqual(int(encoder.virtual_busbar_mask.sum()), 4)
                embedding = encoder(tensor_graph)
                self.assertEqual(tuple(embedding.shape), (2, 4))
                self.assertTrue(bool(th.isfinite(embedding).all()))
                embedding.sum().backward()
                self.assertIsNotNone(encoder.virtual_node_embedding.grad)
                self.assertIsNotNone(encoder.substation_node_encoder.weight.grad)

        sparse_tensor_graph = {
            key: th.stack([th.tensor(value), th.tensor(value)])
            for key, value in graph.items()
            if key
            in {
                "node_features",
                "edge_features",
                "node_mask",
                "edge_mask",
                "edge_type",
                "controlled_node_mask",
            }
        }
        sparse_encoder = SparseGraphTransformerEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=2,
            heads=2,
            readout_aggr="virtual_node",
            edge_pre_encoder=True,
            node_id_embeddings=True,
            node_id_emb_dim=2,
            add_substation_nodes=True,
        )
        self.assertEqual(int(sparse_encoder.virtual_busbar_mask.sum()), 4)
        self.assertEqual(sparse_encoder.n_edge_types, spec["n_edge_types"] + 2)
        sparse_embedding = sparse_encoder(sparse_tensor_graph)
        self.assertEqual(tuple(sparse_embedding.shape), (2, 4))
        self.assertTrue(bool(th.isfinite(sparse_embedding).all()))
        sparse_embedding.sum().backward()
        self.assertIsNotNone(sparse_encoder.virtual_node_embedding.grad)
        self.assertIsNotNone(
            sparse_encoder.substation_node_encoder.weight.grad
        )

    def test_substation_nodes_connect_only_to_their_busbars(self):
        spec = self.builder.specs["state"]
        graph = self.builder.build(make_obs())["state"]
        tensor_graph = {
            key: th.tensor(value)
            for key, value in graph.items()
            if key in {"node_features", "edge_features", "node_mask", "edge_mask"}
        }
        encoder = GraphEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=2,
            conv_type="gine",
            readout_aggr="virtual_node",
            node_pre_encoder=True,
            add_substation_nodes=True,
        )

        (
            x,
            edge_index,
            edge_attr,
            batch,
            node_mask,
            flat_node_ids,
            _controlled_node_mask,
            _energized_node_mask,
        ) = encoder._to_pyg_batch(tensor_graph)
        x = encoder.node_pre_encoder(x)
        (
            x,
            edge_index,
            edge_attr,
            batch,
            node_mask,
            _,
            _,
            virtual_indices,
        ) = encoder._append_hierarchy_nodes(
            x,
            edge_index,
            edge_attr,
            batch,
            flat_node_ids,
            node_mask=node_mask,
        )

        # Seven physical nodes + two substation nodes + one graph node.
        self.assertEqual(x.shape[0], 10)
        self.assertEqual(virtual_indices.tolist(), [9])
        # Eight active physical edges + eight busbar/substation edges + four
        # substation/virtual-node edges.
        self.assertEqual(edge_index.shape[1], 20)
        substation_edges = {
            tuple(edge)
            for edge in edge_index[:, -12:-4].T.tolist()
        }
        self.assertEqual(
            substation_edges,
            {
                (0, 7),
                (1, 7),
                (7, 0),
                (7, 1),
                (2, 8),
                (3, 8),
                (8, 2),
                (8, 3),
            },
        )
        virtual_edges = {tuple(edge) for edge in edge_index[:, -4:].T.tolist()}
        self.assertEqual(
            virtual_edges,
            {(7, 9), (8, 9), (9, 7), (9, 8)},
        )
        self.assertTrue(bool(th.all(edge_attr[-12:] == 0.0)))

        embedding = encoder(tensor_graph)
        self.assertEqual(tuple(embedding.shape), (4,))
        self.assertTrue(bool(th.isfinite(embedding).all()))

    def test_toward_summary_hierarchy_removes_reverse_messages(self):
        spec = self.builder.specs["state"]
        graph = self.builder.build(make_obs())["state"]
        tensor_graph = {
            key: th.tensor(value)
            for key, value in graph.items()
            if key in {"node_features", "edge_features", "node_mask", "edge_mask"}
        }
        encoder = GraphEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=2,
            conv_type="gine",
            readout_aggr="virtual_node",
            node_pre_encoder=True,
            add_substation_nodes=True,
            summary_edge_direction="toward_summary",
        )

        (
            x,
            edge_index,
            edge_attr,
            batch,
            node_mask,
            flat_node_ids,
            _controlled_node_mask,
            _energized_node_mask,
        ) = encoder._to_pyg_batch(tensor_graph)
        x = encoder.node_pre_encoder(x)
        (
            _,
            edge_index,
            edge_attr,
            _,
            _,
            _,
            _,
            virtual_indices,
        ) = encoder._append_hierarchy_nodes(
            x,
            edge_index,
            edge_attr,
            batch,
            flat_node_ids,
            node_mask=node_mask,
        )

        self.assertEqual(virtual_indices.tolist(), [9])
        self.assertEqual(edge_index.shape[1], 14)
        self.assertEqual(
            {tuple(edge) for edge in edge_index[:, -6:-2].T.tolist()},
            {(0, 7), (1, 7), (2, 8), (3, 8)},
        )
        self.assertEqual(
            {tuple(edge) for edge in edge_index[:, -2:].T.tolist()},
            {(7, 9), (8, 9)},
        )
        self.assertTrue(bool(th.all(edge_attr[-6:] == 0.0)))

    def test_only_virtual_node_edges_can_be_one_way(self):
        spec = self.builder.specs["state"]
        graph = self.builder.build(make_obs())["state"]
        tensor_graph = {
            key: th.tensor(value)
            for key, value in graph.items()
            if key in {"node_features", "edge_features", "node_mask", "edge_mask"}
        }
        encoder = GraphEncoder(
            spec,
            hidden_dim=8,
            out_dim=4,
            n_layers=2,
            conv_type="gine",
            readout_aggr="virtual_node",
            node_pre_encoder=True,
            add_substation_nodes=True,
            summary_edge_direction="bidirectional",
            virtual_edge_direction="toward_virtual",
        )

        (
            x,
            edge_index,
            edge_attr,
            batch,
            node_mask,
            flat_node_ids,
            _controlled_node_mask,
            _energized_node_mask,
        ) = encoder._to_pyg_batch(tensor_graph)
        x = encoder.node_pre_encoder(x)
        (
            _,
            edge_index,
            edge_attr,
            _,
            _,
            _,
            _,
            virtual_indices,
        ) = encoder._append_hierarchy_nodes(
            x,
            edge_index,
            edge_attr,
            batch,
            flat_node_ids,
            node_mask=node_mask,
        )

        self.assertEqual(virtual_indices.tolist(), [9])
        self.assertEqual(edge_index.shape[1], 18)
        self.assertEqual(
            {tuple(edge) for edge in edge_index[:, -10:-2].T.tolist()},
            {
                (0, 7),
                (1, 7),
                (7, 0),
                (7, 1),
                (2, 8),
                (3, 8),
                (8, 2),
                (8, 3),
            },
        )
        self.assertEqual(
            {tuple(edge) for edge in edge_index[:, -2:].T.tolist()},
            {(7, 9), (8, 9)},
        )
        self.assertTrue(bool(th.all(edge_attr[-10:] == 0.0)))

        embedding = encoder(tensor_graph)
        self.assertEqual(tuple(embedding.shape), (4,))
        self.assertTrue(bool(th.isfinite(embedding).all()))


if __name__ == "__main__":
    unittest.main()
