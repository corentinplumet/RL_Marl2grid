"""A contextual node is observable only while a live line ties it to the agent.

Before this rule the local graph carried every busbar of every neighbouring
substation, whatever the topology, so an agent pooled state it had no
electrical path to.
"""

from types import SimpleNamespace
import unittest

import numpy as np

import torch as th

from common.gnn import GraphEncoder
from common.graph import (
    GridGraphBuilder,
    HeterogeneousGridGraphBuilder,
    HeterogeneousLineGraphBuilder,
)


class ChainEnv:
    """sub0 --line0-- sub1 --line1-- sub2, a generator at sub0, a load at sub2."""

    n_sub, n_line, n_gen, n_load = 3, 2, 1, 1
    n_busbar_per_sub = np.asarray([2, 2, 2], dtype=np.int64)
    line_or_to_subid = np.asarray([0, 1], dtype=np.int64)
    line_ex_to_subid = np.asarray([1, 2], dtype=np.int64)
    gen_to_subid = np.asarray([0], dtype=np.int64)
    load_to_subid = np.asarray([2], dtype=np.int64)
    line_or_pos_topo_vect = np.asarray([0, 2], dtype=np.int64)
    line_ex_pos_topo_vect = np.asarray([1, 3], dtype=np.int64)
    gen_pos_topo_vect = np.asarray([4], dtype=np.int64)
    load_pos_topo_vect = np.asarray([5], dtype=np.int64)


def observation(topo_vect, line_status=(1, 1)):
    return SimpleNamespace(
        topo_vect=np.asarray(topo_vect, dtype=np.int64),
        line_status=np.asarray(line_status, dtype=np.float32),
        rho=np.zeros(2, dtype=np.float32),
        timestep_overflow=np.zeros(2, dtype=np.float32),
        time_before_cooldown_line=np.zeros(2, dtype=np.float32),
        time_before_cooldown_sub=np.zeros(3, dtype=np.float32),
        gen_p=np.asarray([5.0], dtype=np.float32),
        gen_theta=np.zeros(1, dtype=np.float32),
        load_p=np.asarray([4.0], dtype=np.float32),
        load_theta=np.zeros(1, dtype=np.float32),
    )


def busbar_labels(spec, n_busbar=2):
    """``s{substation}b{busbar}`` for busbar rows, ``None`` for other types.

    The disaggregated schemas append asset and line rows whose identifiers are
    offset past the busbar block and use a wider node-id stride. Their
    ``entity_ids`` retain the global busbar indexing used for these labels.
    """
    node_type = spec.get("node_type")
    busbar_type = (
        spec.get("node_type_names", {}).get("busbar") if node_type is not None else None
    )
    labels = []
    for index, node_id in enumerate(spec["node_ids"]):
        if node_type is not None and node_type[index] != busbar_type:
            labels.append(None)
        else:
            busbar_id = (
                spec["entity_ids"][index]
                if "entity_ids" in spec
                else node_id
            )
            labels.append(
                f"s{busbar_id // n_busbar}b{busbar_id % n_busbar}"
            )
    return labels


class ContextVisibilityTest(unittest.TestCase):
    """The agent controls substation 1; substations 0 and 2 are contextual."""

    def build(self, builder_cls, topo, line_status=(1, 1), gated=True, **kwargs):
        builder = builder_cls(
            ChainEnv(),
            {"agent_0": [1]},
            include_neighbors=True,
            context_requires_connection=gated,
            **kwargs,
        )
        spec = builder.specs["agent_0"]
        graph = builder.build(observation(topo, line_status))["agent_0"]
        return spec, graph

    def visible_busbars(self, spec, graph):
        names = busbar_labels(spec)
        return {
            name
            for name, keep in zip(names, graph["node_mask"])
            if name is not None and keep > 0
        }

    def test_only_the_attached_busbar_of_a_neighbour_is_visible(self):
        spec, graph = self.build(GridGraphBuilder, [1, 1, 1, 1, 1, 1])
        visible = self.visible_busbars(spec, graph)
        self.assertIn("s1b0", visible)          # controlled, always kept
        self.assertIn("s1b1", visible)
        self.assertIn("s0b0", visible)          # neighbour, line attached here
        self.assertIn("s2b0", visible)
        self.assertNotIn("s0b1", visible)       # neighbour, nothing attached
        self.assertNotIn("s2b1", visible)

    def test_invisible_candidate_rows_contain_no_neighbor_state(self):
        spec, graph = self.build(GridGraphBuilder, [1, 1, 1, 1, 1, 1])
        hidden_context = (
            (graph["node_mask"] == 0)
            & (spec["controlled_node_mask"] == 0)
        )
        self.assertTrue(bool(np.any(hidden_context)))
        self.assertTrue(
            bool(np.all(graph["node_features"][hidden_context] == 0.0))
        )
        touches_hidden = np.isin(
            spec["edge_index"], np.nonzero(hidden_context)[0]
        ).any(axis=0)
        self.assertTrue(bool(np.any(touches_hidden)))
        self.assertTrue(
            bool(np.all(graph["edge_mask"][touches_hidden] == 0.0))
        )

    def test_structural_edges_cannot_reactivate_an_invisible_candidate(self):
        spec, graph = self.build(
            GridGraphBuilder,
            [1, 1, 1, 1, 1, 1],
            add_substation_edges=True,
        )
        hidden = graph["node_mask"] == 0
        touches_hidden = np.isin(
            spec["edge_index"], np.nonzero(hidden)[0]
        ).any(axis=0)
        self.assertTrue(bool(np.any(touches_hidden)))
        self.assertTrue(
            bool(np.all(graph["edge_mask"][touches_hidden] == 0.0))
        )

    def test_visibility_follows_the_topology(self):
        # Move line 0's far end onto the second busbar of substation 0.
        spec, graph = self.build(GridGraphBuilder, [2, 1, 1, 1, 1, 1])
        visible = self.visible_busbars(spec, graph)
        self.assertIn("s0b1", visible)
        self.assertNotIn("s0b0", visible)

    def test_a_disconnected_line_hides_its_whole_substation(self):
        spec, graph = self.build(
            GridGraphBuilder, [1, 1, 1, 1, 1, 1], line_status=(1, 0)
        )
        visible = self.visible_busbars(spec, graph)
        self.assertNotIn("s2b0", visible)
        self.assertNotIn("s2b1", visible)
        self.assertIn("s1b0", visible)

    def test_controlled_nodes_are_never_hidden(self):
        # Both of substation 1's own busbars stay valid even with every line out.
        spec, graph = self.build(
            GridGraphBuilder, [1, 1, 1, 1, 1, 1], line_status=(0, 0)
        )
        controlled = np.asarray(spec["controlled_node_mask"], dtype=bool)
        self.assertTrue(np.all(graph["node_mask"][controlled] > 0))

    def test_same_substation_edges_do_not_make_a_neighbour_visible(self):
        # A same-substation relation is computational, not electrical, so it
        # must not carry visibility to the busbar the line is not attached to.
        spec, graph = self.build(
            GridGraphBuilder, [1, 1, 1, 1, 1, 1], add_substation_edges=True
        )
        self.assertNotIn("s0b1", self.visible_busbars(spec, graph))

    def test_energized_mask_uses_only_active_electrical_relations(self):
        spec, graph = self.build(
            GridGraphBuilder,
            [1, 1, 1, 1, 1, 1],
            add_self_edges=True,
            add_substation_edges=True,
        )
        labels = busbar_labels(spec)
        energized = {
            label
            for label, keep in zip(labels, graph["energized_node_mask"])
            if keep > 0
        }
        self.assertEqual(energized, {"s0b0", "s1b0", "s2b0"})

        _, disconnected = self.build(
            GridGraphBuilder,
            [1, 1, 1, 1, 1, 1],
            line_status=(0, 0),
            add_self_edges=True,
            add_substation_edges=True,
        )
        self.assertEqual(float(disconnected["energized_node_mask"].sum()), 0.0)

    def test_disabling_the_rule_restores_the_previous_behaviour(self):
        _, graph = self.build(GridGraphBuilder, [1, 1, 1, 1, 1, 1], gated=False)
        self.assertTrue(np.all(graph["node_mask"] > 0))

    def test_summary_node_parameters_are_grid_independent(self):
        """Every learned tensor must keep its shape on a larger grid.

        A per-substation embedding would be shaped by the number of
        substations, which makes a checkpoint unusable on any other network.
        """

        class LongerChain(ChainEnv):
            n_sub, n_line = 6, 5
            n_busbar_per_sub = np.asarray([2] * 6, dtype=np.int64)
            line_or_to_subid = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
            line_ex_to_subid = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
            gen_to_subid = np.asarray([0], dtype=np.int64)
            load_to_subid = np.asarray([5], dtype=np.int64)
            line_or_pos_topo_vect = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
            line_ex_pos_topo_vect = np.asarray([1, 3, 5, 7, 9], dtype=np.int64)
            gen_pos_topo_vect = np.asarray([10], dtype=np.int64)
            load_pos_topo_vect = np.asarray([11], dtype=np.int64)

        shapes = []
        for env in (ChainEnv(), LongerChain()):
            builder = GridGraphBuilder(env, {"a": [1]}, include_neighbors=True)
            encoder = GraphEncoder(
                builder.specs["a"], hidden_dim=8, out_dim=6, n_layers=2,
                conv_type="gine", add_substation_nodes=True,
                readout_aggr="virtual_node",
            )
            shapes.append(
                {name: tuple(tensor.shape)
                 for name, tensor in encoder.state_dict().items()}
            )
        small, large = shapes
        shared = set(small) & set(large)
        differing = {
            name for name in shared if small[name] != large[name]
        }
        # Only the buffers describing the graph itself may differ.
        self.assertEqual(
            differing, {"edge_index", "node_ids", "edge_type"} & differing
        )
        self.assertNotIn("substation_features", small)

    def test_summary_node_reads_its_substation_structure(self):
        # The descriptor must actually distinguish substations, otherwise the
        # projection degenerates to one shared vector.
        builder = GridGraphBuilder(ChainEnv(), {"a": [1]}, include_neighbors=True)
        features = np.asarray(builder.specs["a"]["substation_features"])
        names = builder.specs["a"]["substation_feature_names"]
        self.assertEqual(names[1], "line_count")
        # The middle substation carries two lines, the outer ones carry one.
        self.assertGreater(features[1, 1], features[0, 1])
        self.assertEqual(features[0, 1], features[2, 1])
        # And the generator and load sit at opposite ends.
        self.assertGreater(features[0, 2], features[2, 2])
        self.assertGreater(features[2, 3], features[0, 3])

    def test_same_substation_edges_stay_inside_the_controlled_domain(self):
        # The relation says an element can be moved between two busbars, which
        # outside the controlled domain is another agent's action. Two busbars
        # of a neighbouring substation must not be related by it.
        spec, _ = self.build(
            GridGraphBuilder, [1, 1, 1, 1, 1, 1], add_substation_edges=True
        )
        controlled = np.asarray(spec["controlled_node_mask"], dtype=bool)
        edge_type = np.asarray(spec["edge_type"])
        structural = edge_type == spec["edge_type_names"]["same_substation_busbar"]
        self.assertTrue(structural.any())
        src, dst = np.asarray(spec["edge_index"])[:, structural]
        self.assertTrue(np.all(controlled[src] & controlled[dst]))

    def test_legacy_flag_restores_contextual_same_substation_edges(self):
        builder = GridGraphBuilder(
            ChainEnv(),
            {"agent_0": [1]},
            include_neighbors=True,
            add_substation_edges=True,
            structural_relations_controlled_only=False,
        )
        spec = builder.specs["agent_0"]
        controlled = np.asarray(spec["controlled_node_mask"], dtype=bool)
        edge_type = np.asarray(spec["edge_type"])
        structural = edge_type == spec["edge_type_names"]["same_substation_busbar"]
        src, dst = np.asarray(spec["edge_index"])[:, structural]
        self.assertFalse(np.all(controlled[src] & controlled[dst]))

    def test_summary_nodes_gather_only_controlled_busbars(self):
        # The same relation reached through a summary node: a contextual busbar
        # must not connect to one, or two neighbouring busbars would exchange
        # through it even with the direct edge removed.
        spec, graph = self.build(GridGraphBuilder, [1, 1, 1, 1, 1, 1])
        tensors = {
            key: th.tensor(np.asarray(value))
            for key, value in graph.items()
            if key in {
                "node_features", "edge_features", "node_mask",
                "edge_mask", "edge_type", "controlled_node_mask",
                "energized_node_mask",
            }
        }
        encoder = GraphEncoder(
            spec, hidden_dim=8, out_dim=6, n_layers=2, conv_type="gine",
            add_substation_nodes=True,
        )
        x, edge_index, edge_attr, batch, node_mask, ids, controlled, _ = (
            encoder._to_pyg_batch(tensors)
        )
        x = encoder.node_pre_encoder(encoder._append_node_id_embeddings(x, ids))
        n_real = x.shape[0]
        _, edge_index = encoder._append_hierarchy_nodes(
            x, edge_index, edge_attr, batch, ids,
            node_mask=node_mask, controlled_node_mask=controlled,
        )[:2]
        is_controlled = np.asarray(spec["controlled_node_mask"], dtype=bool)
        for src, dst in zip(*edge_index.tolist()):
            if src < n_real and dst >= n_real:
                self.assertTrue(is_controlled[src])
            if dst < n_real and src >= n_real:
                self.assertTrue(is_controlled[dst])

    def test_substation_nodes_do_not_wire_an_invisible_busbar(self):
        # Substation summary nodes are appended by the encoder, after the
        # builder has decided visibility, so the visibility rule cannot gate
        # them. They must aggregate only visible busbars, or an unreachable
        # contextual busbar would reach the agent through its own substation's
        # summary node.
        spec, graph = self.build(GridGraphBuilder, [1, 1, 1, 1, 1, 1])
        tensors = {
            key: th.tensor(np.asarray(value))
            for key, value in graph.items()
            if key in {
                "node_features",
                "edge_features",
                "node_mask",
                "edge_mask",
                "edge_type",
                "controlled_node_mask",
                "energized_node_mask",
            }
        }
        encoder = GraphEncoder(
            spec,
            hidden_dim=8,
            out_dim=6,
            n_layers=2,
            conv_type="gine",
            add_substation_nodes=True,
        )
        x, edge_index, edge_attr, batch, node_mask, ids, controlled, _ = (
            encoder._to_pyg_batch(tensors)
        )
        x = encoder.node_pre_encoder(encoder._append_node_id_embeddings(x, ids))
        _, edge_index = encoder._append_hierarchy_nodes(
            x,
            edge_index,
            edge_attr,
            batch,
            ids,
            node_mask=node_mask,
            controlled_node_mask=controlled,
            substation_edge_type=spec["edge_type_names"].get(
                "same_substation_busbar"
            ),
        )[:2]
        hidden = np.nonzero(~(np.asarray(graph["node_mask"]) > 0))[0]
        self.assertGreater(len(hidden), 0)
        for row in hidden:
            incident = int(
                (edge_index[0] == int(row)).sum()
                + (edge_index[1] == int(row)).sum()
            )
            self.assertEqual(incident, 0)

    def test_heterogeneous_context_contains_no_neighbor_assets(self):
        # The generator at substation 0 and load at substation 2 belong to
        # neighboring agents. Only their boundary busbars may enter this local
        # heterogeneous graph; the assets must not even be instantiated.
        spec, graph = self.build(
            HeterogeneousGridGraphBuilder, [1, 1, 1, 1, 1, 1]
        )
        node_type = spec["node_type"]
        self.assertFalse(
            bool(np.any(node_type == spec["node_type_names"]["generator"]))
        )
        self.assertFalse(
            bool(np.any(node_type == spec["node_type_names"]["load"]))
        )
        self.assertEqual(
            self.visible_busbars(spec, graph),
            {"s0b0", "s1b0", "s1b1", "s2b0"},
        )

    def test_explicit_line_schema_has_shared_lines_but_no_neighbor_nodes(self):
        # Both boundary lines are local controlled nodes, but neither far-end
        # substation contributes a busbar, generator, or load node.
        spec, graph = self.build(
            HeterogeneousLineGraphBuilder, [1, 1, 1, 1, 1, 1]
        )
        node_type = spec["node_type"]
        names = spec["node_type_names"]
        self.assertEqual(self.visible_busbars(spec, graph), {"s1b0", "s1b1"})
        self.assertEqual(
            int(np.sum(node_type == names["transmission_line"])), 2
        )
        self.assertFalse(bool(np.any(node_type == names["generator"])))
        self.assertFalse(bool(np.any(node_type == names["load"])))
        self.assertTrue(bool(np.all(spec["controlled_node_mask"] > 0)))


class ControlledReadoutTest(unittest.TestCase):
    """``controlled_*`` pools only the nodes the agent acts on.

    Contextual neighbours still take part in message passing, so they reach the
    readout through the controlled nodes, but they do not enter the average.
    The size of the pooled set is then fixed instead of moving with the
    topology.
    """

    ENCODER_TENSORS = {
        "node_features",
        "edge_features",
        "node_mask",
        "edge_mask",
        "edge_type",
        "controlled_node_mask",
        "energized_node_mask",
    }

    def setUp(self):
        self.builder = GridGraphBuilder(
            ChainEnv(), {"agent_0": [1]}, include_neighbors=True
        )
        self.spec = self.builder.specs["agent_0"]

    def graph(self, topo, line_status=(1, 1)):
        built = self.builder.build(observation(topo, line_status))["agent_0"]
        return {
            key: th.tensor(value)
            for key, value in built.items()
            if key in self.ENCODER_TENSORS
        }

    def encoder(self, readout):
        th.manual_seed(0)
        return GraphEncoder(
            self.spec,
            hidden_dim=8,
            out_dim=6,
            n_layers=2,
            conv_type="gine",
            readout_aggr=readout,
            node_pre_encoder=True,
            edge_pre_encoder=True,
        )

    TOPOLOGIES = {
        "nominal": ([1, 1, 1, 1, 1, 1], (1, 1)),
        "neighbour split": ([2, 1, 1, 1, 1, 1], (1, 1)),
        "line out": ([1, 1, 1, 1, 1, 1], (1, 0)),
    }

    def test_plain_mean_pools_a_topology_dependent_number_of_nodes(self):
        sizes = {
            int(self.graph(topo, status)["node_mask"].sum())
            for topo, status in self.TOPOLOGIES.values()
        }
        self.assertGreater(len(sizes), 1)

    def test_controlled_readout_pools_a_fixed_number_of_nodes(self):
        sizes = {
            int(
                (
                    self.graph(topo, status)["node_mask"]
                    * self.graph(topo, status)["controlled_node_mask"]
                ).sum()
            )
            for topo, status in self.TOPOLOGIES.values()
        }
        self.assertEqual(sizes, {2})

    def test_context_still_reaches_the_readout(self):
        # If contextual nodes were cut out of message passing too, the readout
        # would be blind to them and these embeddings would coincide.
        encoder = self.encoder("controlled_mean")
        embeddings = [
            encoder(self.graph(topo, status)).detach()
            for topo, status in self.TOPOLOGIES.values()
        ]
        for other in embeddings[1:]:
            self.assertFalse(th.allclose(embeddings[0], other, atol=1e-6))

    def test_controlled_mean_is_the_mean_over_controlled_rows(self):
        encoder = self.encoder("controlled_mean")
        graph = self.graph([1, 1, 1, 1, 1, 1])
        readout, nodes = encoder.forward_with_nodes(graph)
        controlled = graph["controlled_node_mask"].bool()
        expected = encoder.readout(nodes[controlled].mean(dim=0, keepdim=True))
        self.assertTrue(th.allclose(readout, expected.squeeze(0), atol=1e-5))

    def test_controlled_max_is_supported(self):
        readout = self.encoder("controlled_max")(self.graph([1, 1, 1, 1, 1, 1]))
        self.assertTrue(bool(th.isfinite(readout).all()))

    def test_energized_mean_is_the_mean_over_energized_rows(self):
        encoder = self.encoder("energized_mean")
        graph = self.graph([1, 1, 1, 1, 1, 1])
        readout, nodes = encoder.forward_with_nodes(graph)
        energized = graph["energized_node_mask"].bool()
        expected = encoder.readout(nodes[energized].mean(dim=0, keepdim=True))
        self.assertTrue(th.allclose(readout, expected.squeeze(0), atol=1e-5))

    def test_controlled_energized_mean_uses_the_mask_intersection(self):
        encoder = self.encoder("controlled_energized_mean")
        graph = self.graph([1, 1, 1, 1, 1, 1])
        readout, nodes = encoder.forward_with_nodes(graph)
        pooled = (
            graph["node_mask"].bool()
            & graph["controlled_node_mask"].bool()
            & graph["energized_node_mask"].bool()
        )
        expected = encoder.readout(nodes[pooled].mean(dim=0, keepdim=True))
        self.assertTrue(th.allclose(readout, expected.squeeze(0), atol=1e-5))

    def test_every_declared_readout_composes(self):
        """``[controlled_][energized_]{mean,sum,max}`` must all be runnable.

        The two restrictions are independent, so the name set is generated
        rather than listed. This checks the generator and the pooling code
        agree, which is what stops a declared option failing only at launch.
        """
        from common.gnn import POOLING_AGGREGATIONS

        graph = self.graph([1, 1, 1, 1, 1, 1])
        for readout in POOLING_AGGREGATIONS:
            if readout == "virtual_node":
                continue
            with self.subTest(readout=readout):
                output = self.encoder(readout)(graph)
                self.assertTrue(bool(th.isfinite(output).all()))

    def test_every_gate_accepts_every_declared_readout(self):
        """The parser and both config validators must admit the same names.

        A readout accepted by the encoder but rejected by a launcher fails only
        when a job starts, after it has queued. That happened once: the guard in
        ``run_from_config`` carried its own hand-written copy of the list.
        """
        import inspect
        import alg.mappo.config as config_module
        import run_from_config
        from common.readouts import POOLING_AGGREGATIONS

        offered = inspect.getsource(config_module.get_alg_args)
        offered = offered.split("--gnn-readout-aggr", 1)[1].split("]", 1)[0]
        for readout in POOLING_AGGREGATIONS:
            with self.subTest(readout=readout, gate="argument parser"):
                self.assertIn(f'"{readout}"', offered)

        # The launcher guard must consult the shared set, not a copy of it.
        guard = inspect.getsource(run_from_config)
        self.assertIn("POOLING_AGGREGATIONS", guard)

    def test_missing_controlled_mask_is_rejected(self):
        graph = self.graph([1, 1, 1, 1, 1, 1])
        graph.pop("controlled_node_mask")
        with self.assertRaisesRegex(ValueError, "controlled_node_mask"):
            self.encoder("controlled_mean")(graph)

    def test_missing_energized_mask_is_rejected(self):
        graph = self.graph([1, 1, 1, 1, 1, 1])
        graph.pop("energized_node_mask")
        with self.assertRaisesRegex(ValueError, "energized_node_mask"):
            self.encoder("energized_mean")(graph)

    def test_gradients_reach_the_contextual_nodes(self):
        encoder = self.encoder("controlled_mean")
        graph = self.graph([1, 1, 1, 1, 1, 1])
        graph["node_features"] = graph["node_features"].clone().requires_grad_(True)
        encoder(graph).sum().backward()
        contextual = ~graph["controlled_node_mask"].bool()
        visible = graph["node_mask"].bool()
        rows = contextual & visible
        self.assertGreater(
            float(graph["node_features"].grad[rows].abs().sum()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
