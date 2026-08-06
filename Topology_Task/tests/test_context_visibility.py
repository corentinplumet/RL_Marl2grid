"""A contextual node is observable only while a live line ties it to the agent.

Before this rule the local graph carried every busbar of every neighbouring
substation, whatever the topology, so an agent pooled state it had no
electrical path to.
"""

from types import SimpleNamespace
import unittest

import numpy as np

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
    offset past the busbar block, so only busbar rows can be named this way.
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
            labels.append(f"s{node_id // n_busbar}b{node_id % n_busbar}")
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

    def test_disabling_the_rule_restores_the_previous_behaviour(self):
        _, graph = self.build(GridGraphBuilder, [1, 1, 1, 1, 1, 1], gated=False)
        self.assertTrue(np.all(graph["node_mask"] > 0))

    def test_contextual_assets_follow_their_busbar(self):
        # The generator sits at substation 0. It is visible while its busbar is
        # attached to the agent's region and hidden once line 0 goes out.
        spec, graph = self.build(
            HeterogeneousGridGraphBuilder, [1, 1, 1, 1, 1, 1]
        )
        node_type = spec["node_type"]
        generator = node_type == spec["node_type_names"]["generator"]
        self.assertTrue(np.all(graph["node_mask"][generator] > 0))

        _, cut = self.build(
            HeterogeneousGridGraphBuilder,
            [1, 1, 1, 1, 1, 1],
            line_status=(0, 1),
        )
        self.assertTrue(np.all(cut["node_mask"][generator] == 0))

    def test_explicit_line_schema_reaches_through_the_line_node(self):
        # Here a busbar connects to its neighbour through a line node, so
        # visibility has to propagate over two hops.
        spec, graph = self.build(
            HeterogeneousLineGraphBuilder, [1, 1, 1, 1, 1, 1]
        )
        visible = self.visible_busbars(spec, graph)
        self.assertIn("s0b0", visible)
        self.assertNotIn("s0b1", visible)


if __name__ == "__main__":
    unittest.main()
