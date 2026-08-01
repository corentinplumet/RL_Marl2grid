from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch as th

from alg.mappo.agent import Actor, CandidateActionScorer
from common.action_metadata import build_action_graph_metadata
from common.gnn import GINEConv, GraphEncoder, SparseGraphTransformerEncoder
from common.graph import HeterogeneousLineGraphBuilder


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


class MockAction:
    def __init__(self, payload):
        self.payload = payload

    def as_dict(self):
        return self.payload


def topology_action(substation_id, object_type, object_id):
    return MockAction(
        {
            "change_bus_vect": {
                "nb_modif_objects": 1,
                str(substation_id): {
                    "object": {"type": object_type, "id": object_id}
                },
                "nb_modif_subs": 1,
                "modif_subs_id": [str(substation_id)],
            }
        }
    )


def make_actions():
    return [
        MockAction({}),
        topology_action(0, "line (origin)", 0),
        topology_action(0, "generator", 0),
        topology_action(1, "load", 0),
        MockAction(
            {"change_line_status": {"nb_changed": 1, "changed_id": [0]}}
        ),
    ]


def make_obs():
    return SimpleNamespace(
        topo_vect=np.asarray([1, 2, 2, 1], dtype=np.float32),
        line_status=np.asarray([1], dtype=np.float32),
        rho=np.asarray([0.75], dtype=np.float32),
        timestep_overflow=np.asarray([2], dtype=np.float32),
        time_before_cooldown_line=np.asarray([3], dtype=np.float32),
        time_before_cooldown_sub=np.asarray([4, 5], dtype=np.float32),
        gen_p=np.asarray([10.0], dtype=np.float32),
        gen_theta=np.asarray([0.1], dtype=np.float32),
        load_p=np.asarray([8.0], dtype=np.float32),
        load_theta=np.asarray([-0.2], dtype=np.float32),
    )


def make_metadata(spec):
    return build_action_graph_metadata(
        spec,
        make_actions(),
        line_or_to_subid=MockGridEnv.line_or_to_subid,
        line_ex_to_subid=MockGridEnv.line_ex_to_subid,
        original_action_ids=[0, 11, 12, 13, 14],
    )


class ActionGraphMetadataTest(unittest.TestCase):
    def setUp(self):
        self.builder = HeterogeneousLineGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0, 1]},
            include_neighbors=True,
        )
        self.spec = self.builder.specs["agent_0"]
        self.metadata = make_metadata(self.spec)

    @staticmethod
    def active_values(indices, mask, action_id):
        return indices[action_id][mask[action_id]].tolist()

    def test_graph_spec_exposes_stable_physical_row_maps(self):
        np.testing.assert_array_equal(
            self.spec["busbar_id_to_node_row"], [0, 1, 2, 3]
        )
        np.testing.assert_array_equal(self.spec["gen_id_to_node_row"], [4])
        np.testing.assert_array_equal(self.spec["load_id_to_node_row"], [5])
        np.testing.assert_array_equal(self.spec["line_id_to_node_row"], [6])
        np.testing.assert_array_equal(
            self.spec["substation_busbar_node_rows"], [[0, 1], [2, 3]]
        )

    def test_actions_map_to_the_objects_and_endpoint_context_they_touch(self):
        metadata = self.metadata
        self.assertEqual(metadata.n_actions, 5)
        self.assertEqual(metadata.original_action_ids.tolist(), [0, 11, 12, 13, 14])
        self.assertFalse(bool(metadata.busbar_mask[0].any()))
        self.assertFalse(bool(metadata.line_mask[0].any()))

        self.assertEqual(
            self.active_values(metadata.busbar_indices, metadata.busbar_mask, 1),
            [0, 1],
        )
        self.assertEqual(
            self.active_values(metadata.line_indices, metadata.line_mask, 1),
            [6],
        )
        self.assertEqual(
            self.active_values(
                metadata.generator_indices, metadata.generator_mask, 2
            ),
            [4],
        )
        self.assertEqual(
            self.active_values(metadata.load_indices, metadata.load_mask, 3),
            [5],
        )
        self.assertEqual(
            self.active_values(metadata.busbar_indices, metadata.busbar_mask, 4),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            self.active_values(metadata.line_indices, metadata.line_mask, 4),
            [6],
        )

    def test_missing_neighbor_line_is_rejected(self):
        local_builder = HeterogeneousLineGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0]},
            include_neighbors=False,
        )
        with self.assertRaisesRegex(ValueError, "gnn_include_neighbors=true"):
            build_action_graph_metadata(
                local_builder.specs["agent_0"],
                [MockAction({}), topology_action(0, "line (origin)", 0)],
                line_or_to_subid=MockGridEnv.line_or_to_subid,
                line_ex_to_subid=MockGridEnv.line_ex_to_subid,
            )


class CandidateActionScorerTest(unittest.TestCase):
    def setUp(self):
        builder = HeterogeneousLineGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0, 1]},
            include_neighbors=True,
        )
        self.metadata = make_metadata(builder.specs["agent_0"])

    def test_typed_and_untyped_pooling_produce_finite_logits_and_gradients(self):
        for pool_mode in ("mean", "typed_mean"):
            with self.subTest(pool_mode=pool_mode):
                scorer = CandidateActionScorer(
                    graph_dim=5,
                    node_dim=7,
                    hidden_layers=[],
                    act_fn_name="relu",
                    metadata=self.metadata,
                    pool_mode=pool_mode,
                )
                graph_embedding = th.randn(3, 5, requires_grad=True)
                node_embeddings = th.randn(3, 7, 7, requires_grad=True)
                logits = scorer(graph_embedding, node_embeddings)

                self.assertEqual(tuple(logits.shape), (3, 5))
                self.assertTrue(bool(th.isfinite(logits).all()))
                logits.sum().backward()
                self.assertIsNotNone(node_embeddings.grad)
                self.assertGreater(float(node_embeddings.grad.abs().sum()), 0.0)

    def test_do_nothing_prior_has_the_requested_initial_probability(self):
        scorer = CandidateActionScorer(
            graph_dim=5,
            node_dim=7,
            hidden_layers=[],
            act_fn_name="relu",
            metadata=self.metadata,
            pool_mode="typed_mean",
        )
        scorer.init_do_nothing_prior(0.7)
        logits = scorer(th.zeros(5), th.zeros(7, 7))
        probability = th.softmax(logits, dim=-1)[0]
        self.assertAlmostEqual(float(probability), 0.7, places=6)


@unittest.skipIf(GINEConv is None, "torch-geometric is not installed")
class CandidateActorIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.builder = HeterogeneousLineGraphBuilder(
            MockGridEnv(),
            {"agent_0": [0, 1]},
            include_neighbors=True,
        )
        self.spec = self.builder.specs["agent_0"]
        self.spec["action_graph_metadata"] = make_metadata(self.spec)
        graph = self.builder.build(make_obs())["agent_0"]
        self.tensor_graph = {
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
            }
        }

    def test_encoder_forward_with_nodes_preserves_graph_readout(self):
        encoder = GraphEncoder(
            self.spec,
            hidden_dim=8,
            out_dim=6,
            n_layers=2,
            conv_type="gine",
            node_pre_encoder=True,
            edge_pre_encoder=True,
        )
        old_style_embedding = encoder(self.tensor_graph)
        graph_embedding, node_embeddings = encoder.forward_with_nodes(
            self.tensor_graph
        )
        self.assertTrue(th.equal(old_style_embedding, graph_embedding))
        self.assertEqual(tuple(graph_embedding.shape), (6,))
        self.assertEqual(tuple(node_embeddings.shape), (7, 8))

        sparse_encoder = SparseGraphTransformerEncoder(
            self.spec,
            hidden_dim=8,
            out_dim=6,
            n_layers=2,
            heads=2,
        )
        sparse_old_style = sparse_encoder(self.tensor_graph)
        sparse_graph, sparse_nodes = sparse_encoder.forward_with_nodes(
            self.tensor_graph
        )
        self.assertTrue(th.equal(sparse_old_style, sparse_graph))
        self.assertEqual(tuple(sparse_graph.shape), (6,))
        self.assertEqual(tuple(sparse_nodes.shape), (7, 8))

    def test_actor_samples_and_evaluates_candidate_logits(self):
        args = SimpleNamespace(
            actor_encoder="gnn",
            actor_action_head="candidate_pool",
            actor_layers=[8],
            actor_act_fn="relu",
            intervention_gate=False,
            init_do_nothing_prob=0.0,
            candidate_action_pool="typed_mean",
            candidate_action_use_features=True,
            candidate_action_do_nothing_head=True,
            gnn_concat_flat=False,
            gnn_type="gine",
            gnn_hidden_dim=8,
            gnn_out_dim=6,
            gnn_layers=2,
            gnn_heads=1,
            gnn_layer_norm=True,
            gnn_readout_aggr="mean",
            gnn_node_pre_encoder=True,
            gnn_edge_pre_encoder=True,
            gnn_node_id_embeddings=False,
        )
        env = SimpleNamespace(
            observation_space={
                "agent_0": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                )
            },
            action_space={"agent_0": gym.spaces.Discrete(5)},
            graph_specs={"agent_0": self.spec},
        )
        actor = Actor(0, env, args, continuous_actions=False)
        batched_graph = {
            key: th.stack([value, value])
            for key, value in self.tensor_graph.items()
        }
        observation = {"graph": batched_graph}

        action, log_prob, entropy = actor.get_discrete_action(observation)
        eval_action = actor.get_eval_discrete_action(observation)
        self.assertEqual(tuple(action.shape), (2,))
        self.assertEqual(tuple(log_prob.shape), (2,))
        self.assertEqual(tuple(entropy.shape), (2,))
        self.assertEqual(tuple(eval_action.shape), (2,))
        self.assertTrue(bool(th.isfinite(log_prob).all()))

        restored_actor = Actor(0, env, args, continuous_actions=False)
        restored_actor.load_state_dict(actor.state_dict())
        self.assertTrue(
            th.equal(
                actor._actor_logits(observation),
                restored_actor._actor_logits(observation),
            )
        )


if __name__ == "__main__":
    unittest.main()
