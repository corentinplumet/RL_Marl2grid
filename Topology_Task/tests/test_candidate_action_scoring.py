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
        self.spec = builder.specs["agent_0"]
        self.metadata = make_metadata(self.spec)
        graph_type_ids = self.spec["node_type_names"]
        self.node_type_ids = {
            "busbar": graph_type_ids["busbar"],
            "line": graph_type_ids["transmission_line"],
            "load": graph_type_ids["load"],
            "generator": graph_type_ids["generator"],
        }

    @staticmethod
    def masked_mean(node_embeddings, indices, mask):
        gathered = node_embeddings[:, indices.clamp_min(0), :]
        weights = mask.to(node_embeddings).unsqueeze(0).unsqueeze(-1)
        return (gathered * weights).sum(dim=2) / weights.sum(
            dim=2
        ).clamp_min(1.0)

    def test_pooling_modes_produce_finite_logits_and_gradients(self):
        for pool_mode in ("mean", "typed_mean", "typed_attention"):
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

    def test_existing_mean_modes_match_the_original_pooling_equations(self):
        node_embeddings = th.randn(2, 7, 7)
        typed_metadata = (
            (self.metadata.busbar_indices, self.metadata.busbar_mask),
            (self.metadata.line_indices, self.metadata.line_mask),
            (self.metadata.load_indices, self.metadata.load_mask),
            (self.metadata.generator_indices, self.metadata.generator_mask),
        )
        expected_typed = th.cat(
            [
                self.masked_mean(node_embeddings, indices, mask)
                for indices, mask in typed_metadata
            ],
            dim=-1,
        )
        expected_mean = self.masked_mean(
            node_embeddings,
            th.cat([indices for indices, _ in typed_metadata], dim=1),
            th.cat([mask for _, mask in typed_metadata], dim=1),
        )

        for mode, expected in (
            ("mean", expected_mean),
            ("typed_mean", expected_typed),
        ):
            scorer = CandidateActionScorer(
                graph_dim=5,
                node_dim=7,
                hidden_layers=[],
                act_fn_name="relu",
                metadata=self.metadata,
                pool_mode=mode,
            )
            output = scorer._local_context(th.randn(2, 5), node_embeddings)
            self.assertTrue(th.equal(output.context, expected))
            self.assertFalse(
                any(key.startswith("pool.") for key in scorer.state_dict())
            )

    def test_affected_attention_respects_masks_and_empty_types(self):
        scorer = CandidateActionScorer(
            graph_dim=5,
            node_dim=7,
            hidden_layers=[],
            act_fn_name="relu",
            metadata=self.metadata,
            pool_mode="typed_attention",
            attention_heads=2,
        )
        graph_embedding = th.randn(3, 5, requires_grad=True)
        node_embeddings = th.randn(3, 7, 7, requires_grad=True)
        logits, attention = scorer(
            graph_embedding,
            node_embeddings,
            return_attention=True,
        )

        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertEqual(tuple(attention.context.shape), (3, 5, 28))
        for node_type, weights in attention.weights.items():
            mask = attention.eligible_masks[node_type]
            self.assertEqual(tuple(weights.shape[:3]), (3, 5, 2))
            outside = ~mask.unsqueeze(0).unsqueeze(2).expand_as(weights)
            outside_weights = weights.masked_select(outside)
            self.assertTrue(
                th.equal(outside_weights, th.zeros_like(outside_weights))
            )
            expected_sum = mask.any(dim=-1).to(weights).view(1, 5, 1)
            expected_sum = expected_sum.expand(3, 5, 2)
            self.assertTrue(
                th.allclose(weights.sum(dim=-1), expected_sum, atol=1e-6)
            )

        self.assertTrue(
            th.equal(
                attention.context[:, 0],
                th.zeros_like(attention.context[:, 0]),
            )
        )
        logits.sum().backward()
        for name in (
            "pool.query_projection.weight",
            "pool.key_projection.weight",
            "pool.value_projection.weight",
            "pool.score_vector",
        ):
            parameter = dict(scorer.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_affected_attention_checkpoint_round_trip_is_exact(self):
        kwargs = dict(
            graph_dim=5,
            node_dim=7,
            hidden_layers=[9],
            act_fn_name="relu",
            metadata=self.metadata,
            pool_mode="typed_attention",
        )
        scorer = CandidateActionScorer(**kwargs)
        restored = CandidateActionScorer(**kwargs)
        restored.load_state_dict(scorer.state_dict())
        graph_embedding = th.randn(2, 5)
        node_embeddings = th.randn(2, 7, 7)
        self.assertTrue(
            th.equal(
                scorer(graph_embedding, node_embeddings),
                restored(graph_embedding, node_embeddings),
            )
        )

    def soft_prior_scorer(self, prior_bias=2.0, chunk_size=64):
        return CandidateActionScorer(
            graph_dim=5,
            node_dim=7,
            hidden_layers=[9],
            act_fn_name="relu",
            metadata=self.metadata,
            pool_mode="typed_attention",
            attention_scope="soft_prior",
            attention_prior_bias=prior_bias,
            attention_chunk_size=chunk_size,
            node_types=self.spec["node_type"],
            node_type_ids=self.node_type_ids,
        )

    def test_soft_prior_attends_to_all_typed_nodes_except_for_action_zero(self):
        scorer = self.soft_prior_scorer()
        graph_embedding = th.randn(2, 5, requires_grad=True)
        node_embeddings = th.randn(2, 7, 7, requires_grad=True)
        logits, attention = scorer(
            graph_embedding,
            node_embeddings,
            return_attention=True,
        )

        self.assertEqual(tuple(logits.shape), (2, 5))
        for node_type, weights in attention.weights.items():
            eligible = attention.eligible_masks[node_type]
            affected = attention.affected_masks[node_type]
            indices = attention.node_indices[node_type]
            self.assertEqual(tuple(eligible.shape), tuple(affected.shape))
            self.assertEqual(tuple(indices.shape), tuple(eligible.shape))
            self.assertTrue(
                th.equal(weights[:, 0], th.zeros_like(weights[:, 0]))
            )
            if weights.shape[-1]:
                expected = th.ones_like(weights[:, 1:].sum(dim=-1))
                self.assertTrue(
                    th.allclose(weights[:, 1:].sum(dim=-1), expected, atol=1e-6)
                )
        logits.sum().backward()
        self.assertGreater(float(node_embeddings.grad.abs().sum()), 0.0)

    def test_larger_soft_prior_bias_increases_hardcoded_attention_mass(self):
        unbiased = self.soft_prior_scorer(prior_bias=0.0)
        with th.no_grad():
            unbiased.pool.score_vector.zero_()
        biased = self.soft_prior_scorer(prior_bias=4.0)
        biased.load_state_dict(unbiased.state_dict())
        graph_embedding = th.randn(2, 5)
        node_embeddings = th.randn(2, 7, 7)
        _, unbiased_attention = unbiased(
            graph_embedding,
            node_embeddings,
            return_attention=True,
        )
        _, biased_attention = biased(
            graph_embedding,
            node_embeddings,
            return_attention=True,
        )

        unbiased_mass = []
        biased_mass = []
        for node_type, affected in unbiased_attention.affected_masks.items():
            valid_actions = affected.any(dim=-1)
            if not bool(valid_actions.any()):
                continue
            expanded_affected = affected.unsqueeze(0).unsqueeze(2)
            unbiased_type_mass = (
                unbiased_attention.weights[node_type]
                * expanded_affected.to(unbiased_attention.weights[node_type])
            ).sum(dim=-1)
            biased_type_mass = (
                biased_attention.weights[node_type]
                * expanded_affected.to(biased_attention.weights[node_type])
            ).sum(dim=-1)
            unbiased_mass.append(unbiased_type_mass[:, valid_actions])
            biased_mass.append(biased_type_mass[:, valid_actions])

        unbiased_mass = th.cat(unbiased_mass, dim=1)
        biased_mass = th.cat(biased_mass, dim=1)
        self.assertTrue(bool(th.all(biased_mass >= unbiased_mass)))
        self.assertGreater(float(biased_mass.mean()), float(unbiased_mass.mean()))

    def test_soft_prior_action_chunking_preserves_outputs(self):
        chunked = self.soft_prior_scorer(chunk_size=1)
        unchunked = self.soft_prior_scorer(chunk_size=64)
        unchunked.load_state_dict(chunked.state_dict())
        graph_embedding = th.randn(2, 5)
        node_embeddings = th.randn(2, 7, 7)
        chunked_logits, chunked_attention = chunked(
            graph_embedding,
            node_embeddings,
            return_attention=True,
        )
        unchunked_logits, unchunked_attention = unchunked(
            graph_embedding,
            node_embeddings,
            return_attention=True,
        )

        self.assertTrue(th.allclose(chunked_logits, unchunked_logits, atol=1e-7))
        for node_type in chunked_attention.weights:
            self.assertTrue(
                th.allclose(
                    chunked_attention.weights[node_type],
                    unchunked_attention.weights[node_type],
                    atol=1e-7,
                )
            )

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
            candidate_action_pool="typed_attention",
            candidate_action_use_features=True,
            candidate_action_do_nothing_head=True,
            candidate_action_attention_scope="affected",
            candidate_action_attention_heads=1,
            candidate_action_attention_dim=0,
            candidate_action_attention_temperature=1.0,
            candidate_action_attention_query="global_action_features",
            candidate_action_attention_normalizer="softmax",
            candidate_action_attention_prior_bias=2.0,
            candidate_action_attention_chunk_size=64,
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

        diagnostics = actor.get_candidate_attention(observation, action_ids=[1, 4])
        self.assertEqual(tuple(diagnostics["logits"].shape), (2, 2))
        self.assertEqual(
            tuple(diagnostics["weights"]["line"].shape[:2]),
            (2, 2),
        )

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
