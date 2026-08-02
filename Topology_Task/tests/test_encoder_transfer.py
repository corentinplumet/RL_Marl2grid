import os
import tempfile
import unittest

import torch as th
import torch.nn as nn

from common.transfer import (
    freeze_graph_encoders,
    load_encoder_from_checkpoint,
    resolve_checkpoint_path,
)


class FakeGraphEncoder(nn.Module):
    """Stands in for GraphEncoder: grid-independent weights, grid-shaped buffers."""

    def __init__(self, n_nodes: int, n_edges: int, hidden: int = 4):
        super().__init__()
        self.node_pre_encoder = nn.Linear(3, hidden)
        self.conv = nn.Linear(hidden, hidden)
        self.register_buffer("edge_index", th.zeros((2, n_edges), dtype=th.long))
        self.register_buffer("node_ids", th.arange(n_nodes, dtype=th.long))


class FakeGraphAndFlatEncoder(nn.Module):
    def __init__(self, graph_encoder: nn.Module):
        super().__init__()
        self.graph_encoder = graph_encoder


class FakeActor(nn.Module):
    def __init__(self, graph_encoder: nn.Module, n_actions: int):
        super().__init__()
        self.encoder = FakeGraphAndFlatEncoder(graph_encoder)
        self.actor = nn.Linear(4, n_actions)


def make_actors(n_nodes, n_edges, n_actions, n_agents=3, hidden=4, shared=True):
    encoder = FakeGraphEncoder(n_nodes, n_edges, hidden) if shared else None
    return {
        f"agent_{idx}": FakeActor(
            encoder if shared else FakeGraphEncoder(n_nodes, n_edges, hidden),
            n_actions,
        )
        for idx in range(n_agents)
    }


def save_checkpoint(actors, path, global_step=1234):
    record = {agent: actor.state_dict() for agent, actor in actors.items()}
    record["critic"] = {"weight": th.zeros(1)}
    record["global_step"] = global_step
    th.save(record, path)


class EncoderTransferTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "source.tar")
        # bus14-like source: 28 nodes / 60 edges, 3 agents, small action space.
        self.source = make_actors(n_nodes=28, n_edges=60, n_actions=50)
        save_checkpoint(self.source, self.path)

    def target(self, **kwargs):
        # wcci-like target: more nodes, more edges, more agents, bigger heads.
        params = dict(n_nodes=72, n_edges=180, n_actions=140, n_agents=4)
        params.update(kwargs)
        return make_actors(**params)

    def test_loads_weights_across_grid_sizes(self):
        actors = self.target()
        source_weight = self.source["agent_0"].encoder.graph_encoder.node_pre_encoder.weight
        target_encoder = actors["agent_0"].encoder.graph_encoder
        self.assertFalse(th.allclose(source_weight, target_encoder.node_pre_encoder.weight))

        report = load_encoder_from_checkpoint(actors, self.path)

        self.assertTrue(th.allclose(source_weight, target_encoder.node_pre_encoder.weight))
        self.assertEqual(report["encoders"], 1)
        self.assertEqual(report["source_global_step"], 1234)
        self.assertFalse(report["frozen"])

    def test_grid_shaped_buffers_are_skipped(self):
        actors = self.target()
        target_encoder = actors["agent_0"].encoder.graph_encoder

        report = load_encoder_from_checkpoint(actors, self.path)

        self.assertEqual(
            sorted(report["skipped_topology_buffers"]), ["edge_index", "node_ids"]
        )
        self.assertEqual(tuple(target_encoder.edge_index.shape), (2, 180))
        self.assertEqual(tuple(target_encoder.node_ids.shape), (72,))

    def test_heads_are_untouched(self):
        actors = self.target()
        before = actors["agent_0"].actor.weight.clone()

        load_encoder_from_checkpoint(actors, self.path)

        self.assertTrue(th.equal(before, actors["agent_0"].actor.weight))

    def test_freeze_leaves_only_heads_trainable(self):
        actors = self.target()

        report = load_encoder_from_checkpoint(actors, self.path, freeze=True)

        encoder_params = list(actors["agent_0"].encoder.graph_encoder.parameters())
        self.assertTrue(all(not param.requires_grad for param in encoder_params))
        self.assertTrue(actors["agent_0"].actor.weight.requires_grad)
        self.assertEqual(
            report["frozen_parameter_count"],
            sum(param.numel() for param in encoder_params),
        )

    def test_shared_encoder_is_loaded_once(self):
        actors = self.target()
        report = load_encoder_from_checkpoint(actors, self.path)
        self.assertEqual(report["encoders"], 1)
        first = actors["agent_0"].encoder.graph_encoder.conv.weight
        for agent in actors:
            self.assertTrue(
                th.equal(first, actors[agent].encoder.graph_encoder.conv.weight)
            )

    def test_per_actor_encoders_all_receive_the_weights(self):
        actors = self.target()
        # Rebuild without sharing, as share_actor_gnn=false would.
        actors = make_actors(
            n_nodes=72, n_edges=180, n_actions=140, n_agents=4, shared=False
        )

        report = load_encoder_from_checkpoint(actors, self.path, freeze=True)

        self.assertEqual(report["encoders"], 4)
        source_weight = self.source["agent_0"].encoder.graph_encoder.conv.weight
        for agent in actors:
            encoder = actors[agent].encoder.graph_encoder
            self.assertTrue(th.allclose(source_weight, encoder.conv.weight))
            self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))

    def test_architecture_mismatch_raises(self):
        actors = self.target(hidden=8)
        with self.assertRaises(ValueError) as ctx:
            load_encoder_from_checkpoint(actors, self.path)
        self.assertIn("architecture mismatch", str(ctx.exception))

    def test_checkpoint_without_graph_encoder_raises(self):
        mlp_actor = nn.Module()
        mlp_actor.actor = nn.Linear(4, 4)
        path = os.path.join(self.tmp.name, "mlp.tar")
        th.save({"agent_0": mlp_actor.state_dict(), "global_step": 1}, path)

        with self.assertRaises(ValueError) as ctx:
            load_encoder_from_checkpoint(self.target(), path)
        self.assertIn("non-GNN actor encoder", str(ctx.exception))

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_encoder_from_checkpoint(
                self.target(), os.path.join(self.tmp.name, "nope.tar")
            )

    def test_freeze_only_helper(self):
        actors = self.target()
        report = freeze_graph_encoders(actors)
        self.assertTrue(report["frozen"])
        self.assertIsNone(report["checkpoint"])
        self.assertTrue(
            all(
                not param.requires_grad
                for param in actors["agent_0"].encoder.graph_encoder.parameters()
            )
        )

    def test_checkpoint_stem_resolves_under_checkpoint_dir(self):
        self.assertEqual(
            resolve_checkpoint_path("best_test_run_s0"),
            os.path.join("checkpoint", "best_test_run_s0.tar"),
        )
        self.assertEqual(resolve_checkpoint_path("/abs/path.tar"), "/abs/path.tar")


if __name__ == "__main__":
    unittest.main()
