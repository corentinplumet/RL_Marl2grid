import tempfile
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from run_from_config import validate_args
from tools.gnn_graph_screening import (
    ENCODERS,
    GRAPH_TYPES,
    NORMALIZATION_MODES,
    create_confirmation,
    create_stage1,
    create_stage2,
    create_stage3,
    validate_configs,
)


def load_args(path: Path):
    with path.open("rb") as file:
        return tomllib.load(file)["args"]


class GraphScreeningConfigTests(unittest.TestCase):
    def test_all_screening_stages_preserve_selected_factors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage1 = create_stage1(root / "stage1")
            self.assertEqual(len(stage1), 12)
            validate_configs([root / "stage1"])

            observed_stage1 = {
                (
                    args["gnn_graph_type"],
                    args["gnn_physical_scaling"],
                    args["gnn_running_norm"],
                )
                for args in map(load_args, stage1)
            }
            expected_stage1 = {
                (graph_type, physical, running)
                for graph_type in GRAPH_TYPES
                for physical, running in NORMALIZATION_MODES.values()
            }
            self.assertEqual(observed_stage1, expected_stage1)
            for path in stage1:
                args = load_args(path)
                self.assertEqual(args["gnn_type"], "gine")
                self.assertTrue(args["share_actor_gnn"])
                self.assertFalse(args["gnn_node_id_embeddings"])
                self.assertFalse(args["gnn_concat_flat"])
                self.assertEqual(args["env_id"], "bus14")
                self.assertEqual(args["n_envs"], 72)
                self.assertEqual(args["n_steps"], 576)
                self.assertEqual(args["eval_freq"], 82_944)
                self.assertNotIn("reduced_action_space", args)
                self.assertEqual(args["total_timesteps"], 8_000_000)

            stage1_winner = next(
                path
                for path in stage1
                if path.stem == "gs_s1_hetero_line_n3_both_s0"
            )
            stage2 = create_stage2(stage1_winner, root / "stage2")
            self.assertEqual(len(stage2), 8)
            validate_configs([root / "stage2"])
            observed_structures = {
                (
                    args["gnn_add_substation_edges"],
                    args["gnn_add_substation_nodes"],
                    args["gnn_readout_aggr"] == "virtual_node",
                )
                for args in map(load_args, stage2)
            }
            self.assertEqual(
                observed_structures,
                {
                    (edges, nodes, virtual)
                    for edges in (False, True)
                    for nodes in (False, True)
                    for virtual in (False, True)
                },
            )

            stage2_winner = next(
                path for path in stage2 if "e1n0v1" in path.stem
            )
            stage3 = create_stage3(stage2_winner, root / "stage3")
            self.assertEqual(len(stage3), 5)
            validate_configs([root / "stage3"])
            self.assertEqual(
                {load_args(path)["gnn_type"] for path in stage3}, set(ENCODERS)
            )
            for path in stage3:
                args = load_args(path)
                self.assertEqual(args["gnn_graph_type"], "heterogeneous_line")
                self.assertTrue(args["gnn_physical_scaling"])
                self.assertTrue(args["gnn_running_norm"])
                self.assertTrue(args["gnn_add_substation_edges"])
                self.assertFalse(args["gnn_add_substation_nodes"])
                self.assertEqual(args["gnn_readout_aggr"], "virtual_node")

            confirmation = create_confirmation(stage3[:2], root / "stage4")
            self.assertEqual(len(confirmation), 6)
            validate_configs([root / "stage4"])
            for path in confirmation:
                args = load_args(path)
                self.assertIn(args["seed"], {0, 1, 2})
                self.assertEqual(args["total_timesteps"], 15_000_000)
                self.assertEqual(args["time_limit"], 1300)

    def test_non_sparse_attention_readout_is_rejected_early(self):
        common = {"n_envs": 4, "n_steps": 8, "eval_freq": 16}
        with self.assertRaises(SystemExit):
            validate_args(
                {
                    **common,
                    "gnn_type": "gine",
                    "gnn_readout_aggr": "controlled_attention",
                }
            )

        validate_args(
            {
                **common,
                "gnn_type": "sparse_transformer",
                "gnn_readout_aggr": "controlled_attention",
            }
        )


if __name__ == "__main__":
    unittest.main()
