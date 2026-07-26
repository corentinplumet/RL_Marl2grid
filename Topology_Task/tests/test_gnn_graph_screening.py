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
    STAGE1D_DIRECTIONS,
    STAGE1B_FINALISTS,
    STAGE1C_STRUCTURES,
    STAGE2_STRUCTURES,
    create_confirmation,
    create_stage1,
    create_stage1b,
    create_stage1c,
    create_stage1d,
    create_stage1e,
    create_stage2,
    create_stage3,
    validate_configs,
)


def load_args(path: Path):
    with path.open("rb") as file:
        return tomllib.load(file)["args"]


class GraphScreeningConfigTests(unittest.TestCase):
    def test_stage2_covers_all_bus_structures_with_three_fresh_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "stage2"
            configs = create_stage2(output_dir)
            self.assertEqual(len(configs), 24)
            validate_configs([output_dir])

            seeds_by_structure = {
                structure: set() for structure in STAGE2_STRUCTURES
            }
            for path in configs:
                with path.open("rb") as file:
                    config = tomllib.load(file)
                args = config["args"]
                structure = (
                    args["gnn_add_substation_edges"],
                    args["gnn_add_substation_nodes"],
                    args["gnn_readout_aggr"] == "virtual_node",
                )
                seeds_by_structure[structure].add(args["seed"])

                self.assertEqual(args["gnn_graph_type"], "bus")
                self.assertEqual(args["gnn_type"], "gine")
                self.assertEqual(args["critic_encoder"], "mlp")
                self.assertFalse(args["gnn_physical_scaling"])
                self.assertFalse(args["gnn_running_norm"])
                self.assertEqual(
                    args["sparse_gt_add_substation_edges"], structure[0]
                )
                self.assertEqual(
                    args["gnn_summary_edge_direction"], "bidirectional"
                )
                self.assertEqual(
                    args["gnn_virtual_edge_direction"], "toward_virtual"
                )
                self.assertEqual(args["total_timesteps"], 15_000_000)
                self.assertEqual(args["time_limit"], 5760)
                self.assertEqual(
                    config["environment"]["MAX_TIME_LIMIT_MINUTES"], "5760"
                )

            self.assertEqual(set(seeds_by_structure), set(STAGE2_STRUCTURES))
            for seeds in seeds_by_structure.values():
                self.assertEqual(seeds, {0, 1, 2})

    def test_stage1e_combines_heterogeneous_structures_and_line_scout(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "stage1e"
            configs = create_stage1e(output_dir)
            self.assertEqual(len(configs), 8)
            validate_configs([output_dir])

            heterogeneous_structures = set()
            line_scouts = []
            for path in configs:
                with path.open("rb") as file:
                    config = tomllib.load(file)
                args = config["args"]
                structure = (
                    args["gnn_add_substation_edges"],
                    args["gnn_add_substation_nodes"],
                    args["gnn_readout_aggr"] == "virtual_node",
                )
                if args["gnn_graph_type"] == "heterogeneous":
                    heterogeneous_structures.add(structure)
                elif args["gnn_graph_type"] == "heterogeneous_line":
                    line_scouts.append(structure)
                else:
                    self.fail(f"Unexpected graph type in {path.name}")

                self.assertEqual(args["gnn_type"], "gine")
                self.assertEqual(args["critic_encoder"], "mlp")
                self.assertFalse(args["gnn_physical_scaling"])
                self.assertFalse(args["gnn_running_norm"])
                self.assertEqual(
                    args["gnn_generator_edge_direction"], "bidirectional"
                )
                self.assertEqual(args["gnn_load_edge_direction"], "bidirectional")
                self.assertEqual(
                    args["gnn_line_node_edge_direction"], "bidirectional"
                )
                self.assertEqual(args["seed"], 0)
                self.assertEqual(args["total_timesteps"], 8_000_000)
                self.assertEqual(args["time_limit"], 2880)
                self.assertEqual(
                    config["environment"]["MAX_TIME_LIMIT_MINUTES"], "2880"
                )

            self.assertEqual(heterogeneous_structures, set(STAGE1C_STRUCTURES))
            self.assertEqual(line_scouts, [(False, False, False)])

    def test_stage1d_covers_the_eight_nonbaseline_direction_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "stage1d"
            configs = create_stage1d(output_dir)
            self.assertEqual(len(configs), 8)
            validate_configs([output_dir])

            observed = set()
            for path in configs:
                with path.open("rb") as file:
                    config = tomllib.load(file)
                args = config["args"]
                directions = (
                    args["gnn_generator_edge_direction"],
                    args["gnn_load_edge_direction"],
                )
                observed.add(directions)
                self.assertEqual(args["gnn_graph_type"], "heterogeneous")
                self.assertEqual(args["gnn_type"], "gine")
                self.assertEqual(args["critic_encoder"], "mlp")
                self.assertFalse(args["gnn_physical_scaling"])
                self.assertFalse(args["gnn_running_norm"])
                self.assertFalse(args["gnn_add_substation_edges"])
                self.assertFalse(args["gnn_add_substation_nodes"])
                self.assertEqual(args["gnn_readout_aggr"], "mean")
                self.assertEqual(args["seed"], 0)
                self.assertEqual(args["total_timesteps"], 8_000_000)
                self.assertEqual(args["time_limit"], 2880)
                self.assertEqual(
                    config["environment"]["MAX_TIME_LIMIT_MINUTES"], "2880"
                )

            self.assertEqual(observed, set(STAGE1D_DIRECTIONS))
            self.assertNotIn(("bidirectional", "bidirectional"), observed)

    def test_stage1c_covers_the_seven_nonbaseline_structures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "stage1c"
            configs = create_stage1c(output_dir)
            self.assertEqual(len(configs), 7)
            validate_configs([output_dir])

            observed = set()
            for path in configs:
                with path.open("rb") as file:
                    config = tomllib.load(file)
                args = config["args"]
                structure = (
                    args["gnn_add_substation_edges"],
                    args["gnn_add_substation_nodes"],
                    args["gnn_readout_aggr"] == "virtual_node",
                )
                observed.add(structure)
                self.assertEqual(
                    args["sparse_gt_add_substation_edges"], structure[0]
                )
                self.assertEqual(args["gnn_graph_type"], "bus")
                self.assertEqual(args["gnn_type"], "gine")
                self.assertEqual(args["critic_encoder"], "mlp")
                self.assertFalse(args["gnn_physical_scaling"])
                self.assertFalse(args["gnn_running_norm"])
                self.assertEqual(args["seed"], 0)
                self.assertEqual(args["total_timesteps"], 8_000_000)
                self.assertEqual(args["time_limit"], 2880)
                self.assertEqual(
                    config["environment"]["MAX_TIME_LIMIT_MINUTES"], "2880"
                )

            self.assertEqual(observed, set(STAGE1C_STRUCTURES))
            self.assertNotIn((False, False, False), observed)

    def test_stage1b_confirms_three_finalists_with_three_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "stage1b"
            configs = create_stage1b(output_dir)
            self.assertEqual(len(configs), 9)
            validate_configs([output_dir])

            observed = {
                (
                    args["gnn_graph_type"],
                    args["gnn_physical_scaling"],
                    args["gnn_running_norm"],
                )
                for args in map(load_args, configs)
            }
            expected = {
                (
                    graph_type,
                    NORMALIZATION_MODES[norm_label][0],
                    NORMALIZATION_MODES[norm_label][1],
                )
                for graph_type, norm_label in STAGE1B_FINALISTS
            }
            self.assertEqual(observed, expected)
            self.assertEqual(
                {load_args(path)["seed"] for path in configs}, {0, 1, 2}
            )
            for path in configs:
                with path.open("rb") as file:
                    config = tomllib.load(file)
                args = config["args"]
                self.assertEqual(
                    config["environment"]["MAX_TIME_LIMIT_MINUTES"], "2880"
                )
                self.assertEqual(args["env_id"], "bus14")
                self.assertEqual(args["gnn_type"], "gine")
                self.assertEqual(args["critic_encoder"], "mlp")
                self.assertEqual(args["total_timesteps"], 8_000_000)
                self.assertEqual(args["time_limit"], 2880)
                self.assertEqual(args["n_envs"], 72)
                self.assertEqual(args["n_steps"], 576)
                self.assertEqual(args["eval_freq"], 82_944)

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
                self.assertEqual(
                    args["gnn_generator_edge_direction"], "bidirectional"
                )
                self.assertEqual(args["gnn_load_edge_direction"], "bidirectional")
                self.assertEqual(
                    args["gnn_line_node_edge_direction"], "bidirectional"
                )
                self.assertEqual(
                    args["gnn_summary_edge_direction"], "bidirectional"
                )
                self.assertNotIn("reduced_action_space", args)
                self.assertEqual(args["total_timesteps"], 8_000_000)

            stage2 = create_stage2(root / "stage2")
            self.assertEqual(len(stage2), 24)
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
                path for path in stage2 if "e1n0v1_s0" in path.stem
            )
            stage3 = create_stage3(stage2_winner, root / "stage3")
            self.assertEqual(len(stage3), 5)
            validate_configs([root / "stage3"])
            self.assertEqual(
                {load_args(path)["gnn_type"] for path in stage3}, set(ENCODERS)
            )
            for path in stage3:
                args = load_args(path)
                self.assertEqual(args["gnn_graph_type"], "bus")
                self.assertFalse(args["gnn_physical_scaling"])
                self.assertFalse(args["gnn_running_norm"])
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

    def test_invalid_relation_direction_is_rejected_early(self):
        with self.assertRaises(SystemExit):
            validate_args(
                {
                    "n_envs": 4,
                    "n_steps": 8,
                    "eval_freq": 16,
                    "gnn_load_edge_direction": "load_to_generator",
                }
            )


if __name__ == "__main__":
    unittest.main()
