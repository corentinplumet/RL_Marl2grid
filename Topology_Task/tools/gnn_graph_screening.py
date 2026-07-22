#!/usr/bin/env python3
"""Create and promote the staged graph-architecture screening configs.

The first stage is a fixed 3 x 4 representation/normalization screen. Later
stages must inherit the actual winner of the preceding stage, so this tool
materializes those configs only after a winner has been selected.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 guard
    import tomli as tomllib


TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
SCREENING_ROOT = TASK_DIR / "configs" / "gnn_graph_screening"
STAGE1_DIR = SCREENING_ROOT / "stage1_representation_normalization"
STAGE2_DIR = SCREENING_ROOT / "stage2_structure"
STAGE3_DIR = SCREENING_ROOT / "stage3_encoder"
STAGE4_DIR = SCREENING_ROOT / "stage4_confirmation"

GRAPH_TYPES = ("bus", "heterogeneous", "heterogeneous_line")
NORMALIZATION_MODES = {
    "n0_none": (False, False),
    "n1_physical": (True, False),
    "n2_running": (False, True),
    "n3_both": (True, True),
}
ENCODERS = ("gcn", "gat", "gine", "graphsage", "sparse_transformer")


BASE_CONFIG = """# Generated graph-architecture screening configuration.
# Keep non-screened settings identical so the comparisons remain paired.

[run]
name = "{name}"
run_dir = "outputs/jed-{{config_name}}-{{job_id}}"
seed_from_slurm_array = false
extra_args = []

[launcher]
use_srun = true
cpu_bind = "cores"
print_python_summary = true

[environment]
PYTHONUNBUFFERED = "1"
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
NUMEXPR_NUM_THREADS = "1"
VECLIB_MAXIMUM_THREADS = "1"
MAX_TIME_LIMIT_MINUTES = "1440"
MPLCONFIGDIR = "{{run_dir}}/matplotlib"
WANDB_DIR = "{{run_dir}}/wandb"
XDG_CACHE_HOME = "{{run_dir}}/cache"

[args]
# Paired bus14 screening protocol
time_limit = 720
checkpoint = true
alg = "MAPPO"
seed = 0
verbose = true
exp_tag = "{name}"
track = true
wandb_project = "Grid2Op"
wandb_entity = "corentin-plumet-epfl"
wandb_mode = "online"
th_deterministic = false
cuda = false
n_threads = 4

# bus14 environment and fixed MLP critic
env_id = "bus14"
n_envs = 72
action_type = "topology"
difficulty = 0
decentralized = true
n1_reward = false
env_config_path = "scenario.json"
norm_obs = true
use_heuristic = false
heuristic_type = "idle"
line_margin_reward_weight = 0.0
topology_reward_weight = 0.0
optimize_mem = true
constraints_type = 0
split_chronics = true
test_chronics_pct = 0.2
chronic_split_seed = 0

# Shared, grid-size-independent actor encoder
actor_encoder = "gnn"
critic_encoder = "mlp"
gnn_type = "gine"
gnn_hidden_dim = 128
gnn_out_dim = 128
gnn_layers = 2
gnn_heads = 4
gnn_readout_aggr = "mean"
sparse_gt_pooling = ""
graphsage_aggr = "mean"
gcn_edge_weight_feature = "none"
gnn_layer_norm = true
gnn_node_pre_encoder = true
gnn_edge_pre_encoder = false
gnn_node_id_embeddings = false
gnn_node_id_emb_dim = 8
gnn_concat_flat = false
share_actor_gnn = true
gnn_graph_type = "{graph_type}"
gnn_include_neighbors = true

# Structural factors (held off in stage 1)
gnn_add_substation_edges = false
gnn_add_substation_nodes = false
sparse_gt_add_self_edges = true
sparse_gt_add_substation_edges = false
sparse_gt_use_edge_attr = true
sparse_gt_use_edge_type_embeddings = true
sparse_gt_relation_bias = true
sparse_gt_dropout = 0.0
sparse_gt_attention_dropout = 0.0
sparse_gt_ffn_multiplier = 4

# Input preprocessing factors
gnn_physical_scaling = {physical_scaling}
gnn_running_norm = {running_norm}
gnn_power_scale_mw = 0.0
gnn_norm_clip = 10.0

# Approximately half of the 15M-step bus14 reference budget
total_timesteps = 8000000
n_steps = 576
eval_freq = 82944
actor_layers = [128, 128]
critic_layers = [256, 256, 256]
actor_act_fn = "relu"
critic_act_fn = "relu"
actor_lr = 0.0001
critic_lr = 0.0001
anneal_lr = true
gamma = 0.9
gae_lambda = 0.95
update_epochs = 10
n_minibatches = 16
max_grad_norm = 1.0
target_kl = 0.02
norm_adv = true
clip_coef = 0.2
clip_vfloss = true
entropy_coef = 0.01
entropy_coef_final = 0.01
vf_coef = 0.5
optimize_critic_updates = true
init_do_nothing_prob = 0.0
norm_reward = true
action0_logit_bonus_init = 0.0
action0_logit_bonus_final = 0.0
action0_logit_bonus_fraction = 1.0
deterministic_eval = true
eval_episodes = 10
eval_all_split_chronics = false
eval_train_chronics = true

# No intervention mechanism is mixed into the representation screen
intervention_gate = false
intervention_gate_eval_mode = "final_action_map"
intervention_penalty = 0.0
safe_intervention_penalty = 0.0
safe_intervention_rho_threshold = 0.90
eval_action_heuristic = "none"
eval_action_rho_threshold = 0.90
"""


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return _bool(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return str(value)
    raise TypeError(f"Unsupported TOML scalar: {value!r}")


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        config = tomllib.load(file)
    if "args" not in config:
        raise ValueError(f"Config has no [args] section: {path}")
    return config


def _set_value(text: str, section: str, key: str, value: Any) -> str:
    """Set or append one scalar key while preserving the rest of a TOML file."""

    lines = text.splitlines()
    section_header = f"[{section}]"
    try:
        section_start = lines.index(section_header)
    except ValueError as exc:
        raise ValueError(f"Missing TOML section {section_header}") from exc

    section_end = len(lines)
    for index in range(section_start + 1, len(lines)):
        if re.fullmatch(r"\s*\[[^]]+\]\s*", lines[index]):
            section_end = index
            break

    replacement = f"{key} = {_toml_value(value)}"
    key_pattern = re.compile(rf"\s*{re.escape(key)}\s*=")
    for index in range(section_start + 1, section_end):
        if key_pattern.match(lines[index]):
            lines[index] = replacement
            return "\n".join(lines) + "\n"

    lines.insert(section_end, replacement)
    return "\n".join(lines) + "\n"


def _apply_values(text: str, values: dict[tuple[str, str], Any]) -> str:
    for (section, key), value in values.items():
        text = _set_value(text, section, key, value)
    return text


def _write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force to replace it.")
    path.write_text(content, encoding="utf-8")


def _relative_config_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(TASK_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_launch_script(output_dir: Path, config_paths: Iterable[Path], force: bool) -> None:
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        *(f"sbatch job_jed.sh {_relative_config_path(path)}" for path in config_paths),
        "",
    ]
    _write(output_dir / "launch_all.sh", "\n".join(commands), force)


def _normalization_label(args: dict[str, Any]) -> str:
    physical = bool(args.get("gnn_physical_scaling", False))
    running = bool(args.get("gnn_running_norm", False))
    for label, pair in NORMALIZATION_MODES.items():
        if pair == (physical, running):
            return label
    raise AssertionError("Unreachable normalization mode")


def _manifest_row(path: Path) -> dict[str, Any]:
    args = _read_config(path)["args"]
    return {
        "config": path.name,
        "seed": args.get("seed", 0),
        "graph_type": args.get("gnn_graph_type", "bus"),
        "normalization": _normalization_label(args),
        "physical_scaling": args.get("gnn_physical_scaling", False),
        "running_normalization": args.get("gnn_running_norm", False),
        "substation_edges": args.get("gnn_add_substation_edges", False),
        "substation_nodes": args.get("gnn_add_substation_nodes", False),
        "virtual_node": args.get("gnn_readout_aggr") == "virtual_node",
        "encoder": args.get("gnn_type", "gine"),
        "timesteps": args.get("total_timesteps"),
    }


def _write_manifest(output_dir: Path, config_paths: list[Path], force: bool) -> None:
    rows = [_manifest_row(path) for path in config_paths]
    fieldnames = list(rows[0]) if rows else []
    lines: list[str] = []
    if fieldnames:
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        lines.append(buffer.getvalue())
    _write(output_dir / "manifest.csv", "".join(lines), force)


def create_stage1(output_dir: Path = STAGE1_DIR, force: bool = False) -> list[Path]:
    config_paths: list[Path] = []
    for graph_type in GRAPH_TYPES:
        graph_label = graph_type.replace("heterogeneous", "hetero")
        for norm_label, (physical, running) in NORMALIZATION_MODES.items():
            name = f"gs_s1_{graph_label}_{norm_label}_s0"
            content = BASE_CONFIG.format(
                name=name,
                graph_type=graph_type,
                physical_scaling=_bool(physical),
                running_norm=_bool(running),
            )
            path = output_dir / f"{name}.toml"
            _write(path, content, force)
            config_paths.append(path)
    _write_launch_script(output_dir, config_paths, force)
    _write_manifest(output_dir, config_paths, force)
    return config_paths


def _promoted_name(prefix: str, source_args: dict[str, Any], suffix: str) -> str:
    graph_type = str(source_args.get("gnn_graph_type", "bus")).replace(
        "heterogeneous", "hetero"
    )
    norm = _normalization_label(source_args)
    seed = int(source_args.get("seed", 0))
    return f"{prefix}_{graph_type}_{norm}_{suffix}_s{seed}"


def create_stage2(
    winner: Path, output_dir: Path = STAGE2_DIR, force: bool = False
) -> list[Path]:
    source = winner.read_text(encoding="utf-8")
    source_args = _read_config(winner)["args"]
    if str(source_args.get("gnn_type", "")).lower() != "gine":
        raise ValueError("Stage 2 must inherit a stage-1 GINE winner.")

    config_paths: list[Path] = []
    for edges, nodes, virtual in itertools.product((False, True), repeat=3):
        suffix = f"e{int(edges)}n{int(nodes)}v{int(virtual)}"
        name = _promoted_name("gs_s2", source_args, suffix)
        content = _apply_values(
            source,
            {
                ("run", "name"): name,
                ("args", "exp_tag"): name,
                ("args", "gnn_add_substation_edges"): edges,
                ("args", "gnn_add_substation_nodes"): nodes,
                ("args", "gnn_readout_aggr"): "virtual_node" if virtual else "mean",
                ("args", "sparse_gt_pooling"): "",
                ("args", "sparse_gt_add_substation_edges"): edges,
            },
        )
        path = output_dir / f"{name}.toml"
        _write(path, content, force)
        config_paths.append(path)
    _write_launch_script(output_dir, config_paths, force)
    _write_manifest(output_dir, config_paths, force)
    return config_paths


def create_stage3(
    winner: Path, output_dir: Path = STAGE3_DIR, force: bool = False
) -> list[Path]:
    source = winner.read_text(encoding="utf-8")
    source_args = _read_config(winner)["args"]
    readout = str(source_args.get("gnn_readout_aggr", "mean"))
    if readout not in {"mean", "virtual_node"}:
        raise ValueError("Stage 3 requires a mean or virtual-node stage-2 winner.")

    structural = "e{}n{}v{}".format(
        int(bool(source_args.get("gnn_add_substation_edges", False))),
        int(bool(source_args.get("gnn_add_substation_nodes", False))),
        int(readout == "virtual_node"),
    )
    config_paths: list[Path] = []
    for encoder in ENCODERS:
        name = _promoted_name("gs_s3", source_args, f"{structural}_{encoder}")
        content = _apply_values(
            source,
            {
                ("run", "name"): name,
                ("args", "exp_tag"): name,
                ("args", "gnn_type"): encoder,
                ("args", "gnn_heads"): 4,
                ("args", "sparse_gt_pooling"): "",
            },
        )
        path = output_dir / f"{name}.toml"
        _write(path, content, force)
        config_paths.append(path)
    _write_launch_script(output_dir, config_paths, force)
    _write_manifest(output_dir, config_paths, force)
    return config_paths


def create_confirmation(
    winners: Iterable[Path], output_dir: Path = STAGE4_DIR, force: bool = False
) -> list[Path]:
    config_paths: list[Path] = []
    for winner in winners:
        source = winner.read_text(encoding="utf-8")
        _read_config(winner)
        family = re.sub(r"_s\d+$", "", winner.stem)
        family = re.sub(r"^gs_s\d_", "", family)
        for seed in (0, 1, 2):
            name = f"gs_s4_{family}_confirm_s{seed}"
            content = _apply_values(
                source,
                {
                    ("run", "name"): name,
                    ("args", "seed"): seed,
                    ("args", "exp_tag"): name,
                    ("args", "time_limit"): 1300,
                    ("args", "total_timesteps"): 15000000,
                    ("environment", "MAX_TIME_LIMIT_MINUTES"): "1440",
                },
            )
            path = output_dir / f"{name}.toml"
            _write(path, content, force)
            config_paths.append(path)
    _write_launch_script(output_dir, config_paths, force)
    _write_manifest(output_dir, config_paths, force)
    return config_paths


def validate_configs(paths: Iterable[Path]) -> list[Path]:
    from run_from_config import validate_args

    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.toml")))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("No TOML configs were found.")
    for path in expanded:
        config = _read_config(path)
        validate_args(dict(config["args"]))
    return expanded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create the 12 fixed stage-1 configs.")
    init.add_argument("--output", type=Path, default=STAGE1_DIR)
    init.add_argument("--force", action="store_true")

    structure = subparsers.add_parser(
        "promote-structure", help="Create eight stage-2 configs from the stage-1 winner."
    )
    structure.add_argument("winner", type=Path)
    structure.add_argument("--output", type=Path, default=STAGE2_DIR)
    structure.add_argument("--force", action="store_true")

    encoder = subparsers.add_parser(
        "promote-encoder", help="Create five stage-3 configs from the stage-2 winner."
    )
    encoder.add_argument("winner", type=Path)
    encoder.add_argument("--output", type=Path, default=STAGE3_DIR)
    encoder.add_argument("--force", action="store_true")

    confirm = subparsers.add_parser(
        "confirm", help="Create full-budget seeds 0, 1, and 2 for selected configs."
    )
    confirm.add_argument("winners", nargs="+", type=Path)
    confirm.add_argument("--output", type=Path, default=STAGE4_DIR)
    confirm.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate TOML files or folders.")
    validate.add_argument("paths", nargs="+", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "init":
        paths = create_stage1(args.output, args.force)
    elif args.command == "promote-structure":
        paths = create_stage2(args.winner, args.output, args.force)
    elif args.command == "promote-encoder":
        paths = create_stage3(args.winner, args.output, args.force)
    elif args.command == "confirm":
        paths = create_confirmation(args.winners, args.output, args.force)
    else:
        paths = validate_configs(args.paths)
    print(f"{args.command}: {len(paths)} config(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
