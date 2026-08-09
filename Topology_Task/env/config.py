import sys

from common.imports import *
from common.utils import str2bool


def _reject_removed_validation_split_args(argv: Optional[List[str]] = None) -> None:
    removed_flags = ("--validation-chronics-pct", "--val-chronics-pct")
    args_to_check = sys.argv[1:] if argv is None else argv
    for arg in args_to_check:
        flag = arg.split("=", 1)[0]
        if flag in removed_flags:
            raise ValueError(
                f"{flag} was removed. Chronic splitting now only supports train/test; "
                "use --test-chronics-pct to control the test split."
            )


def get_env_args(argv: Optional[List[str]] = None) -> Namespace:
    """
    Parse and return the command-line arguments for configuring the environment.

    Returns:
        Namespace: A namespace containing the parsed arguments.
    """
    _reject_removed_validation_split_args(argv)

    parser = ap.ArgumentParser()

    # Settings
    # bus14, bus36, bus118
    parser.add_argument(
        "--env-id", type=str, default="bus14", help="ID of the grid2op environment"
    )
    parser.add_argument(
        "--n-envs", type=int, default=20, help="Number of parallel envs to run"
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default="topology",
        choices=["topology", "redispatch"],
        help="Type of environment: topology (discrete) or redispatch (continuous)",
    )
    parser.add_argument(
        "--reduced-action-space",
        type=str,
        default="",
        help=(
            "Optional path to a teacher_student reduced_action_space.json file. "
            "When set for topology actions, each agent exposes only the selected "
            "actions and maps reduced ids back to original Grid2Op action ids."
        ),
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="Difficulty 0 means splitting the grid in areas controlled by different agents (as specified in scenario.json), difficulty 1 means n-subs = n-agents",
    )
    parser.add_argument(
        "--decentralized",
        type=str2bool,
        default=True,
        help="Toggles partial observability (agents can only observe features related to their substations)",
    )
    parser.add_argument(
        "--n1-reward",
        type=str2bool,
        default=False,
        help="Toggles N1 contintency analysis as an additional reward",
    )

    # Scenarios
    parser.add_argument(
        "--env-config-path",
        type=str,
        default="scenario.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--obs-attrs",
        nargs="+",
        default=None,
        help=(
            "Optional explicit Grid2Op observation attributes to keep in the "
            "flat observation. When unset, attributes are derived from "
            "scenario.json as before."
        ),
    )

    # Normalization
    parser.add_argument(
        "--norm-obs", type=str2bool, default=True, help="Toggle normalize observations"
    )

    parser.add_argument(
        "--use-heuristic",
        type=str2bool,
        default=False,
        help="Toggles heuristics for base operations",
    )

    parser.add_argument(
        "--heuristic-type",
        type=str,
        default="idle",
        choices=["idle"],
        help="Select the type of heuristic to use: idle",
    )
    parser.add_argument(
        "--line-margin-reward-weight",
        type=float,
        default=0.0,
        help="Optional training reward weight for LineMarginReward dense safety shaping.",
    )
    parser.add_argument(
        "--topology-reward-weight",
        type=float,
        default=0.0,
        help="Optional training reward weight for DistanceReward topology-change penalty.",
    )

    parser.add_argument(
        "--optimize-mem",
        type=str2bool,
        default=True,
        help="Whether to load data chunks upon resets (True), or the whole dataset once (False)",
    )

    parser.add_argument(
        "--constraints-type",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Select the type of constraints to use: no constraints (0), failure constraints (1), overloads constraints (2)",
    )
    parser.add_argument(
        "--split-chronics",
        type=str2bool,
        default=False,
        help="Split chronics into train/test sets. Training uses train, periodic eval uses test.",
    )
    parser.add_argument(
        "--test-chronics-pct",
        type=float,
        default=0.2,
        help="Fraction of available chronics reserved for the test split when --split-chronics is enabled. Values > 1 are treated as percentages.",
    )
    parser.add_argument(
        "--chronic-split-seed",
        type=int,
        default=None,
        help="Seed used to create the chronic split. Defaults to --seed.",
    )
    parser.add_argument(
        "--chronic-shard-count",
        type=int,
        default=1,
        help=(
            "Optional number of disjoint chronic shards. Use with "
            "--chronic-shard-index to split collection across SLURM array tasks."
        ),
    )
    parser.add_argument(
        "--chronic-shard-index",
        type=int,
        default=0,
        help="Zero-based chronic shard index for this worker.",
    )
    parser.add_argument(
        "--actor-encoder",
        type=str,
        default="mlp",
        choices=["mlp", "gnn", "transformer"],
        help="Encoder used by each actor policy.",
    )
    parser.add_argument(
        "--critic-encoder",
        type=str,
        default="mlp",
        choices=["mlp", "gnn", "transformer"],
        help="Encoder used by the centralized critic.",
    )
    parser.add_argument(
        "--gnn-include-neighbors",
        type=str2bool,
        default=False,
        help="For the thesis-style gnn encoder, include one-hop neighboring substations in local agent graphs.",
    )
    parser.add_argument(
        "--gnn-context-requires-connection",
        type=str2bool,
        default=True,
        help=(
            "Require contextual graph nodes to have a live electrical "
            "connection to the agent's controlled region. Disable only to "
            "reproduce the legacy always-visible neighboring-node behavior."
        ),
    )
    parser.add_argument(
        "--gnn-structural-relations-controlled-only",
        type=str2bool,
        default=True,
        help=(
            "Build same-substation edges and substation summary nodes only "
            "inside the agent's controlled substations. A same-substation "
            "relation says an element can be moved between two busbars, which "
            "outside the controlled domain is another agent's action and joins "
            "two nodes this agent cannot relate. Disable to reproduce the "
            "legacy behavior, where neighboring busbars of one substation "
            "exchanged messages directly and through their summary node."
        ),
    )

    # Parse the arguments
    params, _ = parser.parse_known_args(argv)

    return params
