import sys

from common.imports import *
from common.utils import str2bool


def _reject_removed_validation_eval_args() -> None:
    removed_flags = ("--validate-on-best-test",)
    for arg in sys.argv[1:]:
        flag = arg.split("=", 1)[0]
        if flag in removed_flags:
            raise ValueError(
                f"{flag} was removed. Chronic splitting now only supports train/test "
                "evaluation on the test split."
            )


def get_alg_args() -> Namespace:
    """Parse command-line arguments for PPO.

    This function sets up and parses arguments for configuring the training and evaluation of a PPO agent.

    Returns:
        A namespace containing the parsed arguments.
    """
    _reject_removed_validation_eval_args()

    parser = ap.ArgumentParser()

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=25000000,
        help="Total timesteps for the experiment",
    )

    parser.add_argument(
        "--n-steps", type=int, default=2000, help="Steps per policy rollout"
    )  # 20k for 1 env

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=80000,
        help="Total timesteps between deterministic evals",
    )

    parser.add_argument(
        "--actor-layers",
        nargs="+",
        type=int,
        default=[256, 256, 256],
        help="Actor network size",
    )
    parser.add_argument(
        "--critic-layers",
        nargs="+",
        type=int,
        default=[256, 256, 256],
        help="Critic network size",
    )
    parser.add_argument(
        "--actor-act-fn", type=str, default="relu", help="Actor activation function"
    )
    parser.add_argument(
        "--critic-act-fn", type=str, default="relu", help="Critic activation function"
    )
    parser.add_argument(
        "--actor-lr", type=float, default=1e-4, help="Learning rate for the actor"
    )
    parser.add_argument(
        "--critic-lr", type=float, default=1e-4, help="Learning rate for the critic"
    )
    parser.add_argument(
        "--anneal-lr",
        type=str2bool,
        default=True,
        help="Toggles learning rate annealing",
    )

    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="Lambda for the genralized advantage estimation",
    )

    parser.add_argument(
        "--update-epochs", type=int, default=10, help="Number of update epochs"
    )

    parser.add_argument(
        "--n-minibatches", type=int, default=8, help="Number of minibatches"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum norm for gradient clipping",
    )
    parser.add_argument(
        "--target-kl", type=float, default=0.02, help="Target KL divergence threshold"
    )

    parser.add_argument(
        "--norm-adv",
        type=str2bool,
        default=True,
        help="Toggles advantage normalization",
    )

    parser.add_argument(
        "--clip-coef", type=float, default=0.2, help="Surrogate clip coefficient"
    )
    parser.add_argument(
        "--clip-vfloss",
        type=str2bool,
        default=True,
        help="Toggles clip for value function loss",
    )

    parser.add_argument(
        "--entropy-coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--entropy-coef-final",
        type=float,
        default=None,
        help="If set, linearly anneal entropy coefficient from --entropy-coef to this value.",
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient"
    )

    parser.add_argument(
        "--init-do-nothing-prob",
        type=float,
        default=0.5,
        help="Initial softmax probability on action 0 (do-nothing) at actor init. "
        "0.0 keeps the default Xavier init; e.g. 0.7 makes do-nothing 70%% likely at every state initially.",
    )

    parser.add_argument(
        "--norm-reward",
        type=str2bool,
        default=True,
        help="Toggle running-stats reward normalization (SB3 VecNormalize style: divide reward by running std of discounted returns).",
    )
    parser.add_argument(
        "--action0-logit-bonus-init",
        type=float,
        default=0.0,
        help="Extra training-time logit bonus for action 0 at the start of training.",
    )
    parser.add_argument(
        "--action0-logit-bonus-final",
        type=float,
        default=0.0,
        help="Final value for the training-time action-0 logit bonus schedule.",
    )
    parser.add_argument(
        "--action0-logit-bonus-fraction",
        type=float,
        default=1.0,
        help="Fraction of total timesteps over which to anneal the action-0 logit bonus.",
    )
    parser.add_argument(
        "--deterministic-eval",
        type=str2bool,
        default=True,
        help="Use greedy argmax actions during evaluation. Set False to sample evaluation actions.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes/chronics used for each evaluation.",
    )
    parser.add_argument(
        "--eval-all-split-chronics",
        type=str2bool,
        default=True,
        help="When chronic splitting is enabled, evaluate every chronic in the test split.",
    )
    parser.add_argument(
        "--eval-train-chronics",
        type=str2bool,
        default=False,
        help="When chronic splitting is enabled, also evaluate on the train split and log it under train_eval/.",
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gat",
        choices=["gcn", "gat", "gine", "graphsage"],
        help="PyTorch Geometric convolution used by thesis-style gnn encoders.",
    )
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for thesis-style gnn encoders.",
    )
    parser.add_argument(
        "--gnn-out-dim",
        type=int,
        default=128,
        help="Output embedding dimension for thesis-style gnn encoders.",
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=2,
        help="Number of message-passing layers for thesis-style gnn encoders.",
    )
    parser.add_argument(
        "--gnn-heads",
        type=int,
        default=1,
        help="Number of attention heads for thesis-style GAT encoders.",
    )
    parser.add_argument(
        "--gnn-readout-aggr",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="Graph-level node pooling used by thesis-style gnn encoders.",
    )
    parser.add_argument(
        "--graphsage-aggr",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="GraphSAGE neighborhood aggregation for thesis-style gnn encoders.",
    )
    parser.add_argument(
        "--gnn-aggr",
        dest="graphsage_aggr",
        type=str,
        choices=["mean", "sum"],
        help=ap.SUPPRESS,
    )
    parser.add_argument(
        "--gnn-layer-norm",
        type=str2bool,
        default=True,
        help="Use layer normalization inside thesis-style gnn layers.",
    )
    parser.add_argument(
        "--gnn-concat-flat",
        type=str2bool,
        default=False,
        help="Concatenate flat observations to thesis-style gnn embeddings before the head.",
    )
    parser.add_argument(
        "--share-actor-gnn",
        type=str2bool,
        default=False,
        help="Share one actor GNN encoder across all actor policies while keeping separate MLP action heads.",
    )
    parser.add_argument(
        "--gnn-graph-type",
        type=str,
        default="bus",
        choices=["bus"],
        help="Graph type for thesis-style gnn encoders.",
    )

    return parser.parse_known_args()[0]
