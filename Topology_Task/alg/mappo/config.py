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
        "--actor-action-head",
        type=str,
        default="mlp",
        choices=["mlp", "candidate_pool"],
        help=(
            "Actor logit head. candidate_pool scores actions from the "
            "heterogeneous-line graph nodes they affect."
        ),
    )
    parser.add_argument(
        "--candidate-action-pool",
        type=str,
        default="typed_mean",
        choices=["mean", "typed_mean", "typed_attention"],
        help=(
            "Pooling used by candidate_pool over the physical nodes touched "
            "by each action."
        ),
    )
    parser.add_argument(
        "--candidate-action-attention-scope",
        type=str,
        default="affected",
        choices=["affected", "soft_prior", "all"],
        help="Nodes eligible for learned candidate-action attention.",
    )
    parser.add_argument(
        "--candidate-action-attention-heads",
        type=int,
        default=1,
        help="Number of candidate-action attention heads.",
    )
    parser.add_argument(
        "--candidate-action-attention-dim",
        type=int,
        default=0,
        help="Per-head attention size. Zero uses the GNN node dimension.",
    )
    parser.add_argument(
        "--candidate-action-attention-temperature",
        type=float,
        default=1.0,
        help="Positive softmax temperature for candidate-action attention.",
    )
    parser.add_argument(
        "--candidate-action-attention-query",
        type=str,
        default="global_action_features",
        choices=["global_action_features", "learned_action", "global_only"],
        help="Information used to construct each candidate-action query.",
    )
    parser.add_argument(
        "--candidate-action-attention-normalizer",
        type=str,
        default="softmax",
        choices=["softmax"],
        help="Normalization applied across eligible graph nodes.",
    )
    parser.add_argument(
        "--candidate-action-attention-prior-bias",
        type=float,
        default=2.0,
        help=(
            "Additive attention-score bonus for hardcoded affected nodes "
            "when scope=soft_prior."
        ),
    )
    parser.add_argument(
        "--candidate-action-attention-chunk-size",
        type=int,
        default=64,
        help="Number of candidate actions scored per dense-attention chunk.",
    )
    parser.add_argument(
        "--candidate-action-use-features",
        type=str2bool,
        default=True,
        help="Append static physical action features to each candidate score.",
    )
    parser.add_argument(
        "--candidate-action-do-nothing-head",
        type=str2bool,
        default=True,
        help=(
            "Score action 0 with a dedicated head over global graph context "
            "when candidate_pool is enabled."
        ),
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
    parser.add_argument(
        "--lr-anneal-timesteps",
        type=int,
        default=None,
        help=(
            "If set, anneal actor/critic learning rates over this many "
            "environment timesteps instead of over --total-timesteps."
        ),
    )
    parser.add_argument(
        "--lr-final-frac",
        type=float,
        default=0.0,
        help=(
            "Final learning-rate fraction after annealing. For example, 0.1 "
            "keeps 10%% of --actor-lr/--critic-lr after the anneal horizon."
        ),
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
        "--optimize-critic-updates",
        type=str2bool,
        default=True,
        help=(
            "Update the shared centralized critic once per rollout minibatch instead "
            "of once inside every actor update loop. True preserves the optimized "
            "branch behavior; false restores the older "
            "training dynamics."
        ),
    )

    parser.add_argument(
        "--init-do-nothing-prob",
        type=float,
        default=0.5,
        help="Initial softmax probability on action 0 (do-nothing) at actor init. "
        "0.0 keeps the default Xavier init; e.g. 0.7 makes do-nothing 70%% likely at every state initially.",
    )
    parser.add_argument(
        "--intervention-gate",
        type=str2bool,
        default=False,
        help=(
            "Use a hierarchical actor with a learned do-nothing/intervene gate "
            "followed by a non-idle local action head. The environment action "
            "space is unchanged; action 0 is produced when the gate selects "
            "do-nothing."
        ),
    )
    parser.add_argument(
        "--intervention-gate-eval-mode",
        type=str,
        default="final_action_map",
        choices=["final_action_map", "hierarchical_greedy"],
        help=(
            "Deterministic evaluation rule for --intervention-gate. "
            "'final_action_map' chooses the most likely executed environment "
            "action under the full hierarchical policy. 'hierarchical_greedy' "
            "first chooses the most likely gate decision, then the most likely "
            "non-idle action if the gate chooses intervene."
        ),
    )
    parser.add_argument(
        "--intervention-gate-entropy-mode",
        type=str,
        default="coupled",
        choices=["coupled", "separate"],
        help=(
            "Entropy formula used by --intervention-gate. 'coupled' preserves "
            "the original H(gate)+P(intervene)*H(non-idle) objective. "
            "'separate' uses independent weighted gate and non-idle entropy "
            "terms, avoiding a direct entropy incentive to raise P(intervene)."
        ),
    )
    parser.add_argument(
        "--intervention-gate-entropy-mult",
        type=float,
        default=1.0,
        help=(
            "Multiplier on H(gate) when "
            "--intervention-gate-entropy-mode=separate."
        ),
    )
    parser.add_argument(
        "--intervention-nonidle-entropy-mult",
        type=float,
        default=1.0,
        help=(
            "Multiplier on H(non-idle action head) when "
            "--intervention-gate-entropy-mode=separate."
        ),
    )
    parser.add_argument(
        "--intervention-penalty",
        type=float,
        default=0.0,
        help=(
            "Training-time reward penalty applied to each agent for its own "
            "non-idle action. This applies to both the flat actor and "
            "--intervention-gate."
        ),
    )
    parser.add_argument(
        "--safe-intervention-penalty",
        type=float,
        default=0.0,
        help=(
            "Additional training-time reward penalty for a non-idle action when "
            "the pre-action max rho is below --safe-intervention-rho-threshold. "
            "This targets unnecessary interventions in safe states."
        ),
    )
    parser.add_argument(
        "--safe-intervention-rho-threshold",
        type=float,
        default=0.90,
        help="Grid state is treated as safe for sparse-intervention penalties when max rho is below this value.",
    )
    parser.add_argument(
        "--adaptive-intervention-budget",
        type=str2bool,
        default=False,
        help=(
            "Enable an adaptive Lagrangian intervention budget. Each agent is "
            "penalized by lambda_i * cost_i for non-idle actions, and lambda_i "
            "is updated after each rollout from the observed budget violation."
        ),
    )
    parser.add_argument(
        "--intervention-budget-target",
        type=float,
        default=0.25,
        help=(
            "Target mean intervention cost per agent and environment step for "
            "--adaptive-intervention-budget. With local_safe/global_safe costs, "
            "this is a target safe-state weighted intervention rate."
        ),
    )
    parser.add_argument(
        "--intervention-budget-lr",
        type=float,
        default=0.01,
        help="Primal-dual learning rate for adaptive intervention lambdas.",
    )
    parser.add_argument(
        "--intervention-budget-init-lambda",
        type=float,
        default=0.0,
        help="Initial Lagrange multiplier for every agent intervention budget.",
    )
    parser.add_argument(
        "--intervention-budget-max-lambda",
        type=float,
        default=10.0,
        help="Upper clamp for adaptive intervention Lagrange multipliers.",
    )
    parser.add_argument(
        "--intervention-budget-warmup-steps",
        type=int,
        default=0,
        help=(
            "Do not update adaptive intervention lambdas before this many "
            "environment steps. Penalties still use the initial lambda."
        ),
    )
    parser.add_argument(
        "--intervention-budget-cost-mode",
        type=str,
        default="local_safe",
        choices=["nonidle", "global_safe", "local_safe"],
        help=(
            "Cost used by --adaptive-intervention-budget. 'nonidle' charges "
            "every non-idle action equally. 'global_safe' weights non-idle "
            "actions by a smooth global max-rho safety score. 'local_safe' "
            "uses each agent's local max-rho score and is decentralized."
        ),
    )
    parser.add_argument(
        "--intervention-budget-rho-threshold",
        type=float,
        default=0.90,
        help=(
            "Center of the smooth safety weight used by global_safe/local_safe "
            "intervention budget costs."
        ),
    )
    parser.add_argument(
        "--intervention-budget-rho-sharpness",
        type=float,
        default=25.0,
        help=(
            "Sharpness of sigmoid(threshold - max_rho) used by smooth safety "
            "weights. Higher values make the budget closer to a hard threshold."
        ),
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
        "--eval-action-heuristic",
        type=str,
        default="none",
        choices=["none", "rho_threshold", "local_rho_threshold"],
        help=(
            "Evaluation-only action override. 'rho_threshold' uses the global "
            "pre-action max rho and forces all agents to action 0 when the grid "
            "is safe. 'local_rho_threshold' computes a pre-action max rho on "
            "each agent's local observation domain and independently forces "
            "only safe agents to action 0."
        ),
    )
    parser.add_argument(
        "--eval-action-rho-threshold",
        type=float,
        default=0.90,
        help=(
            "Pre-action max rho threshold used by "
            "--eval-action-heuristic rho_threshold/local_rho_threshold. States "
            "below this threshold are treated as safe and eval actions are "
            "forced to do nothing."
        ),
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
        "--trace-rollout-actions",
        type=str2bool,
        default=False,
        help=(
            "Log exact per-agent action traces as WandB tables during training "
            "and evaluation. Disabled by default because decoded action traces "
            "can be verbose."
        ),
    )
    parser.add_argument(
        "--trace-rollout-env-idx",
        type=int,
        default=0,
        help="Vectorized training environment index to include in the action trace table.",
    )
    parser.add_argument(
        "--trace-rollout-max-steps",
        type=int,
        default=512,
        help="Maximum number of rows/steps to keep in each logged action trace table.",
    )
    parser.add_argument(
        "--trace-rollout-every",
        type=int,
        default=10,
        help="Log one training action trace every N rollouts when --trace-rollout-actions is true.",
    )
    parser.add_argument(
        "--trace-rollout-decode-actions",
        type=str2bool,
        default=True,
        help="Decode action ids to Grid2Op action descriptions in rollout action trace tables.",
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gat",
        choices=["gcn", "gat", "gine", "graphsage", "sparse_transformer"],
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
        choices=[
            "mean",
            "sum",
            "max",
            "attention",
            "controlled_mean",
            "controlled_attention",
            "virtual_node",
        ],
        help=(
            "Graph-level readout used by thesis-style GNN encoders. "
            "virtual_node appends a learned node connected to every busbar and "
            "uses its final state instead of pooling."
        ),
    )
    parser.add_argument(
        "--sparse-gt-pooling",
        type=str,
        default="",
        choices=[
            "",
            "mean",
            "sum",
            "max",
            "attention",
            "controlled_mean",
            "controlled_attention",
            "virtual_node",
        ],
        help=(
            "Optional readout override for --gnn-type sparse_transformer. "
            "When unset, --gnn-readout-aggr is used."
        ),
    )
    parser.add_argument(
        "--gnn-add-substation-nodes",
        type=str2bool,
        default=False,
        help=(
            "Append one learned node per included substation and connect it "
            "to that substation's busbars according to "
            "--gnn-summary-edge-direction."
        ),
    )
    parser.add_argument(
        "--gnn-summary-edge-direction",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "toward_summary"],
        help=(
            "Direction of busbar/substation hierarchy edges. bidirectional "
            "exchanges messages in both directions; toward_summary keeps only "
            "busbar-to-substation messages. Virtual-node edges use "
            "--gnn-virtual-edge-direction."
        ),
    )
    parser.add_argument(
        "--gnn-virtual-edge-direction",
        type=str,
        default="inherit",
        choices=["inherit", "bidirectional", "toward_virtual"],
        help=(
            "Direction of edges entering the graph-level virtual node. "
            "toward_virtual keeps only busbar-to-virtual or "
            "substation-to-virtual messages. inherit preserves the historical "
            "--gnn-summary-edge-direction behavior."
        ),
    )
    parser.add_argument(
        "--gnn-add-substation-edges",
        type=str2bool,
        default=None,
        help=(
            "Add bidirectional typed edges between distinct busbars belonging "
            "to the same substation. When unset, the legacy sparse-transformer "
            "default is preserved."
        ),
    )
    parser.add_argument(
        "--sparse-gt-add-self-edges",
        type=str2bool,
        default=None,
        help=(
            "Add explicit self-attention graph edges for sparse graph "
            "transformers. Defaults to true only when --gnn-type is "
            "sparse_transformer."
        ),
    )
    parser.add_argument(
        "--sparse-gt-add-substation-edges",
        type=str2bool,
        default=None,
        help=(
            "Legacy sparse-transformer-specific same-substation edge option. "
            "Used only when --gnn-add-substation-edges is unset."
        ),
    )
    parser.add_argument(
        "--sparse-gt-use-edge-attr",
        type=str2bool,
        default=True,
        help="Use Grid2Op line edge attributes in sparse graph transformer attention.",
    )
    parser.add_argument(
        "--sparse-gt-use-edge-type-embeddings",
        type=str2bool,
        default=True,
        help="Use learned relation embeddings for sparse graph transformer edge types.",
    )
    parser.add_argument(
        "--sparse-gt-relation-bias",
        type=str2bool,
        default=True,
        help="Use learned per-relation attention biases in sparse graph transformers.",
    )
    parser.add_argument(
        "--sparse-gt-dropout",
        type=float,
        default=0.0,
        help="Dropout inside sparse graph transformer residual and feed-forward blocks.",
    )
    parser.add_argument(
        "--sparse-gt-attention-dropout",
        type=float,
        default=0.0,
        help="Dropout applied to sparse graph transformer attention weights.",
    )
    parser.add_argument(
        "--sparse-gt-ffn-multiplier",
        type=int,
        default=4,
        help="Feed-forward hidden-size multiplier inside sparse graph transformer layers.",
    )
    parser.add_argument(
        "--graphsage-aggr",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="GraphSAGE neighborhood aggregation for thesis-style gnn encoders.",
    )
    parser.add_argument(
        "--gcn-edge-weight-feature",
        type=str,
        default="none",
        choices=["none", "rho"],
        help=(
            "Optional edge feature used as GCNConv edge_weight. "
            "Set to 'rho' to weight GCN messages by line loading."
        ),
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
        "--gnn-physical-scaling",
        type=str2bool,
        default=False,
        help=(
            "Scale continuous graph inputs to dimensionless physical ranges "
            "before GNN encoding. Disabled by default for checkpoint and "
            "experiment reproducibility."
        ),
    )
    parser.add_argument(
        "--gnn-running-norm",
        type=str2bool,
        default=False,
        help=(
            "Apply mask-aware running standardization to continuous graph "
            "features using statistics shared by all agents."
        ),
    )
    parser.add_argument(
        "--gnn-power-scale-mw",
        type=float,
        default=0.0,
        help=(
            "Power base in MW used by physical graph scaling. Values <= 0 "
            "select the largest installed generator rating automatically."
        ),
    )
    parser.add_argument(
        "--gnn-norm-clip",
        type=float,
        default=10.0,
        help=(
            "Absolute clipping threshold after graph running "
            "standardization; values <= 0 disable clipping."
        ),
    )
    parser.add_argument(
        "--gnn-node-pre-encoder",
        type=str2bool,
        default=False,
        help="Use a small MLP to embed raw node features before thesis-style gnn message passing.",
    )
    parser.add_argument(
        "--gnn-edge-pre-encoder",
        type=str2bool,
        default=False,
        help="Use a small MLP to embed raw edge features before edge-aware gnn message passing.",
    )
    parser.add_argument(
        "--gnn-node-id-embeddings",
        type=str2bool,
        default=False,
        help=(
            "Concatenate learned substation and node-slot embeddings to each "
            "graph node feature."
        ),
    )
    parser.add_argument(
        "--gnn-node-id-emb-dim",
        type=int,
        default=8,
        help="Embedding size for each learned substation id and busbar id embedding.",
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
        choices=["bus", "heterogeneous", "heterogeneous_line"],
        help=(
            "Graph representation used by GNN encoders: aggregated busbar nodes "
            "or typed equipment nodes, optionally with transmission lines as "
            "explicit nodes."
        ),
    )
    parser.add_argument(
        "--gnn-generator-edge-direction",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "asset_to_busbar", "busbar_to_asset"],
        help="Message direction for generator--busbar attachment relations.",
    )
    parser.add_argument(
        "--gnn-load-edge-direction",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "asset_to_busbar", "busbar_to_asset"],
        help="Message direction for load--busbar attachment relations.",
    )
    parser.add_argument(
        "--gnn-line-node-edge-direction",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "line_to_busbar", "busbar_to_line"],
        help=(
            "Message direction for busbar--line-node attachments when "
            "--gnn-graph-type=heterogeneous_line."
        ),
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="group",
        choices=["group", "entity", "hybrid"],
        help="Token schema used by transformer encoders.",
    )
    parser.add_argument(
        "--tokenizer-include-neighbors",
        type=str2bool,
        default=True,
        help=(
            "For transformer local actor tokens, include one-hop neighboring "
            "substations and touching lines."
        ),
    )
    parser.add_argument(
        "--tokenizer-include-maintenance",
        type=str2bool,
        default=True,
        help="Include maintenance features in transformer token observations.",
    )
    parser.add_argument(
        "--tokenizer-max-feature-dim",
        type=int,
        default=128,
        help="Fixed padded feature width for every raw token.",
    )
    parser.add_argument(
        "--tokenizer-include-busbar-tokens",
        type=str2bool,
        default=False,
        help="Add explicit busbar entity tokens to entity/hybrid tokenizers.",
    )
    parser.add_argument(
        "--transformer-d-model",
        type=int,
        default=128,
        help="Transformer token embedding dimension.",
    )
    parser.add_argument(
        "--transformer-n-heads",
        type=int,
        default=4,
        help="Number of transformer attention heads.",
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=3,
        help="Number of transformer encoder layers.",
    )
    parser.add_argument(
        "--transformer-ff-dim",
        type=int,
        default=512,
        help="Feed-forward hidden dimension inside transformer encoder layers.",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.05,
        help="Dropout used inside transformer encoder layers.",
    )
    parser.add_argument(
        "--transformer-activation",
        type=str,
        default="gelu",
        choices=["relu", "gelu"],
        help="Transformer feed-forward activation.",
    )
    parser.add_argument(
        "--transformer-pool",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="How to pool token outputs before actor/critic heads.",
    )
    parser.add_argument(
        "--transformer-concat-flat",
        type=str2bool,
        default=False,
        help="Concatenate the flat observation to the transformer token embedding.",
    )
    parser.add_argument(
        "--transformer-use-entity-id-embeddings",
        type=str2bool,
        default=True,
        help="Add learned entity-id embeddings to token inputs.",
    )
    parser.add_argument(
        "--transformer-use-substation-embeddings",
        type=str2bool,
        default=True,
        help="Add learned substation/endpoint embeddings to token inputs.",
    )
    parser.add_argument(
        "--transformer-use-bus-embeddings",
        type=str2bool,
        default=True,
        help="Add learned busbar embeddings to token inputs.",
    )
    parser.add_argument(
        "--transformer-use-agent-embeddings",
        type=str2bool,
        default=True,
        help="Add learned agent/domain id embeddings to token inputs.",
    )
    parser.add_argument(
        "--transformer-layer-norm-eps",
        type=float,
        default=1e-5,
        help="LayerNorm epsilon used in transformer encoders.",
    )

    return parser.parse_known_args()[0]
