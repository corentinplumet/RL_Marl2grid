from time import time
from pathlib import Path

from alg.mappo.config import get_alg_args
from alg.mappo.core import MAPPO
from common.checkpoint import MAPPOCheckpoint
from common.imports import *
from common.runtime_config import merge_runtime_args
from common.utils import set_random_seed, set_torch, str2bool
from env.config import get_env_args
from env.utils import MAEnvWrapper
from env.wrappers import AsyncMultiAgentVecEnv

# Dictionary mapping algorithm names to their corresponding classes
ALGORITHMS: Dict[str, Type[Any]] = {
    "MAPPO": MAPPO,
}


def _compose_runtime_args(
    base_args: Namespace,
    env_args: Optional[Namespace] = None,
    alg_args: Optional[Namespace] = None,
) -> Namespace:
    """Parse and merge every argument group before constructing environments.

    Structured-observation settings live in the MAPPO parser but are required by
    ``MAEnvWrapper``.  Keeping the merge here ensures training and evaluation
    environments are built from the same graph and normalization configuration.
    """
    env_args = get_env_args() if env_args is None else env_args
    alg_args = get_alg_args() if alg_args is None else alg_args
    return merge_runtime_args(base_args, env_args, alg_args)


def _checkpoint_stem(checkpoint_name: str) -> str:
    stem = Path(checkpoint_name).name
    return stem[:-4] if stem.endswith(".tar") else stem


def _strip_checkpoint_save_prefixes(checkpoint_name: str) -> str:
    stem = _checkpoint_stem(checkpoint_name)
    changed = True
    while changed:
        changed = False
        for prefix in ("final_", "best_test_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                changed = True
    return stem


def _is_completed_final_checkpoint(checkpoint_name: str) -> bool:
    return _checkpoint_stem(checkpoint_name).startswith("final_")


def main(args: Namespace) -> None:

    start_time = time()

    # Graph construction and preprocessing depend on both environment and
    # algorithm arguments, so compose the complete configuration before any
    # training environment (or checkpoint saver) is created.
    args = _compose_runtime_args(args)
    assert args.n_envs >= 1, f"Invalid n° of environments: {args.n_envs}. Must be >= 1"

    alg = args.alg.upper()
    assert alg in ALGORITHMS.keys(), (
        f"Unsupported algorithm: {alg}. Supported algorithms are: {ALGORITHMS}"
    )
    if (alg == "LAGRMAPPO" and args.constraints_type == 0) or (
        alg != "LAGRMAPPO" and args.constraints_type in [1, 2]
    ):
        raise ValueError("Check the constrained version of the alg/env!")

    cli_resume_run_name = args.resume_run_name
    cli_resume_total_timesteps = args.resume_total_timesteps
    cli_resume_time_limit = args.resume_time_limit
    cli_resume_wandb_run_name = args.resume_wandb_run_name
    cli_resume_start_next_rollout = args.resume_start_next_rollout
    cli_resume_delete_checkpoint_after_load = args.resume_delete_checkpoint_after_load
    cli_wandb_mode = args.wandb_mode

    if cli_resume_run_name:
        run_name = cli_resume_wandb_run_name or _strip_checkpoint_save_prefixes(
            cli_resume_run_name
        )
    else:
        run_name = f"{args.alg}_{args.env_id}_{'T' if args.action_type == 'topology' else 'R'}_{args.seed}_{args.difficulty}_{'H' if args.use_heuristic else ''}_{'I' if args.heuristic_type == 'idle' else ''}_{'C1' if args.constraints_type == 1 else 'C2' if args.constraints_type == 2 else ''}_{int(time())}_{np.random.randint(0, 50000)}"

    # Initialize the appropriate checkpoint based on the algorithm
    if alg == "MAPPO":
        checkpoint = MAPPOCheckpoint(run_name, args)
    else:
        pass  # This case should not occur due to earlier assertion

    # Set random seed and Torch configuration
    set_random_seed(args.seed)
    set_torch(args.n_threads, args.th_deterministic, args.cuda)

    # Resume run if checkpoint was resumed
    if checkpoint.resumed:
        args = checkpoint.loaded_run["args"]
        current_global_step = int(checkpoint.loaded_run.get("global_step", 0))
        args.resume_run_name = cli_resume_run_name
        args.resume_total_timesteps = cli_resume_total_timesteps
        args.resume_time_limit = cli_resume_time_limit
        args.resume_wandb_run_name = cli_resume_wandb_run_name
        args.resume_delete_checkpoint_after_load = cli_resume_delete_checkpoint_after_load
        args.wandb_mode = cli_wandb_mode
        args.resume_start_next_rollout = (
            _is_completed_final_checkpoint(cli_resume_run_name)
            if cli_resume_start_next_rollout is None
            else cli_resume_start_next_rollout
        )
        if cli_resume_total_timesteps:
            if cli_resume_total_timesteps <= current_global_step:
                raise ValueError(
                    "--resume-total-timesteps must be larger than the checkpoint "
                    f"global_step ({current_global_step:,}). Got "
                    f"{cli_resume_total_timesteps:,}."
                )
            args.total_timesteps = cli_resume_total_timesteps
        if cli_resume_time_limit:
            args.time_limit = cli_resume_time_limit

    env_fns = [lambda i=i: MAEnvWrapper(args, idx=i) for i in range(args.n_envs)]
    envs = AsyncMultiAgentVecEnv(env_fns)

    # Run the specified algorithm
    ALGORITHMS[alg](envs, run_name, start_time, args, checkpoint)


if __name__ == "__main__":
    # mp.get_context("forkserver")
    parser = ap.ArgumentParser()

    # Cluster
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1300,
        help="Time limit for the action ranking",
    )
    parser.add_argument(
        "--checkpoint", type=str2bool, default=False, help="Toggles checkpoint."
    )
    parser.add_argument(
        "--resume-run-name", type=str, default="", help="Run name to resume"
    )
    parser.add_argument(
        "--resume-total-timesteps",
        type=int,
        default=0,
        help=(
            "When resuming, override the checkpoint's total_timesteps target. "
            "Use this to extend a run to 15M/16M steps."
        ),
    )
    parser.add_argument(
        "--resume-time-limit",
        type=float,
        default=0.0,
        help="When resuming, override the checkpoint's time_limit in minutes.",
    )
    parser.add_argument(
        "--resume-wandb-run-name",
        type=str,
        default="",
        help=(
            "Optional WandB id/name to continue. By default final_ and best_test_ "
            "checkpoint prefixes are stripped from --resume-run-name."
        ),
    )
    parser.add_argument(
        "--resume-start-next-rollout",
        type=str2bool,
        default=None,
        help=(
            "When true, start at last_rollout + 1. Defaults to true for final_ "
            "checkpoints and false for other checkpoints."
        ),
    )
    parser.add_argument(
        "--resume-delete-checkpoint-after-load",
        type=str2bool,
        default=False,
        help="Delete the loaded checkpoint after a successful load.",
    )

    # Reproducibility [MAPPO, QPLEX, LAGRMAPPO]
    parser.add_argument("--alg", type=str, default="MAPPO", help="Algorithm to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Logger
    parser.add_argument("--verbose", type=str2bool, default=True, help="Toggles prints")
    parser.add_argument(
        "--exp-tag", type=str, default="", help="Tag for logging the experiment"
    )
    parser.add_argument(
        "--track", type=str2bool, default=True, help="Tag for logging the experiment"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="Grid2Op",
        help="Wandb's project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="corentin-plumet-epfl",
        help="Entity (team) of wandb's project.",
    )
    parser.add_argument(
        "--wandb-mode", type=str, default="online", help="Online or offline wandb mode."
    )

    # Torch
    parser.add_argument(
        "--th-deterministic",
        type=str2bool,
        default=False,
        help="Enable deterministic in Torch.",
    )
    parser.add_argument(
        "--cuda",
        type=str2bool,
        default=True,
        help="Enable GPU (CUDA or MPS) by default.",
    )

    parser.add_argument(
        "--n-threads", type=int, default=4, help="Max number of torch threads."
    )

    main(parser.parse_known_args()[0])
