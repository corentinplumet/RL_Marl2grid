import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .imports import *

CHECKPOINT_FORMAT_VERSION = 2
EXACT_BOUNDARY_PHASE = "post_update"


def capture_rng_state() -> Dict[str, Any]:
    """Capture every process-global RNG used by training."""
    mps_state = None
    if (
        hasattr(th.backends, "mps")
        and th.backends.mps.is_available()
        and hasattr(th, "mps")
        and hasattr(th.mps, "get_rng_state")
    ):
        mps_state = th.mps.get_rng_state().cpu()
    return {
        "python": rnd.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": th.get_rng_state().cpu(),
        "torch_cuda": (
            [state.cpu() for state in th.cuda.get_rng_state_all()]
            if th.cuda.is_available()
            else None
        ),
        "torch_mps": mps_state,
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    if not state:
        raise ValueError("The checkpoint does not contain an RNG state.")
    rnd.setstate(state["python"])
    np.random.set_state(state["numpy"])
    th.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not th.cuda.is_available():
            raise RuntimeError(
                "The checkpoint contains CUDA RNG states, but CUDA is unavailable."
            )
        th.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in cuda_states])
    mps_state = state.get("torch_mps")
    if mps_state is not None:
        if (
            not hasattr(th.backends, "mps")
            or not th.backends.mps.is_available()
            or not hasattr(th, "mps")
            or not hasattr(th.mps, "set_rng_state")
        ):
            raise RuntimeError(
                "The checkpoint contains an MPS RNG state, but MPS is unavailable."
            )
        th.mps.set_rng_state(mps_state.cpu())


def exact_resume_state(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the exact boundary state, or ``None`` for a legacy/snapshot record."""
    version = int(record.get("checkpoint_format_version", 0))
    if version < CHECKPOINT_FORMAT_VERSION:
        return None
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Checkpoint format is newer than this code supports: "
            f"{version} > {CHECKPOINT_FORMAT_VERSION}."
        )
    boundary = record.get("checkpoint_boundary", {})
    state = record.get("resume_state")
    if (
        not isinstance(boundary, dict)
        or "exact" not in boundary
        or "next_rollout" not in boundary
    ):
        raise ValueError("Versioned checkpoint has no valid boundary metadata.")
    if boundary["exact"] is not True:
        return None
    if boundary.get("phase") != EXACT_BOUNDARY_PHASE:
        raise ValueError(
            "Checkpoint claims exact continuation from an unsupported phase: "
            f"{boundary.get('phase')!r}."
        )
    if not isinstance(state, dict):
        raise ValueError(
            "Checkpoint claims exact continuation but has no resume state."
        )
    required = {
        "global_step",
        "next_rollout",
        "rng_state",
        "environment_states",
        "next_obs",
        "reward_normalizers",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(
            "Exact checkpoint is incomplete; missing: " + ", ".join(missing)
        )
    if int(state["next_rollout"]) != int(boundary["next_rollout"]):
        raise ValueError(
            "Checkpoint next-rollout metadata disagrees with its resume state."
        )
    return state


@dataclass
class CheckpointSaver(ABC):
    """Abstract base class for saving and loading checkpoints.

    Attributes:
        run_name (str): The name of the run.
        args (dict): The arguments for the run configuration.
    """

    run_name: str
    args: dict

    def __post_init__(self):
        """Post-initialization to set up the checkpoint directory and load a checkpoint if resuming a run."""
        self.ckpt_dir = "checkpoint"
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.loaded_run, self.record = {}, {}
        self.loaded_checkpoint_path = None
        self._last_record_is_final = False
        if self.args.resume_run_name:
            checkpoint_name = self._resolve_checkpoint_name(self.args.resume_run_name)
            if not os.path.exists(checkpoint_name):
                raise FileNotFoundError(
                    f"Could not find checkpoint '{checkpoint_name}'. "
                    "Pass either a checkpoint stem from Topology_Task/checkpoint "
                    "or a path to a .tar checkpoint."
                )
            self.loaded_checkpoint_path = checkpoint_name
            self.loaded_run = th.load(checkpoint_name, weights_only=False)
            if getattr(self.args, "resume_delete_checkpoint_after_load", False):
                os.remove(checkpoint_name)

    def _resolve_checkpoint_name(self, resume_run_name: str) -> str:
        """Resolve a checkpoint stem or path to a .tar path."""
        checkpoint_name = resume_run_name
        if not checkpoint_name.endswith(".tar"):
            checkpoint_name += ".tar"
        if os.path.isabs(checkpoint_name) or os.path.dirname(checkpoint_name):
            return checkpoint_name
        return os.path.join(self.ckpt_dir, checkpoint_name)

    @property
    def checkpoint_base_name(self) -> str:
        """Base checkpoint filename stem, using exp_tag when available."""
        exp_tag = str(getattr(self.args, "exp_tag", "") or "").strip()
        base_name = exp_tag or self.run_name
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", base_name).strip("_") or self.run_name

    def _checkpoint_path(self, checkpoint_name: str) -> str:
        """Return the path for a checkpoint stem or filename."""
        stem = checkpoint_name[:-4] if checkpoint_name.endswith(".tar") else checkpoint_name
        return os.path.join(self.ckpt_dir, stem + ".tar")

    def _atomic_save(self, checkpoint_path: str) -> None:
        """Write a checkpoint without exposing a partially overwritten archive."""
        temporary_path = f"{checkpoint_path}.{os.getpid()}.tmp"
        try:
            th.save(self.record, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    @property
    def resumed(self) -> bool:
        """Check if a run was resumed from a checkpoint.

        Returns:
            True if a run was resumed, False otherwise.
        """
        return self.loaded_run != {}

    def _get_base_record(self, global_step: int) -> None:
        """Get a base record with the global step.

        Args:
            global_step: The current global step.
        """
        self.record = {
            "global_step": global_step,
        }

    def save(self) -> None:
        """Save the current record to a checkpoint file."""
        prefix = "final_" if self._last_record_is_final else ""
        self._atomic_save(
            self._checkpoint_path(prefix + self.checkpoint_base_name)
        )

    def save_as(self, checkpoint_name: str) -> None:
        """Save the current record to a specific checkpoint name."""
        self._atomic_save(self._checkpoint_path(checkpoint_name))

    @abstractmethod
    def set_record(self) -> None:
        """Abstract method to set the record with specific run details."""
        pass


class MAPPOCheckpoint(CheckpointSaver):
    def set_record(
        self,
        args: Dict[str, Any],
        actors,
        critic: nn.Sequential,
        global_step: int,
        actor_optim: optim,
        critic_optim: optim,
        wb_run_name: str,
        last_rollout: int = 0,
        mark_final: bool = True,
        training_state: Optional[Dict[str, Any]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        checkpoint_phase: str = "snapshot",
    ) -> None:
        """Set the record for PPO checkpoints.

        Args:
            args: Run arguments.
            actors: Actor networks.
            critic: Critic network.
            global_step : Current global step.
            actor_optim: Actor optimizer.
            critic_optim: Critic optimizer.
            wb_run_name: Weights & Biases run name.
            last_rollout: Last rollout step. Defaults to 0.
            mark_final: Whether to prefix the run name when saving the final checkpoint.
            training_state: Optional non-module state such as adaptive
                intervention Lagrange multipliers.
            resume_state: Complete state needed to continue from a post-update
                rollout boundary. ``None`` marks the record as a non-exact
                policy snapshot.
            checkpoint_phase: Point in the training loop represented by this
                record. Exact checkpoints must use ``post_update``.
        """
        if resume_state is not None and checkpoint_phase != EXACT_BOUNDARY_PHASE:
            raise ValueError(
                "Exact resume state may only be recorded at a post-update boundary."
            )
        self.args = args
        self._last_record_is_final = bool(
            mark_final and global_step >= args.total_timesteps - args.n_envs
        )
        self._get_base_record(global_step)
        self.record["args"] = args
        for agent, model in actors.items():
            self.record[agent] = model.state_dict()
        self.record["critic"] = critic.state_dict()
        self.record["actor_optim"] = actor_optim.state_dict()
        self.record["critic_optim"] = critic_optim.state_dict()
        self.record["wb_run_name"] = wb_run_name
        self.record["last_rollout"] = last_rollout
        self.record["training_state"] = training_state or {}
        self.record["checkpoint_format_version"] = CHECKPOINT_FORMAT_VERSION
        self.record["checkpoint_boundary"] = {
            "phase": checkpoint_phase,
            "exact": resume_state is not None,
            "completed_rollout": int(last_rollout),
            "next_rollout": int(last_rollout) + 1,
        }
        self.record["resume_state"] = resume_state
