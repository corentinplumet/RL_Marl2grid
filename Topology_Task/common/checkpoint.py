import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .imports import *


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
        th.save(self.record, self._checkpoint_path(prefix + self.checkpoint_base_name))

    def save_as(self, checkpoint_name: str) -> None:
        """Save the current record to a specific checkpoint name."""
        th.save(self.record, self._checkpoint_path(checkpoint_name))

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
        """
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
