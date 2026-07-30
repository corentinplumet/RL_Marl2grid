import os
import subprocess
from collections import deque

import shutil

from .imports import *


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class Logger:
    """Logger class for managing and logging metrics to WandB.

    Attributes:
        log_freq (int): Frequency of logging metrics.
        episodic_survival (Deque[float]): Deque to store episodic survival metrics.
        episodic_return (Deque[float]): Deque to store episodic return metrics.
        episodic_length (Deque[float]): Deque to store episodic length metrics.
        wb_mode (str): Mode for WandB (online/offline).
        wb_path (str): Path to WandB logging directory.
    """

    def __init__(self, run_name: str, args: Dict[str, Any], log_freq: int = 1) -> None:
        """Initialize the Logger with given parameters.

        Args:
            run_name: Name of the run for logging.
            args: Arguments containing configuration for logging.
            log_freq: Frequency of logging metrics.
        """
        self.log_freq = log_freq

        self.episodic_survival = deque(maxlen=log_freq)
        self.episodic_return = deque(maxlen=log_freq)
        self.episodic_length = deque(maxlen=log_freq)
        self._eval_buffers = {
            None: {
                "survival": self.episodic_survival,
                "return": self.episodic_return,
            }
        }

        self.wb_mode = str(args.wandb_mode).strip().lower()

        init_timeout = float(os.environ.get("WANDB_INIT_TIMEOUT", "120"))

        def _init_wandb(mode: str):
            return wb.init(
                name=run_name,
                id=run_name,
                config=vars(args),
                mode=mode,
                project=args.wandb_project,
                entity=args.wandb_entity,
                settings=wb.Settings(
                    _disable_stats=True,
                    init_timeout=init_timeout,
                ),
                resume=(
                    True
                    if mode == "online" and args.resume_run_name
                    else None
                ),
                # sync_tensorboard=True,
            )

        try:
            wb_path = _init_wandb(self.wb_mode)
        except wb.errors.CommError as exc:
            allow_fallback = _env_flag("WANDB_OFFLINE_FALLBACK", True)
            if self.wb_mode != "online" or not allow_fallback:
                raise
            print(
                "W&B online initialization failed; continuing in offline mode. "
                f"The local run will be retained for later sync. Error: {exc}",
                flush=True,
            )
            try:
                wb.teardown()
            except Exception as teardown_exc:
                print(
                    "W&B cleanup after the online timeout reported an error; "
                    f"attempting offline initialization anyway: {teardown_exc}",
                    flush=True,
                )
            self.wb_mode = "offline"
            wb_path = _init_wandb(self.wb_mode)
        self.wb_path = os.path.split(wb_path.dir)[0]

    def _get_eval_buffers(self, prefix: Optional[str] = None) -> Dict[str, Deque]:
        if prefix not in self._eval_buffers:
            self._eval_buffers[prefix] = {
                "survival": deque(maxlen=self.log_freq),
                "return": deque(maxlen=self.log_freq),
            }
        return self._eval_buffers[prefix]

    def _survival_keys(self, prefix: Optional[str] = None) -> List[str]:
        if prefix:
            return [
                f"{prefix}/charts/episodic_survival",
                f"{prefix}/episodic_survival",
            ]
        return ["charts/episodic_survival"]

    def store_metrics(
        self,
        global_step: int,
        avg_survival: float,
        avg_return: float,
        tags: List,
        prefix: Optional[str] = None,
    ) -> None:
        """Store the given metrics and log them if the log frequency is met.

        Args:
            global_step: Current global step of training.
            avg_survival: Average survival metric to be stored.
            avg_return: Average return metric to be stored.
        """
        buffers = self._get_eval_buffers(prefix)
        buffers["survival"].append(avg_survival)
        buffers["return"].append(avg_return)
        if global_step % self.log_freq == 0:
            self.log_metrics(global_step, tags, prefix)

    def log_metrics(
        self, global_step: int, tags: List, prefix: Optional[str] = None
    ) -> None:
        """Log the stored metrics to WandB.

        Args:
            global_step: Current global step of training.
        """
        buffers = self._get_eval_buffers(prefix)
        metric_tags = [f"{prefix}/{tag}" for tag in tags] if prefix else tags
        record = dict(zip(metric_tags, buffers["return"][0]))   # assuming log_freq=1
        for survival_key in self._survival_keys(prefix):
            record[survival_key] = np.mean(buffers["survival"])
        record['charts/global_step'] = global_step

        wb.log(record, step=global_step)

    def log_train_metrics(self, global_step: int, metrics: Dict[str, Any]) -> None:
        """Log per-rollout training metrics (entropy, KL, losses, action stats) to WandB.

        Args:
            global_step: Current global step of training.
            metrics: Flat dict of metric name -> scalar value.
        """
        record = dict(metrics)
        record['charts/global_step'] = global_step
        wb.log(record, step=global_step)

    def close(self) -> None:
        """Close the logger and clean up resources."""
        if self.wb_path is None:
            return
        wb.finish()
        if self.wb_mode == "offline":
            sync_timeout = float(os.environ.get("WANDB_SYNC_TIMEOUT", "300"))
            try:
                result = subprocess.run(
                    ["wandb", "sync", "--append", self.wb_path],
                    check=False,
                    timeout=sync_timeout,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(
                    "Automatic W&B sync failed; offline data was retained at "
                    f"{self.wb_path}. Sync it later with: "
                    f"wandb sync --append {self.wb_path}\nError: {exc}",
                    flush=True,
                )
                return
            if result.returncode == 0:
                shutil.rmtree(self.wb_path)
            else:
                print(
                    "Automatic W&B sync failed; offline data was retained at "
                    f"{self.wb_path}. Sync it later with: "
                    f"wandb sync --append {self.wb_path}",
                    flush=True,
                )

class ConstrainedLogger(Logger):
    """Logger class for managing and logging metrics to WandB.

    Attributes:
        log_freq (int): Frequency of logging metrics.
        episodic_survival (Deque[float]): Deque to store episodic survival metrics.
        episodic_return (Deque[float]): Deque to store episodic return metrics.
        episodic_length (Deque[float]): Deque to store episodic length metrics.
        wb_mode (str): Mode for WandB (online/offline).
        wb_path (str): Path to WandB logging directory.
    """

    def __init__(self, run_name: str, args: Dict[str, Any], log_freq: int = 1) -> None:
        """Initialize the Logger with given parameters.

        Args:
            run_name: Name of the run for logging.
            args: Arguments containing configuration for logging.
            log_freq: Frequency of logging metrics.
        """
        super().__init__(run_name, args, log_freq)
        self.episodic_cost = deque(maxlen=log_freq)
        self._eval_buffers[None]["cost"] = self.episodic_cost

    def store_metrics(
        self,
        global_step: int,
        avg_survival: float,
        avg_return: float,
        avg_cost: float,
        tags: List,
        prefix: Optional[str] = None,
    ) -> None:
        """Store the given metrics and log them if the log frequency is met.

        Args:
            global_step: Current global step of training.
            avg_survival: Average survival metric to be stored.
            avg_return: Average return metric to be stored.
            avg_cost: Average cost return metric to be stored.

        """
        buffers = self._get_eval_buffers(prefix)
        buffers["survival"].append(avg_survival)
        buffers["return"].append(avg_return)
        if "cost" not in buffers:
            buffers["cost"] = deque(maxlen=self.log_freq)
        buffers["cost"].append(avg_cost)
        if global_step % self.log_freq == 0:
            self.log_metrics(global_step, tags, prefix)

    def log_metrics(
        self, global_step: int, tags: List, prefix: Optional[str] = None
    ) -> None:
        """Log the stored metrics to WandB.

        Args:
            global_step: Current global step of training.
        """
        buffers = self._get_eval_buffers(prefix)
        if "cost" not in buffers:
            buffers["cost"] = deque(maxlen=self.log_freq)
        metric_tags = [f"{prefix}/{tag}" for tag in tags] if prefix else tags
        record = dict(zip(metric_tags, buffers["return"][0]))   # assuming log_freq=1
        record['charts/global_step'] = global_step
        for survival_key in self._survival_keys(prefix):
            record[survival_key] = np.mean(buffers["survival"])
        cost_keys = (
            [f"{prefix}/charts/episodic_cost", f"{prefix}/episodic_cost"]
            if prefix
            else ["charts/episodic_cost"]
        )
        for cost_key in cost_keys:
            record[cost_key] = np.mean(buffers["cost"])

        wb.log(record, step=global_step)
