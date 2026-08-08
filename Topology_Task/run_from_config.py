#!/usr/bin/env python3
"""Run main.py from a TOML config."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 guard
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Python < 3.11 needs the 'tomli' package to read TOML configs. "
            "Install it with: pip install tomli"
        ) from exc


ENV_ALIASES = {
    "ACTION_TIME_LIMIT": "time_limit",
    "PY_TIME_LIMIT": "time_limit",
    "TIME_LIMIT": "time_limit",
    "TORCH_THREADS": "n_threads",
}


def scheduler_job_id() -> str:
    """Return a stable identifier for either a Slurm or PBS allocation."""
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return slurm_job_id

    pbs_job_id = os.environ.get("PBS_JOBID")
    if pbs_job_id:
        # PBS IDs commonly include the server name (for example
        # 12345.atlas). It is not needed in local output directory names.
        return f"pbs_{pbs_job_id.split('.', 1)[0]}"

    return "local"


def scheduler_array_task_id() -> str | None:
    """Return the current Slurm or PBS array index, when present."""
    return (
        os.environ.get("SLURM_ARRAY_TASK_ID")
        or os.environ.get("PBS_ARRAY_INDEX")
        or os.environ.get("PBS_ARRAYID")
    )


def scheduler_cpu_count() -> str:
    """Return the CPU count exposed by Slurm or PBS."""
    return (
        os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("NCPUS")
        or os.environ.get("PBS_NCPUS")
        or "72"
    )


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def coerce_override(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return parse_bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        parts = [part for part in raw.replace(",", " ").split() if part]
        if not current:
            return parts
        return [coerce_override(part, current[0]) for part in parts]
    return raw


def cli_value(value: Any) -> list[str]:
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(cli_value(item))
        return result
    return [str(value)]


def format_value(value: Any, context: dict[str, str]) -> str:
    return str(value).format(**context)


def build_args(config_args: dict[str, Any], extra_args: list[str]) -> list[str]:
    args: list[str] = []
    for key, value in config_args.items():
        if value == "":
            continue
        args.append("--" + key.replace("_", "-"))
        args.extend(cli_value(value))
    args.extend(extra_args)
    return args


def validate_args(config_args: dict[str, Any]) -> None:
    n_envs = int(config_args.get("n_envs", 1))
    if n_envs < 1:
        raise SystemExit("Invalid config: n_envs must be >= 1.")

    n_steps = int(config_args.get("n_steps", 0))
    eval_freq = int(config_args.get("eval_freq", 0))
    if n_steps <= 0 or eval_freq <= 0:
        raise SystemExit("Invalid config: n_steps and eval_freq must be > 0.")
    if n_steps % n_envs != 0:
        raise SystemExit(
            f"Invalid config: n_steps={n_steps} must be divisible by n_envs={n_envs}."
        )
    if eval_freq % n_envs != 0:
        raise SystemExit(
            f"Invalid config: eval_freq={eval_freq} must be divisible by n_envs={n_envs}."
        )

    gnn_type = str(config_args.get("gnn_type", "gat")).lower()
    readout = str(config_args.get("gnn_readout_aggr", "mean")).lower()
    if gnn_type == "sparse_transformer":
        readout = str(config_args.get("sparse_gt_pooling", "") or readout).lower()
    elif readout not in {
        "mean",
        "sum",
        "max",
        "controlled_mean",
        "controlled_sum",
        "controlled_max",
        "energized_mean",
        "controlled_energized_mean",
        "virtual_node",
    }:
        raise SystemExit(
            "Invalid config: readout "
            f"'{readout}' is only implemented for gnn_type=sparse_transformer. "
            "Use mean, sum, max, controlled_mean, controlled_sum, "
            "controlled_max, energized_mean, controlled_energized_mean, "
            "or virtual_node with ordinary GNN encoders."
        )

    direction_choices = {
        "gnn_generator_edge_direction": {
            "bidirectional",
            "asset_to_busbar",
            "busbar_to_asset",
        },
        "gnn_load_edge_direction": {
            "bidirectional",
            "asset_to_busbar",
            "busbar_to_asset",
        },
        "gnn_line_node_edge_direction": {
            "bidirectional",
            "line_to_busbar",
            "busbar_to_line",
        },
        "gnn_summary_edge_direction": {"bidirectional", "toward_summary"},
        "gnn_virtual_edge_direction": {
            "inherit",
            "bidirectional",
            "toward_virtual",
        },
    }
    for key, allowed in direction_choices.items():
        default = (
            "inherit" if key == "gnn_virtual_edge_direction" else "bidirectional"
        )
        value = str(config_args.get(key, default)).lower()
        if value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise SystemExit(
                f"Invalid config: {key}='{value}'. Use one of: {choices}."
            )


def print_python_summary() -> None:
    print("========== Python environment summary ==========")
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    for name in ["torch", "numpy", "grid2op", "lightsim2grid", "wandb"]:
        try:
            mod = __import__(name)
            print(f"{name}: {getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"{name}: import failed: {exc}")
    try:
        import torch

        print("CUDA available:", torch.cuda.is_available())
        print("Torch threads:", torch.get_num_threads())
    except Exception:
        pass
    print("===============================================")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="TOML run config")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="extra main.py args")
    parser.add_argument("--dry-run", action="store_true", help="print command only")
    ns = parser.parse_args()

    task_dir = Path(__file__).resolve().parent
    project_dir = task_dir.parent
    if not (task_dir / "main.py").is_file():
        raise SystemExit(f"Could not find main.py next to {Path(__file__).name}.")

    config_path = Path(ns.config).expanduser()
    if not config_path.is_absolute():
        candidates = [
            Path.cwd() / config_path,
            task_dir / config_path,
            project_dir / config_path,
        ]
        config_path = next((path for path in candidates if path.exists()), task_dir / config_path)
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    run = config.get("run", {})
    launcher = config.get("launcher", {})
    config_args = dict(config.get("args", {}))

    if run.get("seed_from_slurm_array", False) and "SEED" not in os.environ:
        array_task_id = scheduler_array_task_id()
        if array_task_id:
            config_args["seed"] = int(array_task_id)

    env_to_arg = {key.upper(): key for key in config_args}
    env_to_arg.update(ENV_ALIASES)
    for env_key, arg_key in env_to_arg.items():
        if env_key in os.environ and arg_key in config_args:
            config_args[arg_key] = coerce_override(
                os.environ[env_key], config_args[arg_key]
            )

    job_id = scheduler_job_id()
    array_task_id = scheduler_array_task_id()
    context = {
        "config_name": config_path.stem,
        "job_id": job_id,
        "array_task_id": array_task_id or "0",
        "project_dir": str(project_dir),
        "task_dir": str(task_dir),
    }
    run_dir = Path(
        format_value(run.get("run_dir", "outputs/{config_name}-{job_id}"), context)
    )
    if not run_dir.is_absolute():
        run_dir = project_dir / run_dir
    context["run_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "checkpoint").mkdir(exist_ok=True)

    for key, value in config.get("environment", {}).items():
        os.environ[key] = format_value(value, context)
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    os.environ.setdefault("WANDB_INIT_TIMEOUT", "120")
    os.environ.setdefault("WANDB_OFFLINE_FALLBACK", "true")

    for key in ["MPLCONFIGDIR", "WANDB_DIR", "XDG_CACHE_HOME"]:
        if key in os.environ:
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    if os.environ.get("SLURM_JOB_ID") and Path(f"/tmp/{job_id}").is_dir():
        os.environ.setdefault("TMPDIR", f"/tmp/{job_id}")

    extra_args = list(run.get("extra_args", [])) + list(ns.overrides)
    validate_args(config_args)
    command = [sys.executable, "-u", "main.py", *build_args(config_args, extra_args)]

    if launcher.get("use_srun", True) and os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        command = [
            "srun",
            "--ntasks=1",
            f"--cpus-per-task={os.environ.get('SLURM_CPUS_PER_TASK', '72')}",
            f"--cpu-bind={launcher.get('cpu_bind', 'cores')}",
            *command,
        ]

    print("========== Cluster config run ==========")
    print(f"Config: {config_path}")
    print(f"Run name: {run.get('name', config_path.stem)}")
    print(f"Run dir: {run_dir}")
    print(f"Task dir: {task_dir}")
    print(f"Job id: {job_id}")
    print(f"Array task id: {array_task_id or 'none'}")
    print(f"CPUs per task: {scheduler_cpu_count()}")
    print(f"Command: {' '.join(command)}")
    print("====================================")

    if launcher.get("print_python_summary", True):
        print_python_summary()

    if ns.dry_run:
        return 0
    return subprocess.run(command, cwd=task_dir, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
