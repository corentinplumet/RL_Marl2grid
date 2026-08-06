from __future__ import annotations

from run_from_config import (
    scheduler_array_task_id,
    scheduler_cpu_count,
    scheduler_job_id,
)


def test_scheduler_metadata_uses_pbs_environment(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("PBS_JOBID", "12345.atlas")
    monkeypatch.setenv("PBS_ARRAY_INDEX", "3")
    monkeypatch.setenv("NCPUS", "10")

    assert scheduler_job_id() == "pbs_12345"
    assert scheduler_array_task_id() == "3"
    assert scheduler_cpu_count() == "10"


def test_scheduler_metadata_keeps_slurm_precedence(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "987")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "40")
    monkeypatch.setenv("PBS_JOBID", "12345.atlas")
    monkeypatch.setenv("PBS_ARRAY_INDEX", "3")
    monkeypatch.setenv("NCPUS", "10")

    assert scheduler_job_id() == "987"
    assert scheduler_array_task_id() == "4"
    assert scheduler_cpu_count() == "40"
