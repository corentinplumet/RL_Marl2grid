# Launching training on NUS Atlas

Atlas uses PBS Pro, whereas Izar and Jed use Slurm. `job_atlas.sh` hides that
difference: when executed on an Atlas login node, it submits itself with
`qsub`, then activates the training environment and runs `run_from_config.py`
inside the allocated compute job.

## First-time setup

Clone or update the repository on Atlas and make the launcher executable:

```bash
ssh corentin.plumet@atlas9-c01
cd /path/to/RL_Marl2grid
chmod +x job_atlas.sh
```

The default environment name is `marl2grid`. The repository's
`Topology_Task/conda_env.yml` installs CPU-only PyTorch, which matches the
default Atlas allocation. Atlas compute nodes may not have internet access, so
install dependencies and populate any online caches before submitting.

## Submit a training run

From the repository root:

```bash
./job_atlas.sh configs/pooling_screening/gs_s3dw_mp2_h128_connected_neighbors_controlled_mean_s0.toml
```

No config argument is required for that pooling-screening config because it is
the launcher's default. Overrides work in the same style as the other launchers:

```bash
./job_atlas.sh --deterministic-eval false
SEED=3 TOTAL_TIMESTEPS=1000000 ./job_atlas.sh
```

If the NUS allocation requires a PBS project, specify it at submission time:

```bash
ATLAS_PROJECT=your_project ./job_atlas.sh
```

The defaults request 72 CPUs on one node, no GPU, 128 GB of memory, and 120
hours from the `parallel` queue. This matches the pooling-screening configs'
72 vectorized environments and leaves margin above their 96-hour time limit.
The settings can all be changed without editing the file:

```bash
ATLAS_PROJECT=your_project \
ATLAS_QUEUE=your_larger_single_node_queue \
ATLAS_NCPUS=48 \
ATLAS_MEMORY=192gb \
ATLAS_WALLTIME=24:00:00 \
./job_atlas.sh
```

The environments created by `n_envs` are local Python subprocesses, not MPI
workers. They can use many cores on one node, but requesting several PBS nodes
will not distribute them automatically. Keep `select=1` and use the largest
single-node CPU queue available to your project. Run `gstat`, `qstat -Q`, and
`pbsnodes -avSj` on Atlas to see the current queues and node limits. The
current `hpc parallel` helper still prints obsolete LSF instructions, so do not
use its `bsub` example.

The generic Atlas account limit allows at most 96 CPUs running at once. A
72-CPU default therefore runs one training job at a time; additional submitted
jobs remain queued. To prioritize total screening throughput instead, submit
three 32-CPU jobs so all three can run concurrently:

```bash
ATLAS_NCPUS=32 ./job_atlas.sh CONFIG
```

GPU execution remains available as an opt-in. This also sets `CUDA=true`:

```bash
ATLAS_QUEUE=volta_gpu ATLAS_NCPUS=10 ATLAS_NGPUS=1 ATLAS_MEMORY=64gb \
./job_atlas.sh
```

That mode needs a CUDA-enabled PyTorch environment. Check it with:

```bash
conda activate marl2grid
python -c 'import torch; print(torch.cuda.is_available())'
```

For an array, `PBS_ARRAY_INDEX` is automatically mapped to the array mechanism
already used by `run_from_config.py`. It controls the seed when the chosen
config has `seed_from_slurm_array = true`:

```bash
ATLAS_ARRAY=0-7 ./job_atlas.sh
```

To check the resolved training command without starting training, submit a
short dry run:

```bash
DRY_RUN=true ATLAS_WALLTIME=00:10:00 ./job_atlas.sh
```

To inspect the exact `qsub` command without submitting anything:

```bash
ATLAS_SUBMIT_DRY_RUN=true ./job_atlas.sh
```

## Environment and module variations

The launcher searches for Conda in `PATH`, `~/miniconda3`, `~/miniforge3`, and
`~/anaconda3`. For other installations:

```bash
CONDA_BASE=/path/to/miniforge3 CONDA_ENV=marl2grid ./job_atlas.sh CONFIG
```

If Atlas requires modules before Conda or Python becomes available:

```bash
ATLAS_MODULES="module_name another_module" ./job_atlas.sh CONFIG
```

To bypass Conda activation entirely:

```bash
PYTHON_BIN=/path/to/venv/bin/python ./job_atlas.sh CONFIG
```

## Monitor and manage jobs

```bash
qstat -ans1
qstat -f JOB_ID
qdel JOB_ID
```

PBS normally writes a combined stdout/stderr file in the submission directory.
Use `ATLAS_OUTPUT=/path/to/file.log` if a fixed output path is preferred.

The current Atlas queue listing exposes the CPU queue as `parallel`, and NUS
provides `gstat` for checking the queues currently available. Cluster policy
and project names can change, so use `gstat`, `qstat -Q`, `hpc pbs help`, or
your allocation documentation to confirm the current queue and project, then
set `ATLAS_QUEUE` and `ATLAS_PROJECT` as needed.

Public NUS references:

- [PBS job submission and monitoring](https://nusit.nus.edu.sg/services/getting-started/how-to-run-batch-job/)
- [NUS queue discovery and parallel-job tools](https://nusit.nus.edu.sg/services/hpc-newsletter/handy-tools-customised-for-hpc-parallel-computing/)
- [NUS Volta PBS GPU job example](https://nusit.nus.edu.sg/technus/using-detectron2-on-hpc/)
