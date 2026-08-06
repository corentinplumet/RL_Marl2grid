#!/usr/bin/env bash
# NUS Atlas / Volta PBS Pro launcher.
#
# These directives are defaults for users who call `qsub job_atlas.sh`
# directly. When this file is executed as `./job_atlas.sh`, the submission
# wrapper below passes the configurable values on the qsub command line.
#PBS -N marl2grid_atlas
#PBS -q parallel
#PBS -l select=1:ncpus=72:mem=128gb
#PBS -l walltime=120:00:00
#PBS -j oe

set -euo pipefail

DEFAULT_CONFIG="configs/pooling_screening/gs_s3dw_mp2_h128_connected_neighbors_controlled_mean_s0.toml"

usage() {
    cat <<'EOF'
Usage (recommended, from the repository root on Atlas):
  ./job_atlas.sh [CONFIG] [RUN_FROM_CONFIG_OVERRIDES...]

Examples:
  ./job_atlas.sh
  ./job_atlas.sh configs/pooling_screening/gs_s3dw_mp2_h128_connected_neighbors_controlled_mean_s0.toml
  ./job_atlas.sh --deterministic-eval false
  SEED=3 TOTAL_TIMESTEPS=1000000 ./job_atlas.sh
  DRY_RUN=true ./job_atlas.sh

The script submits itself with qsub when run on an Atlas login node. Inside a
PBS allocation it activates the environment and launches run_from_config.py.
Config paths are resolved relative to Topology_Task.

PBS submission settings:
  ATLAS_PROJECT       PBS project/allocation passed with -P. No default.
  ATLAS_QUEUE         PBS queue. Default: parallel
  ATLAS_NCPUS         CPU cores on one node. Default: 72
  ATLAS_NGPUS         GPUs. Default: 0
  ATLAS_MEMORY        Memory. Default: 128gb
  ATLAS_WALLTIME      Wall time. Default: 120:00:00
  ATLAS_JOB_NAME      PBS job name. Default: marl2grid_atlas
  ATLAS_ARRAY         Optional PBS array range, for example 0-7.
  ATLAS_MAIL_USER     Optional notification email address.
  ATLAS_MAIL_EVENTS   PBS mail events. Default when email is set: abe
  ATLAS_OUTPUT        Optional PBS output path passed with -o.
  ATLAS_SELECT        Complete PBS select expression. Overrides CPU/GPU/memory.
  ATLAS_SUBMIT_DRY_RUN=true
                      Print the qsub command without submitting it.

Training environment:
  ATLAS_CONFIG        Default config path when CONFIG is omitted.
  CONDA_ENV           Conda environment. Default: marl2grid
  CONDA_BASE          Conda installation path when conda is not on PATH.
  ATLAS_MODULES       Space-separated modules to load before activating conda.
  PYTHON_BIN          Python executable; when set, conda activation is skipped.
  CUDA                Override config CUDA setting. Default: false for CPU jobs.
  DRY_RUN=true        Resolve/print the training command without training.
  REPO_DIR            Repository path. Normally detected automatically.
  ATLAS_RUN_DIRECT=true
                      Run in the current shell instead of submitting with PBS.

Uppercase environment variables override TOML [args] values, for example
N_ENVS, SEED, TOTAL_TIMESTEPS, or TORCH_THREADS for n_threads.

Direct qsub is also supported:
  qsub -P PROJECT -v "REPO_DIR=$PWD" job_atlas.sh
EOF
}

is_true() {
    case "${1:-false}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

CONFIG="${ATLAS_CONFIG:-${DEFAULT_CONFIG}}"
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
    CONFIG="$1"
    shift
fi

# Executing this file on an Atlas login node is the convenient submission
# interface. PBS executes the same file again with PBS_JOBID set.
if [[ -z "${PBS_JOBID:-}" ]] && ! is_true "${ATLAS_RUN_DIRECT:-false}"; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "qsub was not found. Run this on an Atlas login node, or set ATLAS_RUN_DIRECT=true for a local run." >&2
        exit 1
    fi

    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
    REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"

    if [[ ! -d "${REPO_DIR}/Topology_Task" ]]; then
        echo "Could not find ${REPO_DIR}/Topology_Task." >&2
        echo "Run the script from the cloned repository or set REPO_DIR explicitly." >&2
        exit 1
    fi

    if [[ "${CONFIG}" = /* ]]; then
        config_candidates=("${CONFIG}")
    else
        config_candidates=("${REPO_DIR}/Topology_Task/${CONFIG}" "${REPO_DIR}/${CONFIG}")
    fi
    config_found=false
    for config_candidate in "${config_candidates[@]}"; do
        if [[ -f "${config_candidate}" ]]; then
            config_found=true
            break
        fi
    done
    if [[ "${config_found}" != true ]]; then
        echo "Could not find training config: ${CONFIG}" >&2
        echo "Config paths are normally relative to ${REPO_DIR}/Topology_Task." >&2
        exit 1
    fi

    if [[ "${REPO_DIR}" == *,* || "${CONFIG}" == *,* ]]; then
        echo "REPO_DIR and CONFIG cannot contain commas because PBS -v uses commas as separators." >&2
        exit 1
    fi

    ATLAS_QUEUE="${ATLAS_QUEUE:-parallel}"
    ATLAS_NCPUS="${ATLAS_NCPUS:-72}"
    ATLAS_NGPUS="${ATLAS_NGPUS:-0}"
    ATLAS_MEMORY="${ATLAS_MEMORY:-128gb}"
    ATLAS_WALLTIME="${ATLAS_WALLTIME:-120:00:00}"
    ATLAS_JOB_NAME="${ATLAS_JOB_NAME:-marl2grid_atlas}"
    if [[ -z "${ATLAS_SELECT:-}" ]]; then
        ATLAS_SELECT="select=1:ncpus=${ATLAS_NCPUS}:mem=${ATLAS_MEMORY}"
        if [[ "${ATLAS_NGPUS}" != "0" ]]; then
            ATLAS_SELECT+=":ngpus=${ATLAS_NGPUS}"
        fi
    fi

    if [[ -z "${CUDA+x}" ]]; then
        if [[ "${ATLAS_NGPUS}" == "0" ]]; then
            CUDA=false
        else
            CUDA=true
        fi
    fi
    export CUDA

    submit=(
        qsub
        -V
        -N "${ATLAS_JOB_NAME}"
        -q "${ATLAS_QUEUE}"
        -l "${ATLAS_SELECT}"
        -l "walltime=${ATLAS_WALLTIME}"
        -v "REPO_DIR=${REPO_DIR},ATLAS_CONFIG=${CONFIG}"
    )

    if [[ -n "${ATLAS_PROJECT:-}" ]]; then
        submit+=(-P "${ATLAS_PROJECT}")
    fi
    if [[ -n "${ATLAS_ARRAY:-}" ]]; then
        submit+=(-J "${ATLAS_ARRAY}")
    fi
    if [[ -n "${ATLAS_MAIL_USER:-}" ]]; then
        submit+=(-M "${ATLAS_MAIL_USER}" -m "${ATLAS_MAIL_EVENTS:-abe}")
    fi
    if [[ -n "${ATLAS_OUTPUT:-}" ]]; then
        submit+=(-o "${ATLAS_OUTPUT}")
    fi

    # PBS Pro's -F option forwards a whitespace-separated argument string to
    # the batch script. The config itself travels through ATLAS_CONFIG, so only
    # CLI overrides need to be forwarded here.
    if [[ $# -gt 0 ]]; then
        for forwarded_arg in "$@"; do
            if [[ "${forwarded_arg}" =~ [[:space:]] ]]; then
                echo "PBS cannot safely forward an override containing whitespace: ${forwarded_arg}" >&2
                echo "Use the corresponding uppercase environment variable instead; qsub -V preserves it." >&2
                exit 1
            fi
        done
        printf -v forwarded_args '%s ' "$@"
        submit+=(-F "${forwarded_args}")
    fi
    submit+=("${SCRIPT_PATH}")

    echo "Submitting Atlas training job"
    echo "  repository: ${REPO_DIR}"
    echo "  config:     ${CONFIG}"
    echo "  resources:  ${ATLAS_SELECT}, walltime=${ATLAS_WALLTIME}, queue=${ATLAS_QUEUE}"
    if [[ -n "${ATLAS_PROJECT:-}" ]]; then
        echo "  project:    ${ATLAS_PROJECT}"
    fi

    if is_true "${ATLAS_SUBMIT_DRY_RUN:-false}"; then
        printf 'qsub command:'
        printf ' %q' "${submit[@]}"
        printf '\n'
        exit 0
    fi

    "${submit[@]}"
    exit 0
fi

REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:-$(pwd)}}"
TASK_DIR="${REPO_DIR}/Topology_Task"

if [[ ! -d "${TASK_DIR}" ]]; then
    echo "Could not find ${TASK_DIR}." >&2
    echo "Submit from the repository root, or set REPO_DIR=/path/to/RL_Marl2grid." >&2
    exit 1
fi

echo "Job ${PBS_JOBID:-direct} started on $(hostname) at $(date)"
if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
    echo "Allocated hosts:"
    sort -u "${PBS_NODEFILE}"
fi

# Atlas installations commonly expose software through environment modules.
# Loading is opt-in so this also works with a user-owned Miniforge install.
if [[ -n "${ATLAS_MODULES:-}" ]]; then
    if ! command -v module >/dev/null 2>&1; then
        for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
            if [[ -r "${module_init}" ]]; then
                # shellcheck disable=SC1090
                source "${module_init}"
                break
            fi
        done
    fi
    if ! command -v module >/dev/null 2>&1; then
        echo "ATLAS_MODULES was set, but the module command is unavailable." >&2
        exit 1
    fi
    read -r -a atlas_modules <<< "${ATLAS_MODULES}"
    module load "${atlas_modules[@]}"
fi

CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-marl2grid}}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${CONDA_BASE:-}" ]]; then
        :
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="${HOME}/miniconda3"
    elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="${HOME}/miniforge3"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="${HOME}/anaconda3"
    else
        echo "Could not find conda. Load a conda module, set CONDA_BASE, or set PYTHON_BIN." >&2
        exit 1
    fi

    if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        echo "Could not find ${CONDA_BASE}/etc/profile.d/conda.sh." >&2
        exit 1
    fi

    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    PYTHON_BIN="python"
    echo "Using conda env: ${CONDA_ENV}"
else
    echo "Using Python executable: ${PYTHON_BIN}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

# Atlas defaults to a CPU-only allocation. GPU submissions made through the
# wrapper export CUDA=true automatically when ATLAS_NGPUS is non-zero.
export CUDA="${CUDA:-false}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

cd "${TASK_DIR}"

run_config_args=()
if is_true "${DRY_RUN:-false}"; then
    run_config_args+=(--dry-run)
fi

echo "Using config: ${CONFIG}"
"${PYTHON_BIN}" -u run_from_config.py "${run_config_args[@]}" "${CONFIG}" "$@"

echo "Job ${PBS_JOBID:-direct} finished at $(date)"
