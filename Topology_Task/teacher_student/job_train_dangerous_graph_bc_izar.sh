#!/usr/bin/env bash
#SBATCH --job-name=train_danger_bc
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Topology_Task/teacher_student/slurm-%x-%j.out
#SBATCH --error=Topology_Task/teacher_student/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  sbatch --exclude=i39 Topology_Task/teacher_student/job_train_dangerous_graph_bc_izar.sh \
    --dataset outputs/teacher_student_datasets/wcci_nomaint_mk64_danger090 \
    --checkpoint checkpoint/final_ft64c_NLS_mean_f1_a0h0_s0.tar \
    --output checkpoint/dangerous_graph_bc/bcg_bal020_w3_aux025_mean_f1_a0h0_s0.tar

Environment:
  CONDA_ENV   Conda environment name. Default: marl2grid
  CONDA_BASE  Conda installation root if conda is not on PATH.
  REPO_DIR    Repository root. Default: SLURM_SUBMIT_DIR or current directory.
EOF
  exit 0
fi

repo_dir="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
task_dir="$repo_dir/Topology_Task"
conda_env="${CONDA_ENV:-${CONDA_ENV_NAME:-marl2grid}}"

if [[ ! -d "$task_dir" ]]; then
  echo "Could not find $task_dir" >&2
  exit 1
fi

if [[ -n "${CONDA_BASE:-}" ]]; then
  conda_base="$CONDA_BASE"
elif command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  conda_base="${HOME}/miniforge3"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  conda_base="${HOME}/miniconda3"
else
  echo "Could not find conda; set CONDA_BASE." >&2
  exit 1
fi

source "$conda_base/etc/profile.d/conda.sh"
conda activate "$conda_env"
cd "$task_dir"

echo "Dangerous graph-BC training job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Arguments: $*"
python -u teacher_student/train_dangerous_graph_bc.py "$@"
