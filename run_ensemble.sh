#!/usr/bin/env bash
#SBATCH --account=rwth1954
#SBATCH --job-name=asbestosis_ensemble
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition c23g
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

echo "=== Job started on $(hostname) at $(date) ==="

PROJECT_DIR="/rwthfs/rz/cluster/home/rwth1954/Asbestosis"
VENV_DIR="${PROJECT_DIR}/.venv"
DATA_DIR="${ASBESTOSIS_DATA_DIR:-/hpcwork/rwth1954/Asbestosis_Data}"
FOLD_DIR="${ASBESTOSIS_FOLD_DIR:-${PROJECT_DIR}/splits}"
OUT_DIR="${ASBESTOSIS_OUT_DIR:-${PROJECT_DIR}/logs}"

MODEL="${ASBESTOSIS_MODEL:-chexnet}"
N_FOLDS="${ASBESTOSIS_N_FOLDS:-5}"
N_BOOTSTRAP="${ASBESTOSIS_N_BOOTSTRAP:-500}"
BATCH_SIZE="${ASBESTOSIS_BATCH_SIZE:-64}"
NUM_WORKERS="${ASBESTOSIS_NUM_WORKERS:-8}"
HEAD_DROPOUT="${ASBESTOSIS_HEAD_DROPOUT:-0.5}"
SEED="${ASBESTOSIS_SEED:-0}"
NO_WANDB="${ASBESTOSIS_NO_WANDB:-0}"
WANDB_PROJECT="${ASBESTOSIS_WANDB_PROJECT:-Asbestosis}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT=true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

nvidia-smi || true
source "${VENV_DIR}/bin/activate"
python -V

ARGS=(
    "--base-folder"    "${DATA_DIR}"
    "--fold-folder"    "${FOLD_DIR}"
    "--output-folder"  "${OUT_DIR}"
    "--model"          "${MODEL}"
    "--n-folds"        "${N_FOLDS}"
    "--n-bootstrap"    "${N_BOOTSTRAP}"
    "--batch-size"     "${BATCH_SIZE}"
    "--num-workers"    "${NUM_WORKERS}"
    "--head-dropout"   "${HEAD_DROPOUT}"
    "--seed"           "${SEED}"
    "--wandb-project"  "${WANDB_PROJECT}"
)

[[ "${NO_WANDB}" == "1" ]] && ARGS+=("--no-wandb")

echo "Command: srun python ensemble.py ${ARGS[*]}"
srun python ensemble.py "${ARGS[@]}"

echo "=== Job finished at $(date) ==="
