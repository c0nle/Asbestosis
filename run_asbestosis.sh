#!/usr/bin/env bash
#SBATCH --account=rwth1954
#SBATCH --job-name=asbestosis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition c23g
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G

set -euo pipefail

echo "=== Job started on $(hostname) at $(date) ==="
echo "PWD: $(pwd)"

mkdir -p logs

# --- Environment (anpassen) ---
PROJECT_DIR="/rwthfs/rz/cluster/home/rwth1954/Asbestosis"
VENV_DIR="${PROJECT_DIR}/.venv"

cd "${PROJECT_DIR}"

# Nicht-interaktiv / schönere Logs
export PYTHONUNBUFFERED=1

# wandb soll nicht fragen -> offline
export WANDB_MODE=offline
export WANDB_SILENT=true

# Optional: Threads sinnvoll begrenzen
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=== Environment ==="
echo "Python before venv: $(which python || true)"
nvidia-smi || true

# venv aktivieren
source "${VENV_DIR}/bin/activate"

echo "Python after venv: $(which python)"
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
PY

echo "=== Starting training ==="
srun python main.py

echo "=== Job finished at $(date) ==="