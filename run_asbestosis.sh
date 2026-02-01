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
DATA_DIR="${ASBESTOSIS_DATA_DIR:-/hpcwork/rwth1954/Asbestosis_Data}"
FOLD_DIR="${ASBESTOSIS_FOLD_DIR:-${PROJECT_DIR}/splits}"
OUT_DIR="${ASBESTOSIS_OUT_DIR:-${PROJECT_DIR}/logs}"

# Model/data options
LABEL_COL="${ASBESTOSIS_LABEL:-mixed_shapes}"
FOLD="${ASBESTOSIS_FOLD:-0}"
EPOCHS="${ASBESTOSIS_EPOCHS:-40}"
BATCH_SIZE="${ASBESTOSIS_BATCH_SIZE:-24}"
LR="${ASBESTOSIS_LR:-1e-4}"
NUM_WORKERS="${ASBESTOSIS_NUM_WORKERS:-8}"
EVAL_EVERY="${ASBESTOSIS_EVAL_EVERY:-1}"
TEST_EVERY="${ASBESTOSIS_TEST_EVERY:-0}"
MAX_TRAIN_STEPS="${ASBESTOSIS_MAX_TRAIN_STEPS:-}"
MAX_EVAL_STEPS="${ASBESTOSIS_MAX_EVAL_STEPS:-}"

# Overfitting controls / model options
MODEL="${ASBESTOSIS_MODEL:-vit_b_16}"
HEAD_DROPOUT="${ASBESTOSIS_HEAD_DROPOUT:-0.1}"
WEIGHT_DECAY="${ASBESTOSIS_WEIGHT_DECAY:-1e-4}"
FREEZE_BACKBONE_EPOCHS="${ASBESTOSIS_FREEZE_BACKBONE_EPOCHS:-5}"
EARLY_STOP_PATIENCE="${ASBESTOSIS_EARLY_STOP_PATIENCE:-12}"
EARLY_STOP_METRIC="${ASBESTOSIS_EARLY_STOP_METRIC:-auc/eval}"
EARLY_STOP_MIN_DELTA="${ASBESTOSIS_EARLY_STOP_MIN_DELTA:-0.0}"
BALANCED_SAMPLER="${ASBESTOSIS_BALANCED_SAMPLER:-1}"
BACKBONE_LR_MULT="${ASBESTOSIS_BACKBONE_LR_MULT:-0.1}"

# Feature toggles (set to 1 to enable)
NO_PRETRAINED="${ASBESTOSIS_NO_PRETRAINED:-0}"
NO_WANDB="${ASBESTOSIS_NO_WANDB:-0}"
DEDUPE_BY_FILEID="${ASBESTOSIS_DEDUPE_BY_FILEID:-0}"
DROP_CONFLICTS="${ASBESTOSIS_DROP_CONFLICTING_FILEID_LABELS:-0}"
CHECK_MAPPING="${ASBESTOSIS_CHECK_MAPPING:-0}"
SANITY_OVERFIT="${ASBESTOSIS_SANITY_OVERFIT:-0}"
SANITY_SAMPLES="${ASBESTOSIS_SANITY_SAMPLES:-32}"
SANITY_EPOCHS="${ASBESTOSIS_SANITY_EPOCHS:-50}"
SANITY_LR="${ASBESTOSIS_SANITY_LR:-1e-3}"
SEED="${ASBESTOSIS_SEED:-0}"

cd "${PROJECT_DIR}"

# Nicht-interaktiv / schönere Logs
export PYTHONUNBUFFERED=1

# wandb soll nicht fragen
export WANDB_MODE="${WANDB_MODE:-online}"  # set WANDB_MODE=offline for offline runs
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
mkdir -p "${FOLD_DIR}" "${OUT_DIR}"

ARGS=(
  "--base-folder" "${DATA_DIR}"
  "--fold-folder" "${FOLD_DIR}"
  "--output-folder" "${OUT_DIR}"
  "--label" "${LABEL_COL}"
  "--model" "${MODEL}"
  "--head-dropout" "${HEAD_DROPOUT}"
  "--fold" "${FOLD}"
  "--epochs" "${EPOCHS}"
  "--batch-size" "${BATCH_SIZE}"
  "--learning-rate" "${LR}"
  "--weight-decay" "${WEIGHT_DECAY}"
  "--backbone-lr-mult" "${BACKBONE_LR_MULT}"
  "--freeze-backbone-epochs" "${FREEZE_BACKBONE_EPOCHS}"
  "--num-workers" "${NUM_WORKERS}"
  "--seed" "${SEED}"
)

if [[ -n "${EVAL_EVERY}" && "${EVAL_EVERY}" != "0" ]]; then
  ARGS+=("--eval-every" "${EVAL_EVERY}")
fi
if [[ -n "${TEST_EVERY}" && "${TEST_EVERY}" != "0" ]]; then
  ARGS+=("--test-every" "${TEST_EVERY}")
fi
if [[ -n "${EARLY_STOP_PATIENCE}" && "${EARLY_STOP_PATIENCE}" != "0" ]]; then
  ARGS+=(
    "--early-stop-patience" "${EARLY_STOP_PATIENCE}"
    "--early-stop-metric" "${EARLY_STOP_METRIC}"
    "--early-stop-min-delta" "${EARLY_STOP_MIN_DELTA}"
  )
fi
if [[ "${BALANCED_SAMPLER}" == "1" ]]; then
  ARGS+=("--balanced-sampler")
fi
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  ARGS+=("--max-train-steps" "${MAX_TRAIN_STEPS}")
fi
if [[ -n "${MAX_EVAL_STEPS}" ]]; then
  ARGS+=("--max-eval-steps" "${MAX_EVAL_STEPS}")
fi

if [[ "${NO_PRETRAINED}" == "1" ]]; then
  ARGS+=("--no-pretrained")
fi
if [[ "${NO_WANDB}" == "1" ]]; then
  ARGS+=("--no-wandb")
fi
if [[ "${DEDUPE_BY_FILEID}" == "1" ]]; then
  ARGS+=("--dedupe-by-fileid")
fi
if [[ "${DROP_CONFLICTS}" == "1" ]]; then
  ARGS+=("--drop-conflicting-fileid-labels")
fi
if [[ "${CHECK_MAPPING}" == "1" ]]; then
  ARGS+=("--check-mapping")
fi
if [[ "${SANITY_OVERFIT}" == "1" ]]; then
  ARGS+=(
    "--sanity-overfit"
    "--sanity-samples" "${SANITY_SAMPLES}"
    "--sanity-epochs" "${SANITY_EPOCHS}"
    "--sanity-lr" "${SANITY_LR}"
  )
fi

echo "Command: srun python main.py ${ARGS[*]}"
srun python main.py "${ARGS[@]}"

echo "=== Job finished at $(date) ==="
