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
SPLIT_LABEL="${ASBESTOSIS_SPLIT_LABEL:-mixed_shapes}"
# Default label set: focus on well-supported tasks (rare/unstable tasks can be enabled via ASBESTOSIS_LABELS).
DEFAULT_LABELS="mixed_shapes,diffuse_pleural_location,local_pleural_location,pleural_calcification_location,occupational_disease"
# Train a subset of labels by setting ASBESTOSIS_LABELS; default is the stable label set.
LABELS="${ASBESTOSIS_LABELS:-${DEFAULT_LABELS}}"
PRIMARY_LABEL="${ASBESTOSIS_PRIMARY_LABEL:-mixed_shapes}"
# Print the same label set in the focus line by default (override with ASBESTOSIS_FOCUS_LABELS).
FOCUS_LABELS="${ASBESTOSIS_FOCUS_LABELS:-${DEFAULT_LABELS}}"
FOLD="${ASBESTOSIS_FOLD:-0}"
EPOCHS="${ASBESTOSIS_EPOCHS:-40}"
BATCH_SIZE="${ASBESTOSIS_BATCH_SIZE:-24}"
LR="${ASBESTOSIS_LR:-5e-5}"
NUM_WORKERS="${ASBESTOSIS_NUM_WORKERS:-8}"
EVAL_EVERY="${ASBESTOSIS_EVAL_EVERY:-1}"
TEST_EVERY="${ASBESTOSIS_TEST_EVERY:-0}"
MAX_TRAIN_STEPS="${ASBESTOSIS_MAX_TRAIN_STEPS:-}"
MAX_EVAL_STEPS="${ASBESTOSIS_MAX_EVAL_STEPS:-}"

# Overfitting controls / model options
HEAD_DROPOUT="${ASBESTOSIS_HEAD_DROPOUT:-0.2}"
WEIGHT_DECAY="${ASBESTOSIS_WEIGHT_DECAY:-1e-3}"
FREEZE_BACKBONE_EPOCHS="${ASBESTOSIS_FREEZE_BACKBONE_EPOCHS:-5}"
EARLY_STOP_PATIENCE="${ASBESTOSIS_EARLY_STOP_PATIENCE:-12}"
EARLY_STOP_METRIC="${ASBESTOSIS_EARLY_STOP_METRIC:-primary_auc/eval}"
EARLY_STOP_MIN_DELTA="${ASBESTOSIS_EARLY_STOP_MIN_DELTA:-0.0}"
BACKBONE_LR_MULT="${ASBESTOSIS_BACKBONE_LR_MULT:-0.1}"
MIN_POS_BACKBONE="${ASBESTOSIS_MIN_POS_BACKBONE:-20}"
MIN_NEG_BACKBONE="${ASBESTOSIS_MIN_NEG_BACKBONE:-20}"
HEAD_ONLY_LOSS_WEIGHT="${ASBESTOSIS_HEAD_ONLY_LOSS_WEIGHT:-0.25}"
TRAIN_SAMPLER="${ASBESTOSIS_TRAIN_SAMPLER:-primary_balanced}"
THRESHOLD_STRATEGY="${ASBESTOSIS_THRESHOLD_STRATEGY:-recall_at_precision}"
TARGET_PRECISION="${ASBESTOSIS_TARGET_PRECISION:-0.5}"
FBETA="${ASBESTOSIS_FBETA:-2.0}"
GROUP_SPLITS_BY="${ASBESTOSIS_GROUP_SPLITS_BY:-auto}"
LEAKAGE_ACTION="${ASBESTOSIS_LEAKAGE_ACTION:-error}"

# Feature toggles (set to 1 to enable)
NO_PRETRAINED="${ASBESTOSIS_NO_PRETRAINED:-0}"
NO_WANDB="${ASBESTOSIS_NO_WANDB:-0}"
WANDB_DETAIL="${ASBESTOSIS_WANDB_DETAIL:-compact}"
CHECK_MAPPING="${ASBESTOSIS_CHECK_MAPPING:-0}"
LABEL_STATS="${ASBESTOSIS_LABEL_STATS:-0}"
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
  "--split-label" "${SPLIT_LABEL}"
  "--labels" "${LABELS}"
  "--primary-label" "${PRIMARY_LABEL}"
  "--head-dropout" "${HEAD_DROPOUT}"
  "--fold" "${FOLD}"
  "--epochs" "${EPOCHS}"
  "--batch-size" "${BATCH_SIZE}"
  "--learning-rate" "${LR}"
  "--weight-decay" "${WEIGHT_DECAY}"
  "--backbone-lr-mult" "${BACKBONE_LR_MULT}"
  "--freeze-backbone-epochs" "${FREEZE_BACKBONE_EPOCHS}"
  "--min-pos-backbone" "${MIN_POS_BACKBONE}"
  "--min-neg-backbone" "${MIN_NEG_BACKBONE}"
  "--head-only-loss-weight" "${HEAD_ONLY_LOSS_WEIGHT}"
  "--train-sampler" "${TRAIN_SAMPLER}"
  "--threshold-strategy" "${THRESHOLD_STRATEGY}"
  "--target-precision" "${TARGET_PRECISION}"
  "--fbeta" "${FBETA}"
  "--group-splits-by" "${GROUP_SPLITS_BY}"
  "--leakage-action" "${LEAKAGE_ACTION}"
  "--num-workers" "${NUM_WORKERS}"
  "--seed" "${SEED}"
  "--wandb-detail" "${WANDB_DETAIL}"
)

if [[ -n "${FOCUS_LABELS}" ]]; then
  ARGS+=("--focus-labels" "${FOCUS_LABELS}")
fi

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
if [[ "${CHECK_MAPPING}" == "1" ]]; then
  ARGS+=("--check-mapping")
fi
if [[ "${LABEL_STATS}" == "1" ]]; then
  ARGS+=("--label-stats")
fi

echo "Command: srun python main.py ${ARGS[*]}"
srun python main.py "${ARGS[@]}"

echo "=== Job finished at $(date) ==="
