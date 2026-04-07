#!/usr/bin/env bash
#SBATCH --account=rwth1954
#SBATCH --job-name=asbestosis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition c23g
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G

set -euo pipefail

echo "=== Job started on $(hostname) at $(date) ==="
echo "PWD: $(pwd)"

mkdir -p logs

# ---------------------------------------------------------------------------
# Paths — override via environment variables before calling sbatch.
# ---------------------------------------------------------------------------
PROJECT_DIR="/rwthfs/rz/cluster/home/rwth1954/Asbestosis"
VENV_DIR="${PROJECT_DIR}/.venv"
DATA_DIR="${ASBESTOSIS_DATA_DIR:-/hpcwork/rwth1954/Asbestosis_Data}"
FOLD_DIR="${ASBESTOSIS_FOLD_DIR:-${PROJECT_DIR}/splits}"
OUT_DIR="${ASBESTOSIS_OUT_DIR:-${PROJECT_DIR}/logs}"

# ---------------------------------------------------------------------------
# Hyperparameters — all overridable via ASBESTOSIS_* environment variables.
# Example: ASBESTOSIS_FOLD=2 sbatch --export=ALL,ASBESTOSIS_FOLD=2 run_asbestosis.sh
# ---------------------------------------------------------------------------

# Backbone: chexnet = DenseNet121 pretrained on 100k+ chest X-rays.
# Requires chexnet_features_state_dict.pt in the project directory.
MODEL="${ASBESTOSIS_MODEL:-chexnet}"

# Labels to classify (comma-separated column names from the metadata CSV).
DEFAULT_LABELS="mixed_shapes,occupational_disease"
LABELS="${ASBESTOSIS_LABELS:-${DEFAULT_LABELS}}"
PRIMARY_LABEL="${ASBESTOSIS_PRIMARY_LABEL:-mixed_shapes}"
FOCUS_LABELS="${ASBESTOSIS_FOCUS_LABELS:-${DEFAULT_LABELS}}"

# K-fold index (0-based). Pass different values via --export to train all folds:
#   for fold in 0 1 2 3 4; do sbatch --export=ALL,ASBESTOSIS_FOLD=$fold run_asbestosis.sh; done
FOLD="${ASBESTOSIS_FOLD:-0}"

EPOCHS="${ASBESTOSIS_EPOCHS:-60}"
BATCH_SIZE="${ASBESTOSIS_BATCH_SIZE:-32}"

# Learning rate for the classification heads. The backbone receives LR * backbone_lr_mult.
# CheXNet heads need 3e-4 to converge; the backbone is protected at 3e-5 (mult=0.1).
if [[ -z "${ASBESTOSIS_LR+x}" ]]; then
    if [[ "${MODEL}" == "vit_b_16" ]]; then
        LR="3e-5"
    else
        LR="3e-4"
    fi
else
    LR="${ASBESTOSIS_LR}"
fi

# Head dropout: slightly lower for ViT (already regularised by attention dropout).
if [[ -z "${ASBESTOSIS_HEAD_DROPOUT+x}" ]]; then
    if [[ "${MODEL}" == "vit_b_16" ]]; then
        HEAD_DROPOUT="0.4"
    else
        HEAD_DROPOUT="0.5"
    fi
else
    HEAD_DROPOUT="${ASBESTOSIS_HEAD_DROPOUT}"
fi

# Weight decay: higher for ViT which has more parameters.
if [[ -z "${ASBESTOSIS_WEIGHT_DECAY+x}" ]]; then
    if [[ "${MODEL}" == "vit_b_16" ]]; then
        WEIGHT_DECAY="3e-3"
    else
        WEIGHT_DECAY="1e-2"
    fi
else
    WEIGHT_DECAY="${ASBESTOSIS_WEIGHT_DECAY}"
fi

# Freeze the backbone for N epochs so the randomly-initialised heads converge first.
# CheXNet needs longer because the chest-X-ray features are already good —
# the heads need time to adapt before the backbone is fine-tuned.
if [[ -z "${ASBESTOSIS_FREEZE_BACKBONE_EPOCHS+x}" ]]; then
    if [[ "${MODEL}" == "chexnet" ]]; then
        FREEZE_BACKBONE_EPOCHS="15"
    elif [[ "${MODEL}" == "vit_b_16" ]]; then
        FREEZE_BACKBONE_EPOCHS="10"
    else
        FREEZE_BACKBONE_EPOCHS="5"
    fi
else
    FREEZE_BACKBONE_EPOCHS="${ASBESTOSIS_FREEZE_BACKBONE_EPOCHS}"
fi

SPLIT_LABEL="${ASBESTOSIS_SPLIT_LABEL:-mixed_shapes}"
NUM_WORKERS="${ASBESTOSIS_NUM_WORKERS:-8}"
EVAL_EVERY="${ASBESTOSIS_EVAL_EVERY:-1}"

EARLY_STOP_PATIENCE="${ASBESTOSIS_EARLY_STOP_PATIENCE:-10}"
EARLY_STOP_METRIC="${ASBESTOSIS_EARLY_STOP_METRIC:-macro_auc/eval}"
EARLY_STOP_MIN_DELTA="${ASBESTOSIS_EARLY_STOP_MIN_DELTA:-0.0}"

BACKBONE_LR_MULT="${ASBESTOSIS_BACKBONE_LR_MULT:-0.1}"
MIN_POS_BACKBONE="${ASBESTOSIS_MIN_POS_BACKBONE:-20}"
MIN_NEG_BACKBONE="${ASBESTOSIS_MIN_NEG_BACKBONE:-20}"
HEAD_ONLY_LOSS_WEIGHT="${ASBESTOSIS_HEAD_ONLY_LOSS_WEIGHT:-0.25}"
MAX_POS_WEIGHT="${ASBESTOSIS_MAX_POS_WEIGHT:-10.0}"
GRAD_CLIP="${ASBESTOSIS_GRAD_CLIP:-1.0}"
LABEL_SMOOTHING="${ASBESTOSIS_LABEL_SMOOTHING:-0.0}"

# SWA (Stochastic Weight Averaging): set to 0 to disable.
# If enabled, use a late start (e.g. last 5 epochs) so only converged weights are averaged.
SWA_START_EPOCH="${ASBESTOSIS_SWA_START_EPOCH:-0}"

# Bootstrap resamples for AUC confidence intervals on the test set (0 = disabled).
N_BOOTSTRAP="${ASBESTOSIS_N_BOOTSTRAP:-500}"

TRAIN_SAMPLER="${ASBESTOSIS_TRAIN_SAMPLER:-primary_balanced}"
PRIMARY_BAL_MAX_POS_WEIGHT="${ASBESTOSIS_PRIMARY_BAL_MAX_POS_WEIGHT:-10}"
TASK_WEIGHTING="${ASBESTOSIS_TASK_WEIGHTING:-equal}"
PRIMARY_TASK_WEIGHT="${ASBESTOSIS_PRIMARY_TASK_WEIGHT:-0}"
THRESHOLD_STRATEGY="${ASBESTOSIS_THRESHOLD_STRATEGY:-recall_at_precision}"
FBETA="${ASBESTOSIS_FBETA:-2.0}"
TARGET_PRECISION="${ASBESTOSIS_TARGET_PRECISION:-0.3}"
GROUP_SPLITS_BY="${ASBESTOSIS_GROUP_SPLITS_BY:-patientID}"
LEAKAGE_ACTION="${ASBESTOSIS_LEAKAGE_ACTION:-error}"
REGENERATE_SPLITS="${ASBESTOSIS_REGENERATE_SPLITS:-0}"

NO_PRETRAINED="${ASBESTOSIS_NO_PRETRAINED:-0}"
NO_WANDB="${ASBESTOSIS_NO_WANDB:-0}"
WANDB_DETAIL="${ASBESTOSIS_WANDB_DETAIL:-compact}"
CHECK_MAPPING="${ASBESTOSIS_CHECK_MAPPING:-0}"
LABEL_STATS="${ASBESTOSIS_LABEL_STATS:-0}"
SEED="${ASBESTOSIS_SEED:-0}"

MAX_TRAIN_STEPS="${ASBESTOSIS_MAX_TRAIN_STEPS:-}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT=true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=== Environment ==="
nvidia-smi || true

source "${VENV_DIR}/bin/activate"

python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
PY

echo "=== Starting training ==="
mkdir -p "${FOLD_DIR}" "${OUT_DIR}"

ARGS=(
    "--base-folder"               "${DATA_DIR}"
    "--fold-folder"               "${FOLD_DIR}"
    "--output-folder"             "${OUT_DIR}"
    "--split-label"               "${SPLIT_LABEL}"
    "--model"                     "${MODEL}"
    "--labels"                    "${LABELS}"
    "--primary-label"             "${PRIMARY_LABEL}"
    "--head-dropout"              "${HEAD_DROPOUT}"
    "--fold"                      "${FOLD}"
    "--epochs"                    "${EPOCHS}"
    "--batch-size"                "${BATCH_SIZE}"
    "--learning-rate"             "${LR}"
    "--weight-decay"              "${WEIGHT_DECAY}"
    "--backbone-lr-mult"          "${BACKBONE_LR_MULT}"
    "--freeze-backbone-epochs"    "${FREEZE_BACKBONE_EPOCHS}"
    "--min-pos-backbone"          "${MIN_POS_BACKBONE}"
    "--min-neg-backbone"          "${MIN_NEG_BACKBONE}"
    "--head-only-loss-weight"     "${HEAD_ONLY_LOSS_WEIGHT}"
    "--max-pos-weight"            "${MAX_POS_WEIGHT}"
    "--grad-clip"                 "${GRAD_CLIP}"
    "--label-smoothing"           "${LABEL_SMOOTHING}"
    "--swa-start-epoch"           "${SWA_START_EPOCH}"
    "--n-bootstrap"               "${N_BOOTSTRAP}"
    "--train-sampler"             "${TRAIN_SAMPLER}"
    "--primary-balanced-max-pos-weight" "${PRIMARY_BAL_MAX_POS_WEIGHT}"
    "--task-weighting"            "${TASK_WEIGHTING}"
    "--primary-task-weight"       "${PRIMARY_TASK_WEIGHT}"
    "--threshold-strategy"        "${THRESHOLD_STRATEGY}"
    "--fbeta"                     "${FBETA}"
    "--target-precision"          "${TARGET_PRECISION}"
    "--group-splits-by"           "${GROUP_SPLITS_BY}"
    "--leakage-action"            "${LEAKAGE_ACTION}"
    "--num-workers"               "${NUM_WORKERS}"
    "--seed"                      "${SEED}"
    "--wandb-detail"              "${WANDB_DETAIL}"
    "--eval-every"                "${EVAL_EVERY}"
    "--early-stop-patience"       "${EARLY_STOP_PATIENCE}"
    "--early-stop-metric"         "${EARLY_STOP_METRIC}"
    "--early-stop-min-delta"      "${EARLY_STOP_MIN_DELTA}"
)

[[ -n "${FOCUS_LABELS}" ]]    && ARGS+=("--focus-labels"    "${FOCUS_LABELS}")
[[ -n "${MAX_TRAIN_STEPS}" ]] && ARGS+=("--max-train-steps" "${MAX_TRAIN_STEPS}")
[[ "${NO_PRETRAINED}"    == "1" ]] && ARGS+=("--no-pretrained")
[[ "${NO_WANDB}"         == "1" ]] && ARGS+=("--no-wandb")
[[ "${CHECK_MAPPING}"    == "1" ]] && ARGS+=("--check-mapping")
[[ "${LABEL_STATS}"      == "1" ]] && ARGS+=("--label-stats")
[[ "${REGENERATE_SPLITS}" == "1" ]] && ARGS+=("--regenerate-splits")

echo "Command: srun python main.py ${ARGS[*]}"
srun python main.py "${ARGS[@]}"

echo "=== Job finished at $(date) ==="
