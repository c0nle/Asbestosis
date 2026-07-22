#!/usr/local_rwth/bin/zsh
#SBATCH --account=p0021834
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --partition c23g
#SBATCH --gres=gpu:1
#SBATCH --time "0-8:00:00"
#SBATCH --job-name cv_all_models
#SBATCH --output=logs/train_test/output/cv_%A_%a.txt
#SBATCH --array=0-59   # 2 models x 5 folds x 6 label = 60 jobs

### zsh arrays are 1-indexed — MODEL_IDX, FOLD and MAIN_LABEL derived from SLURM_ARRAY_TASK_ID
all_models=("vit_b_16" "densenet121")  ##RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x576 and 1024x1) for "mobilenet_v3_small" "mobilenet_v3_large")
all_labels=("mixed_shapes" "diffuse_pleural_location" "local_pleural_location" "pleural_calcification_location" "pleural_calcification_side" "occupational_disease")  # imbalanced: "diffuse_pleural_thickening_width" "diffuse_pleural_thickening_extend" "small_irregular_size" "localized_pleural_thickening_extend" 

MODEL_IDX=$(( $SLURM_ARRAY_TASK_ID / 30 + 1 ))   # 1..2
FOLD=$(( ($SLURM_ARRAY_TASK_ID / 6) % 5 ))      # 0..4
LABEL_IDX=$(( $SLURM_ARRAY_TASK_ID % 6 + 1 ))   # 1..6
MODEL=${all_models[$MODEL_IDX]}
MAIN_LABEL=${all_labels[$LABEL_IDX]}

echo "=== Array task $SLURM_ARRAY_TASK_ID  model=$MODEL  fold=$FOLD ==="
date

module load GCCcore/13.2.0
module load CUDA/12.8.0
module load Python/3.11.5

cd /rwthfs/rz/cluster/home/rwth1954/Asbestosis/
source .venv/bin/activate

# Backbone-specific settings (freeze epochs and LR differ per architecture)
case "$MODEL" in
    vit_b_16)
        FREEZE=10 ; HEAD_DROPOUT=0.4 ; LR=3e-5 ; WD=3e-3 ;;
    *)
        FREEZE=5  ; HEAD_DROPOUT=0.5 ; LR=3e-4 ; WD=1e-2 ;;
esac

python main.py \
    --base-folder        "/hpcwork/rwth1954/Asbestosis_Data" \
    --fold-folder        "/rwthfs/rz/cluster/home/rwth1954/Asbestosis/splits" \
    --output-folder      "/rwthfs/rz/cluster/home/rwth1954/Asbestosis/logs/train_test/models" \
    --model              "$MODEL" \
    --fold               "$FOLD" \
    --labels             "small_irregular_size,mixed_shapes,diffuse_pleural_thickening_width,diffuse_pleural_thickening_extend,diffuse_pleural_location,localized_pleural_thickening_extend,local_pleural_location,pleural_calcification_location,pleural_calcification_side,occupational_disease" \
    --primary-label      "${MAIN_LABEL}" \
    --split-label        "${MAIN_LABEL}" \
    --combine-train-val \
    --eval-every         0 \
    --epochs             60 \
    --batch-size         32 \
    --learning-rate      "$LR" \
    --weight-decay       "$WD" \
    --head-dropout       "$HEAD_DROPOUT" \
    --backbone-lr-mult   0.1 \
    --freeze-backbone-epochs "$FREEZE" \
    --min-pos-backbone   20 \
    --min-neg-backbone   20 \
    --head-only-loss-weight 0.25 \
    --max-pos-weight     10.0 \
    --grad-clip          1.0 \
    --label-smoothing    0.0 \
    --swa-start-epoch    0 \
    --train-sampler      "primary_balanced" \
    --primary-balanced-max-pos-weight 10 \
    --task-weighting     "equal" \
    --primary-task-weight 0 \
    --threshold-strategy "recall_at_precision" \
    --target-precision   0.3 \
    --fbeta              2.0 \
    --n-bootstrap        500 \
    --group-splits-by    "patientID" \
    --leakage-action     "error" \
    --filter-image-quality \
    --num-workers        8 \
    --seed               0 \
    --focus-labels       "mixed_shapes,occupational_disease" \
    --wandb-detail       "compact"

date
echo "=== Done ==="
