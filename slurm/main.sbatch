#!/usr/local_rwth/bin/zsh
#SBATCH --account=p0021834
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G

#SBATCH --partition c23g
#SBATCH --gres=gpu:1
#SBATCH --time "0-8:00:00"

#SBATCH --job-name model_cmp_mixed_shapes
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --array=0-30  # 3 models, 10 parameters -> 30 runs

### beginning of executable commands
module load GCCcore/13.2.0  
module load CUDA/12.8.0  #6.3 
module load Python/3.11.5

cd /rwthfs/rz/cluster/home/rwth1954/Asbestosis/
source .venv/bin/activate

all_models=("chexnet" "efficientnet_b0" "resnet18" "vit_b_16" "densenet121" "mobilenet_v3_small" "mobilenet_v3_large")
all_labels=("small_irregular_size" "mixed_shapes" "diffuse_pleural_thickening_width" "diffuse_pleural_thickening_extend" "diffuse_pleural_location" "localized_pleural_thickening_extend" "local_pleural_location" "pleural_calcification_location" "pleural_calcification_side" "occupational_disease")

MODEL=${all_models[$(($SLURM_ARRAY_TASK_ID / 10))]}
MAIN_LABEL=${all_labels[$(($SLURM_ARRAY_TASK_ID % 10))]}

exec > logs/train_test/${MODEL}_${MAIN_LABEL}.txt 2>&1

date
# models to test:
python main.py --base-folder "/hpcwork/rwth1954/Asbestosis_Data" \
 --fold "4" \
 --split-label "${MAIN_LABEL}" \
 --labels "small_irregular_size,mixed_shapes,diffuse_pleural_thickening_width,diffuse_pleural_thickening_extend,diffuse_pleural_location,localized_pleural_thickening_extend,local_pleural_location,pleural_calcification_location,pleural_calcification_side,occupational_disease" \
 --filter-image-quality
 --group-splits-by "patientID" \
 --epochs 60 --batch-size 32 --learning-rate 3e-4 \
 --model "$MODEL" \
 --head-dropout 0.5 --weight-decay 1e-2 --eval-every 0 \
 --combine-train-val
 --primary-balanced-max-pos-weight 10 \
 --task-weighting "equal" \
 --primary-task-weight 0 --fbeta 2.0 --target-precision 0.3 \
 --threshold-strategy "recall_at_precision" --n-bootstrap 500 \
 --num-workers 8 \
 --fold-folder "/rwthfs/rz/cluster/home/rwth1954/Asbestosis/splits" \
 --output-folder "/rwthfs/rz/cluster/home/rwth1954/Asbestosis/logs/train_test" \
 --backbone-lr-mult 0.1 --freeze-backbone-epochs 15 --min-pos-backbone 20 --min-neg-backbone 20 \
 --head-only-loss-weight 0.25 \
 --max-pos-weight 10.0 \
 --grad-clip 1.0 \
 --label-smoothing 0.0 \
 --swa-start-epoch 0 \
 --train-sampler "primary_balanced" \
 --leakage-action "error" \
 --seed 0 \
 --wandb-detail "compact" \
 --focus-labels mixed_shapes,occupational_disease 
date
