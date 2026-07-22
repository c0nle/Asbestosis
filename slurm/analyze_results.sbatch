#!/usr/local_rwth/bin/zsh
#SBATCH --account=p0021834
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G

#SBATCH --partition c23g
#SBATCH --gres=gpu:1
#SBATCH --time "00:50:00"

#SBATCH --job-name analyze_results
#SBATCH --output=logs/out_%A_%a.txt

### beginning of executable commands
module load GCCcore/13.2.0  
module load CUDA/12.8.0  #6.3 
module load Python/3.11.5

cd /rwthfs/rz/cluster/home/rwth1954/Asbestosis/
source .venv/bin/activate


date
python analyze_results.py \
  --models chexnet resnet18 efficientnet_b0 vit_b_16 densenet121 \
  --base-folder /hpcwork/rwth1954/Asbestosis_Data \
  --fold-folder /rwthfs/rz/cluster/home/rwth1954/Asbestosis/splits \
  --ckpt-folder /rwthfs/rz/cluster/home/rwth1954/Asbestosis/logs/train_test/models_06-18 \
  --output-folder ./logs/train_test/analysis_06-18/07-13-slurm \
  --n-bootstrap 2000 \
  --n-folds 5
date
