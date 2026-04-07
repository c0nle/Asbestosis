# Asbestosis Chest X-Ray Classification

Multi-task binary classification of ILO asbestosis findings on chest X-rays.
A shared DenseNet121 backbone (CheXNet, pretrained on 100k+ chest X-rays) feeds
one binary head per label, trained with BCEWithLogitsLoss and mixed-precision AMP.

## Labels

Two clinically validated labels are trained simultaneously:

| Label | Description | Prevalence |
|---|---|---|
| `mixed_shapes` | Parenchymal opacities (rounded + irregular) | ~41% |
| `occupational_disease` | Occupational disease recognised | ~7% |

Both labels come from the dichotome metadata CSV. The multi-task setup allows
the backbone to share information across tasks, which regularises training and
improves generalisation on the rarer label.

## Model

**CheXNet** — DenseNet121 pretrained on the ChestX-ray14 dataset (Wang et al. 2017)
via the torchxrayvision library. The pretrained weights are extracted once into a
self-contained state dict (`chexnet_features_state_dict.pt`) so no torchxrayvision
dependency is needed at training time.

Input: single-channel (grayscale) 224×224. Normalization maps `[0, 1] → [−1024, 1024]`
to match the original CheXNet training range.

## Project Structure

```
Asbestosis/
├── main.py                          # Training entry point
├── model.py                         # Model builders (CheXNet, CNNs, ViT)
├── training.py                      # Train/eval loop, metrics, W&B logging
├── dataset.py                       # XRayMultiTaskDataset
├── ensemble.py                      # K-fold ensemble evaluation
├── utils.py                         # Shared helpers (splits, leakage check, …)
├── Preprocessor_Metadata.py         # Metadata loading & split generation
├── run_asbestosis.sh                # SLURM launcher (all hyperparameters)
├── chexnet_features_state_dict.pt   # Pre-extracted CheXNet weights (see below)
├── splits/                          # Generated stratified fold CSVs
└── logs/                            # Checkpoints and SLURM output files
```

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision scikit-learn pandas wandb

# 2. Extract CheXNet weights (run once on any node with torchxrayvision)
pip install torchxrayvision
python - <<'PY'
import torch, torchxrayvision as xrv
m = xrv.models.DenseNet(weights="densenet121-res224-all")
sd = {k: v for k, v in list(m.named_parameters()) + list(m.named_buffers())}
torch.save(sd, "chexnet_features_state_dict.pt")
print("Saved", len(sd), "keys")
PY
# torchxrayvision is not needed after this step.
```

## Training a Single Fold

```bash
# Local / interactive
python main.py \
    --base-folder /path/to/Asbestosis_Data \
    --fold 0 \
    --model chexnet \
    --labels mixed_shapes,occupational_disease

# SLURM (all hyperparameters live in run_asbestosis.sh)
sbatch --export=ALL,ASBESTOSIS_FOLD=0 run_asbestosis.sh
```

## K-Fold Training (5 Folds in Parallel)

```bash
for fold in 0 1 2 3 4; do
    sbatch --export=ALL,ASBESTOSIS_FOLD=$fold run_asbestosis.sh
done
```

Wait for all jobs to finish (`squeue -u <user>`), then run the ensemble evaluation.

**Note on early stopping and backbone freezing:** early-stop patience only starts
counting after the backbone unfreeze epoch. During the frozen phase the model
cannot improve at its full capacity, so plateaus there should not trigger
stopping.

## Ensemble Evaluation

Loads all five fold checkpoints, runs inference over each fold's held-out test
split using all available models, and averages predicted probabilities. Reports
per-task and macro-averaged AUC, and logs ROC/PR curves + bootstrap CI to W&B.

```bash
source .venv/bin/activate
python ensemble.py \
    --base-folder /hpcwork/rwth1954/Asbestosis_Data \
    --output-folder /rwthfs/rz/cluster/home/rwth1954/Asbestosis/logs
```

Results are logged as a single W&B run named `ensemble_chexnet_5folds`.

## Key Hyperparameters

All hyperparameters can be overridden via `ASBESTOSIS_*` environment variables.
The defaults in `run_asbestosis.sh` are tuned for CheXNet on this dataset.

| Parameter | Default | Notes |
|---|---|---|
| `EPOCHS` | 60 | Folds converge around epoch 30–50 |
| `LR` | 3e-4 | Head LR; backbone gets `LR × backbone_lr_mult` |
| `FREEZE_BACKBONE_EPOCHS` | 15 | Heads trained alone first |
| `BACKBONE_LR_MULT` | 0.1 | Backbone fine-tuned at 10× lower LR |
| `EARLY_STOP_PATIENCE` | 10 | Stop if macro_auc/eval doesn't improve |
| `SWA_START_EPOCH` | 0 | Set to `EPOCHS - 5` to enable SWA |
| `N_BOOTSTRAP` | 500 | Bootstrap resamples for AUC CI at test time |
| `TRAIN_SAMPLER` | primary_balanced | Oversamples positives for mixed_shapes |
| `THRESHOLD_STRATEGY` | recall_at_precision | Threshold chosen to satisfy precision ≥ 0.3 |

## W&B Logging

Each training run logs to the `Asbestosis` W&B project. Key metrics:

- `macro_auc/eval` — main early-stopping signal
- `task/<label>/auc/<split>` — per-label AUC on train/eval/test
- `roc_curve/final_test/<label>` — ROC curve with random-guess diagonal
- `pr_curve/final_test/<label>` — PR curve with prevalence baseline
- `auc_bootstrap/final_test/<label>` — histogram of 500 bootstrap AUC samples
- `task/<label>/auc_ci_lo/final_test` — 95% CI lower bound

Set `ASBESTOSIS_NO_WANDB=1` to disable W&B logging.

## Results (CheXNet, 5-fold CV, 60 epochs, no SWA)

### Per-fold — `best_test` AUC (best checkpoint evaluated on held-out test split)

| Fold | macro_auc | mixed_shapes | occupational_disease | Best epoch |
|---|---|---|---|---|
| 0 | 0.761 | 0.664 | 0.858 | 57 |
| 1 | **0.774** | 0.649 | **0.898** | 40 |
| 2 | 0.763 | 0.638 | 0.889 | 40 |
| 3 | 0.772 | **0.671** | 0.874 | 57 |
| 4 | 0.752 | 0.606 | **0.898** | 43 |
| **Mean ± std** | **0.764 ± 0.008** | **0.646 ± 0.024** | **0.883 ± 0.018** | — |

### Ensemble — all 5 models, full dataset (n=3 163 / 3 123)

Probabilities from all five fold-models are averaged before computing metrics.

| | mixed_shapes | occupational_disease | macro |
|---|---|---|---|
| **Ensemble AUC** | 0.659 | **0.901** | **0.780** |
| **95% CI (bootstrap n=500)** | [0.638, 0.679] | [0.881, 0.919] | — |
| **Single-model AUC** | 0.639 | 0.878 | 0.758 |
| **Ensemble gain Δ** | +0.020 | +0.023 | **+0.022** |

The ensemble brings a consistent +0.022 macro AUC gain over single-fold models.
Full ROC/PR curves and the bootstrap AUC histogram are logged in the
`ensemble_chexnet_5folds` W&B run.

## Data

Expected layout under `--base-folder`:

```
Asbestosis_Data/
├── dichotome_data_anonymized_with_patientID.csv   # metadata with labels
├── mapping.csv                                     # medicoID → fileID
└── <image files>                                   # resolved via fileID
```

Splits are generated automatically on first run (grouped by `patientID` to
prevent the same patient from appearing in both train and test) and cached in
`--fold-folder` (default: `splits/`).
