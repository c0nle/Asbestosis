"""
main.py
-------
Multi-task chest X-ray classification pipeline for asbestos-related disease
detection.  Trains a shared-backbone CNN or ViT with one binary head per ILO
label using BCEWithLogitsLoss and mixed-precision training.

Workflow
--------
1. Parse CLI arguments.
2. Load (or generate) stratified K-fold splits from the dichotome metadata CSV.
   Splits are grouped by ``patientID`` to prevent the same patient from
   appearing in both train and test.
3. Build train / val / test datasets and data loaders.
4. Construct the multi-task model and AdamW optimizer.
5. Run training with optional backbone freezing, weighted sampling, and
   task-selective backbone gradients.
6. Evaluate on the validation set every ``--eval-every`` epochs.
7. Save the best checkpoint (tracked by ``--early-stop-metric``) and evaluate
   on the test set at the end.
8. Log all metrics to W&B and save the final model weights.
"""

import argparse
import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn, optim
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    Resize,
    ToDtype,
    ToImage,
)

import Preprocessor_Metadata
from dataset import (
    XRayMultiTaskDataset,
    _binary_value_map_from_series,
    _filter_rows_with_images,
    _find_xray_path,
)
from model import MultiTaskModel, _build_multitask_model, _split_head_backbone_params
from training import (
    _print_focus_confusions,
    _print_focus_line,
    log_multitask_metrics,
    run_epoch,
)
from utils import (
    _canonical_label_value,
    _choose_group_col,
    _ensure_file_id,
    _ensure_writable_dir,
    _is_missing_label_value,
    _leakage_check,
    _missing_mask,
    _parse_csv_arg,
    _set_seed,
    _wandb_is_active,
    print_label_stats,
)


def check_mapping_merge(
    base_folder: str,
    image_dir: str,
    metadata_file: str,
    mapping_file: str,
    sample_n: int = 20,
    seed: int = 0,
) -> None:
    """
    Print diagnostics for the metadata ↔ mapping.csv merge.

    Shows how many rows matched, and spot-checks whether the resolved image
    files actually exist on disk.

    Args:
        base_folder:   Root data folder (for context in log output).
        image_dir:     Directory to search for image files.
        metadata_file: Path to the metadata CSV.
        mapping_file:  Path to the medicoID → fileID mapping CSV.
        sample_n:      Number of matched rows to check on disk.
        seed:          Random seed for row sampling.
    """
    random.seed(seed)
    np.random.seed(seed)
    metadata = pd.read_csv(metadata_file)
    if "Anforderungsnummer" not in metadata.columns:
        raise RuntimeError(f"Cannot check mapping: missing column 'Anforderungsnummer' in {metadata_file}")

    merged = _ensure_file_id(metadata, mapping_file)
    total = len(merged)
    matched = int((merged["fileID"] != -1).sum()) if "fileID" in merged.columns else 0
    print(f"=== Mapping check ===")
    print(f"metadata_file: {metadata_file}")
    print(f"mapping_file:  {mapping_file}")
    print(f"image_dir:     {image_dir}")
    print(f"rows:          {total}")
    print(f"matched fileID: {matched} ({(matched / max(1, total)) * 100:.1f}%)")

    sample = (
        merged[merged["fileID"] != -1].sample(n=min(sample_n, matched), random_state=seed)
        if matched else merged.head(0)
    )
    missing_paths = 0
    checked = 0
    print("Sample merged rows (Anforderungsnummer, fileID, exists, path):")
    for _, row in sample.iterrows():
        file_id = str(int(row["fileID"]))
        path = _find_xray_path(image_dir, file_id)
        exists = os.path.exists(path)
        checked += 1
        missing_paths += 0 if exists else 1
        print(f"- {int(row['Anforderungsnummer'])} -> {file_id} | exists={exists} | {path}")
    if checked:
        print(f"Sample missing files: {missing_paths}/{checked}")


def main() -> None:
    """
    Entry point for the asbestosis multi-task training pipeline.

    Parses CLI arguments (or environment variable overrides), sets up the
    data, model, optimizer, and runs the training loop.  See module docstring
    for the high-level workflow and ``run_asbestosis.sh`` for the SLURM
    launcher with all configurable hyperparameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-folder",
        default=os.environ.get("ASBESTOSIS_BASE_FOLDER", "/hpcwork/rwth1954/Asbestosis_Data"),
        help="Root folder containing mapping.csv, metadata CSVs and image subfolders.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Folder containing images (or a folder with `png/` and/or `anon/` subfolders). Defaults to --base-folder.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--split-label",
        default="mixed_shapes",
        help="Label column used only for generating stratified fold splits (training is always multi-task).",
    )
    parser.add_argument(
        "--labels",
        default="all",
        help="Comma-separated label columns, or 'all' to train on all label columns (excluding general metadata columns).",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--model",
        choices=["vit_b_16", "resnet18", "efficientnet_b0", "densenet121", "chexnet", "mobilenet_v3_small", "mobilenet_v3_large"],
        default="chexnet",
        help=(
            "Backbone architecture. "
            "'chexnet' uses DenseNet121 pretrained on 100k+ chest X-rays — "
            "best choice for this task. 'densenet121' uses ImageNet weights only."
        ),
    )
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="Cap training steps per epoch (useful for quick smoke tests).")
    parser.add_argument("--eval-every", type=int, default=1, help="Run eval every N epochs (set 0 to disable periodic eval).")
    parser.add_argument("--test-every", type=int, default=0, help="Run test every N epochs (set 0 to disable periodic test).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--head-dropout", type=float, default=0.1, help="Dropout probability for the classifier heads.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1, help="Multiplier for backbone LR vs head LR after unfreezing.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--early-stop-metric",
        choices=[
            "primary_auc/eval",
            "primary_pr_auc/eval",
            "primary_f1/eval",
            "macro_auc/eval",
            "macro_pr_auc/eval",
            "macro_f1/eval",
            "loss/eval",
        ],
        default="macro_auc/eval",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--primary-label",
        default="mixed_shapes",
        help="Primary label to optimize/early-stop on (must be included in --labels).",
    )
    parser.add_argument("--train-sampler", choices=["random", "primary_balanced"], default="primary_balanced")
    parser.add_argument(
        "--primary-balanced-max-pos-weight",
        type=float,
        default=10.0,
        help="Cap for the positive sampling weight used by --train-sampler primary_balanced.",
    )
    parser.add_argument("--task-weighting", choices=["equal", "present", "sqrt_present"], default="equal")
    parser.add_argument(
        "--primary-task-weight",
        type=float,
        default=1.0,
        help="Extra multiplier applied to the primary task in the multi-task loss. Set <=0 to auto-tune from data.",
    )
    parser.add_argument("--threshold-strategy", choices=["f1", "fbeta", "recall_at_precision"], default="recall_at_precision")
    parser.add_argument("--fbeta", type=float, default=2.0)
    parser.add_argument("--target-precision", type=float, default=0.5)
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=10.0,
        help="Cap for BCE pos_weight across all tasks (0 = no cap). Prevents gradient spikes from rare classes.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 = disabled). Prevents large post-unfreeze updates.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing epsilon applied to binary targets during training (0 = disabled).",
    )
    parser.add_argument(
        "--swa-start-epoch",
        type=int,
        default=0,
        help=(
            "Epoch to start Stochastic Weight Averaging (0 = disabled). "
            "E.g. 14 to average checkpoints from epoch 14 onward. "
            "After training, BN stats are updated and the averaged model is "
            "used for the final_test evaluation."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=0,
        help=(
            "Number of bootstrap resamples for AUC confidence intervals on the test set "
            "(0 = disabled). Logged as a W&B histogram when active. E.g. 500."
        ),
    )
    parser.add_argument(
        "--min-pos-backbone",
        type=int,
        default=5,
        help="Minimum positive samples in train split for a task to influence the backbone.",
    )
    parser.add_argument(
        "--min-neg-backbone",
        type=int,
        default=5,
        help="Minimum negative samples in train split for a task to influence the backbone.",
    )
    parser.add_argument(
        "--head-only-loss-weight",
        type=float,
        default=0.25,
        help="Loss multiplier for tasks that do not influence the backbone (still trains heads).",
    )
    parser.add_argument(
        "--focus-labels",
        default="mixed_shapes,occupational_disease",
        help="Comma-separated labels to print as a compact per-epoch summary line (use 'none' to disable).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-mapping", action="store_true", help="Print diagnostics for metadata<->mapping merge and exit.")
    parser.add_argument("--check-mapping-samples", type=int, default=20)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet pretrained weights (avoids network downloads).",
    )
    parser.add_argument(
        "--fold-folder",
        default=None,
        help="Where to write stratified split CSVs (defaults to <base-folder>/strat_dichotom_splits).",
    )
    parser.add_argument(
        "--group-splits-by",
        default="patientID",
        help=(
            "Column used to group patients when generating splits, preventing the same "
            "patient from appearing in both train and test.  The anonymised CSV "
            "provides 'patientID' as the first column for this purpose.  "
            "Use 'none' to disable grouping, or 'auto' to try a priority list "
            "(patientID → medicoID → Anforderungsnummer → …)."
        ),
    )
    parser.add_argument(
        "--regenerate-splits",
        action="store_true",
        help="Force regeneration of the stratified folds CSV even if it already exists in --fold-folder.",
    )
    parser.add_argument(
        "--leakage-action",
        choices=["error", "warn", "ignore"],
        default="error",
        help="What to do if group IDs overlap between train/val/test within a fold.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Where to save the trained model (defaults to <base-folder>).",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-detail", choices=["compact", "full"], default="compact")
    parser.add_argument(
        "--label-stats",
        action="store_true",
        help="Print per-label dataset counts and exit.",
    )
    parser.add_argument(
        "--filter-image-quality",
        action="store_true",
        help="Exclude rows where technical_quality == 0 from training/validation/test datasets.",
    )
    parser.add_argument(
        "--combine-train-val",
        action="store_true",
        help="Merge train and val splits into a single training set, disabling validation and early stopping.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # CheXNet (torchxrayvision) was trained with pixel values in [-1024, 1024].
    # After ToDtype(float32, scale=True) images are in [0, 1].
    # Normalize(mean=0.5, std=1/2048) maps [0,1] → [-1024, 1024] exactly.
    # All other models use standard [-1, 1] normalization (mean=0.5, std=0.5).
    if str(args.model) == "chexnet":
        _norm = Normalize(mean=[0.5], std=[1.0 / 2048.0])
    else:
        _norm = Normalize(mean=[0.5], std=[0.5])

    preprocess = Compose(
        [
            # Scale jitter keeps most of the image in view (lower/upper lung zones preserved).
            RandomResizedCrop(224, scale=(0.85, 1.0)),
            # Horizontal flip is safe: labels encode presence/severity, not strict laterality.
            RandomHorizontalFlip(p=0.5),
            RandomRotation(10),
            ColorJitter(brightness=0.2, contrast=0.2),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            _norm,
        ]
    )
    preprocess_no_aug = Compose(
        [
            Resize((224, 224)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            _norm,
        ]
    )

    base_folder = args.base_folder
    root_folder = args.image_dir or base_folder
    mapping_file = os.path.join(base_folder, "mapping.csv")
    default_fold_folder = os.path.join(base_folder, "strat_dichotom_splits")
    fold_folder = args.fold_folder or default_fold_folder
    output_folder = args.output_folder or base_folder

    # Anonymised metadata CSV with patientID column for leak-free splitting.
    metadata_file = os.path.join(base_folder, "dichotome_data_anonymized_with_patientID.csv")

    missing = [p for p in [metadata_file, mapping_file] if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "Missing required data files:\n- "
            + "\n- ".join(missing)
            + "\nSet `--base-folder` (or `ASBESTOSIS_BASE_FOLDER`) to the correct dataset root."
        )

    if args.fold_folder is None:
        fallback_fold_folder = os.path.join(os.getcwd(), "splits")
        fold_folder = _ensure_writable_dir(fold_folder, fallback=fallback_fold_folder)
        if fold_folder != default_fold_folder:
            print(f"Warning: cannot write to {default_fold_folder}; using {fold_folder} instead.")
    else:
        fold_folder = _ensure_writable_dir(fold_folder, fallback=None)
    os.makedirs(output_folder, exist_ok=True)

    if args.check_mapping:
        check_mapping_merge(
            base_folder=base_folder,
            image_dir=root_folder,
            metadata_file=metadata_file,
            mapping_file=mapping_file,
            sample_n=args.check_mapping_samples,
            seed=args.seed,
        )
        raise SystemExit(0)

    if args.label_stats:
        md = pd.read_csv(metadata_file)
        md = _ensure_file_id(md, mapping_file)
        md = md[md["fileID"] != -1].reset_index(drop=True)
        column_groups = Preprocessor_Metadata.get_column_name_groups(md, True)
        general_cols = set(column_groups.get("general", []))
        if str(args.labels).strip().lower() == "all":
            label_cols = []
            for group_name, cols in column_groups.items():
                if group_name == "general":
                    continue
                label_cols.extend([c for c in cols if c not in general_cols])
            label_cols = sorted(set(label_cols))
        else:
            label_cols = [c.strip() for c in str(args.labels).split(",") if c.strip()]
        print_label_stats(md, label_cols)
        raise SystemExit(0)

    # --- Splits ---
    fold_splitted_metadata_filename = os.path.join(
        fold_folder,
        os.path.basename(metadata_file).replace(".csv", "_stratified_folds.csv"),
    )
    n_folds = 5
    needs_splits = True
    if os.path.isfile(fold_splitted_metadata_filename):
        metadata = pd.read_csv(fold_splitted_metadata_filename)
        fold_cols = [c for c in (f"Fold{i}" for i in range(n_folds)) if c in metadata.columns]
        has_val = any((metadata[c] == "val").any() for c in fold_cols)
        if has_val and not bool(args.regenerate_splits):
            needs_splits = False
        else:
            reason = "forced by --regenerate-splits" if bool(args.regenerate_splits) else "no 'val' split"
            print(f"Regenerating split file ({reason}): {fold_splitted_metadata_filename}")

    if needs_splits:
        metadata = pd.read_csv(metadata_file)
        metadata = _ensure_file_id(metadata, mapping_file)
        metadata = metadata[metadata["fileID"] != -1]
        if args.split_label not in metadata.columns:
            raise SystemExit(f"Split label '{args.split_label}' not found in metadata.")
        group_col = _choose_group_col(
            metadata,
            requested=str(getattr(args, "group_splits_by", "auto")),
            n_folds=n_folds,
        )
        if group_col is None:
            raise SystemExit(
                f"Cannot generate grouped splits: no suitable grouping column found for "
                f"--group-splits-by={args.group_splits_by!r}. "
                "Try --group-splits-by Anforderungsnummer."
            )
        print(f"Split grouping column: {group_col}")
        metadata = Preprocessor_Metadata.create_splits(
            metadata,
            n_folds,
            fold_folder,
            fold_splitted_metadata_filename,
            args.split_label,
            group_col=group_col,
            strict_group_col=True,
            seed_base=int(args.seed),
        )
    else:
        metadata = _ensure_file_id(metadata, mapping_file)
        metadata = metadata[metadata["fileID"] != -1]

    prepared_metadata = metadata.reset_index(drop=True)

    # Derive presence labels on-the-fly from existing columns (no new CSV column needed).
    # NaN in these columns means "not present"; any non-NaN value means "present".
    _presence_derivations = {
        "diffuse_pleural_thickening_presence": "diffuse_pleural_thickening_width",
        "localized_pleural_thickening_presence": "localized_pleural_thickening_width",
    }
    for derived_col, source_col in _presence_derivations.items():
        if source_col in prepared_metadata.columns and derived_col not in prepared_metadata.columns:
            prepared_metadata[derived_col] = prepared_metadata[source_col].notna().astype(int)

    column_groups = Preprocessor_Metadata.get_column_name_groups(prepared_metadata, True)
    general_cols = set(column_groups.get("general", []))

    # --- Labels (always multi-task) ---
    labels_arg = str(args.labels).strip()
    if labels_arg.lower() == "all":
        label_cols = []
        for group_name, cols in column_groups.items():
            if group_name == "general":
                continue
            label_cols.extend([c for c in cols if c not in general_cols])
        label_cols = sorted(set(label_cols))
    else:
        label_cols = [c.strip() for c in labels_arg.split(",") if c.strip()]

    missing_cols = [c for c in label_cols if c not in prepared_metadata.columns]
    if missing_cols:
        print(f"Warning: ignoring unknown label columns: {missing_cols}")
        label_cols = [c for c in label_cols if c in prepared_metadata.columns]
    if not label_cols:
        raise SystemExit("No valid label columns were found (use --labels all or a comma list).")

    fold = int(args.fold)
    test_metadata  = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "test"]
    val_metadata   = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "val"]
    train_metadata = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "train"]

    group_col = _choose_group_col(
        prepared_metadata,
        requested=str(getattr(args, "group_splits_by", "auto")),
        n_folds=n_folds,
    )
    if group_col is not None:
        ov_tv, ov_tt, ov_vt = _leakage_check(train_metadata, val_metadata, test_metadata, group_col=str(group_col))
        if (ov_tv or ov_tt or ov_vt) and str(args.leakage_action) != "ignore":
            msg = (
                f"Leakage warning: '{group_col}' overlaps between splits "
                f"(train∩val={ov_tv}, train∩test={ov_tt}, val∩test={ov_vt})."
            )
            if str(args.leakage_action) == "warn":
                print("WARNING: " + msg)
            else:
                if not needs_splits:
                    print(msg + " Regenerating splits with strict grouping...")
                    base_md = pd.read_csv(metadata_file)
                    base_md = _ensure_file_id(base_md, mapping_file)
                    base_md = base_md[base_md["fileID"] != -1]
                    group_col = _choose_group_col(
                        base_md,
                        requested=str(getattr(args, "group_splits_by", "auto")),
                        n_folds=n_folds,
                    )
                    if group_col is None:
                        raise SystemExit(msg)
                    metadata = Preprocessor_Metadata.create_splits(
                        base_md,
                        n_folds,
                        fold_folder,
                        fold_splitted_metadata_filename,
                        args.split_label,
                        group_col=group_col,
                        strict_group_col=True,
                        seed_base=int(args.seed),
                    )
                    prepared_metadata = metadata.reset_index(drop=True)
                    column_groups = Preprocessor_Metadata.get_column_name_groups(prepared_metadata, True)
                    general_cols = set(column_groups.get("general", []))
                    test_metadata  = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "test"]
                    val_metadata   = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "val"]
                    train_metadata = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "train"]
                    ov_tv, ov_tt, ov_vt = _leakage_check(train_metadata, val_metadata, test_metadata, group_col=str(group_col))
                    if ov_tv or ov_tt or ov_vt:
                        raise SystemExit(msg)
                    print("Leakage fixed by regenerating splits.")
                else:
                    raise SystemExit(msg)


    def _filter_any_label(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        any_valid = np.zeros(len(df), dtype=bool)
        for col in label_cols:
            s = df[col]
            valid = ~_missing_mask(s)
            any_valid |= valid.to_numpy()
        return df[any_valid]

    train_metadata = _filter_any_label(train_metadata)
    val_metadata   = _filter_any_label(val_metadata)
    test_metadata  = _filter_any_label(test_metadata)

    # Filter out rows without corresponding image files early so that:
    # - train stats / degeneracy checks are accurate,
    # - samplers match the actual dataset length,
    # - eval/test metrics aren't affected by missing images.
    n_tr0, n_va0, n_te0 = len(train_metadata), len(val_metadata), len(test_metadata)
    train_metadata = _filter_rows_with_images(train_metadata, root_folder)
    val_metadata   = _filter_rows_with_images(val_metadata,   root_folder)
    test_metadata  = _filter_rows_with_images(test_metadata,  root_folder)
    
    # Combine train and val if requested
    if args.combine_train_val:
        print("Combining train and val splits into a single training set.")
        train_metadata = pd.concat([train_metadata, val_metadata], ignore_index=True)
        val_metadata = val_metadata.iloc[0:0]  # Empty DataFrame with same columns
        n_va0 = len(val_metadata)
        n_tr0 = len(train_metadata)
        args.eval_every = 0
        if args.early_stop_patience > 0:
            print("Disabling early stopping since validation is disabled (--combine-train-val).")
            args.early_stop_patience = 0

    if (len(train_metadata), len(val_metadata), len(test_metadata)) != (n_tr0, n_va0, n_te0):
        print(
            "Filtered missing images: "
            f"train {n_tr0}->{len(train_metadata)}, "
            f"val {n_va0}->{len(val_metadata)}, "
            f"test {n_te0}->{len(test_metadata)}"
        )

    # Build per-task value maps (always binary 0/1) and label names for logging.
    task_value_maps: Dict[str, Dict[object, int]] = {}
    task_index_to_label: Dict[str, List[str]] = {}
    for col in label_cols:
        mapping, idx_to_label = _binary_value_map_from_series(prepared_metadata[col], task=col)
        task_value_maps[col] = mapping
        task_index_to_label[col] = idx_to_label

    # --- Per-task BCE loss with class-imbalance pos_weight ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion: Dict[str, nn.Module] = {}
    train_task_stats: Dict[str, Dict[str, float]] = {}
    for task in label_cols:
        mapping = task_value_maps.get(task, {})
        s = train_metadata[task]
        miss = _missing_mask(s)
        vals = []
        for v in s[~miss].tolist():
            if _is_missing_label_value(v):
                continue
            k = _canonical_label_value(v)
            if k is None or k not in mapping:
                continue
            vals.append(int(mapping[k]))
        arr = np.asarray(vals, dtype=int) if vals else np.asarray([], dtype=int)
        pos = float((arr == 1).sum())
        neg = float((arr == 0).sum())
        train_task_stats[task] = {"pos": pos, "neg": neg, "present": float(pos + neg)}
        pos_weight_val = (neg / max(1.0, pos)) if (pos + neg) > 0 else 1.0
        # If primary_balanced sampler is active for the primary task, the sampler already
        # rebalances classes via oversampling → applying pos_weight > 1 on top would
        # double-count the imbalance and over-penalise positives. Reset to 1.0 for that task.
        if task == str(args.primary_label).strip() and str(args.train_sampler) == "primary_balanced":
            print(f"[{task}] pos_weight reset to 1.0 (primary_balanced sampler handles imbalance)")
            pos_weight_val = 1.0
        # Cap pos_weight to prevent gradient spikes from extremely rare tasks.
        # With batch_size=32 and pos_weight>>10, a single positive example can dominate
        # the entire batch gradient, destabilising training.
        max_pos_weight = float(args.max_pos_weight)
        if max_pos_weight > 0 and pos_weight_val > max_pos_weight:
            print(f"[{task}] pos_weight capped {pos_weight_val:.2f} → {max_pos_weight:.1f}")
            pos_weight_val = max_pos_weight
        pos_weight = torch.tensor([pos_weight_val], device=device)
        criterion[task] = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        print(f"[{task}] BCEWithLogitsLoss(pos_weight={pos_weight_val:.4f}) [neg={neg:.0f} pos={pos:.0f}]")

    primary_label = str(args.primary_label).strip()
    if primary_label not in label_cols:
        raise SystemExit(f"Primary label '{primary_label}' is not part of training labels: {label_cols}")

    # Drop tasks that have only a single class in the TRAIN split.
    degenerate_tasks = [
        t for t in list(label_cols)
        if float(train_task_stats.get(t, {}).get("pos", 0.0)) == 0.0
        or float(train_task_stats.get(t, {}).get("neg", 0.0)) == 0.0
    ]
    if degenerate_tasks:
        print("Dropping degenerate tasks (train has only one class): " + ", ".join(sorted(degenerate_tasks)))
        label_cols = [t for t in label_cols if t not in set(degenerate_tasks)]
        if primary_label in set(degenerate_tasks):
            raise SystemExit(
                f"Primary label '{primary_label}' is degenerate in train for fold {fold} "
                f"(pos={train_task_stats[primary_label]['pos']:.0f} neg={train_task_stats[primary_label]['neg']:.0f}). "
                "Choose another --primary-label or adjust splits/data."
            )
        if not label_cols:
            raise SystemExit("All selected labels are degenerate in train (pos==0 or neg==0). Nothing to train.")
        criterion         = {t: criterion[t]         for t in label_cols if t in criterion}
        train_task_stats  = {t: train_task_stats[t]  for t in label_cols if t in train_task_stats}
        task_value_maps   = {t: task_value_maps[t]   for t in label_cols if t in task_value_maps}
        task_index_to_label = {t: task_index_to_label[t] for t in label_cols if t in task_index_to_label}
        train_metadata = _filter_any_label(train_metadata)
        val_metadata   = _filter_any_label(val_metadata) if not args.combine_train_val else None
        test_metadata  = _filter_any_label(test_metadata)

    print(f"Active tasks for this fold: {len(label_cols)} -> " + ", ".join(label_cols))

    # --- Optional task loss weighting ---
    task_weights = {t: 1.0 for t in label_cols}
    tw = str(args.task_weighting)
    if tw != "equal":
        raw = {}
        for t in label_cols:
            present = float(train_task_stats.get(t, {}).get("present", 0.0))
            raw[t] = max(1.0, present) if tw == "present" else max(1.0, float(np.sqrt(present)))
        mean_w = float(np.mean(list(raw.values()))) if raw else 1.0
        if mean_w <= 0:
            mean_w = 1.0
        task_weights = {t: float(raw[t]) / mean_w for t in raw}
    primary_tw = float(args.primary_task_weight)
    if primary_tw <= 0:
        presents = [float(train_task_stats.get(t, {}).get("present", 0.0)) for t in label_cols]
        median_present  = float(np.median(presents)) if presents else 0.0
        primary_present = float(train_task_stats.get(primary_label, {}).get("present", 0.0))
        primary_tw = 1.25 if primary_present >= median_present and primary_present > 0 else 1.0
        print(f"Auto primary_task_weight={primary_tw:.2f} (primary_present={primary_present:.0f}, median_present={median_present:.0f})")

    if primary_tw != 1.0 and primary_label in task_weights:
        task_weights[primary_label] = float(task_weights[primary_label]) * float(primary_tw)

    if tw != "equal" or primary_tw != 1.0:
        sample_keys = list(task_weights.keys())[:6]
        sample_s = ", ".join([f"{t}={float(task_weights[t]):.2f}" for t in sample_keys])
        print(
            f"Task weighting: {tw}, primary_task_weight={float(primary_tw):.3f} "
            f"(mean-normalized; sample: {sample_s})"
        )

    focus_raw    = _parse_csv_arg(args.focus_labels)
    focus_labels = [lab for lab in focus_raw if lab in label_cols]
    dropped_focus = [lab for lab in focus_raw if lab not in label_cols]
    if focus_labels:
        print("Focus labels: " + ", ".join(focus_labels))
    if dropped_focus:
        print("Focus labels not active in this fold: " + ", ".join(dropped_focus))

    min_pos = int(args.min_pos_backbone)
    min_neg = int(args.min_neg_backbone)
    backbone_tasks = {
        t for t in label_cols
        if train_task_stats.get(t, {}).get("pos", 0.0) >= min_pos
        and train_task_stats.get(t, {}).get("neg", 0.0) >= min_neg
    }
    head_only_tasks = [t for t in label_cols if t not in backbone_tasks]
    print(
        f"Backbone tasks: {len(backbone_tasks)}/{len(label_cols)} "
        f"(min_pos={min_pos}, min_neg={min_neg}). Head-only tasks: {len(head_only_tasks)}."
    )
    if head_only_tasks:
        print("Head-only (no backbone gradients): " + ", ".join(head_only_tasks))
    if primary_label not in backbone_tasks:
        print(
            f"Warning: primary label '{primary_label}' does not meet backbone thresholds "
            f"(pos={train_task_stats[primary_label]['pos']:.0f} neg={train_task_stats[primary_label]['neg']:.0f})."
        )

    # --- Datasets / data loaders ---
    n_workers = int(args.num_workers)
    gen = torch.Generator()
    gen.manual_seed(int(args.seed))

    # Prepare annotations for datasets
    dataset_columns = ["fileID"] + label_cols
    if args.filter_image_quality and "technical_quality" in prepared_metadata.columns:
        dataset_columns.append("technical_quality")

    train_dataset = XRayMultiTaskDataset(
        train_metadata[dataset_columns],
        root_folder,
        label_columns=label_cols,
        label_value_maps=task_value_maps,
        transform=preprocess,
        filter_image_quality=args.filter_image_quality,
    )
    if not args.combine_train_val:
        val_dataset = XRayMultiTaskDataset(
            val_metadata[dataset_columns],
            root_folder,
            label_columns=label_cols,
            label_value_maps=task_value_maps,
            transform=preprocess_no_aug,
            filter_image_quality=args.filter_image_quality,
        )
    test_dataset = XRayMultiTaskDataset(
        test_metadata[dataset_columns],
        root_folder,
        label_columns=label_cols,
        label_value_maps=task_value_maps,
        transform=preprocess_no_aug,
        filter_image_quality=args.filter_image_quality,
    )

    loader_kwargs = dict(
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=n_workers,
        generator=gen,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(n_workers > 0),
    )
    if str(args.train_sampler) == "primary_balanced":
        vm = task_value_maps.get(primary_label, {})
        # Compute weights AFTER image filtering (train_dataset.img_labels), so
        # len(weights) matches len(train_dataset).
        s = train_dataset.img_labels[primary_label]
        miss = _missing_mask(s)
        pos_n, neg_n = 0, 0
        labels_for_weight = []
        for v, is_miss in zip(s.tolist(), miss.tolist()):
            if is_miss or _is_missing_label_value(v):
                labels_for_weight.append(None)
                continue
            k = _canonical_label_value(v)
            if k is None or k not in vm:
                labels_for_weight.append(None)
                continue
            y = int(vm[k])
            labels_for_weight.append(y)
            if y == 1:
                pos_n += 1
            else:
                neg_n += 1
        raw_pos_w = (neg_n / max(1, pos_n)) if (pos_n + neg_n) > 0 else 1.0
        cap = float(args.primary_balanced_max_pos_weight)
        pos_w = min(raw_pos_w, cap) if cap > 0 else raw_pos_w
        weights = [pos_w if y == 1 else 1.0 for y in labels_for_weight]
        weights = [1.0 if y is None else w for y, w in zip(labels_for_weight, weights)]
        try:
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=gen)
        except TypeError:
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        if raw_pos_w != pos_w:
            print(
                f"Train sampler: primary_balanced ({primary_label}) pos={pos_n} neg={neg_n} "
                f"pos_weight={raw_pos_w:.4f} (capped to {pos_w:.4f})"
            )
        else:
            print(f"Train sampler: primary_balanced ({primary_label}) pos={pos_n} neg={neg_n} pos_weight={pos_w:.4f}")
        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, generator=gen), **loader_kwargs)
    val_loader  = DataLoader(val_dataset,  sampler=SequentialSampler(val_dataset),  **loader_kwargs) if not args.combine_train_val else None
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), **loader_kwargs)

    model_name = str(args.model)
    model = _build_multitask_model(
        model_name,
        label_cols,
        no_pretrained=bool(args.no_pretrained),
        head_dropout=float(args.head_dropout),
    )
    model = model.to(device)

    if args.freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.heads.parameters():
            p.requires_grad = True
        print(f"Freezing backbone for {args.freeze_backbone_epochs} epochs")

    scaler = GradScaler(device=device, enabled=(device == "cuda"))

    def make_optimizer():
        head_params, backbone_params = _split_head_backbone_params(model)
        if backbone_params and float(args.backbone_lr_mult) != 1.0:
            return optim.AdamW(
                [
                    {"params": backbone_params, "lr": float(args.learning_rate) * float(args.backbone_lr_mult)},
                    {"params": head_params,     "lr": float(args.learning_rate)},
                ],
                weight_decay=float(args.weight_decay),
            )
        return optim.AdamW(
            head_params + backbone_params,
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )

    def make_scheduler(optimizer, remaining_epochs: int):
        total_steps = max(1, remaining_epochs * len(train_loader))
        if len(optimizer.param_groups) == 2:
            max_lr = [float(args.learning_rate) * float(args.backbone_lr_mult), float(args.learning_rate)]
        else:
            max_lr = float(args.learning_rate)
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

    if not args.no_wandb:
        wandb.init(
            project="Asbestosis_train-test",
            config={
                "learning_rate":               float(args.learning_rate),
                "dataset":                     root_folder,
                "split_folder":                fold_folder,
                "train_samples":               len(train_loader.dataset),
                "val_samples":                 len(val_loader.dataset) if not args.combine_train_val else 0,
                "test_samples":                len(test_loader.dataset),
                "epochs":                      int(args.epochs),
                "batch_size":                  int(args.batch_size),
                "optimizer":                   "AdamW",
                "head_dropout":                float(args.head_dropout),
                "augmentation":                str(preprocess),
                "machine":                     "HPC",
                "labels":                      label_cols,
                "fold":                        fold,
                "model_name":                  model_name,
                "batch_size":                  int(args.batch_size),
                "metadata":                    metadata_file,
                "primary_label":               primary_label,
                "train_sampler":               str(args.train_sampler),
                "primary_balanced_max_pos_weight": float(args.primary_balanced_max_pos_weight),
                "task_weighting":              str(args.task_weighting),
                "primary_task_weight":         float(args.primary_task_weight),
                "min_pos_backbone":            int(args.min_pos_backbone),
                "min_neg_backbone":            int(args.min_neg_backbone),
                "head_only_loss_weight":       float(args.head_only_loss_weight),
                "threshold_strategy":          str(args.threshold_strategy),
                "fbeta":                       float(args.fbeta),
                "target_precision":            float(args.target_precision),
                "group_splits_by":             str(args.group_splits_by),
                "leakage_action":              str(args.leakage_action),
            },
            name=f"multitask_{model_name}_fold={fold}_labels={primary_label}",
        )

    if args.early_stop_patience > 0 and args.eval_every == 0 and not args.combine_train_val:
        args.eval_every = 1
        print("Early stopping enabled -> forcing --eval-every 1")

    best_score = None
    best_epoch = -1
    best_path = os.path.join(output_folder, f"best_{model_name}_label={primary_label}_fold={fold}.pth")
    no_improve = 0
    best_fixed_thresholds = None

    swa_model = None
    if int(args.swa_start_epoch) > 0:
        swa_model = AveragedModel(model)
        print(f"SWA enabled: averaging model weights from epoch {args.swa_start_epoch} onward")

    optimizer    = make_optimizer()
    lr_scheduler = make_scheduler(optimizer, remaining_epochs=int(args.epochs))

    for epoch in range(int(args.epochs)):
        if args.freeze_backbone_epochs and epoch == int(args.freeze_backbone_epochs):
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer    = make_optimizer()
            lr_scheduler = make_scheduler(optimizer, remaining_epochs=int(args.epochs) - epoch)
            print("Unfroze backbone and reset optimizer/scheduler")

        train_results, train_avg_loss = run_epoch(
            model, optimizer, lr_scheduler, criterion, scaler,
            train_loader, device, train=True,
            max_steps=args.max_train_steps,
            backbone_tasks=backbone_tasks,
            head_only_loss_weight=float(args.head_only_loss_weight),
            task_weights=task_weights,
            grad_clip=float(args.grad_clip),
            label_smoothing=float(args.label_smoothing),
        )

        if swa_model is not None and epoch >= int(args.swa_start_epoch):
            swa_model.update_parameters(model)

        log_multitask_metrics(
            train_results, train_avg_loss, suffix="train", epoch=epoch,
            task_index_to_label=task_index_to_label,
            wandb_detail=args.wandb_detail,
        )

        do_eval = not args.combine_train_val and args.eval_every and (epoch % int(args.eval_every) == 0 or epoch == int(args.epochs) - 1)
        do_test = args.test_every and (epoch % int(args.test_every) == 0 or epoch == int(args.epochs) - 1)

        eval_metrics = None
        fixed_thresholds_eval = {}
        if do_eval:
            eval_results, eval_avg_loss = run_epoch(
                model, optimizer, None, criterion, scaler,
                val_loader, device, train=False,
                max_steps=None,
                backbone_tasks=backbone_tasks,
                head_only_loss_weight=float(args.head_only_loss_weight),
                task_weights=task_weights,
            )
            eval_metrics = log_multitask_metrics(
                eval_results, eval_avg_loss, suffix="eval", epoch=epoch,
                task_index_to_label=task_index_to_label,
                compute_thresholds=True,
                wandb_detail=args.wandb_detail,
                threshold_strategy=str(args.threshold_strategy),
                fbeta=float(args.fbeta),
                target_precision=float(args.target_precision),
            )
            _print_focus_line(eval_metrics, suffix="eval", epoch=epoch, focus_labels=focus_labels)
            for task in label_cols:
                if str(args.threshold_strategy) == "recall_at_precision":
                    fixed_thresholds_eval[task] = eval_metrics.get(
                        f"task/{task}/best_threshold_recall_at_p{float(args.target_precision):.2f}/eval"
                    )
                elif str(args.threshold_strategy) == "fbeta":
                    fixed_thresholds_eval[task] = eval_metrics.get(f"task/{task}/best_threshold_fbeta/eval")
                else:
                    fixed_thresholds_eval[task] = eval_metrics.get(f"task/{task}/best_threshold_f1/eval")
            _print_focus_confusions(
                eval_results, thresholds=fixed_thresholds_eval,
                suffix="eval", epoch=epoch, focus_labels=focus_labels,
            )

        if do_test:
            test_results, test_avg_loss = run_epoch(
                model, optimizer, None, criterion, scaler,
                test_loader, device, train=False,
                max_steps=None,
                backbone_tasks=backbone_tasks,
                head_only_loss_weight=float(args.head_only_loss_weight),
                task_weights=task_weights,
            )
            log_multitask_metrics(
                test_results, test_avg_loss, suffix="test", epoch=epoch,
                task_index_to_label=task_index_to_label,
                wandb_detail=args.wandb_detail,
            )

        # --- Early stopping ---
        if args.early_stop_patience > 0 and do_eval and eval_metrics is not None:
            metric = str(args.early_stop_metric)
            if metric == "primary_auc/eval":
                score = eval_metrics.get(f"task/{primary_label}/auc/eval")
            elif metric == "primary_pr_auc/eval":
                score = eval_metrics.get(f"task/{primary_label}/pr_auc/eval")
            elif metric == "primary_f1/eval":
                score = eval_metrics.get(f"task/{primary_label}/f1/eval")
            elif metric == "macro_auc/eval":
                score = eval_metrics.get("macro_auc/eval")
            elif metric == "macro_pr_auc/eval":
                score = eval_metrics.get("macro_pr_auc/eval")
            elif metric == "macro_f1/eval":
                score = eval_metrics.get("macro_f1/eval")
            else:
                score = -eval_metrics.get("average_loss/eval", float("inf"))

            if score is None:
                continue
            score = float(score)

            improved = best_score is None or (score - float(best_score)) > float(args.early_stop_min_delta)
            if improved:
                best_score = score
                best_epoch = epoch
                no_improve = 0
                best_fixed_thresholds = fixed_thresholds_eval
                checkpoint = {
                    "model_state_dict":       model.state_dict(),
                    "epoch":                  best_epoch,
                    "score":                  float(best_score),
                    "labels":                 label_cols,
                    "task_index_to_label":    task_index_to_label,
                    "task_value_maps":        task_value_maps,
                    "best_fixed_thresholds":  best_fixed_thresholds,
                }
                torch.save(checkpoint, best_path)
                print(f"Saved best model to {best_path} (epoch {epoch}, score={best_score:.4f})")
            else:
                # Don't count non-improvement while the backbone is still frozen —
                # the model hasn't seen its full capacity yet, so plateaus here are
                # expected and should not trigger early stopping.
                if epoch >= int(args.freeze_backbone_epochs):
                    no_improve += 1
                    if no_improve >= int(args.early_stop_patience):
                        print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, score={best_score:.4f})")
                        break

    final_epoch = epoch if int(args.epochs) > 0 else -1

    if swa_model is not None:
        print("SWA: updating BatchNorm statistics over training set...")
        update_bn(train_loader, swa_model, device=device)
        model.load_state_dict(swa_model.module.state_dict())
        print("SWA weights applied to model — final_test uses the averaged model.")

    final_results, final_avg_loss = run_epoch(
        model, optimizer, None, criterion, scaler,
        test_loader, device, train=False,
        max_steps=None,
        backbone_tasks=backbone_tasks,
        head_only_loss_weight=float(args.head_only_loss_weight),
        task_weights=task_weights,
    )
    final_metrics = log_multitask_metrics(
        final_results, final_avg_loss, suffix="final_test", epoch=final_epoch,
        task_index_to_label=task_index_to_label,
        fixed_thresholds=best_fixed_thresholds,
        wandb_detail=args.wandb_detail,
        threshold_strategy=str(args.threshold_strategy),
        fbeta=float(args.fbeta),
        target_precision=float(args.target_precision),
        n_bootstrap=int(args.n_bootstrap),
        log_curves=True,
    )
    _print_focus_line(final_metrics, suffix="final_test", epoch=final_epoch, focus_labels=focus_labels)
    if best_fixed_thresholds:
        _print_focus_confusions(
            final_results, thresholds=best_fixed_thresholds,
            suffix="final_test", epoch=final_epoch, focus_labels=focus_labels,
        )

    if _wandb_is_active():
        try:
            for key in (
                "macro_f1/final_test",
                "macro_auc/final_test",
                "macro_pr_auc/final_test",
                "macro_accuracy/final_test",
                "average_loss/final_test",
            ):
                if key in final_metrics:
                    wandb.summary[key] = float(final_metrics[key])
        except Exception:
            pass

    # Evaluate best checkpoint as well.
    if args.early_stop_patience > 0 and os.path.isfile(best_path):
        try:
            state = torch.load(best_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                best_fixed_thresholds = state.get("best_fixed_thresholds") or best_fixed_thresholds
            best_results, best_avg_loss = run_epoch(
                model, optimizer, None, criterion, scaler,
                test_loader, device, train=False,
                max_steps=None,
                backbone_tasks=backbone_tasks,
                head_only_loss_weight=float(args.head_only_loss_weight),
                task_weights=task_weights,
            )
            best_ep = int(state.get("epoch", best_epoch)) if isinstance(state, dict) else best_epoch
            best_metrics = log_multitask_metrics(
                best_results, best_avg_loss, suffix="best_test", epoch=best_ep,
                task_index_to_label=task_index_to_label,
                fixed_thresholds=best_fixed_thresholds,
                wandb_detail=args.wandb_detail,
                threshold_strategy=str(args.threshold_strategy),
                fbeta=float(args.fbeta),
                target_precision=float(args.target_precision),
                n_bootstrap=int(args.n_bootstrap),
                log_curves=True,
            )
            _print_focus_line(best_metrics, suffix="best_test", epoch=best_ep, focus_labels=focus_labels)
            if best_fixed_thresholds:
                _print_focus_confusions(
                    best_results, thresholds=best_fixed_thresholds,
                    suffix="best_test", epoch=best_ep, focus_labels=focus_labels,
                )
        except Exception as e:
            print(f"Warning: failed to evaluate best checkpoint ({best_path}): {e}")

    torch.save(
        model.state_dict(),
        os.path.join(
            output_folder,
            f"asbestosis_{model_name}_n{int(args.epochs)}_b{int(args.batch_size)}_labels=multitask_fold={fold}.pth",
        ),
    )

    # When early stopping did not run (e.g. --combine-train-val), no best_* checkpoint
    # exists yet.  Save one now so that analyze_results.py can load the model together
    # with its label list.  Does not overwrite an existing best_* (early-stop wins).
    if not os.path.isfile(best_path):
        torch.save(
            {
                "model_state_dict":      model.state_dict(),
                "epoch":                 final_epoch,
                "labels":                label_cols,
                "task_index_to_label":   task_index_to_label,
                "task_value_maps":       task_value_maps,
                "best_fixed_thresholds": best_fixed_thresholds or {},
            },
            best_path,
        )
        print(f"Saved final checkpoint (with metadata) to {best_path}")

    if _wandb_is_active():
        wandb.finish()


if __name__ == "__main__":
    main()
