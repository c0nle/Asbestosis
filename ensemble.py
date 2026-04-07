"""
ensemble.py
-----------
K-Fold Ensemble evaluation for the asbestosis multi-task pipeline.

For each fold's held-out test split, runs inference with ALL available fold
models and averages their predicted probabilities before computing metrics.
Labels are read directly from the checkpoints — no need to pass --labels.

Results (ROC/PR curves, bootstrap AUC CI, macro metrics) are logged as a
single W&B run named "ensemble" in the same "Asbestosis" project.

Usage (after training all 5 folds with run_asbestosis.sh):

    python ensemble.py \
        --base-folder /hpcwork/rwth1954/Asbestosis_Data \
        --fold-folder  /rwthfs/rz/cluster/home/rwth1954/Asbestosis/splits \
        --output-folder /rwthfs/rz/cluster/home/rwth1954/Asbestosis/logs \
        --model chexnet

Partial ensembles (e.g. only folds 0+2 finished) are supported — missing
checkpoints are skipped with a warning.
"""

import argparse
import os
import warnings
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

import Preprocessor_Metadata
from dataset import (
    XRayMultiTaskDataset,
    _binary_value_map_from_series,
    _filter_rows_with_images,
)
from model import _build_multitask_model
from utils import (
    _ensure_file_id,
    _missing_mask,
    _set_seed,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str, model_name: str, device: str, head_dropout: float):
    """
    Load checkpoint saved by main.py.

    Returns (model, label_cols) where label_cols is read from the checkpoint.
    Returns (None, None) if the file does not exist.
    """
    if not os.path.isfile(path):
        return None, None

    state = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(state, dict) and "model_state_dict" in state):
        print(f"  WARNING: {path} has unexpected format, skipping.")
        return None, None

    label_cols = state.get("labels")
    if not label_cols:
        print(f"  WARNING: checkpoint {path} has no 'labels' key, skipping.")
        return None, None

    model = _build_multitask_model(model_name, label_cols, no_pretrained=True, head_dropout=head_dropout)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model, label_cols


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer(model, loader, label_cols: List[str], device: str) -> Dict[str, dict]:
    """Run inference; returns dict task → {prob, y_true, mask} as numpy arrays."""
    model.eval()
    prob_acc:  Dict[str, list] = {t: [] for t in label_cols}
    true_acc:  Dict[str, list] = {t: [] for t in label_cols}
    mask_acc:  Dict[str, list] = {t: [] for t in label_cols}

    with torch.no_grad():
        for data, targets, masks, _ in loader:
            data = data.to(device)
            feats = model.backbone(data)
            for task in label_cols:
                logits = model.heads[task](feats).view(-1)
                prob_acc[task].append(torch.sigmoid(logits).cpu().numpy())
                true_acc[task].append(targets[task].cpu().numpy().reshape(-1))
                mask_acc[task].append(masks[task].cpu().numpy().reshape(-1).astype(bool))

    out = {}
    for task in label_cols:
        prob  = np.concatenate(prob_acc[task])
        y_true = np.concatenate(true_acc[task]).astype(float)
        mask   = np.concatenate(mask_acc[task])
        # mask out missing-label positions so they're ignored in nanmean
        prob[~mask]  = np.nan
        y_true[~mask] = np.nan
        out[task] = {"prob": prob, "y_true": y_true, "mask": mask}
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true_raw: np.ndarray, y_prob_raw: np.ndarray, task: str, prefix: str = "") -> Dict[str, float]:
    valid = np.isfinite(y_true_raw) & np.isfinite(y_prob_raw)
    y_true = y_true_raw[valid].astype(int)
    y_prob = y_prob_raw[valid]
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        print(f"  {prefix}{task}: not enough data (n={len(y_true)})")
        return {}

    y_pred = (y_prob >= 0.5).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        auc     = float(roc_auc_score(y_true, y_prob))
        pr_auc  = float(average_precision_score(y_true, y_prob))

    print(
        f"  {prefix}{task:42s}  n={len(y_true):4d}  pos={y_true.mean():.1%}  "
        f"auc={auc:.3f}  pr_auc={pr_auc:.3f}  "
        f"f1={float(f1):.3f}  prec={float(prec):.3f}  rec={float(rec):.3f}"
    )
    return {"auc": auc, "pr_auc": pr_auc, "f1": float(f1),
            "prec": float(prec), "rec": float(rec),
            "y_true": y_true, "y_prob": y_prob}


# ---------------------------------------------------------------------------
# W&B curve + bootstrap logging
# ---------------------------------------------------------------------------

def _wandb_log_task(
    task: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict[str, float],
    mode: str,          # "ensemble" or "single"
    n_bootstrap: int,
) -> None:
    """Log ROC curve, PR curve, and bootstrap AUC histogram to W&B for one task."""
    if not wandb.run:
        return
    if np.unique(y_true).size < 2:
        return

    prefix = f"{mode}/{task}"

    # --- Scalar metrics ---
    wandb.log({
        f"{prefix}/auc":    metrics.get("auc", float("nan")),
        f"{prefix}/pr_auc": metrics.get("pr_auc", float("nan")),
        f"{prefix}/f1":     metrics.get("f1", float("nan")),
        f"{prefix}/prec":   metrics.get("prec", float("nan")),
        f"{prefix}/rec":    metrics.get("rec", float("nan")),
    }, commit=False)

    # --- ROC curve + random-guess diagonal ---
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        step = max(1, len(fpr) // 300)
        fpr_ds = fpr[::step].tolist()
        tpr_ds = tpr[::step].tolist()
        if fpr_ds[-1] != 1.0:
            fpr_ds.append(1.0); tpr_ds.append(1.0)
        wandb.log(
            {
                f"roc_curve/{mode}/{task}": wandb.plot.line_series(
                    xs=[fpr_ds, [0.0, 1.0]],
                    ys=[tpr_ds, [0.0, 1.0]],
                    keys=["Ensemble", "Random"],
                    title=f"ROC – {task} ({mode})",
                    xname="False Positive Rate",
                )
            },
            commit=False,
        )
    except Exception as e:
        print(f"  WARNING: ROC curve failed for {task}: {e}")

    # --- PR curve + prevalence baseline ---
    try:
        pr_prec_c, pr_rec_c, _ = precision_recall_curve(y_true, y_prob)
        prevalence = float(y_true.mean())
        step = max(1, len(pr_rec_c) // 300)
        rec_ds  = pr_rec_c[::step].tolist()
        prec_ds = pr_prec_c[::step].tolist()
        wandb.log(
            {
                f"pr_curve/{mode}/{task}": wandb.plot.line_series(
                    xs=[rec_ds, [0.0, 1.0]],
                    ys=[prec_ds, [prevalence, prevalence]],
                    keys=["Ensemble", f"Baseline ({prevalence:.1%})"],
                    title=f"PR – {task} ({mode})",
                    xname="Recall",
                )
            },
            commit=False,
        )
    except Exception as e:
        print(f"  WARNING: PR curve failed for {task}: {e}")

    # --- Bootstrap AUC CI ---
    if n_bootstrap > 0:
        try:
            rng = np.random.default_rng(seed=42)
            boot_aucs = []
            n = len(y_true)
            for _ in range(int(n_bootstrap)):
                idx = rng.integers(0, n, size=n)
                yt, yp = y_true[idx], y_prob[idx]
                if np.unique(yt).size < 2:
                    continue
                boot_aucs.append(float(roc_auc_score(yt, yp)))
            if len(boot_aucs) >= 10:
                ci_lo = float(np.percentile(boot_aucs, 2.5))
                ci_hi = float(np.percentile(boot_aucs, 97.5))
                print(
                    f"  bootstrap {task} ({mode}): "
                    f"auc={metrics.get('auc', float('nan')):.3f} "
                    f"95%CI=[{ci_lo:.3f}, {ci_hi:.3f}]"
                )
                wandb.log({
                    f"{prefix}/auc_ci_lo": ci_lo,
                    f"{prefix}/auc_ci_hi": ci_hi,
                    f"auc_bootstrap/{mode}/{task}": wandb.Histogram(np.array(boot_aucs)),
                }, commit=False)
        except Exception as e:
            print(f"  WARNING: bootstrap failed for {task}: {e}")

    wandb.log({}, commit=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="K-Fold ensemble evaluation")
    parser.add_argument("--base-folder",   default="/hpcwork/rwth1954/Asbestosis_Data")
    parser.add_argument("--fold-folder",   default=None)
    parser.add_argument("--output-folder", default=None)
    parser.add_argument(
        "--model",
        choices=["resnet18", "efficientnet_b0", "densenet121", "chexnet",
                 "mobilenet_v3_small", "mobilenet_v3_large"],
        default="chexnet",
    )
    parser.add_argument("--head-dropout", type=float, default=0.5)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--num-workers",  type=int,   default=0,
                        help="DataLoader workers (0 = main process only; use 0 on login nodes).")
    parser.add_argument("--n-folds",      type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--n-bootstrap",  type=int,   default=500,
                        help="Bootstrap resamples for AUC CI (0 = off).")
    parser.add_argument("--no-wandb",     action="store_true",
                        help="Disable W&B logging.")
    parser.add_argument("--wandb-project", default="Asbestosis",
                        help="W&B project name.")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    base_folder   = args.base_folder
    mapping_file  = os.path.join(base_folder, "mapping.csv")
    metadata_file = os.path.join(base_folder, "dichotome_data_anonymized_with_patientID.csv")
    output_folder = args.output_folder or base_folder
    model_name    = str(args.model)

    # fold_folder: prefer explicit arg, then project-local splits/, then data default
    if args.fold_folder:
        fold_folder = args.fold_folder
    else:
        project_splits = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
        fold_folder = project_splits if os.path.isdir(project_splits) \
                      else os.path.join(base_folder, "strat_dichotom_splits")

    # --- Load all available fold checkpoints ---
    models:     Dict[int, torch.nn.Module] = {}
    fold_labels: Dict[int, List[str]] = {}

    for k in range(int(args.n_folds)):
        ckpt = os.path.join(output_folder, f"best_{model_name}_labels=multitask_fold={k}.pth")
        m, lc = _load_checkpoint(ckpt, model_name, device, args.head_dropout)
        if m is None:
            print(f"Fold {k}: checkpoint not found ({ckpt}) — skipping")
        else:
            models[k] = m
            fold_labels[k] = lc
            print(f"Fold {k}: loaded  labels={lc}")

    if not models:
        raise SystemExit("\nNo checkpoints found. Train at least one fold first.")

    label_cols = list(Counter(tuple(v) for v in fold_labels.values()).most_common(1)[0][0])
    print(f"\nUsing label set: {label_cols}")

    for k in list(models.keys()):
        if fold_labels[k] != label_cols:
            print(f"  WARNING: fold {k} has different labels {fold_labels[k]}, skipping.")
            del models[k]

    loaded_folds = sorted(models.keys())
    print(f"\nEnsemble: {len(models)} model(s) — folds {loaded_folds}\n")

    # --- W&B init ---
    if not args.no_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=f"ensemble_{model_name}_{len(models)}folds",
                config={
                    "model":        model_name,
                    "n_folds":      args.n_folds,
                    "loaded_folds": loaded_folds,
                    "labels":       label_cols,
                    "n_bootstrap":  args.n_bootstrap,
                    "head_dropout": args.head_dropout,
                },
                tags=["ensemble"],
            )
        except Exception as e:
            print(f"WARNING: W&B init failed ({e}). Continuing without W&B.")

    # --- Load metadata ---
    fold_csv = os.path.join(
        fold_folder,
        os.path.basename(metadata_file).replace(".csv", "_stratified_folds.csv"),
    )
    if not os.path.isfile(fold_csv):
        raise SystemExit(f"Folds CSV not found: {fold_csv}\nRun main.py first to generate splits.")

    metadata = pd.read_csv(fold_csv)
    metadata = _ensure_file_id(metadata, mapping_file)
    metadata = metadata[metadata["fileID"] != -1].reset_index(drop=True)

    _presence_derivations = {
        "diffuse_pleural_thickening_presence": "diffuse_pleural_thickening_width",
        "localized_pleural_thickening_presence": "localized_pleural_thickening_width",
    }
    for derived_col, source_col in _presence_derivations.items():
        if source_col in metadata.columns and derived_col not in metadata.columns:
            metadata[derived_col] = metadata[source_col].notna().astype(int)

    missing_cols = [c for c in label_cols if c not in metadata.columns]
    if missing_cols:
        raise SystemExit(f"Label columns not in metadata: {missing_cols}")

    task_value_maps: Dict[str, Dict] = {}
    for col in label_cols:
        vm, _ = _binary_value_map_from_series(metadata[col], task=col)
        task_value_maps[col] = vm

    _norm = Normalize(mean=[0.5], std=[1.0 / 2048.0]) if model_name == "chexnet" \
            else Normalize(mean=[0.5], std=[0.5])
    preprocess = Compose([Resize((224, 224)), ToImage(), ToDtype(torch.float32, scale=True), _norm])

    root_folder = base_folder

    # --- Per-fold inference ---
    all_true:     Dict[str, List[float]] = {t: [] for t in label_cols}
    all_ensemble: Dict[str, List[float]] = {t: [] for t in label_cols}
    all_single:   Dict[str, List[float]] = {t: [] for t in label_cols}

    for fold_idx in range(int(args.n_folds)):
        fold_col = f"Fold{fold_idx}"
        if fold_col not in metadata.columns:
            continue

        test_meta = metadata[metadata[fold_col] == "test"].copy()
        test_meta = _filter_rows_with_images(test_meta, root_folder)
        if test_meta.empty:
            continue

        dataset = XRayMultiTaskDataset(
            test_meta[["fileID"] + label_cols], root_folder,
            label_columns=label_cols, label_value_maps=task_value_maps,
            transform=preprocess,
        )
        loader = DataLoader(
            dataset, sampler=SequentialSampler(dataset),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device == "cuda"),
        )

        print(f"--- Fold {fold_idx} test set ({len(dataset)} samples) ---")

        fold_preds: Dict[int, Dict[str, dict]] = {}
        for k, m in models.items():
            fold_preds[k] = _infer(m, loader, label_cols, device)

        if not fold_preds:
            print("  No models available for this fold, skipping.\n")
            continue

        ref_key = next(iter(fold_preds))

        for task in label_cols:
            y_true = fold_preds[ref_key][task]["y_true"]
            mask   = fold_preds[ref_key][task]["mask"]

            prob_stack = np.stack([fold_preds[k][task]["prob"] for k in fold_preds], axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ens_prob = np.nanmean(prob_stack, axis=0)

            all_true[task].extend(y_true[mask].tolist())
            all_ensemble[task].extend(ens_prob[mask].tolist())

            own_key = fold_idx if fold_idx in fold_preds else ref_key
            all_single[task].extend(fold_preds[own_key][task]["prob"][mask].tolist())

        own_key = fold_idx if fold_idx in fold_preds else ref_key
        own_label = f"fold {fold_idx} single" if fold_idx in fold_preds else f"fold {ref_key} (proxy)"
        print(f"  {own_label}:")
        for task in label_cols:
            p = fold_preds[own_key][task]
            _metrics(p["y_true"], p["prob"], task, prefix="    ")
        print()

    # --- Global results: Ensemble ---
    print("=" * 80)
    print(f"ENSEMBLE ({len(models)} model(s), folds {loaded_folds}) — full dataset:")
    print("=" * 80)
    ens_aucs, ens_pr_aucs = [], []
    for task in label_cols:
        yt = np.array(all_true[task])
        yp = np.array(all_ensemble[task])
        m = _metrics(yt, yp, task, prefix="  ")
        if not m:
            continue
        if "auc" in m:
            ens_aucs.append(m["auc"])
        if "pr_auc" in m:
            ens_pr_aucs.append(m["pr_auc"])
        _wandb_log_task(task, m["y_true"], m["y_prob"], m, mode="ensemble", n_bootstrap=args.n_bootstrap)

    macro_ens_auc    = float(np.mean(ens_aucs))    if ens_aucs    else float("nan")
    macro_ens_pr_auc = float(np.mean(ens_pr_aucs)) if ens_pr_aucs else float("nan")
    if ens_aucs:
        print(f"\n  MACRO ensemble   auc={macro_ens_auc:.3f}  pr_auc={macro_ens_pr_auc:.3f}")
    if wandb.run:
        wandb.log({
            "macro_auc/ensemble":    macro_ens_auc,
            "macro_pr_auc/ensemble": macro_ens_pr_auc,
        }, commit=True)

    # --- Global results: Single (each fold's own model) ---
    print("\n" + "=" * 80)
    print("SINGLE model (each fold's own test split) — full dataset:")
    print("=" * 80)
    sng_aucs, sng_pr_aucs = [], []
    for task in label_cols:
        yt = np.array(all_true[task])
        yp = np.array(all_single[task])
        m = _metrics(yt, yp, task, prefix="  ")
        if not m:
            continue
        if "auc" in m:
            sng_aucs.append(m["auc"])
        if "pr_auc" in m:
            sng_pr_aucs.append(m["pr_auc"])
        _wandb_log_task(task, m["y_true"], m["y_prob"], m, mode="single", n_bootstrap=0)

    macro_sng_auc    = float(np.mean(sng_aucs))    if sng_aucs    else float("nan")
    macro_sng_pr_auc = float(np.mean(sng_pr_aucs)) if sng_pr_aucs else float("nan")
    if sng_aucs:
        print(f"\n  MACRO single     auc={macro_sng_auc:.3f}  pr_auc={macro_sng_pr_auc:.3f}")
        if ens_aucs:
            delta_auc    = macro_ens_auc    - macro_sng_auc
            delta_pr_auc = macro_ens_pr_auc - macro_sng_pr_auc
            print(
                f"  Ensemble gain    Δauc={delta_auc:+.3f}  "
                f"Δpr_auc={delta_pr_auc:+.3f}"
            )
    if wandb.run:
        wandb.log({
            "macro_auc/single":    macro_sng_auc,
            "macro_pr_auc/single": macro_sng_pr_auc,
            "ensemble_gain_auc":    macro_ens_auc - macro_sng_auc if ens_aucs and sng_aucs else float("nan"),
        }, commit=True)
        wandb.finish()


if __name__ == "__main__":
    main()
