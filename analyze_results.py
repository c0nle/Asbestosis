"""
analyze_results.py
------------------
Post-training analysis for the asbestosis multi-task pipeline.

Produces
--------
  1. metrics_per_fold.csv   — AUC, sensitivity, specificity per fold × label × model
  2. metrics_summary.csv    — mean ± SD across folds (same metrics)
  3. pairwise_pvalues.csv   — bootstrap two-sided p-value for every model pair × label
  4. roc_<model>.pdf/.png   — ROC figure per model (all labels in one plot)

Threshold note
--------------
Sensitivity and specificity are evaluated at the threshold that was optimised
on the *validation* set during training and stored in each fold checkpoint
(key: 'best_fixed_thresholds').  This keeps threshold selection and test-set
evaluation strictly separate, avoiding any circularity.

If a checkpoint does not contain a threshold (e.g. early-stop was disabled),
the script falls back to the Youden-index threshold on each fold's test set and
prints a warning.  In that case the sensitivity/specificity estimate is slightly
optimistic (upper bound).

Usage (on HPC after training all folds):

    python analyze_results.py \\
        --models chexnet resnet18 \\
        --base-folder   /hpcwork/rwth1954/Asbestosis_Data \\
        --fold-folder   /rwthfs/rz/cluster/home/rwth1954/Asbestosis/splits \\
        --ckpt-folder   /rwthfs/rz/cluster/home/rwth1954/Asbestosis/logs \\
        --output-folder ./analysis_output \\
        --n-bootstrap   2000 \\
        --n-folds       5
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

from dataset import (
    XRayMultiTaskDataset,
    _binary_value_map_from_series,
    _filter_rows_with_images,
)
from model import _build_multitask_model
from utils import _ensure_file_id, _set_seed


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(
    path: str, model_name: str, device: str, head_dropout: float = 0.5
) -> Tuple[Optional[object], Optional[List[str]], Optional[Dict[str, float]]]:
    """
    Load a best-checkpoint saved by main.py.

    Returns (model, label_cols, best_fixed_thresholds).
    All three are None if the file does not exist or has unexpected format.
    'best_fixed_thresholds' maps task → optimal threshold from the validation set.
    It is None when the checkpoint was saved without early-stopping.
    """
    if not os.path.isfile(path):
        return None, None, None
    state = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(state, dict) and "model_state_dict" in state):
        print(f"  WARNING: {path} has unexpected format, skipping.")
        return None, None, None
    label_cols = state.get("labels")
    if not label_cols:
        print(f"  WARNING: {path} has no 'labels' key, skipping.")
        return None, None, None
    thresholds = state.get("best_fixed_thresholds") or {}

    model = _build_multitask_model(model_name, label_cols, no_pretrained=True, head_dropout=head_dropout)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model, label_cols, thresholds


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer(model, loader, label_cols: List[str], device: str) -> Dict[str, dict]:
    """Run inference; returns {task: {prob, y_true, mask}} as numpy arrays."""
    model.eval()
    prob_acc  = {t: [] for t in label_cols}
    true_acc  = {t: [] for t in label_cols}
    mask_acc  = {t: [] for t in label_cols}
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
        prob   = np.concatenate(prob_acc[task])
        y_true = np.concatenate(true_acc[task]).astype(float)
        mask   = np.concatenate(mask_acc[task])
        prob[~mask]   = np.nan
        y_true[~mask] = np.nan
        out[task] = {"prob": prob, "y_true": y_true, "mask": mask}
    return out


# ---------------------------------------------------------------------------
# Per-fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(
    y_true_raw: np.ndarray,
    y_prob_raw: np.ndarray,
    threshold: Optional[float],
    task: str,
) -> Optional[dict]:
    """
    Compute AUC, sensitivity, specificity for one fold/task.

    Sensitivity and specificity are computed at *threshold* (from the
    validation set).  If threshold is None the Youden-index optimum on the
    test set is used as a fallback and a warning is printed.

    Returns None when fewer than two classes are present.
    """
    valid = np.isfinite(y_true_raw) & np.isfinite(y_prob_raw)
    yt = y_true_raw[valid].astype(int)
    yp = y_prob_raw[valid]

    if len(yt) == 0 or np.unique(yt).size < 2:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc = float(roc_auc_score(yt, yp))
        fpr_arr, tpr_arr, thresholds_arr = roc_curve(yt, yp)

    # Determine operating-point threshold
    if threshold is not None:
        thr = float(threshold)
        source = "val"
    else:
        # Fallback: Youden index on test set (optimistic — warns the caller)
        j = tpr_arr - fpr_arr
        best_idx = int(np.argmax(j))
        thr = float(thresholds_arr[best_idx])
        source = "youden_testset_FALLBACK"

    # Sensitivity / Specificity at chosen threshold
    y_pred = (yp >= thr).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()

    sens = float(tp / max(1, tp + fn))
    spec = float(tn / max(1, tn + fp))

    return {
        "auc":             auc,
        "sensitivity":     sens,
        "specificity":     spec,
        "threshold":       thr,
        "threshold_source": source,
        "n":               len(yt),
        "n_pos":           int(yt.sum()),
        "n_neg":           int((yt == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Bootstrap pairwise p-value
# ---------------------------------------------------------------------------

def _bootstrap_pvalue(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_bootstrap: int,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap two-sided p-value for H0: AUC(A) == AUC(B).

    Returns (observed_diff, p_value, SE) where
      observed_diff = AUC(A) - AUC(B),
      SE            = bootstrap standard error of the difference,
      p_value       = 2 * min(P(Δboot > 0), P(Δboot < 0)).

    Reference: Efron & Tibshirani (1993), An Introduction to the Bootstrap.
    """
    valid = np.isfinite(y_true) & np.isfinite(y_prob_a) & np.isfinite(y_prob_b)
    yt = y_true[valid].astype(int)
    ya = y_prob_a[valid]
    yb = y_prob_b[valid]

    if len(yt) == 0 or np.unique(yt).size < 2:
        return float("nan"), float("nan"), float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_a = float(roc_auc_score(yt, ya))
        obs_b = float(roc_auc_score(yt, yb))
    obs_diff = obs_a - obs_b

    rng = np.random.default_rng(seed)
    n = len(yt)
    boot_diffs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt_b = yt[idx]
        if np.unique(yt_b).size < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                da = float(roc_auc_score(yt_b, ya[idx]))
                db = float(roc_auc_score(yt_b, yb[idx]))
                boot_diffs.append(da - db)
            except Exception:
                continue

    if len(boot_diffs) < 10:
        return obs_diff, float("nan"), float("nan")

    boot_arr = np.array(boot_diffs)
    se = float(np.std(boot_arr, ddof=1))
    # Two-sided p-value (floor at resolution 1/n_bootstrap)
    p_positive = float(np.mean(boot_arr > 0))
    p_negative = float(np.mean(boot_arr < 0))
    p = max(2.0 * min(p_positive, p_negative), 1.0 / len(boot_diffs))
    return obs_diff, p, se


# ---------------------------------------------------------------------------
# ROC figure
# ---------------------------------------------------------------------------

_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

_LABEL_DISPLAY = {
    "mixed_shapes":                          "Pneumoconiosis (ILO mixed shapes)",
    "occupational_disease":                  "Occupational disease",
    "small_rounded_right":                   "Small rounded (right lung)",
    "small_rounded_left":                    "Small rounded (left lung)",
    "small_irregular_right":                 "Small irregular (right lung)",
    "small_irregular_left":                  "Small irregular (left lung)",
    "diffuse_pleural_thickening_presence":   "Diffuse pleural thickening",
    "localized_pleural_thickening_presence": "Localized pleural thickening",
    "pleural_calcification_location":        "Pleural calcification",
}

_MODEL_DISPLAY = {
    "chexnet":            "CheXNet (DenseNet-121, chest X-ray pretrained)",
    "densenet121":        "DenseNet-121 (ImageNet pretrained)",
    "resnet18":           "ResNet-18 (ImageNet pretrained)",
    "efficientnet_b0":    "EfficientNet-B0 (ImageNet pretrained)",
    "mobilenet_v3_small": "MobileNet-V3 small (ImageNet pretrained)",
    "mobilenet_v3_large": "MobileNet-V3 large (ImageNet pretrained)",
    "vit_b_16":           "ViT-B/16 (ImageNet pretrained)",
}

def plot_auc_by_model(
    all_preds: Dict[str, Dict[int, Dict[str, dict]]],   # model -> fold_idx -> task -> {"y_true", "prob"}
    output_dir: str,
) -> None:
    """
    Boxplot of per-fold AUC across models, one subplot per label.

    Consumes the same `all_preds` structure used to build `fold_preds` in the
    pooling loop, and computes AUC per fold per task with the same validity
    filtering as `_plot_roc` (finite values, cast y_true to int, require both
    classes present). Each box = distribution of per-fold AUCs for one model.
    """
    # Discover label set the same way the pooling loop does (from first available fold)
    label_cols: list = []
    for fold_preds in all_preds.values():
        if fold_preds:
            label_cols = list(next(iter(fold_preds.values())).keys())
            break
    if not label_cols:
        print("  plot_auc_by_model: no predictions found, skipping.")
        return

    models = sorted(all_preds.keys())

    # per_label_aucs[label][model] = list of per-fold AUCs
    per_label_aucs: Dict[str, Dict[str, list]] = {lbl: {m: [] for m in models} for lbl in label_cols}

    for model_name in models:
        fold_preds = all_preds.get(model_name, {})
        for task in label_cols:
            for fold_idx, preds in fold_preds.items():
                if task not in preds:
                    continue
                p = preds[task]
                yt = np.asarray(p["y_true"])
                yp = np.asarray(p["prob"])
                valid = np.isfinite(yt) & np.isfinite(yp)
                yt = yt[valid].astype(int)
                yp = yp[valid]
                if len(yt) == 0 or np.unique(yt).size < 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auc = roc_auc_score(yt, yp)
                per_label_aucs[task][model_name].append(auc)

    labels = sorted(label_cols)
    fig, axes = plt.subplots(1, len(labels), figsize=(16, 5))
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        model_aucs = [per_label_aucs[label][m] for m in models]
        # Drop models with no valid fold AUCs for this label (keeps boxplot from erroring)
        plotted_models, plotted_data = [], []
        for m, vals in zip(models, model_aucs):
            if len(vals) > 0:
                plotted_models.append(m)
                plotted_data.append(vals)

        disp_label = _LABEL_DISPLAY.get(label, label.replace("_", " ").title())
        if not plotted_data:
            ax.set_title(f"{disp_label}\n(no data)", fontsize=12, fontweight="bold")
            ax.axis("off")
            continue

        bp = ax.boxplot(plotted_data, labels=plotted_models, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightcoral")

        ax.set_title(disp_label, fontsize=12, fontweight="bold")
        ax.set_ylabel("AUC")
        ax.set_ylim([0.4, 1.0])
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "auc_by_model_per_label.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.close(fig)

def _plot_roc(
    model_name: str,
    pooled_true:  Dict[str, np.ndarray],
    pooled_prob:  Dict[str, np.ndarray],
    pooled_thr:   Dict[str, Optional[float]],   # val-set threshold per task
    summary:      Dict[str, dict],               # label → {auc_mean, auc_sd, ...}
    output_folder: str,
) -> None:
    """
    One ROC figure per model.

    Each label gets its own curve (pooled cross-validation test predictions).
    The operating point derived from the validation-set threshold is shown as a
    filled circle on each curve; the corresponding sensitivity and specificity
    values appear in the legend.
    """
    label_cols = list(pooled_true.keys())
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.9, label="Random (AUC = 0.50)")

    for i, task in enumerate(label_cols):
        yt = pooled_true[task]
        yp = pooled_prob[task]
        valid = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[valid].astype(int)
        yp = yp[valid]
        if len(yt) == 0 or np.unique(yt).size < 2:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fpr, tpr, thrs = roc_curve(yt, yp)

        color    = _PALETTE[i % len(_PALETTE)]
        disp     = _LABEL_DISPLAY.get(task, task.replace("_", " ").title())
        s        = summary.get(task, {})
        auc_mean = s.get("auc_mean", float("nan"))
        auc_sd   = s.get("auc_sd",   float("nan"))
        sens_mean = s.get("sens_mean", float("nan"))
        spec_mean = s.get("spec_mean", float("nan"))

        label_str = (
            f"{disp}\n"
            f"  AUC {auc_mean:.3f} ± {auc_sd:.3f}  |  "
            f"Sens {sens_mean:.3f}  Spec {spec_mean:.3f}"
        )
        ax.plot(fpr, tpr, color=color, lw=1.8, label=label_str)

        # Mark the operating point (val-set threshold) on the curve
        thr = pooled_thr.get(task)
        if thr is not None:
            # Find closest index in the ROC threshold array
            diffs = np.abs(thrs - thr)
            op_idx = int(np.argmin(diffs))
            ax.plot(
                fpr[op_idx], tpr[op_idx],
                "o", color=color, ms=7, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8,
            )

    disp_model = _MODEL_DISPLAY.get(model_name, model_name)
    ax.set_xlabel("1 − Specificity  (False Positive Rate)", fontsize=11)
    ax.set_ylabel("Sensitivity  (True Positive Rate)", fontsize=11)
    ax.set_title(f"ROC Curves\n{disp_model}", fontsize=11, pad=10)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92, handlelength=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25, lw=0.6)

    for fmt in ("pdf", "png"):
        out = os.path.join(output_folder, f"roc_{model_name}.{fmt}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_folder, f'roc_{model_name}.pdf')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-training analysis: metrics (mean ± SD), bootstrap p-values, ROC figures."
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["chexnet"],
        choices=["chexnet", "densenet121", "resnet18", "efficientnet_b0",
                 "mobilenet_v3_small", "mobilenet_v3_large", "vit_b_16"],
        help="One or more model names to analyse (pairwise comparison if ≥ 2).",
    )
    parser.add_argument("--base-folder", default="/hpcwork/rwth1954/Asbestosis_Data")
    parser.add_argument(
        "--fold-folder", default=None,
        help="Folder with stratified split CSVs. Default: <project>/splits.",
    )
    parser.add_argument(
        "--ckpt-folder", default=None,
        help=(
            "Folder with best_<model>_labels=multitask_fold=<k>.pth checkpoints. "
            "Default: <project>/logs  (as set by run_asbestosis.sh)."
        ),
    )
    parser.add_argument("--output-folder", default="./analysis_output")
    parser.add_argument("--n-folds",     type=int,   default=5)
    parser.add_argument("--n-bootstrap", type=int,   default=2000,
                        help="Bootstrap samples for pairwise p-values (0 = skip).")
    parser.add_argument("--head-dropout", type=float, default=0.5)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--num-workers",  type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    os.makedirs(args.output_folder, exist_ok=True)

    base_folder   = args.base_folder
    mapping_file  = os.path.join(base_folder, "mapping.csv")
    metadata_file = os.path.join(base_folder, "dichotome_data_anonymized_with_patientID.csv")
    project_dir   = os.path.dirname(os.path.abspath(__file__))

    fold_folder = args.fold_folder or os.path.join(project_dir, "splits")
    ckpt_folder = args.ckpt_folder or os.path.join(project_dir, "logs")

    fold_csv = os.path.join(
        fold_folder,
        os.path.basename(metadata_file).replace(".csv", "_stratified_folds.csv"),
    )
    if not os.path.isfile(fold_csv):
        raise SystemExit(f"Splits CSV not found: {fold_csv}\nRun main.py first.")

    metadata = pd.read_csv(fold_csv)
    metadata = _ensure_file_id(metadata, mapping_file)
    metadata = metadata[metadata["fileID"] != -1].reset_index(drop=True)

    # Derived presence columns (same logic as main.py / ensemble.py)
    for derived, source in {
        "diffuse_pleural_thickening_presence":  "diffuse_pleural_thickening_width",
        "localized_pleural_thickening_presence": "localized_pleural_thickening_width",
    }.items():
        if source in metadata.columns and derived not in metadata.columns:
            metadata[derived] = metadata[source].notna().astype(int)

    # -----------------------------------------------------------------------
    # Per-model inference
    # -----------------------------------------------------------------------
    # Structure: model_name → fold_idx → task → {prob, y_true, mask, threshold}
    all_preds:     Dict[str, Dict[int, Dict[str, dict]]] = {}
    all_thresholds: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}

    for model_name in args.models:
        print(f"\n{'='*70}")
        print(f"Model: {_MODEL_DISPLAY.get(model_name, model_name)}")
        print(f"{'='*70}")

        _norm = (
            Normalize(mean=[0.5], std=[1.0 / 2048.0])
            if model_name == "chexnet"
            else Normalize(mean=[0.5], std=[0.5])
        )
        preprocess = Compose([Resize((224, 224)), ToImage(), ToDtype(torch.float32, scale=True), _norm])

        fold_preds:  Dict[int, Dict[str, dict]] = {}
        fold_thrs:   Dict[int, Dict[str, Optional[float]]] = {}

        for fold_idx in range(args.n_folds):
            ckpt_path = os.path.join(
                ckpt_folder, f"best_{model_name}_labels=multitask_fold={fold_idx}.pth"
            )
            model, label_cols, val_thresholds = _load_checkpoint(
                ckpt_path, model_name, device, args.head_dropout
            )
            if model is None:
                print(f"  Fold {fold_idx}: checkpoint not found — skipping")
                continue

            # Warn when threshold is missing (fold trained without early-stopping)
            if not val_thresholds:
                print(
                    f"  Fold {fold_idx}: WARNING — no validation threshold in checkpoint. "
                    "Sensitivity/specificity will use Youden fallback on the test set "
                    "(slightly optimistic)."
                )

            fold_col = f"Fold{fold_idx}"
            if fold_col not in metadata.columns:
                print(f"  Fold {fold_idx}: '{fold_col}' not in metadata — skipping")
                continue

            test_meta = metadata[metadata[fold_col] == "test"].copy()
            test_meta = _filter_rows_with_images(test_meta, base_folder)
            if test_meta.empty:
                print(f"  Fold {fold_idx}: no test images found — skipping")
                continue

            task_value_maps: Dict[str, Dict] = {}
            for col in label_cols:
                vm, _ = _binary_value_map_from_series(metadata[col], task=col)
                task_value_maps[col] = vm

            dataset = XRayMultiTaskDataset(
                test_meta[["fileID"] + label_cols], base_folder,
                label_columns=label_cols,
                label_value_maps=task_value_maps,
                transform=preprocess,
            )
            loader = DataLoader(
                dataset, sampler=SequentialSampler(dataset),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
            )

            n_with_thr = sum(1 for v in val_thresholds.values() if v is not None)
            print(
                f"  Fold {fold_idx}: {len(dataset)} test samples, "
                f"{len(label_cols)} labels, "
                f"{n_with_thr}/{len(label_cols)} val-thresholds loaded"
            )

            preds = _infer(model, loader, label_cols, device)
            fold_preds[fold_idx] = preds

            # Store val-set threshold per task for this fold
            fold_thrs[fold_idx] = {}
            for task in label_cols:
                fold_thrs[fold_idx][task] = val_thresholds.get(task)

            # Print per-task metrics for this fold
            for task in label_cols:
                p   = preds[task]
                thr = fold_thrs[fold_idx][task]
                m   = _fold_metrics(p["y_true"], p["prob"], thr, task)
                if m:
                    src = "" if m["threshold_source"] == "val" else " [FALLBACK]"
                    print(
                        f"    [{task}] n={m['n']} pos={m['n_pos']}  "
                        f"AUC={m['auc']:.3f}  "
                        f"Sens={m['sensitivity']:.3f}  "
                        f"Spec={m['specificity']:.3f}  "
                        f"thr={m['threshold']:.3f}{src}"
                    )

            # Free GPU memory between folds
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        all_preds[model_name]      = fold_preds
        all_thresholds[model_name] = fold_thrs

    # -----------------------------------------------------------------------
    # Aggregate: mean ± SD across folds
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("SUMMARY — Mean ± SD across folds  (at validation-set threshold)")
    print(f"{'='*70}")

    summary_rows = []
    # model_name → task → aggregated stats (for ROC plot)
    model_task_summary: Dict[str, Dict[str, dict]] = {}

    for model_name in args.models:
        fold_preds = all_preds.get(model_name, {})
        fold_thrs  = all_thresholds.get(model_name, {})
        if not fold_preds:
            print(f"\n{model_name}: no folds available, skipping.")
            continue

        first_fold = next(iter(fold_preds.values()))
        label_cols = list(first_fold.keys())
        model_task_summary[model_name] = {}

        print(f"\n{_MODEL_DISPLAY.get(model_name, model_name)}:")

        for task in label_cols:
            fold_metric_list = []
            for fold_idx, preds in fold_preds.items():
                if task not in preds:
                    continue
                thr = (fold_thrs.get(fold_idx) or {}).get(task)
                m = _fold_metrics(preds[task]["y_true"], preds[task]["prob"], thr, task)
                if m:
                    fold_metric_list.append((fold_idx, m))

            if not fold_metric_list:
                continue

            aucs  = [m["auc"]         for _, m in fold_metric_list]
            senss = [m["sensitivity"] for _, m in fold_metric_list]
            specs = [m["specificity"] for _, m in fold_metric_list]
            n_folds_used = len(fold_metric_list)
            ddof = 1 if n_folds_used > 1 else 0

            auc_mean  = float(np.mean(aucs));   auc_sd  = float(np.std(aucs,  ddof=ddof))
            sens_mean = float(np.mean(senss));  sens_sd = float(np.std(senss, ddof=ddof))
            spec_mean = float(np.mean(specs));  spec_sd = float(np.std(specs, ddof=ddof))

            fallback_folds = [
                fi for fi, m in fold_metric_list
                if m["threshold_source"] != "val"
            ]

            thr_note = ""
            if fallback_folds:
                thr_note = f"  [Youden fallback in folds {fallback_folds}]"

            print(
                f"  {task:45s}  "
                f"AUC  {auc_mean:.3f} ± {auc_sd:.3f}  |  "
                f"Sens {sens_mean:.3f} ± {sens_sd:.3f}  |  "
                f"Spec {spec_mean:.3f} ± {spec_sd:.3f}"
                f"{thr_note}"
            )

            model_task_summary[model_name][task] = {
                "auc_mean":  auc_mean,  "auc_sd":  auc_sd,
                "sens_mean": sens_mean, "sens_sd": sens_sd,
                "spec_mean": spec_mean, "spec_sd": spec_sd,
            }

            for fold_idx, m in fold_metric_list:
                summary_rows.append({
                    "model":            model_name,
                    "label":            task,
                    "fold":             fold_idx,
                    "auc":              m["auc"],
                    "sensitivity":      m["sensitivity"],
                    "specificity":      m["specificity"],
                    "threshold":        m["threshold"],
                    "threshold_source": m["threshold_source"],
                    "n":                m["n"],
                    "n_pos":            m["n_pos"],
                    "n_neg":            m["n_neg"],
                })

    # Save per-fold CSV
    if summary_rows:
        df_folds = pd.DataFrame(summary_rows)
        path_folds = os.path.join(args.output_folder, "metrics_per_fold.csv")
        df_folds.to_csv(path_folds, index=False)
        print(f"\nSaved: {path_folds}")

    # Save aggregate summary
    agg_rows = []
    for model_name, task_dict in model_task_summary.items():
        for task, s in task_dict.items():
            agg_rows.append({"model": model_name, "label": task, **s})
    if agg_rows:
        df_agg = pd.DataFrame(agg_rows)
        path_agg = os.path.join(args.output_folder, "metrics_summary.csv")
        df_agg.to_csv(path_agg, index=False)
        print(f"Saved: {path_agg}")

    # -----------------------------------------------------------------------
    # Pairwise bootstrap p-values
    # -----------------------------------------------------------------------
    if args.n_bootstrap > 0 and len(args.models) >= 2:
        print(f"\n\n{'='*70}")
        print(
            f"Pairwise bootstrap p-values  (n={args.n_bootstrap}, two-sided)\n"
            "* p < 0.05   ** p < 0.01   ns = not significant"
        )
        print(f"{'='*70}")

        def _pool_preds(model_name: str, task: str) -> Tuple[np.ndarray, np.ndarray]:
            """Pool test predictions for *task* across all folds of *model_name*."""
            yt_all, yp_all = [], []
            for fold_idx, preds in all_preds.get(model_name, {}).items():
                if task not in preds:
                    continue
                p = preds[task]
                mask = p["mask"]
                yt_all.append(p["y_true"][mask])
                yp_all.append(p["prob"][mask])
            if not yt_all:
                return np.array([]), np.array([])
            return np.concatenate(yt_all), np.concatenate(yp_all)

        model_pairs = [
            (args.models[i], args.models[j])
            for i in range(len(args.models))
            for j in range(i + 1, len(args.models))
        ]
        all_tasks = []
        for mn in args.models:
            if mn in model_task_summary:
                all_tasks = list(model_task_summary[mn].keys())
                break

        pw_rows = []
        for task in all_tasks:
            print(f"\n  [{task}]")
            for model_a, model_b in model_pairs:
                yt_a, yp_a = _pool_preds(model_a, task)
                yt_b, yp_b = _pool_preds(model_b, task)

                if len(yt_a) == 0 or len(yt_b) == 0:
                    print(f"    {model_a} vs {model_b}: no predictions available")
                    continue

                # Both models are evaluated on the same fold test sets, so pooled
                # y_true should be identical.  If lengths differ (one model
                # skipped a fold), truncate to the shorter set with a warning.
                if len(yt_a) != len(yt_b):
                    n = min(len(yt_a), len(yt_b))
                    print(
                        f"    WARNING: {model_a} ({len(yt_a)}) and {model_b} ({len(yt_b)}) "
                        f"have different sample counts — truncating to {n}."
                    )
                    yt_a, yp_a = yt_a[:n], yp_a[:n]
                    yt_b, yp_b = yt_b[:n], yp_b[:n]

                obs_diff, p_val, se = _bootstrap_pvalue(
                    yt_a, yp_a, yp_b,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
                print(
                    f"    {model_a:20s} vs {model_b:20s}  "
                    f"ΔAUC (A−B) = {obs_diff:+.4f}  "
                    f"p = {p_val:.4f}  SE = {se:.4f}  [{sig}]"
                )
                pw_rows.append({
                    "label":             task,
                    "model_A":           model_a,
                    "model_B":           model_b,
                    "delta_AUC_A_minus_B": obs_diff,
                    "p_value":           p_val,
                    "SE":                se,
                })

        if pw_rows:
            df_pw = pd.DataFrame(pw_rows)
            path_pw = os.path.join(args.output_folder, "pairwise_pvalues.csv")
            df_pw.to_csv(path_pw, index=False)
            print(f"\nSaved: {path_pw}")

    # -----------------------------------------------------------------------
    # ROC figures
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("Generating ROC figures...")
    print(f"{'='*70}")

    for model_name in args.models:
        fold_preds = all_preds.get(model_name, {})
        fold_thrs  = all_thresholds.get(model_name, {})
        if not fold_preds:
            print(f"  {model_name}: no predictions, skipping.")
            continue

        first_fold  = next(iter(fold_preds.values()))
        label_cols  = list(first_fold.keys())

        # Pool all fold predictions for the ROC curve
        pooled_true: Dict[str, np.ndarray] = {}
        pooled_prob: Dict[str, np.ndarray] = {}
        pooled_thr:  Dict[str, Optional[float]] = {}

        for task in label_cols:
            yt_parts, yp_parts, thr_parts = [], [], []
            for fold_idx, preds in fold_preds.items():
                if task not in preds:
                    continue
                p = preds[task]
                yt_parts.append(p["y_true"])
                yp_parts.append(p["prob"])
                t = (fold_thrs.get(fold_idx) or {}).get(task)
                if t is not None:
                    thr_parts.append(t)

            if yt_parts:
                pooled_true[task] = np.concatenate(yt_parts)
                pooled_prob[task] = np.concatenate(yp_parts)
                # Use the mean val-threshold across folds as operating point on pooled curve
                pooled_thr[task] = float(np.mean(thr_parts)) if thr_parts else None

        summary = model_task_summary.get(model_name, {})
        _plot_roc(model_name, pooled_true, pooled_prob, pooled_thr, summary, args.output_folder)
    plot_auc_by_model(all_preds, args.output_folder)
    print("\nDone.")


if __name__ == "__main__":
    main()
