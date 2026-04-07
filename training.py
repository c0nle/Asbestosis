"""
training.py
-----------
Training and evaluation loop for the asbestosis multi-task pipeline.

Public API
----------
run_epoch             : Run one training or evaluation epoch.
log_multitask_metrics : Compute and log per-task and macro classification metrics.
_print_focus_line     : Print a compact per-epoch summary for selected tasks.
_print_focus_confusions: Print per-task confusion matrix stats for focus tasks.
"""

import gc
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.amp import GradScaler

from model import MultiTaskModel
from utils import _wandb_is_active, _wandb_log


def run_epoch(
    model: MultiTaskModel,
    optimizer,
    lr_scheduler,
    criterion: Dict[str, nn.Module],
    scaler: GradScaler,
    data_loader,
    device: str,
    train: bool,
    max_steps: Optional[int],
    backbone_tasks: Optional[set] = None,
    head_only_loss_weight: float = 0.25,
    task_weights: Optional[Dict[str, float]] = None,
    grad_clip: float = 0.0,
    label_smoothing: float = 0.0,
) -> Tuple[Dict[str, dict], float]:
    """
    Run one training or evaluation epoch over the data loader.

    In training mode the multi-task BCE loss is back-propagated and the
    optimizer + LR scheduler are stepped.  Tasks in ``backbone_tasks``
    contribute full gradients to the backbone; all other tasks have their
    backbone gradients detached and their loss down-weighted by
    ``head_only_loss_weight`` (their heads still train normally).

    Args:
        model:                Shared-backbone multi-task model.
        optimizer:            AdamW optimizer (ignored during eval).
        lr_scheduler:         OneCycleLR scheduler stepped per batch (ignored
                              during eval; pass ``None``).
        criterion:            Dict mapping task name → ``BCEWithLogitsLoss``.
        scaler:               AMP ``GradScaler`` (enabled on CUDA only).
        data_loader:          DataLoader yielding
                              ``(image, targets_dict, masks_dict, path)``.
        device:               ``"cuda"`` or ``"cpu"``.
        train:                ``True`` for training mode, ``False`` for eval.
        max_steps:            Stop early after this many batches
                              (``None`` = full epoch).
        backbone_tasks:       Set of task names whose loss flows through the
                              backbone.  ``None`` means all tasks do.
        head_only_loss_weight: Loss multiplier for head-only tasks.
        task_weights:         Optional per-task loss multiplier dict.
        grad_clip:            Max gradient norm for clipping (0 = disabled).
        label_smoothing:      Epsilon for binary label smoothing during training
                              (0 = disabled).  Applied before loss computation.

    Returns:
        Tuple of:

        - ``results_by_task``: dict mapping task name →
          ``{y_true, y_prob, y_mask}`` CPU tensors.
        - ``avg_epoch_loss``: mean batch loss for the epoch.
    """
    if train:
        model.train()
        suffix = "train"
    else:
        model.eval()
        suffix = "eval"

    epoch_loss = 0.0
    n_steps = 0
    y_true_by_task: Dict[str, List[torch.Tensor]] = {}
    y_prob_by_task: Dict[str, List[torch.Tensor]] = {}
    y_mask_by_task: Dict[str, List[torch.Tensor]] = {}

    autocast_enabled = device == "cuda"
    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    for step, (data, targets, masks, path) in enumerate(data_loader):
        if max_steps is not None and step >= max_steps:
            break
        if step % 100 == 0:
            gc.collect()

        data = data.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        masks = {k: v.to(device) for k, v in masks.items()}

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
            feats = model.backbone(data)
            feats_detached = feats.detach()
            logits = {}
            for task in criterion.keys():
                use_feats = feats if (backbone_tasks is None or task in backbone_tasks) else feats_detached
                logits[task] = model.heads[task](use_feats).view(-1)
            loss_num = torch.tensor(0.0, device=device)
            weight_sum = 0.0
            for task, loss_fn in criterion.items():
                task_logits = logits[task].view(-1)
                task_target = targets[task].view(-1).float()
                if train and label_smoothing > 0.0:
                    task_target = task_target * (1.0 - label_smoothing) + (1.0 - task_target) * label_smoothing
                task_mask = masks[task].view(-1)
                per = loss_fn(task_logits, task_target)
                per = per * task_mask
                denom = task_mask.sum().clamp(min=1.0)
                task_loss = per.sum() / denom
                weight = 1.0
                if backbone_tasks is not None and task not in backbone_tasks:
                    weight = float(head_only_loss_weight)
                if task_weights is not None and task in task_weights:
                    weight = weight * float(task_weights[task])
                loss_num = loss_num + task_loss * weight
                weight_sum += weight
            loss = loss_num / max(1.0, weight_sum)

        if train:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        with torch.no_grad():
            for task in criterion.keys():
                probs = torch.sigmoid(logits[task]).view(-1)
                y_true_by_task.setdefault(task, []).append(targets[task].detach().cpu().view(-1))
                y_prob_by_task.setdefault(task, []).append(probs.detach().cpu().view(-1))
                y_mask_by_task.setdefault(task, []).append(masks[task].detach().cpu().view(-1))

        epoch_loss += float(loss.item())
        n_steps += 1

    avg_epoch_loss = epoch_loss / max(1, n_steps)
    _wandb_log({f"average_loss/{suffix}": avg_epoch_loss})
    results = {}
    for task in y_true_by_task.keys():
        if not y_true_by_task[task]:
            continue
        results[task] = {
            "y_true": torch.cat(y_true_by_task[task], dim=0),
            "y_prob": torch.cat(y_prob_by_task[task], dim=0),
            "y_mask": torch.cat(y_mask_by_task[task], dim=0),
        }
    return results, avg_epoch_loss


def log_multitask_metrics(
    results_by_task: dict,
    avg_loss: float,
    suffix: str,
    epoch: int,
    task_index_to_label: Dict[str, List[str]],
    fixed_thresholds: Optional[Dict[str, float]] = None,
    compute_thresholds: bool = False,
    wandb_detail: str = "compact",
    threshold_strategy: str = "fbeta",
    fbeta: float = 2.0,
    target_precision: float = 0.5,
    n_bootstrap: int = 0,
    log_curves: bool = False,
) -> dict:
    """
    Compute and log per-task and macro-averaged classification metrics.

    For each task computes: accuracy, balanced accuracy, precision, recall,
    F1, ROC-AUC, and PR-AUC.  Optionally searches for optimal decision
    thresholds on the precision-recall curve using one of three strategies:

    - ``"f1"``                 – maximise F1.
    - ``"fbeta"``              – maximise F-beta (default beta = 2, recall-heavy).
    - ``"recall_at_precision"``– maximise recall subject to precision ≥ target.

    Logs to W&B when a run is active.  In ``"compact"`` mode only key summary
    metrics are sent; ``"full"`` mode additionally logs per-task confusion
    matrices.

    Args:
        results_by_task:     Dict from :func:`run_epoch`.
        avg_loss:            Mean epoch loss for the split.
        suffix:              Split identifier, e.g. ``"train"``, ``"eval"``,
                             ``"final_test"``, ``"best_test"``.
        epoch:               Current epoch index (for W&B x-axis).
        task_index_to_label: Dict mapping task name → ``["neg_label", "pos_label"]``.
        fixed_thresholds:    Pre-computed per-task thresholds to evaluate.
        compute_thresholds:  Search the PR curve for the best threshold.
        wandb_detail:        ``"compact"`` or ``"full"`` W&B logging verbosity.
        threshold_strategy:  One of ``"f1"``, ``"fbeta"``,
                             ``"recall_at_precision"``.
        fbeta:               Beta for the F-beta threshold search.
        target_precision:    Minimum precision constraint for
                             ``"recall_at_precision"``.

    Returns:
        Dict of all computed metric values (keys follow the pattern
        ``"metric/task/suffix"`` or ``"macro_metric/suffix"``).
    """
    metrics_all: dict = {
        f"epoch/{suffix}": epoch,
        f"average_loss/{suffix}": float(avg_loss),
    }

    per_task_f1 = []
    per_task_auc = []
    per_task_pr_auc = []
    per_task_acc = []

    def _should_log_task_metrics() -> bool:
        if wandb_detail != "full":
            return False
        return suffix in {"eval", "final_test", "best_test"}

    for task, data in results_by_task.items():
        y_true = data["y_true"].detach().cpu().numpy()
        y_prob = data["y_prob"].detach().cpu().numpy()
        y_mask = data["y_mask"].detach().cpu().numpy().astype(bool)
        if y_true.size == 0 or y_mask.sum() == 0:
            continue

        y_true = y_true[y_mask].astype(int)
        scores = y_prob.reshape(-1)[y_mask]
        unique_true = np.unique(y_true)

        y_pred = (scores >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if unique_true.size >= 2 else float("nan")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0,
        )

        auc_val = float("nan")
        if unique_true.size >= 2:
            try:
                auc_val = float(roc_auc_score(y_true, scores))
            except Exception:
                auc_val = float("nan")

        pr_auc_val = float("nan")
        if unique_true.size >= 2:
            try:
                pr_auc_val = float(average_precision_score(y_true, scores))
            except Exception:
                pr_auc_val = float("nan")

        task_metrics = {
            f"task/{task}/accuracy/{suffix}": acc,
            f"task/{task}/balanced_accuracy/{suffix}": bal_acc,
            f"task/{task}/precision/{suffix}": float(precision),
            f"task/{task}/recall/{suffix}": float(recall),
            f"task/{task}/f1/{suffix}": float(f1),
            f"task/{task}/auc/{suffix}": float(auc_val),
            f"task/{task}/pr_auc/{suffix}": float(pr_auc_val),
        }

        if compute_thresholds and unique_true.size >= 2:
            try:
                pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true, scores)
                if pr_thr.size > 0:
                    pr_f1 = (2 * pr_prec[:-1] * pr_rec[:-1]) / (pr_prec[:-1] + pr_rec[:-1] + 1e-12)
                    best_f1_idx = int(np.nanargmax(pr_f1))
                    task_metrics[f"task/{task}/best_threshold_f1/{suffix}"] = float(pr_thr[best_f1_idx])

                    beta2 = float(fbeta) ** 2
                    pr_fbeta = (
                        (1 + beta2) * pr_prec[:-1] * pr_rec[:-1]
                        / (beta2 * pr_prec[:-1] + pr_rec[:-1] + 1e-12)
                    )
                    best_fbeta_idx = int(np.nanargmax(pr_fbeta))
                    task_metrics[f"task/{task}/best_threshold_fbeta/{suffix}"] = float(pr_thr[best_fbeta_idx])

                    p_mask = pr_prec[:-1] >= float(target_precision)
                    if np.any(p_mask):
                        cand = np.where(p_mask)[0]
                        best_idx = int(cand[np.nanargmax(pr_rec[:-1][cand])])
                        task_metrics[
                            f"task/{task}/best_threshold_recall_at_p{float(target_precision):.2f}/{suffix}"
                        ] = float(pr_thr[best_idx])
            except Exception:
                pass

        _is_test_suffix = suffix in {"final_test", "best_test"}

        # --- Bootstrap CI for AUC ---
        if n_bootstrap > 0 and unique_true.size >= 2 and _is_test_suffix:
            rng = np.random.default_rng(seed=42)
            boot_aucs = []
            n = len(y_true)
            for _ in range(int(n_bootstrap)):
                idx = rng.integers(0, n, size=n)
                yt, ys_ = y_true[idx], scores[idx]
                if np.unique(yt).size < 2:
                    continue
                try:
                    boot_aucs.append(float(roc_auc_score(yt, ys_)))
                except Exception:
                    pass
            if len(boot_aucs) >= 10:
                ci_lo = float(np.percentile(boot_aucs, 2.5))
                ci_hi = float(np.percentile(boot_aucs, 97.5))
                task_metrics[f"task/{task}/auc_ci_lo/{suffix}"] = ci_lo
                task_metrics[f"task/{task}/auc_ci_hi/{suffix}"] = ci_hi
                print(
                    f"  bootstrap ({len(boot_aucs)} samples) {task}: "
                    f"auc={float(auc_val):.3f} 95%CI=[{ci_lo:.3f}, {ci_hi:.3f}]"
                )
                if _wandb_is_active():
                    try:
                        wandb.log(
                            {f"auc_bootstrap/{suffix}/{task}": wandb.Histogram(np.array(boot_aucs))},
                            commit=False,
                        )
                    except Exception:
                        pass

        # --- ROC and PR curves ---
        if log_curves and _wandb_is_active() and unique_true.size >= 2 and _is_test_suffix:
            try:
                # ROC curve (downsample to ≤300 points for W&B)
                fpr, tpr, _ = roc_curve(y_true, scores)
                step = max(1, len(fpr) // 300)
                fpr_ds = fpr[::step].tolist()
                tpr_ds = tpr[::step].tolist()
                # ensure endpoints
                if fpr_ds[-1] != 1.0:
                    fpr_ds.append(1.0); tpr_ds.append(1.0)
                wandb.log(
                    {
                        f"roc_curve/{suffix}/{task}": wandb.plot.line_series(
                            xs=[fpr_ds, [0.0, 1.0]],
                            ys=[tpr_ds, [0.0, 1.0]],
                            keys=["Model", "Random"],
                            title=f"ROC – {task} ({suffix})",
                            xname="False Positive Rate",
                        )
                    },
                    commit=False,
                )
            except Exception:
                pass

            try:
                # PR curve with prevalence baseline
                pr_prec_c, pr_rec_c, _ = precision_recall_curve(y_true, scores)
                prevalence = float(y_true.mean())
                step = max(1, len(pr_rec_c) // 300)
                rec_ds  = pr_rec_c[::step].tolist()
                prec_ds = pr_prec_c[::step].tolist()
                wandb.log(
                    {
                        f"pr_curve/{suffix}/{task}": wandb.plot.line_series(
                            xs=[rec_ds, [0.0, 1.0]],
                            ys=[prec_ds, [prevalence, prevalence]],
                            keys=["Model", f"Baseline ({prevalence:.1%})"],
                            title=f"PR – {task} ({suffix})",
                            xname="Recall",
                        )
                    },
                    commit=False,
                )
            except Exception:
                pass

        if fixed_thresholds and task in fixed_thresholds and fixed_thresholds[task] is not None:
            thr = float(fixed_thresholds[task])
            y_pred_fx = (scores >= thr).astype(int)
            pfx, rfx, ffx, _ = precision_recall_fscore_support(
                y_true, y_pred_fx, average="binary", zero_division=0
            )
            task_metrics.update({
                f"task/{task}/fixed_threshold/{suffix}": thr,
                f"task/{task}/f1@fixed_thr/{suffix}": float(ffx),
                f"task/{task}/precision@fixed_thr/{suffix}": float(pfx),
                f"task/{task}/recall@fixed_thr/{suffix}": float(rfx),
            })

        metrics_all.update(task_metrics)
        per_task_f1.append(float(f1))
        per_task_auc.append(float(auc_val) if np.isfinite(auc_val) else float("nan"))
        per_task_pr_auc.append(float(pr_auc_val) if np.isfinite(pr_auc_val) else float("nan"))
        per_task_acc.append(float(acc))

        if _wandb_is_active() and _should_log_task_metrics():
            try:
                class_names = task_index_to_label.get(task, ["0", "1"])
                wandb.log(
                    {
                        f"confusion_matrix/{suffix}/{task}": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=y_true,
                            preds=y_pred,
                            class_names=class_names,
                        )
                    },
                    commit=False,
                )
            except Exception:
                pass

    metrics_all[f"macro_accuracy/{suffix}"] = float(np.nanmean(per_task_acc)) if per_task_acc else float("nan")
    metrics_all[f"macro_f1/{suffix}"] = float(np.nanmean(per_task_f1)) if per_task_f1 else float("nan")
    metrics_all[f"macro_auc/{suffix}"] = float(np.nanmean(per_task_auc)) if per_task_auc else float("nan")
    metrics_all[f"macro_pr_auc/{suffix}"] = float(np.nanmean(per_task_pr_auc)) if per_task_pr_auc else float("nan")

    if str(wandb_detail) == "full":
        wandb_metrics = dict(metrics_all)
    else:
        # Compact mode: only send key ranking metrics to keep W&B clean.
        wandb_metrics = {
            f"epoch/{suffix}": epoch,
            f"average_loss/{suffix}": float(avg_loss),
            f"macro_f1/{suffix}": metrics_all.get(f"macro_f1/{suffix}", float("nan")),
            f"macro_auc/{suffix}": metrics_all.get(f"macro_auc/{suffix}", float("nan")),
            f"macro_pr_auc/{suffix}": metrics_all.get(f"macro_pr_auc/{suffix}", float("nan")),
        }
        for task in results_by_task.keys():
            for k in (f"task/{task}/auc/{suffix}", f"task/{task}/pr_auc/{suffix}", f"task/{task}/f1/{suffix}"):
                if k in metrics_all:
                    wandb_metrics[k] = metrics_all[k]
    _wandb_log(wandb_metrics)
    print(
        f"{suffix} macro: "
        f"loss={metrics_all.get(f'average_loss/{suffix}', float('nan')):.4f} "
        f"macro_auc={metrics_all.get(f'macro_auc/{suffix}', float('nan')):.3f} "
        f"macro_pr_auc={metrics_all.get(f'macro_pr_auc/{suffix}', float('nan')):.3f}"
    )
    return metrics_all


def _print_focus_line(
    metrics: dict,
    suffix: str,
    epoch: int,
    focus_labels: List[str],
) -> None:
    """Print a compact per-epoch AUC / PR-AUC line for the focus labels."""
    if not focus_labels:
        return
    parts = []
    for lab in focus_labels:
        auc = metrics.get(f"task/{lab}/auc/{suffix}")
        pr_auc = metrics.get(f"task/{lab}/pr_auc/{suffix}")
        auc_s = f"{float(auc):.3f}" if auc is not None and np.isfinite(auc) else "nan"
        pr_auc_s = f"{float(pr_auc):.3f}" if pr_auc is not None and np.isfinite(pr_auc) else "nan"
        parts.append(f"{lab} auc={auc_s} pr_auc={pr_auc_s}")
    print(f"{suffix} focus (epoch {epoch}): " + " | ".join(parts))


def _print_focus_confusions(
    results_by_task: dict,
    thresholds: Optional[Dict[str, Optional[float]]],
    suffix: str,
    epoch: int,
    focus_labels: List[str],
) -> None:
    """Print per-task confusion matrix stats (TP/FP/TN/FN) for focus labels."""
    if not focus_labels:
        return
    lines = []
    for task in focus_labels:
        data = results_by_task.get(task)
        if not data:
            continue
        y_true = data["y_true"].detach().cpu().numpy()
        y_prob = data["y_prob"].detach().cpu().numpy()
        y_mask = data["y_mask"].detach().cpu().numpy().astype(bool)
        if y_true.size == 0 or y_mask.sum() == 0:
            continue
        y_true = y_true[y_mask].astype(int)
        scores = y_prob.reshape(-1)[y_mask]
        if np.unique(y_true).size < 2:
            continue
        thr_raw = thresholds.get(task) if thresholds else None
        thr = 0.5 if thr_raw is None else float(thr_raw)
        y_pred = (scores >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        lines.append(
            f"{task} thr={thr:.3f} tp={tp} fp={fp} tn={tn} fn={fn} "
            f"prec={float(precision):.3f} rec={float(recall):.3f} f1={float(f1):.3f}"
        )
    if lines:
        print(f"{suffix} focus confusion (epoch {epoch}): " + " | ".join(lines))
