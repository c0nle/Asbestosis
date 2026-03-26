import argparse
import functools
import gc
import glob
import io
import os
import random
import tempfile
import warnings
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import PIL.Image
import pydicom
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
from torch import nn, optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, WeightedRandomSampler
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    Normalize,
    RandomRotation,
    Resize,
    ToDtype,
    ToImage,
)

import Preprocessor_Metadata


def _wandb_is_active() -> bool:
    return getattr(wandb, "run", None) is not None


def _wandb_log(data: dict, commit: bool = True) -> None:
    if _wandb_is_active():
        wandb.log(data, commit=commit)


def _is_missing_label_value(value) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and np.isnan(value):
            return True
    except Exception:
        pass
    if value == -1:
        return True
    if isinstance(value, str):
        v = value.strip()
        if v == "" or v.lower() in {"nan", "none"}:
            return True
        if v in {"-1", "-1.0"}:
            return True
    return False


def _canonical_label_value(value):
    if _is_missing_label_value(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if np.isfinite(fv) and fv.is_integer():
            return int(fv)
        return fv
    return str(value).strip()


def _coerce_binary_label(value) -> Optional[int]:
    """
    Convert common binary encodings to {0,1}. Returns None if not coercible.
    """
    if _is_missing_label_value(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return 1 if bool(value) else 0
    if isinstance(value, (int, np.integer)):
        iv = int(value)
        return iv if iv in (0, 1) else None
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if np.isfinite(fv) and fv.is_integer():
            iv = int(fv)
            return iv if iv in (0, 1) else None
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"0", "0.0", "false", "no"}:
            return 0
        if v in {"1", "1.0", "true", "yes"}:
            return 1
        # numeric strings like "0.00" / "1.00"
        try:
            fv = float(v)
            if np.isfinite(fv) and float(fv).is_integer() and int(fv) in (0, 1):
                return int(fv)
        except Exception:
            pass
    return None


def _missing_mask(series: pd.Series) -> pd.Series:
    s = series
    mask = s.isna()
    try:
        mask = mask | (s == -1)
    except Exception:
        pass
    try:
        ss = s.astype("string")
        mask = mask | ss.str.strip().isin(["", "nan", "none", "-1", "-1.0"])
    except Exception:
        pass
    return mask


def print_label_stats(metadata: pd.DataFrame, label_cols: List[str]) -> None:
    total = len(metadata)
    print(f"=== Label stats (rows={total}) ===")
    for col in label_cols:
        if col not in metadata.columns:
            print(f"- {col}: missing column")
            continue
        s = metadata[col]
        miss = _missing_mask(s)
        present = int((~miss).sum())
        missing_n = int(miss.sum())
        uniq = int(s[~miss].astype("string").nunique()) if present else 0
        head = s[~miss].astype("string").value_counts().head(8) if present else None
        if head is not None and len(head) > 0:
            top = ", ".join([f"{idx}={int(cnt)}" for idx, cnt in head.items()])
        else:
            top = ""
        print(f"- {col}: present={present} missing={missing_n} uniq={uniq}" + (f" top: {top}" if top else ""))


def _ensure_file_id(metadata: pd.DataFrame, mapping_file: str) -> pd.DataFrame:
    has_file_id = "fileID" in metadata.columns and (metadata["fileID"] != -1).any()
    if has_file_id:
        return metadata

    mapping = pd.read_csv(mapping_file, dtype={"medicoID": str})
    mapping["fileID"] = pd.to_numeric(mapping.get("fileID"), errors="coerce")
    mapping["Anforderungsnummer"] = mapping["medicoID"].str[:-2]
    mask = mapping["Anforderungsnummer"].str.fullmatch(r"\d+").eq(True)
    mapping = mapping[mask]
    mapping["Anforderungsnummer"] = mapping["Anforderungsnummer"].astype(int)

    merged = metadata.drop(columns=["fileID"], errors="ignore").merge(
        mapping[["Anforderungsnummer", "fileID"]],
        on="Anforderungsnummer",
        how="left",
    )
    merged["fileID"] = merged["fileID"].fillna(-1).astype(int)
    return merged


def _ensure_writable_dir(path: str, fallback: Optional[str] = None) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write_test_", delete=True):
            pass
        return path
    except Exception:
        if fallback is None:
            raise
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _read_dicom_from_bytes(data: bytes) -> PIL.Image.Image:
    dicom_ds = pydicom.dcmread(io.BytesIO(data), force=True)
    pixel_array = dicom_ds.pixel_array
    try:
        from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

        try:
            pixel_array = apply_modality_lut(pixel_array, dicom_ds)
        except Exception:
            pass
        try:
            pixel_array = apply_voi_lut(pixel_array, dicom_ds)
        except Exception:
            pass
    except Exception:
        pass

    pixel_array = np.asarray(pixel_array).astype(np.float32)
    try:
        if str(getattr(dicom_ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            pixel_array = float(np.nanmax(pixel_array)) - pixel_array
    except Exception:
        pass
    min_val = np.percentile(pixel_array, 0.5)
    max_val = np.percentile(pixel_array, 99.5)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    normalized = np.clip((pixel_array - min_val) / denom, 0, 1)
    normalized = (normalized * 255).astype(np.uint8)
    return PIL.Image.fromarray(normalized).convert("L")


def _read_image_from_zip(zip_path: str) -> PIL.Image.Image:
    with zipfile.ZipFile(zip_path, "r") as zip_dir:
        candidates = [
            name
            for name in zip_dir.namelist()
            if (name.endswith("IM_0001") or (os.sep + "IM_0001") in name or "IM_" in name)
        ]
        if not candidates:
            candidates = [name for name in zip_dir.namelist() if not name.endswith("/")]
        if not candidates:
            raise FileNotFoundError(f"No files found inside zip: {zip_path}")

        name = candidates[0]
        with zip_dir.open(name) as f:
            data = f.read()

        try:
            return _read_dicom_from_bytes(data)
        except Exception:
            return PIL.Image.open(io.BytesIO(data)).convert("L")


def _read_xray_image(path: str) -> PIL.Image.Image:
    lower = path.lower()
    if lower.endswith(".zip"):
        return _read_image_from_zip(path)
    return PIL.Image.open(path).convert("L")


def _zip_has_files(zip_path: str) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_dir:
            return any(not name.endswith("/") for name in zip_dir.namelist())
    except Exception:
        return False


@functools.lru_cache(maxsize=200_000)
def _find_valid_xray_path(img_dir: str, file_id: str) -> Optional[str]:
    png_dir = os.path.join(img_dir, "png")
    anon_dir = os.path.join(img_dir, "anon")
    candidate_roots = []
    if os.path.isdir(png_dir):
        candidate_roots.append(png_dir)
    candidate_roots.append(img_dir)
    if os.path.isdir(anon_dir):
        candidate_roots.append(anon_dir)
    candidate_roots = [d for d in candidate_roots if os.path.isdir(d)]

    for root in candidate_roots:
        for pattern in (
            os.path.join(root, f"{file_id}-*.png"),
            os.path.join(root, f"{file_id}-*.jpg"),
            os.path.join(root, f"{file_id}-*.jpeg"),
        ):
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

        for p in (
            os.path.join(root, f"{file_id}.png"),
            os.path.join(root, f"{file_id}.jpg"),
            os.path.join(root, f"{file_id}.jpeg"),
        ):
            if os.path.exists(p):
                return p

        zip_path = os.path.join(root, f"{file_id}.zip")
        if os.path.exists(zip_path) and _zip_has_files(zip_path):
            return zip_path

    return None


def _find_xray_path(img_dir: str, file_id: str) -> str:
    valid = _find_valid_xray_path(img_dir, file_id)
    if valid is not None:
        return valid
    return os.path.join(img_dir, f"{file_id}.zip")


def _filter_rows_with_images(df: pd.DataFrame, img_dir: str) -> pd.DataFrame:
    if df.empty:
        return df
    keep = []
    for fid in df["fileID"].tolist():
        try:
            file_id = str(int(fid))
        except Exception:
            keep.append(False)
            continue
        keep.append(_find_valid_xray_path(img_dir, file_id) is not None)
    keep = np.asarray(keep, dtype=bool)
    return df.loc[keep].reset_index(drop=True)


def _binary_value_map_from_series(series: pd.Series, task: str) -> Tuple[Dict[object, int], List[str]]:
    """
    Build a stable {canonical_value -> 0/1} mapping for a label column.
    This allows BCEWithLogitsLoss for categorical-but-binary columns (e.g. "left"/"right").
    """
    s = series.copy()
    miss = _missing_mask(s)
    valid = s[~miss].tolist()
    canon = []
    for v in valid:
        c = _canonical_label_value(v)
        if c is not None:
            canon.append(c)
    uniq = list(dict.fromkeys(canon))

    if len(uniq) == 0:
        return {}, ["0", "1"]
    if len(uniq) > 2:
        raise SystemExit(
            f"Label '{task}' has >2 unique non-missing values ({len(uniq)}). "
            "This run is configured for BCE (binary) only."
        )

    if len(uniq) == 1:
        # Only one observed category -> still define a mapping; training may be degenerate.
        v = uniq[0]
        return {v: 1}, ["0", str(v)]

    # Prefer preserving true 0/1 encodings if possible.
    a, b = uniq[0], uniq[1]
    ca, cb = _coerce_binary_label(a), _coerce_binary_label(b)
    if ca is not None and cb is not None and set([ca, cb]).issubset({0, 1}):
        inv = {a: int(ca), b: int(cb)}
        idx_to_label = ["0", "1"]
        return inv, idx_to_label

    # Otherwise map by a stable ordering (stringified)
    ordered = sorted(uniq, key=lambda x: str(x))
    mapping = {ordered[0]: 0, ordered[1]: 1}
    idx_to_label = [str(ordered[0]), str(ordered[1])]
    return mapping, idx_to_label


class XRayMultiTaskDataset(Dataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        img_dir: str,
        label_columns: List[str],
        label_value_maps: Dict[str, Dict[object, int]],
        transform=None,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.label_columns = list(label_columns)
        self.label_value_maps = {k: dict(v) for k, v in label_value_maps.items()}
        self.img_labels = self._get_available_data(annotations)

    def _get_available_data(self, annotations: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for filename in annotations["fileID"]:
            file_id = str(int(filename))
            keep.append(_find_valid_xray_path(self.img_dir, file_id) is not None)
        keep = np.asarray(keep, dtype=bool)
        return annotations.loc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        file_id = str(int(row["fileID"]))
        img_path = _find_valid_xray_path(self.img_dir, file_id)
        if img_path is None:
            raise FileNotFoundError(f"No valid image found for fileID={file_id} under {self.img_dir}")

        image = _read_xray_image(img_path)
        if self.transform:
            image = self.transform(image)

        targets: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        for col in self.label_columns:
            raw = row.get(col)
            if _is_missing_label_value(raw):
                masks[col] = torch.tensor(0.0, dtype=torch.float32)
                targets[col] = torch.tensor(0.0, dtype=torch.float32)
                continue

            key = _canonical_label_value(raw)
            mapping = self.label_value_maps.get(col, {})
            if key is None or key not in mapping:
                masks[col] = torch.tensor(0.0, dtype=torch.float32)
                targets[col] = torch.tensor(0.0, dtype=torch.float32)
                continue

            masks[col] = torch.tensor(1.0, dtype=torch.float32)
            targets[col] = torch.tensor(float(mapping[key]), dtype=torch.float32)

        return image, targets, masks, img_path


class MultiTaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, head_in_features: int, task_names: List[str], head_dropout: float):
        super().__init__()
        self.backbone = backbone
        self.task_names = list(task_names)
        heads = {}
        for name in self.task_names:
            if head_dropout and head_dropout > 0:
                heads[name] = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(head_in_features, 1))
            else:
                heads[name] = nn.Linear(head_in_features, 1)
        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        feats = self.backbone(x)
        return {name: head(feats).view(-1) for name, head in self.heads.items()}


def _build_vit_multitask(task_names: List[str], no_pretrained: bool, head_dropout: float) -> MultiTaskModel:
    weights = None if no_pretrained else ViT_B_16_Weights.DEFAULT
    try:
        base = vit_b_16(weights=weights)
    except Exception as e:
        print(f"Warning: failed to load pretrained weights ({e}); falling back to random init.")
        base = vit_b_16(weights=None)

    old = base.conv_proj
    new = torch.nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        if old.weight.shape[1] == 3:
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        else:
            new.weight.copy_(old.weight)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)
    base.conv_proj = new

    head_in_features = base.hidden_dim
    base.heads = nn.Identity()
    return MultiTaskModel(base, head_in_features=head_in_features, task_names=task_names, head_dropout=head_dropout)


def _split_head_backbone_params(model: MultiTaskModel):
    head_ids = {id(p) for p in model.heads.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return head_params, backbone_params


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
) -> Tuple[Dict[str, dict], float]:
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
                task_target = targets[task].view(-1)
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
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
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
) -> dict:
    metrics_all: dict = {
        f"epoch/{suffix}": epoch,
        f"average_loss/{suffix}": float(avg_loss),
    }

    per_task_f1 = []
    per_task_auc = []
    per_task_pr_auc = []
    per_task_acc = []

    def _should_log_task_metrics() -> bool:
        # Keep W&B clean: only log rich artifacts like confusion matrices in full mode.
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
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
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
                    best_thr_f1 = float(pr_thr[best_f1_idx])
                    task_metrics[f"task/{task}/best_threshold_f1/{suffix}"] = best_thr_f1

                    beta2 = float(fbeta) ** 2
                    pr_fbeta = (1 + beta2) * pr_prec[:-1] * pr_rec[:-1] / (beta2 * pr_prec[:-1] + pr_rec[:-1] + 1e-12)
                    best_fbeta_idx = int(np.nanargmax(pr_fbeta))
                    task_metrics[f"task/{task}/best_threshold_fbeta/{suffix}"] = float(pr_thr[best_fbeta_idx])

                    # Max recall subject to precision >= target_precision.
                    p_mask = pr_prec[:-1] >= float(target_precision)
                    if np.any(p_mask):
                        cand = np.where(p_mask)[0]
                        best_idx = int(cand[np.nanargmax(pr_rec[:-1][cand])])
                        task_metrics[
                            f"task/{task}/best_threshold_recall_at_p{float(target_precision):.2f}/{suffix}"
                        ] = float(pr_thr[best_idx])
            except Exception:
                pass

        if fixed_thresholds and task in fixed_thresholds and fixed_thresholds[task] is not None:
            thr = float(fixed_thresholds[task])
            y_pred_fx = (scores >= thr).astype(int)
            pfx, rfx, ffx, _ = precision_recall_fscore_support(y_true, y_pred_fx, average="binary", zero_division=0)
            task_metrics.update(
                {
                    f"task/{task}/fixed_threshold/{suffix}": thr,
                    f"task/{task}/f1@fixed_thr/{suffix}": float(ffx),
                    f"task/{task}/precision@fixed_thr/{suffix}": float(pfx),
                    f"task/{task}/recall@fixed_thr/{suffix}": float(rfx),
                }
            )

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
        # Compact mode: only log the key ranking metrics per task for overview.
        wandb_metrics = {
            f"epoch/{suffix}": epoch,
            f"average_loss/{suffix}": float(avg_loss),
            f"macro_auc/{suffix}": metrics_all.get(f"macro_auc/{suffix}", float("nan")),
            f"macro_pr_auc/{suffix}": metrics_all.get(f"macro_pr_auc/{suffix}", float("nan")),
        }
        for task in results_by_task.keys():
            k_auc = f"task/{task}/auc/{suffix}"
            k_pr = f"task/{task}/pr_auc/{suffix}"
            if k_auc in metrics_all:
                wandb_metrics[k_auc] = metrics_all[k_auc]
            if k_pr in metrics_all:
                wandb_metrics[k_pr] = metrics_all[k_pr]
    _wandb_log(wandb_metrics)
    print(
        f"{suffix} macro: "
        f"loss={metrics_all.get(f'average_loss/{suffix}', float('nan')):.4f} "
        f"macro_auc={metrics_all.get(f'macro_auc/{suffix}', float('nan')):.3f} "
        f"macro_pr_auc={metrics_all.get(f'macro_pr_auc/{suffix}', float('nan')):.3f}"
    )
    return metrics_all


def _parse_csv_arg(value: str) -> List[str]:
    v = str(value).strip()
    if not v:
        return []
    lower = v.lower()
    if lower in {"none", "null", "off", "disable", "disabled"}:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def _print_focus_line(metrics: dict, suffix: str, epoch: int, focus_labels: List[str]) -> None:
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


def _normalize_group_id(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _leakage_check(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, group_col: str) -> Tuple[int, int, int]:
    if group_col not in train_df.columns or group_col not in val_df.columns or group_col not in test_df.columns:
        return 0, 0, 0

    def _to_set(df: pd.DataFrame) -> set:
        vals = [_normalize_group_id(v) for v in df[group_col].tolist()]
        return {v for v in vals if v is not None}

    tr = _to_set(train_df)
    va = _to_set(val_df)
    te = _to_set(test_df)
    return len(tr & va), len(tr & te), len(va & te)


def _choose_group_col(metadata: pd.DataFrame, requested: Optional[str], n_folds: int) -> Optional[str]:
    req = (requested or "").strip()
    if req.lower() in {"", "none", "null", "off", "disable", "disabled"}:
        return None
    if req.lower() == "auto":
        candidates = ["medicoID", "Anforderungsnummer", "Aufnahmenummer", "fileID"]
    else:
        candidates = [req]

    for c in candidates:
        if c in metadata.columns and int(metadata[c].nunique(dropna=True)) >= int(n_folds):
            return c
    return None


def _print_focus_confusions(
    results_by_task: dict,
    thresholds: Optional[Dict[str, Optional[float]]],
    suffix: str,
    epoch: int,
    focus_labels: List[str],
) -> None:
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
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        lines.append(
            f"{task} thr={thr:.3f} tp={tp} fp={fp} tn={tn} fn={fn} "
            f"prec={float(precision):.3f} rec={float(recall):.3f} f1={float(f1):.3f}"
        )
    if lines:
        print(f"{suffix} focus confusion (epoch {epoch}): " + " | ".join(lines))


def check_mapping_merge(
    base_folder: str,
    image_dir: str,
    metadata_file: str,
    mapping_file: str,
    sample_n: int = 20,
    seed: int = 0,
) -> None:
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
    print(f"matched fileID: {matched} ({(matched / max(1,total))*100:.1f}%)")

    sample = (
        merged[merged["fileID"] != -1].sample(n=min(sample_n, matched), random_state=seed) if matched else merged.head(0)
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
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
        default="primary_auc/eval",
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
        help="Cap for the positive sampling weight used by --train-sampler primary_balanced (prevents extreme oversampling).",
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
        help="Loss multiplier for tasks that do not influence the backbone (still trains heads, but less).",
    )
    parser.add_argument(
        "--focus-labels",
        default="mixed_shapes,diffuse_pleural_location,local_pleural_location,pleural_calcification_location,occupational_disease",
        help="Comma-separated labels to print as a compact focus line each eval epoch (use 'none' to disable).",
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
        default="Untersuchungsnummer",
        help="Column used to group samples when generating splits (prevents patient leakage). Use 'none' to disable override.",
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
    args = parser.parse_args()

    _set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    preprocess = Compose(
        [
            # Avoid aggressive cropping for CXR; cropping can remove clinically relevant regions.
            Resize((224, 224)),
            RandomRotation(3),
            ColorJitter(brightness=0.15, contrast=0.15),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    preprocess_no_aug = Compose(
        [
            Resize((224, 224)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    base_folder = args.base_folder
    root_folder = args.image_dir or base_folder
    mapping_file = os.path.join(base_folder, "mapping.csv")
    default_fold_folder = os.path.join(base_folder, "strat_dichotom_splits")
    fold_folder = args.fold_folder or default_fold_folder
    output_folder = args.output_folder or base_folder

    raw_metadata_file = os.path.join(base_folder, "dichotome_data_pseudonym.csv")
    metadata_file = os.path.join(base_folder, "dichotome_data_pseudonym.csv")
    anford_nr_file = os.path.join(base_folder, "table_pseudonym.csv")
    nan_thresh = 999

    if os.path.isfile(metadata_file):
        required_files = [metadata_file, mapping_file]
    else:
        required_files = [raw_metadata_file, anford_nr_file, mapping_file]

    missing = [p for p in required_files if not os.path.isfile(p)]
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
        check_file = metadata_file if os.path.isfile(metadata_file) else raw_metadata_file
        check_mapping_merge(
            base_folder=base_folder,
            image_dir=root_folder,
            metadata_file=check_file,
            mapping_file=mapping_file,
            sample_n=args.check_mapping_samples,
            seed=args.seed,
        )
        raise SystemExit(0)

    if args.label_stats:
        if os.path.isfile(metadata_file):
            md = pd.read_csv(metadata_file)
        else:
            md = Preprocessor_Metadata.prepare_metadata(raw_metadata_file, anford_nr_file, mapping_file, nan_thresh, True)
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
        if os.path.isfile(metadata_file):
            metadata = pd.read_csv(metadata_file)
        else:
            metadata = Preprocessor_Metadata.prepare_metadata(raw_metadata_file, anford_nr_file, mapping_file, nan_thresh, True)
        metadata = _ensure_file_id(metadata, mapping_file)
        metadata = metadata[metadata["fileID"] != -1]
        if args.split_label not in metadata.columns:
            raise SystemExit(f"Split label '{args.split_label}' not found in metadata.")
        group_col = _choose_group_col(metadata, requested=str(getattr(args, "group_splits_by", "auto")), n_folds=n_folds)
        if group_col is None:
            raise SystemExit(
                f"Cannot generate grouped splits: no suitable grouping column found for --group-splits-by={args.group_splits_by!r}. "
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
    test_metadata = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "test"]
    val_metadata = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "val"]
    train_metadata = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "train"]

    group_col = _choose_group_col(prepared_metadata, requested=str(getattr(args, "group_splits_by", "auto")), n_folds=n_folds)
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
                # If splits were loaded from disk, try to regenerate once with strict grouping.
                if not needs_splits:
                    print(msg + " Regenerating splits with strict grouping...")
                    if os.path.isfile(metadata_file):
                        base_md = pd.read_csv(metadata_file)
                    else:
                        base_md = Preprocessor_Metadata.prepare_metadata(
                            raw_metadata_file, anford_nr_file, mapping_file, nan_thresh, True
                        )
                    base_md = _ensure_file_id(base_md, mapping_file)
                    base_md = base_md[base_md["fileID"] != -1]
                    group_col = _choose_group_col(base_md, requested=str(getattr(args, "group_splits_by", "auto")), n_folds=n_folds)
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
                    test_metadata = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "test"]
                    val_metadata = prepared_metadata[prepared_metadata[f"Fold{fold}"] == "val"]
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
    val_metadata = _filter_any_label(val_metadata)
    test_metadata = _filter_any_label(test_metadata)

    # Filter out rows without corresponding image files early, so:
    # - train stats/degeneracy checks are accurate,
    # - samplers match the actual dataset length,
    # - eval/test metrics aren't affected by missing images.
    n_tr0, n_va0, n_te0 = len(train_metadata), len(val_metadata), len(test_metadata)
    train_metadata = _filter_rows_with_images(train_metadata, root_folder)
    val_metadata = _filter_rows_with_images(val_metadata, root_folder)
    test_metadata = _filter_rows_with_images(test_metadata, root_folder)
    if (len(train_metadata), len(val_metadata), len(test_metadata)) != (n_tr0, n_va0, n_te0):
        print(
            "Filtered missing images: "
            f"train {n_tr0}->{len(train_metadata)}, "
            f"val {n_va0}->{len(val_metadata)}, "
            f"test {n_te0}->{len(test_metadata)}"
        )

    # Build per-task value maps (always binary 0/1) and names for logging.
    task_value_maps: Dict[str, Dict[object, int]] = {}
    task_index_to_label: Dict[str, List[str]] = {}
    for col in label_cols:
        mapping, idx_to_label = _binary_value_map_from_series(prepared_metadata[col], task=col)
        task_value_maps[col] = mapping
        task_index_to_label[col] = idx_to_label

    # --- Datasets/loaders ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Per-task BCE (all tasks are binary)
    criterion: Dict[str, nn.Module] = {}
    train_task_stats: Dict[str, Dict[str, float]] = {}
    for task in label_cols:
        mapping = task_value_maps.get(task, {})
        s = train_metadata[task]
        miss = _missing_mask(s)
        vals = []
        for v in s[~miss].tolist():
            key = _canonical_label_value(v)
            if key is None or key not in mapping:
                continue
            vals.append(int(mapping[key]))
        arr = np.asarray(vals, dtype=int) if vals else np.asarray([], dtype=int)
        pos = float((arr == 1).sum())
        neg = float((arr == 0).sum())
        train_task_stats[task] = {"pos": pos, "neg": neg, "present": float(pos + neg)}
        pos_weight_val = (neg / max(1.0, pos)) if (pos + neg) > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_val], device=device)
        criterion[task] = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        print(f"[{task}] BCEWithLogitsLoss(pos_weight={pos_weight_val:.4f}) [neg={neg:.0f} pos={pos:.0f}]")

    primary_label = str(args.primary_label).strip()
    if primary_label not in label_cols:
        raise SystemExit(f"Primary label '{primary_label}' is not part of training labels: {label_cols}")

    # Drop tasks that have only a single class in the TRAIN split (pos==0 or neg==0).
    # These tasks cannot be learned from this fold and would distort the objective/metrics.
    degenerate_tasks = []
    for t in list(label_cols):
        pos = float(train_task_stats.get(t, {}).get("pos", 0.0))
        neg = float(train_task_stats.get(t, {}).get("neg", 0.0))
        if pos == 0.0 or neg == 0.0:
            degenerate_tasks.append(t)

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
        # Keep only active tasks everywhere downstream.
        criterion = {t: criterion[t] for t in label_cols if t in criterion}
        train_task_stats = {t: train_task_stats[t] for t in label_cols if t in train_task_stats}
        task_value_maps = {t: task_value_maps[t] for t in label_cols if t in task_value_maps}
        task_index_to_label = {t: task_index_to_label[t] for t in label_cols if t in task_index_to_label}
        # Re-filter splits so we don't keep samples that only had labels in dropped tasks.
        train_metadata = _filter_any_label(train_metadata)
        val_metadata = _filter_any_label(val_metadata)
        test_metadata = _filter_any_label(test_metadata)

    print(f"Active tasks for this fold: {len(label_cols)} -> " + ", ".join(label_cols))

    # Optional: task loss weighting to reduce noise from very sparse tasks.
    task_weights = {t: 1.0 for t in label_cols}
    tw = str(args.task_weighting)
    if tw != "equal":
        raw = {}
        for t in label_cols:
            present = float(train_task_stats.get(t, {}).get("present", 0.0))
            if tw == "present":
                raw[t] = max(1.0, present)
            else:
                raw[t] = max(1.0, float(np.sqrt(present)))
        mean_w = float(np.mean(list(raw.values()))) if raw else 1.0
        if mean_w <= 0:
            mean_w = 1.0
        task_weights = {t: float(raw[t]) / mean_w for t in raw}
    primary_tw = float(args.primary_task_weight)
    if primary_tw <= 0:
        # Auto: bump the primary task slightly if it is at least as well-supported as the median task.
        # Keep it modest to avoid overfitting the validation set via the primary objective.
        presents = [float(train_task_stats.get(t, {}).get("present", 0.0)) for t in label_cols]
        median_present = float(np.median(presents)) if presents else 0.0
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

    focus_raw = _parse_csv_arg(args.focus_labels)
    focus_labels = [lab for lab in focus_raw if lab in label_cols]
    dropped_focus = [lab for lab in focus_raw if lab not in label_cols]
    if focus_labels:
        print("Focus labels: " + ", ".join(focus_labels))
    if dropped_focus:
        print("Focus labels not active in this fold: " + ", ".join(dropped_focus))

    min_pos = int(args.min_pos_backbone)
    min_neg = int(args.min_neg_backbone)
    backbone_tasks = {
        t for t in label_cols if train_task_stats.get(t, {}).get("pos", 0.0) >= min_pos and train_task_stats.get(t, {}).get("neg", 0.0) >= min_neg
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

    # --- Datasets/loaders (active tasks only) ---
    n_workers = int(args.num_workers)
    gen = torch.Generator()
    gen.manual_seed(int(args.seed))
    train_dataset = XRayMultiTaskDataset(
        train_metadata[["fileID"] + label_cols],
        root_folder,
        label_columns=label_cols,
        label_value_maps=task_value_maps,
        transform=preprocess,
    )
    val_dataset = XRayMultiTaskDataset(
        val_metadata[["fileID"] + label_cols],
        root_folder,
        label_columns=label_cols,
        label_value_maps=task_value_maps,
        transform=preprocess_no_aug,
    )
    test_dataset = XRayMultiTaskDataset(
        test_metadata[["fileID"] + label_cols],
        root_folder,
        label_columns=label_cols,
        label_value_maps=task_value_maps,
        transform=preprocess_no_aug,
    )

    loader_kwargs = dict(
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=n_workers,
        generator=gen,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(n_workers > 0),
    )
    if str(args.train_sampler) == "primary_balanced":
        mapping = task_value_maps.get(primary_label, {})
        # Important: compute weights AFTER filtering missing images (train_dataset.img_labels),
        # otherwise len(weights) can differ from len(train_dataset).
        s = train_dataset.img_labels[primary_label]
        miss = _missing_mask(s)
        pos_n = 0
        neg_n = 0
        labels_for_weight = []
        for v, is_miss in zip(s.tolist(), miss.tolist()):
            if is_miss:
                labels_for_weight.append(None)
                continue
            key = _canonical_label_value(v)
            if key is None or key not in mapping:
                labels_for_weight.append(None)
                continue
            y = int(mapping[key])
            labels_for_weight.append(y)
            if y == 1:
                pos_n += 1
            else:
                neg_n += 1
        raw_pos_w = (neg_n / max(1, pos_n)) if (pos_n + neg_n) > 0 else 1.0
        cap = float(args.primary_balanced_max_pos_weight)
        pos_w = min(raw_pos_w, cap) if cap > 0 else raw_pos_w
        weights = [pos_w if y == 1 else 1.0 for y in labels_for_weight]
        # Missing/unknown labels get neutral weight.
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
    # Deterministic evaluation: no shuffling/sampling noise.
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), **loader_kwargs)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), **loader_kwargs)

    model = _build_vit_multitask(label_cols, no_pretrained=bool(args.no_pretrained), head_dropout=float(args.head_dropout))
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
                    {"params": head_params, "lr": float(args.learning_rate)},
                ],
                weight_decay=float(args.weight_decay),
            )
        return optim.AdamW(head_params + backbone_params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    def make_scheduler(optimizer, remaining_epochs: int):
        total_steps = max(1, remaining_epochs * len(train_loader))
        if len(optimizer.param_groups) == 2:
            max_lr = [float(args.learning_rate) * float(args.backbone_lr_mult), float(args.learning_rate)]
        else:
            max_lr = float(args.learning_rate)
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

    if not args.no_wandb:
        wandb.init(
            project="Asbestosis",
            config={
                "learning_rate": float(args.learning_rate),
                "dataset": root_folder,
                "split_folder": fold_folder,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "test_samples": len(test_loader.dataset),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "optimizer": "AdamW",
                "head_dropout": float(args.head_dropout),
                "augmentation": str(preprocess),
                "machine": "HPC",
                "labels": label_cols,
                "fold": fold,
                "metadata": metadata_file,
                "bce_only": True,
                "primary_label": primary_label,
                "train_sampler": str(args.train_sampler),
                "primary_balanced_max_pos_weight": float(args.primary_balanced_max_pos_weight),
                "task_weighting": str(args.task_weighting),
                "primary_task_weight": float(args.primary_task_weight),
                "min_pos_backbone": int(args.min_pos_backbone),
                "min_neg_backbone": int(args.min_neg_backbone),
                "head_only_loss_weight": float(args.head_only_loss_weight),
                "threshold_strategy": str(args.threshold_strategy),
                "fbeta": float(args.fbeta),
                "target_precision": float(args.target_precision),
                "group_splits_by": str(args.group_splits_by),
                "leakage_action": str(args.leakage_action),
            },
            name=f"multitask_vit_b={int(args.batch_size)}_l={float(args.learning_rate)}_n={int(args.epochs)}_fold={fold}",
        )

    if args.early_stop_patience > 0 and args.eval_every == 0:
        args.eval_every = 1
        print("Early stopping enabled -> forcing --eval-every 1")

    best_score = None
    best_epoch = -1
    best_path = os.path.join(output_folder, f"best_vit_labels=multitask_fold={fold}.pth")
    no_improve = 0
    best_fixed_thresholds = None

    optimizer = make_optimizer()
    lr_scheduler = make_scheduler(optimizer, remaining_epochs=int(args.epochs))

    for epoch in range(int(args.epochs)):
        if args.freeze_backbone_epochs and epoch == int(args.freeze_backbone_epochs):
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = make_optimizer()
            lr_scheduler = make_scheduler(optimizer, remaining_epochs=int(args.epochs) - epoch)
            print("Unfroze backbone and reset optimizer/scheduler")

        train_results, train_avg_loss = run_epoch(
            model,
            optimizer,
            lr_scheduler,
            criterion,
            scaler,
            train_loader,
            device,
            train=True,
            max_steps=args.max_train_steps,
            backbone_tasks=backbone_tasks,
            head_only_loss_weight=float(args.head_only_loss_weight),
            task_weights=task_weights,
        )
        log_multitask_metrics(
            train_results,
            train_avg_loss,
            suffix="train",
            epoch=epoch,
            task_index_to_label=task_index_to_label,
            wandb_detail=args.wandb_detail,
        )

        do_eval = args.eval_every and (epoch % int(args.eval_every) == 0 or epoch == int(args.epochs) - 1)
        do_test = args.test_every and (epoch % int(args.test_every) == 0 or epoch == int(args.epochs) - 1)

        eval_metrics = None
        fixed_thresholds_eval = {}
        if do_eval:
            eval_results, eval_avg_loss = run_epoch(
                model,
                optimizer,
                None,
                criterion,
                scaler,
                val_loader,
                device,
                train=False,
                max_steps=None,
                backbone_tasks=backbone_tasks,
                head_only_loss_weight=float(args.head_only_loss_weight),
                task_weights=task_weights,
            )
            eval_metrics = log_multitask_metrics(
                eval_results,
                eval_avg_loss,
                suffix="eval",
                epoch=epoch,
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
                eval_results,
                thresholds=fixed_thresholds_eval,
                suffix="eval",
                epoch=epoch,
                focus_labels=focus_labels,
            )

        if do_test:
            test_results, test_avg_loss = run_epoch(
                model,
                optimizer,
                None,
                criterion,
                scaler,
                test_loader,
                device,
                train=False,
                max_steps=None,
                backbone_tasks=backbone_tasks,
                head_only_loss_weight=float(args.head_only_loss_weight),
                task_weights=task_weights,
            )
            log_multitask_metrics(
                test_results,
                test_avg_loss,
                suffix="test",
                epoch=epoch,
                task_index_to_label=task_index_to_label,
                wandb_detail=args.wandb_detail,
            )

        # Early stopping
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
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "score": float(best_score),
                    "labels": label_cols,
                    "task_index_to_label": task_index_to_label,
                    "task_value_maps": task_value_maps,
                    "best_fixed_thresholds": best_fixed_thresholds,
                }
                torch.save(checkpoint, best_path)
                print(f"Saved best model to {best_path} (epoch {epoch}, score={best_score:.4f})")
            else:
                no_improve += 1
                if no_improve >= int(args.early_stop_patience):
                    print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, score={best_score:.4f})")
                    break

    final_epoch = epoch if int(args.epochs) > 0 else -1
    final_results, final_avg_loss = run_epoch(
        model,
        optimizer,
        None,
        criterion,
        scaler,
        test_loader,
        device,
        train=False,
        max_steps=None,
        backbone_tasks=backbone_tasks,
        head_only_loss_weight=float(args.head_only_loss_weight),
        task_weights=task_weights,
    )
    final_metrics = log_multitask_metrics(
        final_results,
        final_avg_loss,
        suffix="final_test",
        epoch=final_epoch,
        task_index_to_label=task_index_to_label,
        fixed_thresholds=best_fixed_thresholds,
        wandb_detail=args.wandb_detail,
        threshold_strategy=str(args.threshold_strategy),
        fbeta=float(args.fbeta),
        target_precision=float(args.target_precision),
    )
    _print_focus_line(final_metrics, suffix="final_test", epoch=final_epoch, focus_labels=focus_labels)
    if best_fixed_thresholds:
        _print_focus_confusions(
            final_results,
            thresholds=best_fixed_thresholds,
            suffix="final_test",
            epoch=final_epoch,
            focus_labels=focus_labels,
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

    # Evaluate best checkpoint as well (optional)
    if args.early_stop_patience > 0 and os.path.isfile(best_path):
        try:
            state = torch.load(best_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                best_fixed_thresholds = state.get("best_fixed_thresholds") or best_fixed_thresholds
            best_results, best_avg_loss = run_epoch(
                model,
                optimizer,
                None,
                criterion,
                scaler,
                test_loader,
                device,
                train=False,
                max_steps=None,
                backbone_tasks=backbone_tasks,
                head_only_loss_weight=float(args.head_only_loss_weight),
                task_weights=task_weights,
            )
            best_metrics = log_multitask_metrics(
                best_results,
                best_avg_loss,
                suffix="best_test",
                epoch=int(state.get("epoch", best_epoch)) if isinstance(state, dict) else best_epoch,
                task_index_to_label=task_index_to_label,
                fixed_thresholds=best_fixed_thresholds,
                wandb_detail=args.wandb_detail,
                threshold_strategy=str(args.threshold_strategy),
                fbeta=float(args.fbeta),
                target_precision=float(args.target_precision),
            )
            _print_focus_line(best_metrics, suffix="best_test", epoch=int(state.get("epoch", best_epoch)) if isinstance(state, dict) else best_epoch, focus_labels=focus_labels)
            if best_fixed_thresholds:
                _print_focus_confusions(
                    best_results,
                    thresholds=best_fixed_thresholds,
                    suffix="best_test",
                    epoch=int(state.get("epoch", best_epoch)) if isinstance(state, dict) else best_epoch,
                    focus_labels=focus_labels,
                )
        except Exception as e:
            print(f"Warning: failed to evaluate best checkpoint ({best_path}): {e}")

    torch.save(
        model.state_dict(),
        os.path.join(output_folder, f"asbestosis_vit_n{int(args.epochs)}_b{int(args.batch_size)}_labels=multitask_fold={fold}.pth"),
    )
    if _wandb_is_active():
        wandb.finish()


if __name__ == "__main__":
    main()
