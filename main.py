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
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import nn, optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, RandomSampler
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
    pixel_array = dicom_ds.pixel_array.astype(np.float32)
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

        _wandb_log({f"loss/{suffix}": float(loss.item())}, commit=False)
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
) -> dict:
    metrics_all: dict = {
        f"epoch/{suffix}": epoch,
        f"average_loss/{suffix}": float(avg_loss),
    }

    per_task_f1 = []
    per_task_auc = []
    per_task_acc = []

    def _should_log_task_metrics() -> bool:
        if wandb_detail == "full":
            return True
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

        task_metrics = {
            f"task/{task}/accuracy/{suffix}": acc,
            f"task/{task}/balanced_accuracy/{suffix}": bal_acc,
            f"task/{task}/precision/{suffix}": float(precision),
            f"task/{task}/recall/{suffix}": float(recall),
            f"task/{task}/f1/{suffix}": float(f1),
            f"task/{task}/auc/{suffix}": float(auc_val),
        }

        if compute_thresholds and unique_true.size >= 2:
            try:
                pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true, scores)
                if pr_thr.size > 0:
                    pr_f1 = (2 * pr_prec[:-1] * pr_rec[:-1]) / (pr_prec[:-1] + pr_rec[:-1] + 1e-12)
                    best_f1_idx = int(np.nanargmax(pr_f1))
                    best_thr_f1 = float(pr_thr[best_f1_idx])
                    task_metrics[f"task/{task}/best_threshold_f1/{suffix}"] = best_thr_f1
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

    _wandb_log(metrics_all)
    print(
        f"{suffix} macro: "
        f"loss={metrics_all.get(f'average_loss/{suffix}', float('nan')):.4f} "
        f"macro_acc={metrics_all.get(f'macro_accuracy/{suffix}', float('nan')):.3f} "
        f"macro_f1={metrics_all.get(f'macro_f1/{suffix}', float('nan')):.3f} "
        f"macro_auc={metrics_all.get(f'macro_auc/{suffix}', float('nan')):.3f}"
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
        f1 = metrics.get(f"task/{lab}/f1/{suffix}")
        auc_s = f"{float(auc):.3f}" if auc is not None and np.isfinite(auc) else "nan"
        f1_s = f"{float(f1):.3f}" if f1 is not None and np.isfinite(f1) else "nan"
        parts.append(f"{lab} auc={auc_s} f1={f1_s}")
    print(f"{suffix} focus (epoch {epoch}): " + " | ".join(parts))


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
        choices=["primary_auc/eval", "primary_f1/eval", "macro_auc/eval", "macro_f1/eval", "loss/eval"],
        default="primary_auc/eval",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--primary-label",
        default="mixed_shapes",
        help="Primary label to optimize/early-stop on (must be included in --labels).",
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
        if has_val:
            needs_splits = False
        else:
            print(f"Split file has no 'val' split -> regenerating: {fold_splitted_metadata_filename}")

    if needs_splits:
        if os.path.isfile(metadata_file):
            metadata = pd.read_csv(metadata_file)
        else:
            metadata = Preprocessor_Metadata.prepare_metadata(raw_metadata_file, anford_nr_file, mapping_file, nan_thresh, True)
        metadata = _ensure_file_id(metadata, mapping_file)
        metadata = metadata[metadata["fileID"] != -1]
        if args.split_label not in metadata.columns:
            raise SystemExit(f"Split label '{args.split_label}' not found in metadata.")
        metadata = Preprocessor_Metadata.create_splits(
            metadata,
            n_folds,
            fold_folder,
            fold_splitted_metadata_filename,
            args.split_label,
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
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, generator=gen), **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset, generator=gen), **loader_kwargs)
    test_loader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset, generator=gen), **loader_kwargs)

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
                "min_pos_backbone": int(args.min_pos_backbone),
                "min_neg_backbone": int(args.min_neg_backbone),
                "head_only_loss_weight": float(args.head_only_loss_weight),
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
                max_steps=args.max_eval_steps,
                backbone_tasks=backbone_tasks,
                head_only_loss_weight=float(args.head_only_loss_weight),
            )
            eval_metrics = log_multitask_metrics(
                eval_results,
                eval_avg_loss,
                suffix="eval",
                epoch=epoch,
                task_index_to_label=task_index_to_label,
                compute_thresholds=True,
                wandb_detail=args.wandb_detail,
            )
            _print_focus_line(eval_metrics, suffix="eval", epoch=epoch, focus_labels=focus_labels)
            for task in label_cols:
                fixed_thresholds_eval[task] = eval_metrics.get(f"task/{task}/best_threshold_f1/eval")

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
                max_steps=args.max_eval_steps,
                backbone_tasks=backbone_tasks,
                head_only_loss_weight=float(args.head_only_loss_weight),
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
            elif metric == "primary_f1/eval":
                score = eval_metrics.get(f"task/{primary_label}/f1/eval")
            elif metric == "macro_auc/eval":
                score = eval_metrics.get("macro_auc/eval")
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
        max_steps=args.max_eval_steps,
        backbone_tasks=backbone_tasks,
        head_only_loss_weight=float(args.head_only_loss_weight),
    )
    final_metrics = log_multitask_metrics(
        final_results,
        final_avg_loss,
        suffix="final_test",
        epoch=final_epoch,
        task_index_to_label=task_index_to_label,
        fixed_thresholds=best_fixed_thresholds,
        wandb_detail=args.wandb_detail,
    )
    _print_focus_line(final_metrics, suffix="final_test", epoch=final_epoch, focus_labels=focus_labels)

    if _wandb_is_active():
        try:
            for key in ("macro_f1/final_test", "macro_auc/final_test", "macro_accuracy/final_test", "average_loss/final_test"):
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
                max_steps=args.max_eval_steps,
                backbone_tasks=backbone_tasks,
                head_only_loss_weight=float(args.head_only_loss_weight),
            )
            best_metrics = log_multitask_metrics(
                best_results,
                best_avg_loss,
                suffix="best_test",
                epoch=int(state.get("epoch", best_epoch)) if isinstance(state, dict) else best_epoch,
                task_index_to_label=task_index_to_label,
                fixed_thresholds=best_fixed_thresholds,
                wandb_detail=args.wandb_detail,
            )
            _print_focus_line(best_metrics, suffix="best_test", epoch=int(state.get("epoch", best_epoch)) if isinstance(state, dict) else best_epoch, focus_labels=focus_labels)
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
