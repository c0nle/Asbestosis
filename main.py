import gc
import os
import io
import argparse
import zipfile
import glob
import tempfile
import random
import functools
import warnings
from typing import Optional, List
import numpy as np
import pandas as pd

import PIL.Image
import pydicom
import torch
from torch import nn, optim, GradScaler
from torch.utils.data import Dataset, RandomSampler, DataLoader, Subset, WeightedRandomSampler
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.transforms.v2 import ColorJitter, RandomResizedCrop, RandomRotation, ToImage, ToDtype, Compose, Resize, Normalize
import torch.multiprocessing as mp
from sklearn.metrics import (
    RocCurveDisplay,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
import matplotlib.pyplot as plt

import Preprocessor_Metadata
import wandb


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


def _coerce_binary_label(value) -> Optional[int]:
    """
    Convert common binary encodings to {0,1}. Returns None if not coercible.
    """
    if _is_missing_label_value(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return 1 if bool(value) else 0
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
        return None
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer() and int(value) in (0, 1):
            return int(value)
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"0", "false", "no"}:
            return 0
        if v in {"1", "true", "yes"}:
            return 1
    return None


def _canonical_label_value(value):
    """
    Canonicalize values for stable mapping keys across pandas dtype quirks.
    """
    if _is_missing_label_value(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if np.isfinite(fv) and fv.is_integer():
            return int(fv)
        return fv
    return str(value)


def _missing_mask(series: pd.Series) -> pd.Series:
    """
    Treat -1, NaN, None, empty string, 'nan', 'none' as missing.
    """
    s = series
    mask = s.isna()
    try:
        mask = mask | (s == -1)
    except Exception:
        pass
    # Handle common string encodings
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

        # Try binary coercion
        vals = []
        for v in s[~miss].tolist():
            b = _coerce_binary_label(v)
            if b is None:
                vals.append(None)
            else:
                vals.append(int(b))
        non_null = [v for v in vals if v is not None]
        is_binary = len(non_null) == len(vals) and set(non_null).issubset({0, 1})

        if is_binary:
            pos = int(sum(1 for v in non_null if v == 1))
            neg = int(sum(1 for v in non_null if v == 0))
            print(f"- {col}: present={present} missing={missing_n} | pos={pos} neg={neg}")
        else:
            # Non-binary: show presence and top categories (limited).
            try:
                vc = s[~miss].astype("string").value_counts().head(8)
                top = ", ".join([f"{idx}={int(cnt)}" for idx, cnt in vc.items()])
                uniq = int(s[~miss].astype("string").nunique())
                print(f"- {col}: present={present} missing={missing_n} | non-binary uniq={uniq} top: {top}")
            except Exception:
                print(f"- {col}: present={present} missing={missing_n} | non-binary")


def _ensure_file_id(metadata: pd.DataFrame, mapping_file: str) -> pd.DataFrame:
    has_file_id = "fileID" in metadata.columns and (metadata["fileID"] != -1).any()
    if has_file_id:
        return metadata

    mapping = pd.read_csv(mapping_file, dtype={"medicoID": str})
    mapping["fileID"] = pd.to_numeric(mapping.get("fileID"), errors="coerce")
    # `mapping.csv` encodes the (pseudonymized) Anforderungsnummer in the `medicoID` field,
    # with two trailing digits used for internal versioning.
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


class MultiOutputResNet(nn.Module):
    def __init__(self, number_of_classes: list):  # Define number of outputs per task
        super(MultiOutputResNet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove original FC layer

        # Create multiple output heads dynamically
        self.output_heads = nn.ModuleList([
            nn.Linear(in_features, num_classes) for num_classes in number_of_classes
        ])

    def forward(self, x):
        x = self.resnet(x)
        return [head(x) for head in self.output_heads]  # Return multiple outputs


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
        # Prefer classic DICOM names from this dataset, but fall back to any file.
        candidates = [
            name for name in zip_dir.namelist()
            if (name.endswith("IM_0001") or (os.sep + "IM_0001") in name or "IM_" in name)
        ]
        if not candidates:
            candidates = [name for name in zip_dir.namelist() if not name.endswith("/")]
        if not candidates:
            raise FileNotFoundError(f"No files found inside zip: {zip_path}")

        name = candidates[0]
        with zip_dir.open(name) as f:
            data = f.read()

        # Try DICOM first, then fall back to PIL for image formats.
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
    """
    Prefer already-extracted PNG/JPG. Fall back to anon zip only if it contains files.
    """
    # Prefer `png/` if present (common for this dataset and avoids expensive scans in the base folder).
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
        # Prefer extracted images first
        pattern_candidates = [
            os.path.join(root, f"{file_id}-*.png"),
            os.path.join(root, f"{file_id}-*.jpg"),
            os.path.join(root, f"{file_id}-*.jpeg"),
        ]
        for pattern in pattern_candidates:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

        direct_images = [
            os.path.join(root, f"{file_id}.png"),
            os.path.join(root, f"{file_id}.jpg"),
            os.path.join(root, f"{file_id}.jpeg"),
        ]
        for p in direct_images:
            if os.path.exists(p):
                return p

        zip_path = os.path.join(root, f"{file_id}.zip")
        if os.path.exists(zip_path) and _zip_has_files(zip_path):
            return zip_path

    return None


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


def _find_xray_path(img_dir: str, file_id: str) -> str:
    valid = _find_valid_xray_path(img_dir, file_id)
    if valid is not None:
        return valid
    return os.path.join(img_dir, f"{file_id}.zip")


class X_rayImageDataset(Dataset):
    def __init__(self, annotations, img_dir, label_column: str, transform=None):
        self.img_dir = img_dir
        self.img_labels = self.get_available_data(annotations)
        self.transform = transform
        self.label_column = label_column
        self.label_index_mapping = {label: idx for idx, label in enumerate(sorted(self.img_labels[self.label_column].unique()))}

    def get_available_data(self, annotations):
        available_annotations = []
        for filename in annotations["fileID"]:
            file_id = str(int(filename))
            valid = _find_valid_xray_path(self.img_dir, file_id)
            if valid is not None:
                available_annotations.append(int(filename))
        return annotations[annotations['fileID'].isin(available_annotations)]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        file_id = str(int(self.img_labels.iloc[idx]['fileID']))
        img_path = _find_valid_xray_path(self.img_dir, file_id)
        if img_path is None:
            raise FileNotFoundError(f"No valid image found for fileID={file_id} under {self.img_dir}")
        image = _read_xray_image(img_path)
        label = self.img_labels.iloc[idx][self.label_column]
        label = self.label_index_mapping[label]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), img_path


class XRayMultiTaskDataset(Dataset):
    """
    Returns multi-task targets for multiple label columns at once.

    For each task/label:
      - `targets[label]` is either a float tensor (binary, encoded 0/1) or long tensor (multiclass index).
      - `masks[label]` is float tensor in {0,1} indicating if the label is present for that sample.
    """

    def __init__(
        self,
        annotations: pd.DataFrame,
        img_dir: str,
        label_columns: List[str],
        label_value_maps: dict,
        label_output_dims: dict,
        tabular_features: Optional[np.ndarray] = None,
        transform=None,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.label_columns = list(label_columns)
        self.label_value_maps = dict(label_value_maps)
        self.label_output_dims = dict(label_output_dims)
        self.tabular_features = None
        if tabular_features is not None:
            if len(tabular_features) != len(annotations):
                raise ValueError("tabular_features must be aligned with annotations rows (same length).")
            tabular_features = np.asarray(tabular_features, dtype=np.float32)
        self.img_labels, self.tabular_features = self._get_available_data(annotations, tabular_features)

    def _get_available_data(self, annotations: pd.DataFrame, tabular_features: Optional[np.ndarray]):
        keep_idx = []
        for filename in annotations["fileID"]:
            file_id = str(int(filename))
            valid = _find_valid_xray_path(self.img_dir, file_id)
            if valid is not None:
                keep_idx.append(True)
            else:
                keep_idx.append(False)
        keep_idx = np.asarray(keep_idx, dtype=bool)
        filtered = annotations.loc[keep_idx].reset_index(drop=True)
        if tabular_features is None:
            return filtered, None
        return filtered, tabular_features[keep_idx]

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

        tabular = None
        if self.tabular_features is not None:
            tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        targets = {}
        masks = {}
        for col in self.label_columns:
            raw = row.get(col)
            missing = _is_missing_label_value(raw)

            if missing:
                masks[col] = torch.tensor(0.0, dtype=torch.float32)
                # placeholder target (ignored via mask)
                if self.label_output_dims[col] == 1:
                    targets[col] = torch.tensor(0.0, dtype=torch.float32)
                else:
                    targets[col] = torch.tensor(0, dtype=torch.long)
                continue

            masks[col] = torch.tensor(1.0, dtype=torch.float32)
            if self.label_output_dims[col] == 1:
                b = _coerce_binary_label(raw)
                if b is None:
                    # Value is not coercible -> ignore sample for this task.
                    masks[col] = torch.tensor(0.0, dtype=torch.float32)
                    targets[col] = torch.tensor(0.0, dtype=torch.float32)
                else:
                    targets[col] = torch.tensor(float(b), dtype=torch.float32)
            else:
                mapping = self.label_value_maps[col]
                key = _canonical_label_value(raw)
                if key is None or key not in mapping:
                    masks[col] = torch.tensor(0.0, dtype=torch.float32)
                    targets[col] = torch.tensor(0, dtype=torch.long)
                else:
                    targets[col] = torch.tensor(int(mapping[key]), dtype=torch.long)

        if tabular is None:
            return image, targets, masks, img_path
        return image, tabular, targets, masks, img_path


def run_epoch(model, optimizer, lr_scheduler, criterion, scaler, data_loader, device, train=True, show_figs=True, max_steps=None):
    if train:
        model.train()
        suffix = "train"
    else:
        model.eval()
        suffix = "eval"

    epoch_loss = 0.0
    n_steps = 0
    y_true_batches = []
    y_prob_batches = []
    y_mask_batches = []
    is_multitask = getattr(model, "is_multitask", False)
    y_true_by_task = {}
    y_prob_by_task = {}
    y_mask_by_task = {}

    autocast_enabled = device == "cuda"
    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    for step, batch in enumerate(data_loader):
        if max_steps is not None and step >= max_steps:
            break
        if len(batch) == 3:
            data, ground_truth, path = batch
            targets = None
            masks = None
            tabular = None
        elif len(batch) == 4:
            data, targets, masks, path = batch
            tabular = None
        else:
            data, tabular, targets, masks, path = batch
        if step % 100 == 0:
            gc.collect()
            if show_figs:
                plt.imshow(torch.clip(data[0, 0], 0, 100), cmap='gray', interpolation=None)
                plt.title(path[0].split(os.sep)[-1])
                plt.show()
                plt.close()

        data = data.to(device)
        if tabular is not None:
            tabular = tabular.to(device)
        if targets is None:
            ground_truth = ground_truth.to(device)
        else:
            targets = {k: v.to(device) for k, v in targets.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
            logits = model(data) if tabular is None else model(data, tabular)
            if targets is None:
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    target = ground_truth.float().view(-1, 1)
                    loss = criterion(logits, target)
                else:
                    loss = criterion(logits, ground_truth)
            else:
                # Multi-task: criterion is a dict {task: (loss_fn, output_dim)}.
                losses = []
                for task, (loss_fn, output_dim) in criterion.items():
                    task_logits = logits[task]
                    task_mask = masks[task].view(-1)
                    if output_dim == 1:
                        task_target = targets[task].float().view(-1)
                        per = loss_fn(task_logits.view(-1), task_target)
                        per = per * task_mask
                        denom = task_mask.sum().clamp(min=1.0)
                        losses.append(per.sum() / denom)
                    else:
                        task_target = targets[task].long().view(-1)
                        per = loss_fn(task_logits, task_target)
                        per = per * task_mask
                        denom = task_mask.sum().clamp(min=1.0)
                        losses.append(per.sum() / denom)
                loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

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
            if targets is None:
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    probs = torch.sigmoid(logits).view(-1)
                else:
                    probs = torch.softmax(logits, dim=1)
                y_true_batches.append(ground_truth.detach().cpu())
                y_prob_batches.append(probs.detach().cpu())
            else:
                for task, (_, output_dim) in criterion.items():
                    task_logits = logits[task]
                    if output_dim == 1:
                        probs = torch.sigmoid(task_logits).view(-1)
                        y_true_by_task.setdefault(task, []).append(targets[task].detach().cpu().view(-1))
                        y_prob_by_task.setdefault(task, []).append(probs.detach().cpu().view(-1))
                        y_mask_by_task.setdefault(task, []).append(masks[task].detach().cpu().view(-1))
                    else:
                        probs = torch.softmax(task_logits, dim=1)
                        y_true_by_task.setdefault(task, []).append(targets[task].detach().cpu().view(-1))
                        y_prob_by_task.setdefault(task, []).append(probs.detach().cpu())
                        y_mask_by_task.setdefault(task, []).append(masks[task].detach().cpu().view(-1))

        _wandb_log({f"loss/{suffix}": float(loss.item())}, commit=False)
        epoch_loss += float(loss.item())
        n_steps += 1
        print(f"{suffix} step loss: {loss.item()}")

    avg_epoch_loss = epoch_loss / max(1, n_steps)
    _wandb_log({f"average_loss/{suffix}": avg_epoch_loss})
    print(f"{suffix}: \tAverage Loss: {avg_epoch_loss:.6f}")
    if is_multitask:
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
    if not y_true_batches or not y_prob_batches:
        return torch.empty(0), torch.empty(0), float("nan")
    return torch.cat(y_true_batches, dim=0), torch.cat(y_prob_batches, dim=0), avg_epoch_loss


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head_in_features: int,
        task_output_dims: dict,
        head_dropout: float,
        tabular_in_dim: int = 0,
        tabular_hidden_dim: int = 128,
        tabular_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.task_output_dims = dict(task_output_dims)
        self.task_names = list(task_output_dims.keys())
        self.is_multitask = True

        self.tabular_in_dim = int(tabular_in_dim or 0)
        self.tabular_hidden_dim = int(tabular_hidden_dim)
        self.tabular = None
        fusion_in_features = int(head_in_features)
        if self.tabular_in_dim > 0:
            self.tabular = nn.Sequential(
                nn.Linear(self.tabular_in_dim, self.tabular_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=float(tabular_dropout)),
                nn.Linear(self.tabular_hidden_dim, self.tabular_hidden_dim),
                nn.ReLU(),
            )
            fusion_in_features = fusion_in_features + self.tabular_hidden_dim

        heads = {}
        for name, out_dim in self.task_output_dims.items():
            if head_dropout and head_dropout > 0:
                heads[name] = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(fusion_in_features, out_dim))
            else:
                heads[name] = nn.Linear(fusion_in_features, out_dim)
        self.heads = nn.ModuleDict(heads)

    def forward(self, x, tabular: Optional[torch.Tensor] = None):
        feats = self.backbone(x)
        if self.tabular is not None:
            if tabular is None:
                raise ValueError("MultiTaskModel was built with tabular features but none were provided.")
            tab = self.tabular(tabular)
            feats = torch.cat([feats, tab], dim=1)
        return {name: head(feats) for name, head in self.heads.items()}


def _build_multitask_model(
    model_name: str,
    task_output_dims: dict,
    no_pretrained: bool,
    head_dropout: float,
    tabular_in_dim: int = 0,
    tabular_hidden_dim: int = 128,
    tabular_dropout: float = 0.1,
) -> MultiTaskModel:
    if model_name == "resnet50":
        weights = None if no_pretrained else ResNet50_Weights.DEFAULT
        try:
            base = resnet50(weights=weights)
        except Exception as e:
            print(f"Warning: failed to load pretrained weights ({e}); falling back to random init.")
            base = resnet50(weights=None)

        base.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        return MultiTaskModel(
            base,
            head_in_features=in_features,
            task_output_dims=task_output_dims,
            head_dropout=head_dropout,
            tabular_in_dim=tabular_in_dim,
            tabular_hidden_dim=tabular_hidden_dim,
            tabular_dropout=tabular_dropout,
        )

    if model_name == "vit_b_16":
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
        return MultiTaskModel(
            base,
            head_in_features=head_in_features,
            task_output_dims=task_output_dims,
            head_dropout=head_dropout,
            tabular_in_dim=tabular_in_dim,
            tabular_hidden_dim=tabular_hidden_dim,
            tabular_dropout=tabular_dropout,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _balanced_subset_indices(labels: pd.Series, n: int, seed: int) -> List[int]:
    labels = labels.reset_index(drop=True)
    unique = sorted(labels.unique().tolist())
    if not unique:
        return []

    rng = np.random.default_rng(seed)
    per_class = max(1, n // len(unique))
    chosen: set[int] = set()

    for value in unique:
        choices = np.flatnonzero(labels.to_numpy() == value)
        if len(choices) == 0:
            continue
        take = min(per_class, len(choices), n - len(chosen))
        if take <= 0:
            break
        chosen.update(rng.choice(choices, size=take, replace=False).tolist())

    max_n = min(n, len(labels))
    while len(chosen) < max_n:
        chosen.add(int(rng.integers(0, len(labels))))

    return sorted(chosen)


@torch.no_grad()
def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def _freeze_backbone(model: nn.Module, model_name: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if getattr(model, "is_multitask", False):
        for p in model.heads.parameters():
            p.requires_grad = True
        return
    if model_name == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name == "vit_b_16":
        for p in model.heads.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown model_name for freezing: {model_name}")


def _unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def _split_head_backbone_params(model: nn.Module, model_name: str):
    if getattr(model, "is_multitask", False):
        head_module = model.heads
    elif model_name == "resnet50":
        head_module = model.fc
    elif model_name == "vit_b_16":
        head_module = model.heads
    else:
        raise ValueError(f"Unknown model_name for param split: {model_name}")

    head_ids = {id(p) for p in head_module.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return head_params, backbone_params


def _build_model(model_name: str, out_features: int, no_pretrained: bool, head_dropout: float) -> nn.Module:
    if model_name == "resnet50":
        weights = None if no_pretrained else ResNet50_Weights.DEFAULT
        try:
            model = resnet50(weights=weights)
        except Exception as e:
            print(f"Warning: failed to load pretrained weights ({e}); falling back to random init.")
            model = resnet50(weights=None)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if head_dropout and head_dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(model.fc.in_features, out_features))
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, out_features)
        return model

    if model_name == "vit_b_16":
        weights = None if no_pretrained else ViT_B_16_Weights.DEFAULT
        try:
            model = vit_b_16(weights=weights)
        except Exception as e:
            print(f"Warning: failed to load pretrained weights ({e}); falling back to random init.")
            model = vit_b_16(weights=None)

        # Adapt patch embedding to 1-channel input (use mean over RGB channels if pretrained).
        old = model.conv_proj
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
        model.conv_proj = new
        if head_dropout and head_dropout > 0:
            model.heads = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(model.hidden_dim, out_features))
        else:
            model.heads = nn.Sequential(nn.Linear(model.hidden_dim, out_features))
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def overfit_sanity_check(
    model: nn.Module,
    dataset: "X_rayImageDataset",
    device: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int = 0,
    max_samples: int = 32,
) -> None:
    """
    Sanity check: try to overfit on a tiny (roughly balanced) subset.
    If this does not reach very high train accuracy, something is likely wrong (labels/images/pipeline).
    """
    _set_seed(seed)
    subset_idx = _balanced_subset_indices(dataset.img_labels[dataset.label_column], max_samples, seed=seed)
    if not subset_idx:
        raise RuntimeError("Sanity check failed: no samples available in dataset.")

    subset = Subset(dataset, subset_idx)
    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler(device=device, enabled=(device == "cuda"))
    model.train()

    autocast_enabled = device == "cuda"
    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    print(f"=== Sanity overfit: samples={len(subset)} epochs={epochs} lr={learning_rate} device={device} ===")
    for epoch in range(epochs):
        losses = []
        accs = []
        y_true_all = []
        y_score_all = []
        for data, target, _ in loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
                logits = model(data)
                if logits.ndim == 2 and logits.shape[1] == 1:
                    loss = criterion_bce(logits, target.float().view(-1, 1))
                else:
                    loss = criterion_ce(logits, target)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            losses.append(float(loss.item()))
            with torch.no_grad():
                if logits.ndim == 2 and logits.shape[1] == 1:
                    probs = torch.sigmoid(logits).view(-1)
                    preds = (probs >= 0.5).long()
                    accs.append(float((preds == target).float().mean().item()))
                    y_true_all.append(target.detach().cpu())
                    y_score_all.append(probs.detach().cpu())
                else:
                    accs.append(_accuracy_from_logits(logits.detach(), target.detach()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_acc = float(np.mean(accs)) if accs else float("nan")
        auc_val = None
        if y_true_all and y_score_all:
            y_true_np = torch.cat(y_true_all, dim=0).numpy()
            y_score_np = torch.cat(y_score_all, dim=0).numpy()
            if len(np.unique(y_true_np)) > 1:
                try:
                    auc_val = float(roc_auc_score(y_true_np, y_score_np))
                except Exception:
                    auc_val = None
        if auc_val is None:
            print(f"sanity epoch {epoch + 1}/{epochs}: loss={mean_loss:.4f} acc={mean_acc:.3f}")
        else:
            print(f"sanity epoch {epoch + 1}/{epochs}: loss={mean_loss:.4f} acc={mean_acc:.3f} auc={auc_val:.3f}")

        if mean_acc >= 0.98 and mean_loss <= 0.10:
            print("=== Sanity overfit PASSED (early stop) ===")
            return

    print("=== Sanity overfit finished (did not early-stop) ===")


def log_classification_metrics(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    avg_loss: float,
    suffix: str,
    epoch: int,
    fixed_threshold: float = None,
) -> dict:
    y_true_np = y_true.detach().cpu().numpy()
    y_prob_np = y_probs.detach().cpu().numpy()
    if y_true_np.size == 0:
        metrics = {
            f"epoch/{suffix}": epoch,
            f"average_loss/{suffix}": float(avg_loss) if avg_loss == avg_loss else float("nan"),
            f"accuracy/{suffix}": float("nan"),
            f"balanced_accuracy/{suffix}": float("nan"),
            f"precision/{suffix}": float("nan"),
            f"recall/{suffix}": float("nan"),
            f"f1/{suffix}": float("nan"),
            f"pos_rate/{suffix}": float("nan"),
        }
        _wandb_log(metrics)
        print(f"{suffix} metrics (epoch {epoch}): empty split (no samples)")
        return metrics

    is_binary_scores = y_prob_np.ndim == 1 or (y_prob_np.ndim == 2 and y_prob_np.shape[1] == 1)
    if is_binary_scores:
        scores = y_prob_np.reshape(-1)
        y_pred_np = (scores >= 0.5).astype(int)
        pos_rate = float((y_true_np == 1).mean())
    else:
        scores = y_prob_np[:, 1] if y_prob_np.shape[1] == 2 else None
        y_pred_np = y_prob_np.argmax(axis=1)
        pos_rate = float((y_true_np == 1).mean()) if y_prob_np.shape[1] == 2 else float("nan")

    unique_true = np.unique(y_true_np) if y_true_np.size else np.array([])
    bal_acc_val = float("nan")
    try:
        if unique_true.size >= 2:
            bal_acc_val = float(balanced_accuracy_score(y_true_np, y_pred_np))
    except Exception:
        pass

    metrics = {
        f"epoch/{suffix}": epoch,
        f"average_loss/{suffix}": float(avg_loss),
        f"accuracy/{suffix}": float(accuracy_score(y_true_np, y_pred_np)),
        f"balanced_accuracy/{suffix}": bal_acc_val,
        f"pos_rate/{suffix}": pos_rate,
        f"pred_pos_rate@0.5/{suffix}": float((y_pred_np == 1).mean()) if np.isfinite(pos_rate) else float("nan"),
    }

    # Metrics at a fixed threshold (e.g. chosen on val and reused on test to avoid leakage).
    if fixed_threshold is not None and (is_binary_scores or (y_prob_np.ndim == 2 and y_prob_np.shape[1] == 2)):
        score_vec = scores if scores is not None else y_prob_np[:, 1]
        y_pred_fixed = (score_vec >= float(fixed_threshold)).astype(int)
        precision_fx, recall_fx, f1_fx, _ = precision_recall_fscore_support(
            y_true_np,
            y_pred_fixed,
            average="binary",
            zero_division=0,
        )
        metrics.update(
            {
                f"fixed_threshold/{suffix}": float(fixed_threshold),
                f"accuracy@fixed_thr/{suffix}": float(accuracy_score(y_true_np, y_pred_fixed)),
                f"balanced_accuracy@fixed_thr/{suffix}": float(balanced_accuracy_score(y_true_np, y_pred_fixed)) if unique_true.size >= 2 else float("nan"),
                f"precision@fixed_thr/{suffix}": float(precision_fx),
                f"recall@fixed_thr/{suffix}": float(recall_fx),
                f"f1@fixed_thr/{suffix}": float(f1_fx),
                f"pred_pos_rate@fixed_thr/{suffix}": float((y_pred_fixed == 1).mean()),
            }
        )

    # Precision/Recall/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        average="binary" if (is_binary_scores or (y_prob_np.ndim == 2 and y_prob_np.shape[1] == 2)) else "macro",
        zero_division=0,
    )
    metrics.update({
        f"precision/{suffix}": float(precision),
        f"recall/{suffix}": float(recall),
        f"f1/{suffix}": float(f1),
    })

    # AUC
    try:
        if is_binary_scores:
            metrics[f"auc/{suffix}"] = float(roc_auc_score(y_true_np, scores))
        elif y_prob_np.shape[1] == 2:
            metrics[f"auc/{suffix}"] = float(roc_auc_score(y_true_np, y_prob_np[:, 1]))
        else:
            metrics[f"auc_macro_ovr/{suffix}"] = float(
                roc_auc_score(y_true_np, y_prob_np, multi_class="ovr", average="macro")
            )
    except Exception:
        pass

    # Confusion matrix (numeric + optional wandb plot)
    try:
        if is_binary_scores or (y_prob_np.ndim == 2 and y_prob_np.shape[1] == 2):
            cm = confusion_matrix(y_true_np.astype(int), y_pred_np.astype(int), labels=[0, 1])
            tn, fp, fn, tp = cm.ravel().tolist()
            metrics.update(
                {
                    f"cm/tn/{suffix}": int(tn),
                    f"cm/fp/{suffix}": int(fp),
                    f"cm/fn/{suffix}": int(fn),
                    f"cm/tp/{suffix}": int(tp),
                }
            )
        else:
            labels = np.arange(int(y_prob_np.shape[1]))
            cm = confusion_matrix(y_true_np.astype(int), y_pred_np.astype(int), labels=labels)
            # Log the diagonal sum as a sanity metric (equal to accuracy * n) without shipping a big matrix.
            metrics[f"cm/diag_sum/{suffix}"] = int(np.trace(cm))
    except Exception:
        cm = None

    # Score histogram + best threshold (binary only)
    if is_binary_scores or (y_prob_np.ndim == 2 and y_prob_np.shape[1] == 2):
        score_vec = scores if scores is not None else y_prob_np[:, 1]
        try:
            fpr, tpr, thresholds = roc_curve(y_true_np, score_vec)
            best_idx = int(np.argmax(tpr - fpr))
            best_thr = float(thresholds[best_idx])
            y_pred_best = (score_vec >= best_thr).astype(int)
            precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(
                y_true_np,
                y_pred_best,
                average="binary",
                zero_division=0,
            )
            metrics.update(
                {
                    f"best_threshold/{suffix}": best_thr,
                    f"tpr_at_best/{suffix}": float(tpr[best_idx]),
                    f"fpr_at_best/{suffix}": float(fpr[best_idx]),
                    f"accuracy@best_thr/{suffix}": float(accuracy_score(y_true_np, y_pred_best)),
                    f"balanced_accuracy@best_thr/{suffix}": float(balanced_accuracy_score(y_true_np, y_pred_best)) if unique_true.size >= 2 else float("nan"),
                    f"precision@best_thr/{suffix}": float(precision_b),
                    f"recall@best_thr/{suffix}": float(recall_b),
                    f"f1@best_thr/{suffix}": float(f1_b),
                    f"pred_pos_rate@best_thr/{suffix}": float((y_pred_best == 1).mean()),
                }
            )

            # Also compute the threshold that maximizes F1 directly (often more meaningful under imbalance).
            try:
                pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true_np, score_vec)
                if pr_thr.size > 0:
                    pr_f1 = (2 * pr_prec[:-1] * pr_rec[:-1]) / (pr_prec[:-1] + pr_rec[:-1] + 1e-12)
                    best_f1_idx = int(np.nanargmax(pr_f1))
                    best_thr_f1 = float(pr_thr[best_f1_idx])
                    y_pred_f1 = (score_vec >= best_thr_f1).astype(int)
                    precision_f, recall_f, f1_f, _ = precision_recall_fscore_support(
                        y_true_np,
                        y_pred_f1,
                        average="binary",
                        zero_division=0,
                    )
                    metrics.update(
                        {
                            f"best_threshold_f1/{suffix}": best_thr_f1,
                            f"precision@best_f1_thr/{suffix}": float(precision_f),
                            f"recall@best_f1_thr/{suffix}": float(recall_f),
                            f"f1@best_f1_thr/{suffix}": float(f1_f),
                            f"pred_pos_rate@best_f1_thr/{suffix}": float((y_pred_f1 == 1).mean()),
                        }
                    )
            except Exception:
                pass
        except Exception:
            pass

        if _wandb_is_active():
            try:
                wandb.log({f"scores/{suffix}": wandb.Histogram(score_vec)}, commit=False)
            except Exception:
                pass

    _wandb_log(metrics)
    print(
        f"{suffix} metrics (epoch {epoch}): "
        f"loss={metrics[f'average_loss/{suffix}']:.4f} "
        f"acc={metrics[f'accuracy/{suffix}']:.3f} "
        f"bal_acc={metrics[f'balanced_accuracy/{suffix}']:.3f} "
        f"prec={metrics[f'precision/{suffix}']:.3f} "
        f"rec={metrics[f'recall/{suffix}']:.3f} "
        f"f1={metrics[f'f1/{suffix}']:.3f}"
        + (f" pos_rate={metrics.get(f'pos_rate/{suffix}', float('nan')):.3f}")
        + (f" auc={metrics.get(f'auc/{suffix}', metrics.get(f'auc_macro_ovr/{suffix}', float('nan'))):.3f}" if any(k.startswith("auc/") or k.startswith("auc_macro_ovr/") for k in metrics) else "")
        + (f" best_thr={metrics.get(f'best_threshold/{suffix}', float('nan')):.4f}" if f"best_threshold/{suffix}" in metrics else "")
        + (f" f1@best={metrics.get(f'f1@best_thr/{suffix}', float('nan')):.3f}" if f"f1@best_thr/{suffix}" in metrics else "")
        + (f" f1@bestF1={metrics.get(f'f1@best_f1_thr/{suffix}', float('nan')):.3f}" if f"f1@best_f1_thr/{suffix}" in metrics else "")
    )

    # Confusion matrix (wandb plot)
    if _wandb_is_active():
        try:
            if is_binary_scores:
                class_names = ["0", "1"]
            else:
                class_names = [str(i) for i in range(int(y_prob_np.shape[1]))]
            wandb.log(
                {
                    f"confusion_matrix/{suffix}": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true_np,
                        preds=y_pred_np,
                        class_names=class_names,
                    )
                }
            )
        except Exception:
            pass

    return metrics


def log_multitask_metrics(
    results_by_task: dict,
    avg_loss: float,
    suffix: str,
    epoch: int,
    task_output_dims: dict,
    task_index_to_label: dict,
    fixed_thresholds: Optional[dict] = None,
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
        # compact: only for eval + final/best outputs (avoid 16x charts for train)
        return suffix in {"eval", "final_test", "best_test"}

    for task, data in results_by_task.items():
        y_true = data["y_true"].detach().cpu().numpy()
        y_prob = data["y_prob"].detach().cpu().numpy()
        y_mask = data["y_mask"].detach().cpu().numpy().astype(bool)
        out_dim = int(task_output_dims[task])

        if y_true.size == 0 or y_mask.sum() == 0:
            continue

        y_true = y_true[y_mask]
        unique_true = np.unique(y_true) if y_true.size else np.array([])
        if out_dim == 1:
            scores = y_prob.reshape(-1)[y_mask]
            y_pred = (scores >= 0.5).astype(int)

            acc = float(accuracy_score(y_true, y_pred))
            if unique_true.size >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            else:
                bal_acc = float("nan")
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="binary",
                zero_division=0,
            )
            task_metrics = {}
            if _should_log_task_metrics() or wandb_detail == "full":
                if wandb_detail == "full":
                    task_metrics.update(
                        {
                            f"task/{task}/accuracy/{suffix}": acc,
                            f"task/{task}/balanced_accuracy/{suffix}": bal_acc,
                            f"task/{task}/precision/{suffix}": float(precision),
                            f"task/{task}/recall/{suffix}": float(recall),
                            f"task/{task}/f1/{suffix}": float(f1),
                            f"task/{task}/pos_rate/{suffix}": float((y_true == 1).mean()),
                            f"task/{task}/pred_pos_rate@0.5/{suffix}": float((y_pred == 1).mean()),
                        }
                    )
                else:
                    # compact
                    task_metrics.update(
                        {
                            f"task/{task}/balanced_accuracy/{suffix}": bal_acc,
                            f"task/{task}/f1/{suffix}": float(f1),
                            f"task/{task}/recall/{suffix}": float(recall),
                        }
                    )
            try:
                if len(np.unique(y_true)) > 1:
                    task_metrics[f"task/{task}/auc/{suffix}"] = float(roc_auc_score(y_true, scores))
            except Exception:
                pass

            try:
                cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
                tn, fp, fn, tp = cm.ravel().tolist()
                if wandb_detail == "full":
                    task_metrics.update(
                        {
                            f"task/{task}/cm/tn/{suffix}": int(tn),
                            f"task/{task}/cm/fp/{suffix}": int(fp),
                            f"task/{task}/cm/fn/{suffix}": int(fn),
                            f"task/{task}/cm/tp/{suffix}": int(tp),
                        }
                    )
            except Exception:
                pass

            # Optional fixed threshold (e.g. from val).
            if fixed_thresholds and task in fixed_thresholds and fixed_thresholds[task] is not None:
                thr = float(fixed_thresholds[task])
                y_pred_fx = (scores >= thr).astype(int)
                p_fx, r_fx, f1_fx, _ = precision_recall_fscore_support(
                    y_true,
                    y_pred_fx,
                    average="binary",
                    zero_division=0,
                )
                if wandb_detail == "full":
                    task_metrics.update(
                        {
                            f"task/{task}/fixed_threshold/{suffix}": thr,
                            f"task/{task}/f1@fixed_thr/{suffix}": float(f1_fx),
                            f"task/{task}/precision@fixed_thr/{suffix}": float(p_fx),
                            f"task/{task}/recall@fixed_thr/{suffix}": float(r_fx),
                        }
                    )
                else:
                    task_metrics.update(
                        {
                            f"task/{task}/fixed_threshold/{suffix}": thr,
                            f"task/{task}/f1@fixed_thr/{suffix}": float(f1_fx),
                        }
                    )

            # Optional threshold search on this split (for logging / selecting on eval).
            if compute_thresholds and unique_true.size > 1:
                try:
                    fpr, tpr, thresholds = roc_curve(y_true, scores)
                    best_idx = int(np.argmax(tpr - fpr))
                    best_thr_youden = float(thresholds[best_idx])
                    task_metrics[f"task/{task}/best_threshold_youden/{suffix}"] = best_thr_youden
                except Exception:
                    pass
                try:
                    pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true, scores)
                    if pr_thr.size > 0:
                        pr_f1 = (2 * pr_prec[:-1] * pr_rec[:-1]) / (pr_prec[:-1] + pr_rec[:-1] + 1e-12)
                        best_f1_idx = int(np.nanargmax(pr_f1))
                        best_thr_f1 = float(pr_thr[best_f1_idx])
                        task_metrics[f"task/{task}/best_threshold_f1/{suffix}"] = best_thr_f1
                except Exception:
                    pass

            metrics_all.update(task_metrics)
            per_task_acc.append(acc)
            per_task_f1.append(float(f1))
            auc_val = task_metrics.get(f"task/{task}/auc/{suffix}")
            if auc_val is not None:
                per_task_auc.append(float(auc_val))

            if wandb_detail == "full":
                print(
                    f"{suffix}/{task}: "
                    f"acc={acc:.3f} bal_acc={bal_acc:.3f} "
                    f"prec={float(precision):.3f} rec={float(recall):.3f} f1={float(f1):.3f}"
                    + (f" auc={float(auc_val):.3f}" if auc_val is not None else "")
                )

                if _wandb_is_active() and suffix in {"final_test", "best_test"}:
                    try:
                        wandb.log(
                            {
                                f"confusion_matrix/{suffix}/{task}": wandb.plot.confusion_matrix(
                                    probs=None,
                                    y_true=y_true.astype(int),
                                    preds=y_pred.astype(int),
                                    class_names=["0", "1"],
                                )
                            },
                            commit=False,
                        )
                    except Exception:
                        pass
            continue

        # Multiclass task
        probs = y_prob[y_mask]
        y_pred = probs.argmax(axis=1)
        acc = float(accuracy_score(y_true, y_pred))
        if unique_true.size >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        else:
            bal_acc = float("nan")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        task_metrics = {}
        if _should_log_task_metrics() or wandb_detail == "full":
            if wandb_detail == "full":
                task_metrics.update(
                    {
                        f"task/{task}/accuracy/{suffix}": acc,
                        f"task/{task}/balanced_accuracy/{suffix}": bal_acc,
                        f"task/{task}/precision_macro/{suffix}": float(precision),
                        f"task/{task}/recall_macro/{suffix}": float(recall),
                        f"task/{task}/f1_macro/{suffix}": float(f1),
                    }
                )
            else:
                task_metrics.update(
                    {
                        f"task/{task}/balanced_accuracy/{suffix}": bal_acc,
                        f"task/{task}/f1_macro/{suffix}": float(f1),
                        f"task/{task}/recall_macro/{suffix}": float(recall),
                    }
                )

        try:
            if unique_true.size > 1:
                task_metrics[f"task/{task}/auc_macro_ovr/{suffix}"] = float(
                    roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
                )
        except Exception:
            pass

        try:
            cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=np.arange(out_dim))
            if wandb_detail == "full":
                task_metrics[f"task/{task}/cm/diag_sum/{suffix}"] = int(np.trace(cm))
        except Exception:
            pass

        metrics_all.update(task_metrics)
        per_task_acc.append(acc)
        per_task_f1.append(float(f1))
        auc_val = task_metrics.get(f"task/{task}/auc_macro_ovr/{suffix}")
        if auc_val is not None:
            per_task_auc.append(float(auc_val))

        if wandb_detail == "full":
            print(
                f"{suffix}/{task}: "
                f"acc={acc:.3f} bal_acc={bal_acc:.3f} "
                f"prec_macro={float(precision):.3f} rec_macro={float(recall):.3f} f1_macro={float(f1):.3f}"
                + (f" auc_macro_ovr={float(auc_val):.3f}" if auc_val is not None else "")
            )

            if _wandb_is_active() and suffix in {"final_test", "best_test"}:
                try:
                    class_names = task_index_to_label.get(task) or [str(i) for i in range(out_dim)]
                    wandb.log(
                        {
                            f"confusion_matrix/{suffix}/{task}": wandb.plot.confusion_matrix(
                                probs=None,
                                y_true=y_true.astype(int),
                                preds=y_pred.astype(int),
                                class_names=[str(x) for x in class_names],
                            )
                        },
                        commit=False,
                    )
                except Exception:
                    pass

    if per_task_acc:
        metrics_all[f"macro_accuracy/{suffix}"] = float(np.mean(per_task_acc))
    if per_task_f1:
        metrics_all[f"macro_f1/{suffix}"] = float(np.mean(per_task_f1))
    if per_task_auc:
        metrics_all[f"macro_auc/{suffix}"] = float(np.mean(per_task_auc))

    # Reduce wandb clutter in compact mode: always log macro metrics, and only selected task metrics.
    metrics_log = {
        f"epoch/{suffix}": metrics_all[f"epoch/{suffix}"],
        f"average_loss/{suffix}": metrics_all[f"average_loss/{suffix}"],
    }
    for k in (f"macro_accuracy/{suffix}", f"macro_f1/{suffix}", f"macro_auc/{suffix}"):
        if k in metrics_all:
            metrics_log[k] = metrics_all[k]

    if _should_log_task_metrics():
        for k, v in metrics_all.items():
            if not k.startswith("task/"):
                continue
            # compact: only log f1/f1_macro + auc* + balanced_accuracy + fixed_threshold + f1@fixed_thr
            if wandb_detail == "full":
                metrics_log[k] = v
                continue
            if any(
                token in k
                for token in (
                    f"/balanced_accuracy/{suffix}",
                    f"/f1/{suffix}",
                    f"/recall/{suffix}",
                    f"/f1_macro/{suffix}",
                    f"/recall_macro/{suffix}",
                    f"/auc/{suffix}",
                    f"/auc_macro_ovr/{suffix}",
                    f"/fixed_threshold/{suffix}",
                    f"/f1@fixed_thr/{suffix}",
                    f"/best_threshold_f1/{suffix}",
                )
            ):
                metrics_log[k] = v

    _wandb_log(metrics_log)
    print(
        f"{suffix} macro: "
        f"loss={metrics_all.get(f'average_loss/{suffix}', float('nan')):.4f} "
        f"macro_acc={metrics_all.get(f'macro_accuracy/{suffix}', float('nan')):.3f} "
        f"macro_f1={metrics_all.get(f'macro_f1/{suffix}', float('nan')):.3f} "
        f"macro_auc={metrics_all.get(f'macro_auc/{suffix}', float('nan')):.3f}"
    )

    return metrics_all


def check_mapping_merge(
    base_folder: str,
    image_dir: str,
    metadata_file: str,
    mapping_file: str,
    sample_n: int = 20,
    seed: int = 0,
) -> None:
    """
    Diagnose whether the metadata<->mapping merge looks plausible.
    Prints merge coverage, key consistency, and whether referenced image files exist on disk.
    """
    _set_seed(seed)
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

    if "fileID" not in merged.columns:
        print("No 'fileID' column after merge.")
        return

    # Consistency checks
    anf_to_fileid = merged[merged["fileID"] != -1].groupby("Anforderungsnummer")["fileID"].nunique()
    inconsistent_anf = anf_to_fileid[anf_to_fileid > 1]
    print(f"Anforderungsnummer with >1 fileID: {len(inconsistent_anf)}")
    if len(inconsistent_anf) > 0:
        print(inconsistent_anf.sort_values(ascending=False).head(10))

    fileid_counts = merged[merged["fileID"] != -1]["fileID"].value_counts()
    print(f"Unique fileIDs: {int(fileid_counts.index.nunique())}")
    print("Top repeated fileIDs (should be low):")
    print(fileid_counts.head(10))

    # File existence check
    sample = merged[merged["fileID"] != -1].sample(n=min(sample_n, matched), random_state=seed) if matched else merged.head(0)
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

    # Global missing rate (can be slow, so keep it simple)
    # Only check the first N matched rows deterministically
    head_n = min(1000, matched)
    if head_n > 0:
        head = merged[merged["fileID"] != -1].head(head_n)
        miss = 0
        for fid in head["fileID"].astype(int).tolist():
            if not os.path.exists(_find_xray_path(image_dir, str(fid))):
                miss += 1
        print(f"Missing files in first {head_n} matched rows: {miss}/{head_n}")


def _dedupe_by_file_id(metadata: pd.DataFrame, label_col, drop_conflicts: bool = True) -> pd.DataFrame:
    """
    Make training/eval rows 1:1 with images by collapsing duplicate `fileID` rows.
    For the label column, uses the mode (most frequent value). If there's a tie / conflict and
    `drop_conflicts=True`, the entire fileID group is dropped.
    Other columns keep the first row (they should be identical for duplicates anyway).
    """
    if "fileID" not in metadata.columns:
        return metadata
    label_cols = [label_col] if isinstance(label_col, str) else list(label_col)
    label_cols = [c for c in label_cols if c in metadata.columns]
    if not label_cols:
        return metadata

    grouped = metadata.groupby("fileID", dropna=False)
    rows = []
    dropped = 0

    for _, group in grouped:
        row = group.iloc[0].copy()
        conflict = False
        for col in label_cols:
            counts = group[col].value_counts(dropna=False)
            if len(counts) == 0:
                continue
            top = counts.iloc[0]
            modes = counts[counts == top].index.tolist()
            if drop_conflicts and len(modes) > 1:
                conflict = True
                break
            row[col] = modes[0]
        if conflict:
            dropped += 1
            continue
        rows.append(row)

    result = pd.DataFrame(rows).reset_index(drop=True)
    before = len(metadata)
    after = len(result)
    dupes = int(before - grouped.ngroups)
    print(
        f"Deduped by fileID: rows {before} -> {after}, "
        f"duplicate_rows={dupes}, dropped_conflicts={dropped}"
    )
    return result


def compute_roc_one_vs_rest(y_true, y_scores, classes, category_label):
    label_binarizer = LabelBinarizer().fit(classes)
    y_onehot_test = label_binarizer.transform(y_true)
    fpr, tpr, _ = roc_curve(y_onehot_test.ravel(), y_scores.ravel())
    roc_auc_value = auc(fpr, tpr)
    _wandb_log({"multi_roc_auc/" + category_label: roc_auc_value})

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(classes, colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_scores[:, class_id],
            name=f"ROC curve for {str(class_id)}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
            despine=True,
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Receiver Operating Characteristic (One-vs-Rest multiclass) for " + category_label,
    )


def compute_roc_curve(y_true, y_scores, plot=False, title_suffix=''):
    """
    Compute the ROC curve and AUC (Area Under the Curve).

    Parameters:
    - y_true: Ground truth labels (true binary labels).
    - y_scores: Predicted scores or probabilities for positive class.
    - plot: Boolean to define if computed ROC AUC shall be plotted or not. Default is False.
    - title_suffix: string to be appended to the plot title in case plot=True.
    If plot is False, this parameter is ignored.

    Returns:
    - fpr: False Positive Rate (1 - Specificity).
    - tpr: True Positive Rate (Sensitivity).
    - roc_auc: Area Under the ROC Curve (AUC).
    """
    y_true_np = y_true.detach().cpu().numpy()
    if y_scores.ndim == 2 and y_scores.shape[1] >= 2:
        y_score_np = y_scores[:, 1].detach().cpu().numpy()
    else:
        y_score_np = y_scores.detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true_np, y_score_np)
    best_index = np.argmax(tpr - fpr)
    print("Youden Index: ", thresholds[best_index])
    print(f"FPR: {fpr[best_index]}, TPR: {tpr[best_index]}")
    print(f"Specificity: {1 - fpr[best_index]}, Sensitivity: {tpr[best_index]}")
    roc_auc = auc(fpr, tpr)
    auc_key = "AUC/" + title_suffix
    _wandb_log({auc_key: roc_auc})

    if plot:
        # ROC plot
        if title_suffix is None:
            title_suffix = ''
        plt.title('Receiver Operating Characteristic: ' + title_suffix)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        #plt.show()

        title_suffix = title_suffix.replace(' ', '_')
        title_suffix = title_suffix.replace('resnet50_', '')
        title_suffix = title_suffix.replace('resnet18_', '')
        roc_key = "roc_auc/" + title_suffix
        spec_key = "Specificity (Youden)/" + title_suffix
        sens_key = "Sensitivity (Youden)/" + title_suffix
        if _wandb_is_active():
            wandb.log({
                roc_key: wandb.Image(plt),
                spec_key: 1 - fpr[best_index],
                sens_key: tpr[best_index],
            })
        plt.close()

    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "best_index_val": best_index, "roc_auc_val": roc_auc}


def _prepare_tabular_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Basic tabular preprocessing:
    - numeric columns: fill with train median
    - categorical columns: fill with "missing" and one-hot encode (train vocabulary), then align val/test
    """
    if not feature_cols:
        empty = np.zeros((len(train_df), 0), dtype=np.float32)
        return empty, np.zeros((len(val_df), 0), dtype=np.float32), np.zeros((len(test_df), 0), dtype=np.float32), []

    train_x = train_df[feature_cols].copy()
    val_x = val_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_x[c])]
    cat_cols = [c for c in feature_cols if c not in numeric_cols]

    for c in numeric_cols:
        med = train_x[c].median()
        train_x[c] = train_x[c].fillna(med)
        val_x[c] = val_x[c].fillna(med)
        test_x[c] = test_x[c].fillna(med)

    for c in cat_cols:
        train_x[c] = train_x[c].astype("string").fillna("missing")
        val_x[c] = val_x[c].astype("string").fillna("missing")
        test_x[c] = test_x[c].astype("string").fillna("missing")

    train_enc = pd.get_dummies(train_x, columns=cat_cols, dummy_na=False)
    val_enc = pd.get_dummies(val_x, columns=cat_cols, dummy_na=False)
    test_enc = pd.get_dummies(test_x, columns=cat_cols, dummy_na=False)

    val_enc = val_enc.reindex(columns=train_enc.columns, fill_value=0)
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

    feature_names = train_enc.columns.tolist()
    return (
        train_enc.to_numpy(dtype=np.float32),
        val_enc.to_numpy(dtype=np.float32),
        test_enc.to_numpy(dtype=np.float32),
        feature_names,
    )


if __name__ == '__main__':
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
    parser.add_argument("--label", default="mixed_shapes")
    parser.add_argument(
        "--labels",
        default=None,
        help="Multi-task mode: comma-separated label columns, or 'all' to train on all label columns (excluding general metadata columns).",
    )
    parser.add_argument(
        "--label-group",
        default="mixed",
        choices=["mixed", "rounded", "irregular", "large", "pleural", "occupational", "symbol", "general", "all"],
        help="Which metadata column group to load (used for feature/metadata passthrough; labels are controlled via --label or --labels).",
    )
    parser.add_argument(
        "--no-metadata-features",
        action="store_true",
        help="Disable using non-label metadata columns as additional tabular input features in multi-task mode.",
    )
    parser.add_argument("--tabular-hidden-dim", type=int, default=128)
    parser.add_argument("--tabular-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1, help="Run eval every N epochs (set 0 to disable periodic eval).")
    parser.add_argument("--test-every", type=int, default=0, help="Run test every N epochs (set 0 to disable periodic test).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", choices=["vit_b_16", "resnet50"], default="vit_b_16")
    parser.add_argument("--head-dropout", type=float, default=0.1, help="Dropout probability for the classifier head.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1, help="Multiplier for backbone LR vs head LR after unfreezing.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--balanced-sampler", action="store_true", help="Use a WeightedRandomSampler to balance classes in the train loader (binary only).")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--early-stop-metric",
        choices=["auc/eval", "f1/eval", "loss/eval", "macro_auc/eval", "macro_f1/eval"],
        default="auc/eval",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--sanity-overfit", action="store_true", help="Overfit a tiny subset to validate pipeline.")
    parser.add_argument("--sanity-samples", type=int, default=32)
    parser.add_argument("--sanity-epochs", type=int, default=50)
    parser.add_argument("--sanity-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-mapping", action="store_true", help="Print diagnostics for metadata<->mapping merge and exit.")
    parser.add_argument("--check-mapping-samples", type=int, default=20)
    parser.add_argument("--dedupe-by-fileid", action="store_true", help="Collapse duplicate fileID rows before splitting/training.")
    parser.add_argument("--drop-conflicting-fileid-labels", action="store_true", help="When deduping, drop fileIDs with conflicting labels.")
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
    parser.add_argument(
        "--wandb-detail",
        choices=["compact", "full"],
        default="compact",
        help="Controls how many per-task metrics are logged to wandb (compact reduces dashboard clutter).",
    )
    parser.add_argument(
        "--label-stats",
        action="store_true",
        help="Print per-label dataset counts (pos/neg for binary; category counts for non-binary) and exit.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    preprocess = Compose([
        RandomResizedCrop(224),
        RandomRotation(5),
        ColorJitter(0.3),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.5], std=[0.5]),
        #transforms.Normalize(0.5, 0.5),
    ])
    preprocess_no_aug = Compose([
        Resize((224, 224)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.5], std=[0.5]),
    ])

    # Datenvorverarbeitung: fehlende Werte handeln; einlesen
    base_folder = args.base_folder  #  "D:\\Projects\\Thorax\\DeboraThorax\\"  #
    root_folder = args.image_dir or base_folder
    mapping_file = os.path.join(base_folder, "mapping.csv")
    default_fold_folder = os.path.join(base_folder, "strat_dichotom_splits")
    fold_folder = args.fold_folder or default_fold_folder
    output_folder = args.output_folder or base_folder

    raw_metadata_file = os.path.join(base_folder, "dichotome_data_pseudonym.csv")
    metadata_file = os.path.join(base_folder, "dichotome_data_pseudonym.csv")
    anford_nr_file = os.path.join(base_folder, "table_pseudonym.csv")
    nan_thresh = 999
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    number_of_epochs = args.epochs
    fold = args.fold
    # column_groups keys are general, symbol, rounded, irregular, mixed, large, pleural, occupational
    column_group = args.label_group
    criterion_label = args.label  # single-label mode default

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

    # Resolve label columns (same behavior as training, but without starting training)
    if args.label_stats:
        # Load base metadata (same path logic as training)
        if os.path.isfile(metadata_file):
            md = pd.read_csv(metadata_file)
        else:
            md = Preprocessor_Metadata.prepare_metadata(raw_metadata_file, anford_nr_file, mapping_file, nan_thresh, True)
        md = _ensure_file_id(md, mapping_file)
        md = md[md["fileID"] != -1].reset_index(drop=True)
        if args.dedupe_by_fileid:
            # Use provided labels list if available, otherwise fall back to criterion_label for dedupe.
            dedupe_cols = args.labels if args.labels not in (None, "all") else criterion_label
            md = _dedupe_by_file_id(md, dedupe_cols, drop_conflicts=args.drop_conflicting_fileid_labels)

        # Determine label list
        column_groups = Preprocessor_Metadata.get_column_name_groups(md, True)
        general_cols = set(column_groups.get("general", []))
        if args.labels is None:
            label_cols = [criterion_label]
        else:
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

        print_label_stats(md, label_cols)
        raise SystemExit(0)

    # Split Train-Test:
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
        if args.dedupe_by_fileid:
            metadata = _dedupe_by_file_id(metadata, criterion_label, drop_conflicts=args.drop_conflicting_fileid_labels)
        metadata = Preprocessor_Metadata.create_splits(
            metadata,
            n_folds,
            fold_folder,
            fold_splitted_metadata_filename,
            criterion_label,
        )
    else:
        metadata = _ensure_file_id(metadata, mapping_file)
        metadata = metadata[metadata["fileID"] != -1]
        if args.dedupe_by_fileid:
            metadata = _dedupe_by_file_id(metadata, criterion_label, drop_conflicts=args.drop_conflicting_fileid_labels)

    prepared_metadata = metadata
    column_groups = Preprocessor_Metadata.get_column_name_groups(metadata, True)  # group columns into logical

    general_cols = set(column_groups.get("general", []))

    # Multi-task / multi-label mode
    if args.labels is not None:
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
            raise SystemExit("Multi-task mode requested via --labels, but no valid label columns were found.")

        if args.dedupe_by_fileid:
            prepared_metadata = _dedupe_by_file_id(
                prepared_metadata,
                label_cols,
                drop_conflicts=args.drop_conflicting_fileid_labels,
            )

        # Keep all rows; mask missing labels per-task.
        metadata = prepared_metadata
        test_metadata = metadata[metadata[f"Fold{fold}"] == "test"]
        val_metadata = metadata[metadata[f"Fold{fold}"] == "val"]
        train_metadata = metadata[metadata[f"Fold{fold}"] == "train"]

        def _filter_any_label(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            any_valid = np.zeros(len(df), dtype=bool)
            for col in label_cols:
                s = df[col]
                valid = ~s.isna()
                try:
                    valid = valid & (s != -1)
                except Exception:
                    pass
                any_valid |= valid.to_numpy()
            return df[any_valid]

        train_metadata = _filter_any_label(train_metadata)
        val_metadata = _filter_any_label(val_metadata)
        test_metadata = _filter_any_label(test_metadata)

        if column_group == "all":
            base_cols = [c for c in metadata.columns]
        else:
            base_cols = [c for c in column_groups.get(column_group, []) if c in metadata.columns]

        cols_keep = sorted(set(base_cols) | set(label_cols) | {"fileID"})
        current_train_metadata = train_metadata[cols_keep]
        current_val_metadata = val_metadata[cols_keep]
        current_test_metadata = test_metadata[cols_keep]

        # Tabular features: all non-label, non-general columns in the kept metadata.
        fold_cols = {f"Fold{i}" for i in range(n_folds)}
        excluded = set(label_cols) | set(general_cols) | {"fileID"} | fold_cols
        tabular_cols = [c for c in cols_keep if c not in excluded]
        use_tabular = (not args.no_metadata_features) and len(tabular_cols) > 0
        x_train = x_val = x_test = None
        tabular_feature_names = []
        if use_tabular:
            x_train, x_val, x_test, tabular_feature_names = _prepare_tabular_matrices(
                current_train_metadata,
                current_val_metadata,
                current_test_metadata,
                tabular_cols,
            )
            print(f"Tabular features: raw_cols={len(tabular_cols)} encoded_dim={x_train.shape[1]}")
        else:
            print("Tabular features: disabled or none available.")

        # Build per-task value maps and output sizes.
        task_value_maps = {}
        task_output_dims = {}
        task_index_to_label = {}
        for col in label_cols:
            series = metadata[col]
            valid = series[~series.isna()]
            try:
                valid = valid[valid != -1]
            except Exception:
                pass
            raw_unique = valid.unique().tolist()
            canon_unique = []
            for v in raw_unique:
                c = _canonical_label_value(v)
                if c is not None:
                    canon_unique.append(c)
            canon_unique = list(dict.fromkeys(canon_unique))

            # Binary labels: only if every value is coercible to {0,1}.
            binary_vals = []
            non_binary = False
            for v in canon_unique:
                b = _coerce_binary_label(v)
                if b is None:
                    non_binary = True
                    break
                binary_vals.append(int(b))
            is_binary = (not non_binary) and (set(binary_vals).issubset({0, 1})) and (len(set(binary_vals)) <= 2)

            if is_binary:
                task_output_dims[col] = 1
                task_value_maps[col] = {}
                task_index_to_label[col] = ["0", "1"]
            else:
                # Stable ordering
                unique_vals = sorted(canon_unique, key=lambda x: str(x))
                task_output_dims[col] = len(unique_vals)
                task_value_maps[col] = {v: i for i, v in enumerate(unique_vals)}
                task_index_to_label[col] = [str(v) for v in unique_vals]

        model = _build_multitask_model(
            args.model,
            task_output_dims=task_output_dims,
            no_pretrained=args.no_pretrained,
            head_dropout=args.head_dropout,
            tabular_in_dim=int(x_train.shape[1]) if (use_tabular and x_train is not None) else 0,
            tabular_hidden_dim=args.tabular_hidden_dim,
            tabular_dropout=args.tabular_dropout,
        )

        # dataloader
        n_workers = args.num_workers
        gen = torch.Generator()
        train_transform = preprocess_no_aug if args.sanity_overfit else preprocess
        eval_transform = preprocess_no_aug

        train_dataset = XRayMultiTaskDataset(
            current_train_metadata,
            root_folder,
            label_columns=label_cols,
            label_value_maps=task_value_maps,
            label_output_dims=task_output_dims,
            tabular_features=x_train if use_tabular else None,
            transform=train_transform,
        )
        val_dataset = XRayMultiTaskDataset(
            current_val_metadata,
            root_folder,
            label_columns=label_cols,
            label_value_maps=task_value_maps,
            label_output_dims=task_output_dims,
            tabular_features=x_val if use_tabular else None,
            transform=eval_transform,
        )
        test_dataset = XRayMultiTaskDataset(
            current_test_metadata,
            root_folder,
            label_columns=label_cols,
            label_value_maps=task_value_maps,
            label_output_dims=task_output_dims,
            tabular_features=x_test if use_tabular else None,
            transform=eval_transform,
        )

        if args.balanced_sampler:
            print("Warning: --balanced-sampler is only supported for single-label binary training; ignoring in multi-task mode.")

        train_sampler = RandomSampler(train_dataset, replacement=False, generator=gen)
        val_sampler = RandomSampler(val_dataset, replacement=False, generator=gen)
        test_sampler = RandomSampler(test_dataset, replacement=False, generator=gen)

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            generator=gen,
            drop_last=False,
            pin_memory=(torch.cuda.is_available()),
            persistent_workers=(n_workers > 0),
        )
        train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Per-task criteria
        criterion = {}
        for task, out_dim in task_output_dims.items():
            if out_dim == 1:
                # pos_weight from train only (ignore missing labels)
                s = current_train_metadata[task]
                vals = []
                for v in s.tolist():
                    b = _coerce_binary_label(v)
                    if b is None:
                        continue
                    vals.append(int(b))
                arr = np.asarray(vals, dtype=int) if vals else np.asarray([], dtype=int)
                pos = float((arr == 1).sum())
                neg = float((arr == 0).sum())
                pos_weight_val = (neg / max(1.0, pos)) if (pos + neg) > 0 else 1.0
                pos_weight = torch.tensor([pos_weight_val], device=device)
                bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
                criterion[task] = (bce, 1)
                print(f"[{task}] BCEWithLogitsLoss(pos_weight={pos_weight_val:.4f}) [neg={neg:.0f} pos={pos:.0f}]")
            else:
                ce = nn.CrossEntropyLoss(reduction="none")
                criterion[task] = (ce, out_dim)
                print(f"[{task}] CrossEntropyLoss(classes={out_dim})")

        scaler = GradScaler(device=device, enabled=(device == "cuda"))

        def make_optimizer():
            head_params, backbone_params = _split_head_backbone_params(model, args.model)
            if backbone_params and args.backbone_lr_mult != 1.0:
                return optim.AdamW(
                    [
                        {"params": backbone_params, "lr": learning_rate * args.backbone_lr_mult},
                        {"params": head_params, "lr": learning_rate},
                    ],
                    weight_decay=args.weight_decay,
                )
            return optim.AdamW(head_params + backbone_params, lr=learning_rate, weight_decay=args.weight_decay)

        def make_scheduler(optimizer, remaining_epochs: int):
            total_steps = max(1, remaining_epochs * len(train_loader))
            if len(optimizer.param_groups) == 2:
                max_lr = [learning_rate * args.backbone_lr_mult, learning_rate]
            else:
                max_lr = learning_rate
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

        if not args.no_wandb:
            wandb.init(
                project="Asbestosis",
                config={
                    "learning-rate": learning_rate,
                    "dataset:": root_folder,
                    "split folder": fold_folder,
                    "number of training samples": len(train_loader.dataset),
                    "number of val samples": len(val_loader.dataset),
                    "number of test samples": len(test_loader.dataset),
                    "epochs": number_of_epochs,
                    "batch size": batch_size,
                    "optimizer": "AdamW",
                    "head_dropout": args.head_dropout,
                    "Augmentation": str(preprocess),
                    "Machine": "HPC",
                    "model": args.model,
                    "weight_decay": args.weight_decay,
                    "labels": label_cols,
                    "tabular_cols_raw": tabular_cols,
                    "tabular_encoded_dim": int(x_train.shape[1]) if (use_tabular and x_train is not None) else 0,
                    "Fold": fold,
                    "metadata": metadata_file,
                },
                name=f"multitask_b={batch_size}_l={learning_rate}_n={number_of_epochs}_fold={fold}",
            )
            try:
                wandb.define_metric("epoch/*", step_metric="epoch/*")
                wandb.define_metric("average_loss/*", step_metric="epoch/*")
                wandb.define_metric("macro_*/*", step_metric="epoch/*")
            except Exception:
                pass

        if args.freeze_backbone_epochs > 0:
            _freeze_backbone(model, args.model)
            print(f"Freezing backbone for {args.freeze_backbone_epochs} epochs")
        optimizer = make_optimizer()
        lr_scheduler = make_scheduler(optimizer, remaining_epochs=number_of_epochs)

        if args.sanity_overfit:
            print("Warning: --sanity-overfit is implemented for single-label runs only; skipping.")

        if args.early_stop_patience > 0 and args.eval_every == 0:
            args.eval_every = 1
            print("Early stopping enabled -> forcing --eval-every 1")

        best_score = None
        best_epoch = -1
        best_path = os.path.join(output_folder, f"best_{args.model}_labels=multitask_fold={fold}.pth")
        no_improve = 0
        best_fixed_thresholds = None

        # Map legacy early-stop metric to macro metrics in multi-task mode.
        early_metric = args.early_stop_metric
        if early_metric == "auc/eval":
            early_metric = "macro_auc/eval"
        if early_metric == "f1/eval":
            early_metric = "macro_f1/eval"

        for epoch in range(number_of_epochs):
            if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs:
                _unfreeze_all(model)
                optimizer = make_optimizer()
                lr_scheduler = make_scheduler(optimizer, remaining_epochs=number_of_epochs - epoch)
                print("Unfroze backbone and reset optimizer/scheduler")

            train_results, train_avg_loss = run_epoch(
                model, optimizer, lr_scheduler, criterion, scaler, train_loader, device, train=True, show_figs=False, max_steps=args.max_train_steps
            )
            train_metrics = log_multitask_metrics(
                train_results,
                train_avg_loss,
                suffix="train",
                epoch=epoch,
                task_output_dims=task_output_dims,
                task_index_to_label=task_index_to_label,
                wandb_detail=args.wandb_detail,
            )

            do_eval = args.eval_every and (epoch % args.eval_every == 0 or epoch == number_of_epochs - 1)
            do_test = args.test_every and (epoch % args.test_every == 0 or epoch == number_of_epochs - 1)

            eval_metrics = None
            if do_eval:
                eval_results, eval_avg_loss = run_epoch(
                    model, optimizer, None, criterion, scaler, val_loader, device, train=False, show_figs=False, max_steps=args.max_eval_steps
                )
                eval_metrics = log_multitask_metrics(
                    eval_results,
                    eval_avg_loss,
                    suffix="eval",
                    epoch=epoch,
                    task_output_dims=task_output_dims,
                    task_index_to_label=task_index_to_label,
                    compute_thresholds=True,
                    wandb_detail=args.wandb_detail,
                )
                fixed_thresholds_eval = {}
                for task, out_dim in task_output_dims.items():
                    if out_dim == 1:
                        fixed_thresholds_eval[task] = eval_metrics.get(f"task/{task}/best_threshold_f1/eval")

            if do_test:
                test_results, test_avg_loss = run_epoch(
                    model, optimizer, None, criterion, scaler, test_loader, device, train=False, show_figs=False, max_steps=args.max_eval_steps
                )
                log_multitask_metrics(
                    test_results,
                    test_avg_loss,
                    suffix="test",
                    epoch=epoch,
                    task_output_dims=task_output_dims,
                    task_index_to_label=task_index_to_label,
                    wandb_detail=args.wandb_detail,
                )

            # Early stopping on eval metrics
            if args.early_stop_patience > 0 and do_eval and eval_metrics is not None:
                if early_metric == "macro_auc/eval":
                    score = eval_metrics.get("macro_auc/eval")
                elif early_metric == "macro_f1/eval":
                    score = eval_metrics.get("macro_f1/eval")
                else:
                    score = -eval_metrics.get("average_loss/eval", float("inf"))
                if score is None:
                    continue

                improved = False
                if best_score is None:
                    improved = True
                else:
                    delta = score - best_score
                    improved = delta > args.early_stop_min_delta

                if improved:
                    best_score = float(score)
                    best_epoch = epoch
                    no_improve = 0
                    best_fixed_thresholds = fixed_thresholds_eval
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "epoch": best_epoch,
                        "score": float(best_score),
                        "labels": label_cols,
                        "task_output_dims": task_output_dims,
                        "task_index_to_label": task_index_to_label,
                        "best_fixed_thresholds": best_fixed_thresholds,
                    }
                    torch.save(checkpoint, best_path)
                    print(f"Saved best model to {best_path} (epoch {epoch}, score={best_score:.4f})")
                else:
                    no_improve += 1
                    if no_improve >= args.early_stop_patience:
                        print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, score={best_score:.4f})")
                        break

        final_epoch = epoch if number_of_epochs > 0 else -1

        final_results, final_avg_loss = run_epoch(
            model, optimizer, None, criterion, scaler, test_loader, device, train=False, max_steps=args.max_eval_steps
        )
        final_metrics = log_multitask_metrics(
            final_results,
            final_avg_loss,
            suffix="final_test",
            epoch=final_epoch,
            task_output_dims=task_output_dims,
            task_index_to_label=task_index_to_label,
            fixed_thresholds=best_fixed_thresholds,
            wandb_detail=args.wandb_detail,
        )
        if _wandb_is_active():
            try:
                for key in ("macro_f1/final_test", "macro_auc/final_test", "macro_accuracy/final_test", "average_loss/final_test"):
                    if key in final_metrics:
                        wandb.summary[key] = float(final_metrics[key])
                # Also store per-task headline metrics (compact) in summary.
                for task in label_cols:
                    out_dim = int(task_output_dims[task])
                    if out_dim == 1:
                        for k in (f"task/{task}/auc/final_test", f"task/{task}/f1/final_test", f"task/{task}/f1@fixed_thr/final_test"):
                            if k in final_metrics:
                                wandb.summary[k] = float(final_metrics[k])
                    else:
                        for k in (f"task/{task}/auc_macro_ovr/final_test", f"task/{task}/f1_macro/final_test"):
                            if k in final_metrics:
                                wandb.summary[k] = float(final_metrics[k])
            except Exception:
                pass

        # Evaluate best checkpoint if available.
        if args.early_stop_patience > 0 and os.path.isfile(best_path):
            try:
                state = torch.load(best_path, map_location=device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                    best_fixed_thresholds = state.get("best_fixed_thresholds") or best_fixed_thresholds
                else:
                    model.load_state_dict(state)
                best_results, best_avg_loss = run_epoch(
                    model, optimizer, None, criterion, scaler, test_loader, device, train=False, max_steps=args.max_eval_steps
                )
                log_multitask_metrics(
                    best_results,
                    best_avg_loss,
                    suffix="best_test",
                    epoch=best_epoch,
                    task_output_dims=task_output_dims,
                    task_index_to_label=task_index_to_label,
                    fixed_thresholds=best_fixed_thresholds,
                    wandb_detail=args.wandb_detail,
                )
            except Exception as e:
                print(f"Warning: failed to evaluate best checkpoint ({best_path}): {e}")

        torch.save(
            model.state_dict(),
            os.path.join(output_folder, f"asbestosis_{args.model}_n{number_of_epochs}_b{batch_size}_labels=multitask_fold={fold}.pth"),
        )
        if _wandb_is_active():
            wandb.finish()
        raise SystemExit(0)

    # Single-label mode
    #for criterion_label in column_groups[column_group]:
    if criterion_label in metadata.keys() and criterion_label not in column_groups["general"]:
        metadata = prepared_metadata[prepared_metadata[criterion_label] != -1]
        test_metadata = metadata[metadata[f'Fold{fold}'] == 'test']
        val_metadata = metadata[metadata[f'Fold{fold}'] == 'val']
        train_metadata = metadata[metadata[f'Fold{fold}'] == 'train']
        #print(criterion_label + ":\n\ttrain --- pos: " + str(len(train_metadata[train_metadata[criterion_label] == 1])) + " neg: " + str(len(train_metadata[train_metadata[criterion_label] == 0])) +
        #      "\n\ttest --- pos: " + str(len(test_metadata[test_metadata[criterion_label] == 1])) + " neg: " + str(len(test_metadata[test_metadata[criterion_label] == 0])))

        current_train_metadata = train_metadata[[col for col in column_groups[column_group] if col in metadata.columns]]
        current_val_metadata = val_metadata[[col for col in column_groups[column_group] if col in metadata.columns]]
        current_test_metadata = test_metadata[[col for col in column_groups[column_group] if col in metadata.columns]]
        num_classes = len(metadata[criterion_label].unique())
        is_binary_classification = num_classes == 2

        # For binary classification, use a single logit + BCEWithLogitsLoss (enables pos_weight).
        out_features = 1 if is_binary_classification else num_classes
        model = _build_model(args.model, out_features=out_features, no_pretrained=args.no_pretrained, head_dropout=args.head_dropout)

        # dataloader
        n_workers = args.num_workers
        gen = torch.Generator()
        train_transform = preprocess_no_aug if args.sanity_overfit else preprocess
        eval_transform = preprocess_no_aug
        train_dataset = X_rayImageDataset(current_train_metadata, root_folder, label_column=criterion_label, transform=train_transform)

        if args.balanced_sampler and is_binary_classification and len(train_dataset) > 0:
            # Build per-sample weights from the dataset's encoded labels (0/1).
            encoded = train_dataset.img_labels[criterion_label].map(train_dataset.label_index_mapping).astype(int).to_numpy()
            counts = np.bincount(encoded, minlength=2)
            w0 = 1.0 / max(1, counts[0])
            w1 = 1.0 / max(1, counts[1])
            weights = np.where(encoded == 1, w1, w0).astype(np.float64)
            train_sampler = WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
                generator=gen,
            )
            drop_last_train = False
            print(f"Using balanced sampler: class0={counts[0]} class1={counts[1]}")
        else:
            train_sampler = RandomSampler(train_dataset, replacement=False, generator=gen)
            drop_last_train = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            sampler=train_sampler,
            generator=gen,
            drop_last=drop_last_train,
            pin_memory=(torch.cuda.is_available()),
            persistent_workers=(n_workers > 0),
        )

        val_dataset = X_rayImageDataset(current_val_metadata, root_folder, label_column=criterion_label, transform=eval_transform)
        val_sampler = RandomSampler(val_dataset, replacement=False, generator=gen)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            sampler=val_sampler,
            generator=gen,
            drop_last=False,
            pin_memory=False,
            persistent_workers=(n_workers > 0),
        )

        test_dataset = X_rayImageDataset(current_test_metadata, root_folder, label_column=criterion_label, transform=eval_transform)
        test_sampler = RandomSampler(test_dataset, replacement=False, generator=gen)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            sampler=test_sampler,
            generator=gen,
            drop_last=False,
            pin_memory=False,
            persistent_workers=(n_workers > 0),
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        def make_optimizer():
            head_params, backbone_params = _split_head_backbone_params(model, args.model)
            if backbone_params and args.backbone_lr_mult != 1.0:
                return optim.AdamW(
                    [
                        {"params": backbone_params, "lr": learning_rate * args.backbone_lr_mult},
                        {"params": head_params, "lr": learning_rate},
                    ],
                    weight_decay=args.weight_decay,
                )
            return optim.AdamW(head_params + backbone_params, lr=learning_rate, weight_decay=args.weight_decay)

        def make_scheduler(optimizer, remaining_epochs: int):
            total_steps = max(1, remaining_epochs * len(train_loader))
            if len(optimizer.param_groups) == 2:
                max_lr = [learning_rate * args.backbone_lr_mult, learning_rate]
            else:
                max_lr = learning_rate
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

        if not args.no_wandb:
            wandb.init(project="Asbestosis", config={
                "learning-rate": learning_rate,
                "dataset:": root_folder,
                "split folder": fold_folder,
                "number of training samples": len(train_loader.dataset),
                "number of val samples": len(val_loader.dataset),
                "number of test samples": len(test_loader.dataset),
                "epochs": number_of_epochs,
                "batch size": batch_size,
                "optimizer": "AdamW",
                "head_dropout": args.head_dropout,
                "Augmentation": str(preprocess),
                "Machine": "HPC",
                "model": args.model,
                "weight_decay": args.weight_decay,
                "train criteria": criterion_label,
                "Fold": fold,
                "metadata": metadata_file,
            }, name="b={}_l={}_n={}_fold={}_{}".format(batch_size, learning_rate, number_of_epochs, fold, criterion_label))

        if is_binary_classification:
            # class imbalance handling
            pos = float((train_metadata[criterion_label] == 1).sum())
            neg = float((train_metadata[criterion_label] == 0).sum())
            if args.balanced_sampler:
                # Avoid double-compensating when using a balanced sampler.
                criterion = nn.BCEWithLogitsLoss()
                print(f"Using BCEWithLogitsLoss(pos_weight=1.0) with balanced sampler [neg={neg:.0f} pos={pos:.0f}]")
            else:
                pos_weight_val = (neg / max(1.0, pos))
                pos_weight = torch.tensor([pos_weight_val], device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"Using BCEWithLogitsLoss(pos_weight={pos_weight_val:.4f}) [neg={neg:.0f} pos={pos:.0f}]")
        else:
            criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(device=device, enabled=(device == "cuda"))
        if args.freeze_backbone_epochs > 0:
            _freeze_backbone(model, args.model)
            print(f"Freezing backbone for {args.freeze_backbone_epochs} epochs")
        optimizer = make_optimizer()
        lr_scheduler = make_scheduler(optimizer, remaining_epochs=number_of_epochs)

        if args.sanity_overfit:
            overfit_sanity_check(
                model=model,
                dataset=train_dataset,
                device=device,
                batch_size=batch_size,
                epochs=args.sanity_epochs,
                learning_rate=args.sanity_lr,
                seed=args.seed,
                max_samples=args.sanity_samples,
            )
            raise SystemExit(0)

        if args.early_stop_patience > 0 and args.eval_every == 0:
            args.eval_every = 1
            print("Early stopping enabled -> forcing --eval-every 1")

        best_score = None
        best_epoch = -1
        best_path = os.path.join(output_folder, f"best_{args.model}_label={criterion_label}_fold={fold}.pth")
        no_improve = 0
        last_eval_epoch = -1
        last_test_epoch = -1
        best_eval_threshold_f1 = None
        best_eval_threshold_youden = None

        for epoch in range(number_of_epochs):
            if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs:
                _unfreeze_all(model)
                optimizer = make_optimizer()
                lr_scheduler = make_scheduler(optimizer, remaining_epochs=number_of_epochs - epoch)
                print("Unfroze backbone and reset optimizer/scheduler")

            y_true, y_probs, train_avg_loss = run_epoch(model, optimizer, lr_scheduler, criterion, scaler, train_loader, device, train=True, show_figs=False, max_steps=args.max_train_steps)
            train_metrics = log_classification_metrics(y_true, y_probs, train_avg_loss, suffix="train", epoch=epoch)
            if epoch % 10 == 0 and is_binary_classification:
                compute_roc_curve(y_true, y_probs, plot=False, title_suffix="train")

            do_eval = args.eval_every and (epoch % args.eval_every == 0 or epoch == number_of_epochs - 1)
            do_test = args.test_every and (epoch % args.test_every == 0 or epoch == number_of_epochs - 1)
            if do_eval or do_test:
                if do_eval:
                    y_true_eval, y_probs_eval, eval_avg_loss = run_epoch(
                        model, optimizer, None, criterion, scaler, val_loader, device, train=False, show_figs=False, max_steps=args.max_eval_steps
                    )
                    eval_metrics = log_classification_metrics(y_true_eval, y_probs_eval, eval_avg_loss, suffix="eval", epoch=epoch)
                    last_eval_epoch = epoch
                    if is_binary_classification and (epoch % 10 == 0 or epoch == number_of_epochs - 1):
                        compute_roc_curve(y_true_eval, y_probs_eval, plot=False, title_suffix="eval")
                if do_test:
                    y_true_test, y_probs_test, test_avg_loss = run_epoch(
                        model, optimizer, None, criterion, scaler, test_loader, device, train=False, show_figs=False, max_steps=args.max_eval_steps
                    )
                    log_classification_metrics(y_true_test, y_probs_test, test_avg_loss, suffix="test", epoch=epoch)
                    last_test_epoch = epoch
                    if is_binary_classification and (epoch % 10 == 0 or epoch == number_of_epochs - 1):
                        compute_roc_curve(y_true_test, y_probs_test, plot=False, title_suffix="test")

                # Early stopping on eval metrics
                if args.early_stop_patience > 0 and do_eval:
                    if args.early_stop_metric in {"auc/eval", "macro_auc/eval"}:
                        score = eval_metrics.get("auc/eval") or eval_metrics.get("auc_macro_ovr/eval")
                        higher_is_better = True
                    elif args.early_stop_metric in {"f1/eval", "macro_f1/eval"}:
                        score = eval_metrics.get("f1/eval")
                        higher_is_better = True
                    else:
                        score = -eval_metrics.get("average_loss/eval", float("inf"))
                        higher_is_better = True

                    if score is None:
                        continue

                    improved = False
                    if best_score is None:
                        improved = True
                    else:
                        delta = (score - best_score) if higher_is_better else (best_score - score)
                        improved = delta > args.early_stop_min_delta

                    if improved:
                        best_score = score
                        best_epoch = epoch
                        no_improve = 0
                        best_eval_threshold_f1 = eval_metrics.get("best_threshold_f1/eval")
                        best_eval_threshold_youden = eval_metrics.get("best_threshold/eval")
                        checkpoint = {
                            "model_state_dict": model.state_dict(),
                            "epoch": best_epoch,
                            "score": float(best_score),
                            "best_threshold_f1/eval": best_eval_threshold_f1,
                            "best_threshold_youden/eval": best_eval_threshold_youden,
                        }
                        torch.save(checkpoint, best_path)
                        thr_note = ""
                        if best_eval_threshold_f1 is not None:
                            thr_note = f", val_thr_f1={best_eval_threshold_f1:.4f}"
                        print(f"Saved best model to {best_path} (epoch {epoch}, score={best_score:.4f}{thr_note})")
                    else:
                        no_improve += 1
                        if no_improve >= args.early_stop_patience:
                            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, score={best_score:.4f})")
                            break

        final_epoch = epoch if number_of_epochs > 0 else -1

        # Always log a final evaluation (for easy comparison in wandb / logs).
        y_true_final, y_probs_final, final_avg_loss = run_epoch(
            model, optimizer, None, criterion, scaler, test_loader, device, train=False, max_steps=args.max_eval_steps
        )
        fixed_thr_for_test = best_eval_threshold_f1 if best_eval_threshold_f1 is not None else best_eval_threshold_youden
        log_classification_metrics(
            y_true_final,
            y_probs_final,
            final_avg_loss,
            suffix="final_test",
            epoch=final_epoch,
            fixed_threshold=fixed_thr_for_test,
        )
        if is_binary_classification:
            compute_roc_curve(y_true_final, y_probs_final, plot=True, title_suffix="final_test")
        else:
            compute_roc_one_vs_rest(y_true_final, y_probs_final, np.arange(num_classes), criterion_label)

        # If early stopping is enabled and a best checkpoint was saved, evaluate it as well.
        if args.early_stop_patience > 0 and os.path.isfile(best_path):
            try:
                state = torch.load(best_path, map_location=device)
                fixed_thr_best = None
                if isinstance(state, dict) and "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                    fixed_thr_best = state.get("best_threshold_f1/eval") or state.get("best_threshold_youden/eval")
                else:
                    model.load_state_dict(state)
                y_true_best, y_probs_best, best_avg_loss = run_epoch(
                    model, optimizer, None, criterion, scaler, test_loader, device, train=False, max_steps=args.max_eval_steps
                )
                log_classification_metrics(
                    y_true_best,
                    y_probs_best,
                    best_avg_loss,
                    suffix="best_test",
                    epoch=best_epoch,
                    fixed_threshold=fixed_thr_best,
                )
                if is_binary_classification:
                    compute_roc_curve(y_true_best, y_probs_best, plot=True, title_suffix="best_test")
            except Exception as e:
                print(f"Warning: failed to evaluate best checkpoint ({best_path}): {e}")

        torch.save(model.state_dict(), os.path.join(output_folder, f'asbestosis_{args.model}_n{number_of_epochs}_b{batch_size}_label={criterion_label}_fold={fold}.pth'))
        if _wandb_is_active():
            wandb.finish()
    else:
        raise SystemExit(f"Label '{criterion_label}' not found in metadata (or is a general column). Use --label or --labels.")
