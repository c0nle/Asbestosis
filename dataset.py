"""
dataset.py
----------
Image loading and PyTorch dataset for the asbestosis multi-task pipeline.

Public API
----------
XRayMultiTaskDataset      : Multi-task chest X-ray dataset.
_binary_value_map_from_series : Build a {canonical_value → 0/1} mapping for a
                           label column.

Image helpers (used internally and by main.py diagnostics):
_read_xray_image, _find_valid_xray_path, _find_xray_path,
_filter_rows_with_images
"""

import functools
import glob
import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import PIL.Image
import pydicom
import torch
from torch.utils.data import Dataset

from utils import (
    _canonical_label_value,
    _coerce_binary_label,
    _is_missing_label_value,
    _missing_mask,
)


# ---------------------------------------------------------------------------
# DICOM / image I/O
# ---------------------------------------------------------------------------

def _read_dicom_from_bytes(data: bytes) -> PIL.Image.Image:
    """Decode a DICOM byte buffer to a normalised 8-bit grayscale PIL image."""
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
    """Extract and decode the first DICOM (or image) file inside a zip archive."""
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
    """Load a chest X-ray image from a ZIP archive, DICOM, PNG, or JPEG path."""
    if path.lower().endswith(".zip"):
        return _read_image_from_zip(path)
    return PIL.Image.open(path).convert("L")


def _zip_has_files(zip_path: str) -> bool:
    """Return True if the zip archive contains at least one non-directory entry."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_dir:
            return any(not name.endswith("/") for name in zip_dir.namelist())
    except Exception:
        return False


@functools.lru_cache(maxsize=200_000)
def _find_valid_xray_path(img_dir: str, file_id: str) -> Optional[str]:
    """
    Find the path of the image file for ``file_id`` under ``img_dir``.

    Searches in ``png/``, the root, and ``anon/`` sub-directories, trying
    PNG/JPEG patterns first then ZIP files.

    Returns:
        Valid path string, or ``None`` if no image is found.
    """
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
    """
    Return a path for ``file_id`` under ``img_dir``, falling back to a
    non-existent ``.zip`` path if no real file is found.
    """
    valid = _find_valid_xray_path(img_dir, file_id)
    if valid is not None:
        return valid
    return os.path.join(img_dir, f"{file_id}.zip")


def _filter_rows_with_images(df: pd.DataFrame, img_dir: str) -> pd.DataFrame:
    """Drop rows from ``df`` whose ``fileID`` has no associated image file."""
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


# ---------------------------------------------------------------------------
# Label value mapping
# ---------------------------------------------------------------------------

def _binary_value_map_from_series(
    series: pd.Series,
    task: str,
) -> Tuple[Dict[object, int], List[str]]:
    """
    Build a stable ``{canonical_value → 0/1}`` mapping for a label column.

    Enables ``BCEWithLogitsLoss`` for categorical-but-binary columns (e.g.
    ``"left"``/``"right"``).

    Args:
        series: Full metadata column for the task.
        task:   Task name (used in error messages).

    Returns:
        ``(value_map, idx_to_label)`` where ``idx_to_label`` is
        ``["neg_label", "pos_label"]``.

    Raises:
        SystemExit: If the column has more than 2 unique non-missing values.
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class XRayMultiTaskDataset(Dataset):
    """
    PyTorch Dataset for multi-task chest X-ray classification.

    Loads grayscale X-ray images by ``fileID`` and returns per-task binary
    targets together with a validity mask (``1.0`` = label present,
    ``0.0`` = label missing).  Missing labels are excluded from the loss
    computation via the mask.

    Supports ZIP-packed DICOM files and PNG / JPEG images.
    Only rows for which an image file actually exists are kept.

    Args:
        annotations:      DataFrame with at minimum a ``fileID`` column plus one
                          column per task.
        img_dir:          Root directory containing image files (may contain
                          ``png/`` and ``anon/`` sub-directories).
        label_columns:    Ordered list of task names (= column names in
                          ``annotations``).
        label_value_maps: Dict mapping task name → ``{canonical_value → 0/1}``.
        transform:        Optional torchvision transform applied to each image.
    """

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
