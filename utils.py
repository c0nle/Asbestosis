"""
utils.py
--------
Shared utility functions for the asbestosis multi-task training pipeline.

Covers:
- W&B wrapper helpers
- Binary label normalisation and missing-value detection
- Dataset statistics printing
- fileID resolution from mapping.csv
- Filesystem helpers (writable-directory check)
- Pipeline helpers: CSV-arg parsing, group-ID normalisation, leakage
  detection, grouping-column selection, seed setting
"""

import io
import os
import random
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _wandb_is_active() -> bool:
    """Return True if a W&B run is currently active."""
    return getattr(wandb, "run", None) is not None


def _wandb_log(data: dict, commit: bool = True) -> None:
    """Log ``data`` to W&B if a run is active; otherwise no-op."""
    if _wandb_is_active():
        wandb.log(data, commit=commit)


# ---------------------------------------------------------------------------
# Label value helpers
# ---------------------------------------------------------------------------

def _is_missing_label_value(value) -> bool:
    """
    Return True for any value that should be treated as a missing label.

    Recognised missing sentinels: ``None``, ``NaN``, ``-1``, ``"-1"``,
    ``"-1.0"``, empty string, ``"nan"``, ``"none"``.
    """
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
    """
    Normalise a raw label value to a hashable canonical form.

    - ``None`` / missing  → ``None``
    - Integer-valued float (e.g. ``1.0``) → ``int``
    - Other float → ``float``
    - Anything else → stripped ``str``
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
    return str(value).strip()


def _coerce_binary_label(value) -> Optional[int]:
    """
    Convert common binary encodings to {0, 1}.

    Accepts booleans, integers, floats, and strings (``"0"``, ``"1"``,
    ``"true"``, ``"false"``, ``"yes"``, ``"no"``).

    Returns:
        ``0`` or ``1`` on success, ``None`` if the value cannot be coerced.
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
        try:
            fv = float(v)
            if np.isfinite(fv) and float(fv).is_integer() and int(fv) in (0, 1):
                return int(fv)
        except Exception:
            pass
    return None


def _missing_mask(series: pd.Series) -> pd.Series:
    """
    Return a boolean Series that is ``True`` wherever a label is missing.

    Covers ``NaN``, ``-1`` (integer sentinel), and common string sentinels
    (``""``, ``"nan"``, ``"none"``, ``"-1"``, ``"-1.0"``).
    """
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
    """Print per-label presence counts and value distributions."""
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


# ---------------------------------------------------------------------------
# fileID resolution
# ---------------------------------------------------------------------------

def _ensure_file_id(metadata: pd.DataFrame, mapping_file: str) -> pd.DataFrame:
    """
    Add a ``fileID`` column to ``metadata`` by joining with ``mapping_file``.

    If ``fileID`` is already present and non-empty, the DataFrame is returned
    unchanged.  Otherwise the mapping CSV is used to resolve
    ``Anforderungsnummer`` → ``fileID``.  Rows without a match receive ``-1``.

    Args:
        metadata:     Metadata DataFrame (must contain ``Anforderungsnummer``).
        mapping_file: Path to the medicoID → fileID mapping CSV.

    Returns:
        DataFrame with a ``fileID`` column of integer type.
    """
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


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _ensure_writable_dir(path: str, fallback: Optional[str] = None) -> str:
    """
    Ensure ``path`` is a writable directory, falling back to ``fallback`` on failure.

    Args:
        path:     Preferred directory path.
        fallback: Alternative directory to use if ``path`` is not writable.

    Returns:
        The path that was successfully made writable.
    """
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


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _parse_csv_arg(value: str) -> List[str]:
    """Parse a comma-separated string argument into a list of stripped tokens."""
    v = str(value).strip()
    if not v:
        return []
    lower = v.lower()
    if lower in {"none", "null", "off", "disable", "disabled"}:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def _normalize_group_id(value) -> Optional[str]:
    """Normalise a group ID to a canonical string, returning ``None`` for missing values."""
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


def _leakage_check(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
) -> Tuple[int, int, int]:
    """
    Check for patient-group overlap between train, val, and test splits.

    Returns:
        ``(train∩val, train∩test, val∩test)`` overlap counts.
    """
    if group_col not in train_df.columns or group_col not in val_df.columns or group_col not in test_df.columns:
        return 0, 0, 0

    def _to_set(df: pd.DataFrame) -> set:
        vals = [_normalize_group_id(v) for v in df[group_col].tolist()]
        return {v for v in vals if v is not None}

    tr = _to_set(train_df)
    va = _to_set(val_df)
    te = _to_set(test_df)
    return len(tr & va), len(tr & te), len(va & te)


def _choose_group_col(
    metadata: pd.DataFrame,
    requested: Optional[str],
    n_folds: int,
) -> Optional[str]:
    """
    Select the column to use for grouping patients when generating splits.

    Prevents the same patient from appearing in multiple folds (data leakage).

    When ``requested`` is ``"auto"``, columns are tried in this priority order:
    ``patientID`` → ``medicoID`` → ``Anforderungsnummer`` → ``Aufnahmenummer``
    → ``fileID``.

    Args:
        metadata:  Metadata DataFrame.
        requested: Explicit column name, ``"auto"``, or a disable-keyword
                   (``"none"``, ``"null"``, ``"off"``, ``"disabled"``).
        n_folds:   Minimum number of unique groups required.

    Returns:
        Column name to use, or ``None`` if grouping should be disabled.
    """
    req = (requested or "").strip()
    if req.lower() in {"", "none", "null", "off", "disable", "disabled"}:
        return None
    if req.lower() == "auto":
        candidates = ["patientID", "medicoID", "Anforderungsnummer", "Aufnahmenummer", "fileID"]
    else:
        candidates = [req]

    for c in candidates:
        if c in metadata.columns and int(metadata[c].nunique(dropna=True)) >= int(n_folds):
            return c
    return None


def _set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
