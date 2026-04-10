"""
Preprocessor_Metadata.py
------------------------
Split generation and column-group utilities for the asbestosis training pipeline.

Only these two functions are called by ``main.py``:

- :func:`create_splits` : Generate stratified group K-fold train / val / test splits.
- :func:`get_column_name_groups` : Group metadata columns by clinical category.

One-time preprocessing functions (``create_dichotome_metadata``,
``prepare_metadata``, ``get_label_encoding``, etc.) have been moved to
``preprocessing_scripts.py``.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mode_or_first(values: pd.Series):
    """Return the most common value in a Series, or the first element on ties."""
    values = values.dropna()
    if len(values) == 0:
        return None
    counts = values.value_counts()
    return counts[counts == counts.iloc[0]].index.tolist()[0]


def _stratified_group_split(
    groups: np.ndarray,
    group_labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple:
    """
    Split group IDs into train and test subsets while approximately preserving
    the label distribution across groups.

    Falls back to a random (non-stratified) split if stratification is not
    possible (e.g. too few groups per class).

    Args:
        groups:       1-D array of group IDs (one per unique group).
        group_labels: 1-D array of group-level label values (aligned with ``groups``).
        test_size:    Fraction of groups to assign to the test set.
        seed:         Random seed for reproducibility.

    Returns:
        ``(train_groups, test_groups)`` – subsets of the ``groups`` array.
    """
    groups = np.asarray(groups)
    group_labels = np.asarray(group_labels)
    if len(groups) == 0:
        return np.array([], dtype=groups.dtype), np.array([], dtype=groups.dtype)

    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_train, idx_test = next(splitter.split(groups, group_labels))
        return groups[idx_train], groups[idx_test]
    except Exception:
        # Fallback: random split without stratification.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(groups))
        n_test = max(1, min(int(round(test_size * len(groups))), len(groups) - 1)) if len(groups) > 1 else 0
        return groups[perm[n_test:]], groups[perm[:n_test]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_splits(
    metadata: pd.DataFrame,
    n_testfolds: int,
    output_folder: str,
    output_filename: str,
    training_label: str = "mixed_shapes",
    group_col: Optional[str] = None,
    strict_group_col: bool = False,
    seed_base: int = 0,
    train_frac: float = 0.5,
    val_frac: float = 0.2,
    test_frac: float = 0.3,
) -> pd.DataFrame:
    """
    Generate stratified group K-fold train / val / test splits.

    Uses ``StratifiedGroupKFold`` so that every patient appears in exactly one
    test fold and the full dataset is covered without double-counting.  The test
    fraction is always ``1 / n_testfolds`` (~20% for 5 folds); ``test_frac`` is
    accepted for API compatibility but ignored.  The validation set is carved out
    of the training portion using ``val_frac``.

    Each fold assigns non-overlapping patient groups to train, val, and test
    sets while approximately preserving the class distribution of
    ``training_label``.  Per-fold CSV files are written to ``output_folder``,
    and a single fold-annotated metadata CSV is written to ``output_filename``.

    Grouping columns are tried in priority order if ``group_col`` is ``None``:
    ``patientID`` → ``medicoID``.

    Args:
        metadata:         Full metadata DataFrame.
        n_testfolds:      Number of cross-validation folds to generate.
        output_folder:    Directory for per-fold CSV files.
        output_filename:  Path for the fold-annotated metadata CSV.
        training_label:   Column used to stratify the splits (class distribution
                          is preserved at the group level).
        group_col:        Column whose unique values define patient groups
                          (prevents the same patient appearing in multiple
                          splits).  Auto-detected if ``None``.
        strict_group_col: Raise a ``ValueError`` if the requested column has
                          fewer unique groups than ``n_testfolds``.
        seed_base:        Random seed for reproducibility.
        train_frac:       Accepted for API compatibility; ignored (derived from
                          ``n_testfolds`` and ``val_frac``).
        val_frac:         Fraction of the full dataset reserved for validation
                          (carved from the training portion of each fold).
        test_frac:        Accepted for API compatibility; ignored (always
                          ``1 / n_testfolds``).

    Returns:
        ``metadata`` DataFrame with added ``Fold{i}`` columns marking each
        row's split assignment (``"train"``, ``"val"``, or ``"test"``).

    Raises:
        ValueError: If ``training_label`` is missing or a valid grouping column
                    cannot be found.
    """
    if training_label not in metadata.columns:
        raise ValueError(f"training_label '{training_label}' not found in metadata columns.")

    metadata = metadata[metadata[training_label] != -1].copy()
    os.makedirs(output_folder, exist_ok=True)

    for fold in range(n_testfolds):
        metadata[f"Fold{fold}"] = ""

    # --- Resolve grouping column ---
    if group_col is None:
        for candidate in ["patientID", "medicoID"]:
            if candidate in metadata.columns:
                group_col = candidate
                break
    elif group_col not in metadata.columns:
        raise ValueError(f"Requested group_col '{group_col}' not found in metadata columns.")

    if group_col is None:
        raise ValueError(
            "No suitable grouping column found (tried 'patientID', 'medicoID'). "
            "Pass group_col explicitly."
        )

    if metadata[group_col].nunique() < n_testfolds:
        if strict_group_col:
            raise ValueError(
                f"Grouping column '{group_col}' has too few unique groups "
                f"({metadata[group_col].nunique()}) for n_testfolds={n_testfolds}."
            )
        for candidate in ["patientID", "grouping", "Anforderungsnummer", "fileID", "Aufnahmenummer"]:
            if candidate in metadata.columns and metadata[candidate].nunique() >= n_testfolds:
                group_col = candidate
                break

    # Normalise group IDs (prevents duplicates from dtype quirks like "123.0" vs "123").
    try:
        gs = metadata[group_col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        gs = gs.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        metadata[group_col] = gs
        metadata = metadata.dropna(subset=[group_col])
    except Exception:
        pass

    # Build group-level label dict for stratification of the val split.
    group_stats = (
        metadata.groupby(group_col)[training_label]
        .apply(_mode_or_first)
        .reset_index()
        .rename(columns={training_label: "group_label"})
        .dropna(subset=["group_label"])
    )
    group_label_dict = dict(zip(group_stats[group_col], group_stats["group_label"]))

    if len(group_label_dict) < 3:
        raise ValueError(
            f"Not enough groups for train/val/test split using '{group_col}': "
            f"only {len(group_label_dict)} groups available."
        )

    # Row-level arrays for StratifiedGroupKFold.
    g_rows = metadata[group_col].to_numpy()
    y_rows = metadata[training_label].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_testfolds, shuffle=True, random_state=int(seed_base))
    # val_frac expressed relative to the train+val portion (≈ 1 - 1/n_testfolds).
    trainval_frac = 1.0 - 1.0 / n_testfolds
    val_rel = min(0.999, max(0.001, val_frac / trainval_frac))

    for fold_idx, (trainval_row_idx, test_row_idx) in enumerate(
        sgkf.split(np.arange(len(metadata)), y_rows, g_rows)
    ):
        test_patients    = set(g_rows[test_row_idx])
        trainval_patients = sorted(set(g_rows[trainval_row_idx]))
        tv_arr    = np.array(trainval_patients)
        tv_labels = np.array([group_label_dict[p] for p in trainval_patients])

        train_patients, val_patients = _stratified_group_split(
            tv_arr, tv_labels, test_size=val_rel, seed=10_000 + fold_idx
        )
        train_patients = set(train_patients)
        val_patients   = set(val_patients)

        train_split = metadata[metadata[group_col].isin(train_patients)]
        val_split   = metadata[metadata[group_col].isin(val_patients)]
        test_split  = metadata[metadata[group_col].isin(test_patients)]

        train_split.to_csv(os.path.join(output_folder, f"stratified_train_set-f{fold_idx}.csv"), index=False)
        val_split.to_csv(  os.path.join(output_folder, f"stratified_val_set-f{fold_idx}.csv"),   index=False)
        test_split.to_csv( os.path.join(output_folder, f"stratified_test_set-f{fold_idx}.csv"),  index=False)

        metadata.loc[metadata[group_col].isin(train_patients), f"Fold{fold_idx}"] = "train"
        metadata.loc[metadata[group_col].isin(val_patients),   f"Fold{fold_idx}"] = "val"
        metadata.loc[metadata[group_col].isin(test_patients),  f"Fold{fold_idx}"] = "test"

        n_test = len(test_split)
        n_val  = len(val_split)
        n_tr   = len(train_split)
        print(
            f"  Fold {fold_idx}: train={n_tr}  val={n_val}  test={n_test}  "
            f"(patients: {len(train_patients)}/{len(val_patients)}/{len(test_patients)})"
        )

    metadata.to_csv(output_filename, index=False)
    return metadata


def get_column_name_groups(metadata: pd.DataFrame, dichotome: bool = True) -> dict:
    """
    Group metadata column names by clinical category.

    Args:
        metadata:  Prepared metadata DataFrame.
        dichotome: If ``True``, omit ``medicoID_y`` from the general group
                   (it is not present in the binary-label version).

    Returns:
        Dictionary mapping category name → list of column names.
        Keys: ``"general"``, ``"symbol"``, ``"rounded"``, ``"irregular"``,
        ``"mixed"``, ``"large"``, ``"pleural"``, ``"occupational"``.
    """
    general_columns = [
        "Nachname", "Vorname", "Geburtsdatum", "medicoID", "Untersuchungsdatum",
        "Untersuchung_Ort", "Untersuchung_Art", "id", "Untersuchung_id",
        "technical_quality", "Aufnahmenummer", "Anforderungsnummer", "Untersuchungsnummer",
        "Station Anfordernd", "Fachrichtung Anfordernd",
        "Untersuchung Dokumentiert", "fileID", "patientID",
    ]
    if not dichotome:
        general_columns.append("medicoID_y")

    symbol_columns    = [c for c in metadata.columns if c.startswith("symbol_")] + general_columns
    rounded_columns   = [c for c in metadata.columns if c.startswith("small_rounded_")] + general_columns
    irregular_columns = [c for c in metadata.columns if c.startswith("small_irregular_")] + general_columns
    mixed_columns     = [c for c in metadata.columns if c.startswith("mixed_")] + general_columns
    large_columns     = (
        [c for c in metadata.columns if c.startswith("large_")]
        + ["costophrenic_angle_obliteration_nad",
           "costophrenic_angle_obliteration_right",
           "costophrenic_angle_obliteration_left"]
        + general_columns
    )
    pleural_columns    = [c for c in metadata.columns if "pleural" in c] + general_columns
    occupation_columns = [c for c in metadata.columns if c.startswith("occupational_disease_")] + general_columns

    return {
        "general":      general_columns,
        "symbol":       symbol_columns,
        "rounded":      rounded_columns,
        "irregular":    irregular_columns,
        "mixed":        mixed_columns,
        "large":        large_columns,
        "pleural":      pleural_columns,
        "occupational": occupation_columns,
    }
