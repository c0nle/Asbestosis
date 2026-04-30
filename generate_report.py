"""
generate_report.py
------------------
Generates a two-sheet XLSX report:

  Sheet 1 – Label Distribution  (based on training-ready rows after all filters)
  Sheet 2 – Data Audit          (step-by-step funnel from raw CSV to training-ready rows)

Usage:
    python generate_report.py --output report.xlsx
"""

import argparse
import os
import re

import numpy as np
import pandas as pd


BASE = "/hpcwork/rwth1954/Asbestosis_Data"
META_FILE = os.path.join(BASE, "dichotome_data_anonymized_with_patientID.csv")
MAPPING_FILE = os.path.join(BASE, "mapping.csv")
PNG_DIR = os.path.join(BASE, "png")

LABEL_COLS = [
    "small_rounded_right",
    "small_rounded_left",
    "small_rounded_size",
    "small_irregular_right",
    "small_irregular_left",
    "small_irregular_size",
    "mixed_shapes",
    "diffuse_pleural_thickening_width",
    "diffuse_pleural_thickening_extend",
    "diffuse_pleural_location",
    "localized_pleural_thickening_width",
    "localized_pleural_thickening_extend",
    "local_pleural_location",
    "pleural_calcification_location",
    "pleural_calcification_side",
    "occupational_disease",
]

TRAINING_LABELS = ["mixed_shapes", "occupational_disease"]


def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip().lower() in {"", "nan", "none", "-1", "-1.0"}:
        return True
    try:
        return float(val) == -1
    except Exception:
        return False


def _apply_mapping(df: pd.DataFrame, mapping_file: str) -> pd.DataFrame:
    """Join Anforderungsnummer → fileID via mapping.csv (mirrors utils._ensure_file_id)."""
    mapping = pd.read_csv(mapping_file, dtype={"medicoID": str})
    mapping["fileID"] = pd.to_numeric(mapping["fileID"], errors="coerce")
    mapping["Anforderungsnummer"] = mapping["medicoID"].str[:-2]
    mask = mapping["Anforderungsnummer"].str.fullmatch(r"\d+").eq(True)
    mapping = mapping[mask]
    mapping["Anforderungsnummer"] = mapping["Anforderungsnummer"].astype(int)
    merged = df.drop(columns=["fileID"], errors="ignore").merge(
        mapping[["Anforderungsnummer", "fileID"]], on="Anforderungsnummer", how="left"
    )
    merged["fileID"] = merged["fileID"].fillna(-1).astype(int)
    return merged


def _disk_file_ids(png_dir: str) -> set:
    ids = set()
    for f in os.listdir(png_dir):
        m = re.match(r"^(\d+)-", f)
        if m:
            ids.add(int(m.group(1)))
    return ids


# ---------------------------------------------------------------------------
# Build the filtered training-ready DataFrame (mirrors main.py pipeline)
# ---------------------------------------------------------------------------

def build_training_df(df_raw: pd.DataFrame) -> tuple:
    """
    Returns (df_img, audit_steps) where df_img is the training-ready DataFrame
    and audit_steps is a list of dicts describing each filter step.
    """
    steps = []

    def record(step, desc, df_before, df_after, note=""):
        r_before = len(df_before)
        p_before = df_before["patientID"].nunique()
        r_after  = len(df_after)
        p_after  = df_after["patientID"].nunique()
        steps.append({
            "Step":              step,
            "Description":       desc,
            "Rows":              r_after,
            "Patients":          p_after,
            "Rows dropped":      r_before - r_after,
            "Patients dropped":  p_before - p_after,
            "Reason / note":     note,
        })
        return df_after

    # Step 0 — raw (no before/after delta)
    r0 = len(df_raw)
    p0 = df_raw["patientID"].nunique()
    steps.append({
        "Step": 0,
        "Description": "Raw metadata CSV",
        "Rows": r0,
        "Patients": p0,
        "Rows dropped": "—",
        "Patients dropped": "—",
        "Reason / note": "dichotome_data_anonymized_with_patientID.csv",
    })

    # Step 1 — left-join adds rows where one Anforderungsnummer maps to multiple fileIDs
    df_merged = _apply_mapping(df_raw, MAPPING_FILE)
    dup_added = len(df_merged) - r0
    r1 = len(df_merged)
    p1 = df_merged["patientID"].nunique()
    steps.append({
        "Step": 1,
        "Description": "Left-join Anforderungsnummer → fileID (mapping.csv)",
        "Rows": r1,
        "Patients": p1,
        "Rows dropped": f"+{dup_added}",
        "Patients dropped": 0,
        "Reason / note": (
            f"{dup_added} Anforderungsnummern map to more than one fileID → "
            f"duplicate rows added"
        ),
    })

    # Step 2 — drop rows without a fileID match
    df_mapped = df_merged[df_merged["fileID"] != -1].copy()
    r2 = len(df_mapped)
    p2 = df_mapped["patientID"].nunique()
    steps.append({
        "Step": 2,
        "Description": "Drop rows without fileID match (fileID == −1)",
        "Rows": r2,
        "Patients": p2,
        "Rows dropped": r1 - r2,
        "Patients dropped": p1 - p2,
        "Reason / note": "Anforderungsnummer not found in mapping.csv → no image can be located",
    })

    return df_mapped, steps


# ---------------------------------------------------------------------------
# Sheet 1: Label Distribution  (on training-ready data)
# ---------------------------------------------------------------------------

def build_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    rows = []
    for col in LABEL_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        nan_count = int(series.apply(_is_missing).sum())
        valid = series[~series.apply(_is_missing)]

        unique_vals = sorted(valid.unique(), key=lambda x: str(x))

        if set(map(str, unique_vals)) <= {"0", "1", "0.0", "1.0"}:
            pos_val   = "1"
            pos_count = int(valid.astype(str).str.strip().isin({"1", "1.0"}).sum())
            neg_count = int(valid.astype(str).str.strip().isin({"0", "0.0"}).sum())
        elif len(unique_vals) == 2:
            pos_val   = str(unique_vals[1])
            pos_count = int((valid == unique_vals[1]).sum())
            neg_count = int((valid == unique_vals[0]).sum())
        else:
            # Categorical / ordinal: show total valid vs missing; positive = any non-missing
            pos_val   = "any non-missing"
            pos_count = len(valid)
            neg_count = 0

        pos_pct = round(100 * pos_count / total_rows, 2) if total_rows > 0 else float("nan")

        rows.append({
            "Label":                col,
            "Positive count":       pos_count,
            "Negative count":       neg_count,
            "NaN / missing count":  nan_count,
            "Positive [%]":         pos_pct,
            "Positive value":       pos_val,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="report.xlsx")
    args = parser.parse_args()

    print("Loading metadata...")
    df_raw = pd.read_csv(META_FILE)

    print("Building training-ready DataFrame and audit steps...")
    df_training, audit_steps = build_training_df(df_raw)
    print(f"  Training-ready: {len(df_training)} rows, {df_training['patientID'].nunique()} patients")

    # Label distribution is based on the splits CSV (2086 rows) — this is what
    # create_splits assigns to folds. The 1 image missing on disk is dropped at
    # training time, not at split-assignment time, so the splits CSV is the
    # canonical reference for label counts.
    splits_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "splits",
        os.path.basename(META_FILE).replace(".csv", "_stratified_folds.csv"),
    )
    if os.path.isfile(splits_csv):
        df_for_dist = pd.read_csv(splits_csv)
        print(f"  Label distribution from splits CSV: {len(df_for_dist)} rows")
    else:
        # Fallback: use the fileID-mapped data (before image-on-disk check)
        df_for_dist = _apply_mapping(df_raw, MAPPING_FILE)
        df_for_dist = df_for_dist[df_for_dist["fileID"] != -1].copy()
        print(f"  Label distribution from mapped data (splits CSV not found): {len(df_for_dist)} rows")

    print("Building label distribution (on splits-CSV data)...")
    df_labels = build_label_distribution(df_for_dist)

    df_audit = pd.DataFrame(audit_steps)

    print(f"Writing {args.output}...")
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df_labels.to_excel(writer, sheet_name="Label Distribution", index=False)
        df_audit.to_excel(writer, sheet_name="Data Audit", index=False)

        for sheet_name, df in [("Label Distribution", df_labels), ("Data Audit", df_audit)]:
            ws = writer.sheets[sheet_name]
            for col_idx, col in enumerate(df.columns, 1):
                max_len = max(len(str(col)), df[col].astype(str).str.len().max())
                ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 4, 60)

    print(f"Done: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
