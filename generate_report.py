"""
generate_report.py
------------------
Generates a two-sheet XLSX report:

  Sheet 1 - Label Distribution  (based on training-ready rows after all filters)
  Sheet 2 - Data Audit          (step-by-step funnel from raw CSV to training-ready rows)

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
BEFUND_FILE = os.path.join(BASE, "st_Befundtext_RO_Thorax_AR_noName.xlsx")


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
    """
    Join Untersuchungsnummer → fileID via mapping.csv (mirrors utils._ensure_file_id).

    Uses Untersuchungsnummer as the primary key (1:1 per image).  Falls back to
    Anforderungsnummer (de-duplicated, first match) for rows that don't match via
    Untersuchungsnummer.

    Background: medicoID encodes both keys:
        medicoID = Anforderungsnummer + two-digit index
        Untersuchungsnummer = Anforderungsnummer + '-' + index (no leading zero)

    Joining on Anforderungsnummer alone caused cross-matches: when one exam
    produced two images (e.g. PA + lateral), both rows received both fileIDs,
    assigning the wrong image to 8 rows in the old pipeline.  The
    Untersuchungsnummer join resolves each row to exactly its own image.
    """
    mapping = pd.read_csv(mapping_file, dtype={"medicoID": str})
    mapping["fileID"] = pd.to_numeric(mapping["fileID"], errors="coerce")
    anf_series = mapping["medicoID"].str[:-2]
    mask = anf_series.str.fullmatch(r"\d+").eq(True)
    mapping = mapping[mask].copy()
    mapping["Anforderungsnummer"] = anf_series[mask].astype(int)

    raw_idx = mapping["medicoID"].str[-2:].str.lstrip("0")
    mapping["_unt_key"] = (
        mapping["Anforderungsnummer"].astype(str) + "-" +
        raw_idx.where(raw_idx != "", other="0")
    )
    # 7 _unt_key values appear twice in mapping.csv (same Untersuchungsnummer,
    # two fileIDs on disk). Keep the first occurrence so each key maps to
    # exactly one fileID — avoids duplicate training rows with identical labels.
    mapping = mapping.drop_duplicates("_unt_key")

    meta = df.drop(columns=["fileID"], errors="ignore").copy()

    if "Untersuchungsnummer" in meta.columns:
        meta["Untersuchungsnummer"] = meta["Untersuchungsnummer"].astype(str)
        merged = meta.merge(
            mapping[["_unt_key", "fileID"]],
            left_on="Untersuchungsnummer", right_on="_unt_key",
            how="left",
        ).drop(columns=["_unt_key"], errors="ignore")
        merged["fileID"] = merged["fileID"].fillna(-1).astype(int)

        unmatched = merged["fileID"] == -1
        if unmatched.any():
            anf_map = (
                mapping[["Anforderungsnummer", "fileID"]]
                .drop_duplicates("Anforderungsnummer")
                .set_index("Anforderungsnummer")["fileID"]
            )
            merged.loc[unmatched, "fileID"] = (
                merged.loc[unmatched, "Anforderungsnummer"]
                .map(anf_map)
                .fillna(-1)
                .astype(int)
                .values
            )
    else:
        merged = meta.merge(
            mapping[["Anforderungsnummer", "fileID"]]
            .drop_duplicates("Anforderungsnummer"),
            on="Anforderungsnummer", how="left",
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

    # Step 1 — left-join via Untersuchungsnummer (1:1 per image)
    # A few exams have 2 images (PA + lateral): the old Anforderungsnummer join
    # added duplicate rows AND cross-matched fileIDs (row for image A got fileID B).
    # The Untersuchungsnummer join is exact (1:1) so it adds at most 1 row per
    # unmatched Untersuchungsnummer that falls back to Anforderungsnummer.
    df_merged = _apply_mapping(df_raw, MAPPING_FILE)
    dup_added = len(df_merged) - r0
    r1 = len(df_merged)
    p1 = df_merged["patientID"].nunique()
    extra_str = f"+{dup_added}" if dup_added >= 0 else str(dup_added)
    steps.append({
        "Step": 1,
        "Description": "Left-join Untersuchungsnummer → fileID (mapping.csv)",
        "Rows": r1,
        "Patients": p1,
        "Rows dropped": extra_str,
        "Patients dropped": 0,
        "Reason / note": (
            "medicoID encodes Anforderungsnummer + two-digit image index. "
            "Untersuchungsnummer = Anforderungsnummer-index gives a 1:1 key per image. "
            f"{dup_added} extra rows from the fallback Anforderungsnummer join "
            "(exams where Untersuchungsnummer was absent in metadata). "
            "Previous Anforderungsnummer-only join caused 8 cross-matched rows "
            "(exam with 2 images: each row received the other image's fileID)."
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
        "Reason / note": (
            "Untersuchungsnummer / Anforderungsnummer not found in mapping.csv "
            "→ no image can be located for these rows."
        ),
    })

    return df_mapped, steps


# ---------------------------------------------------------------------------
# Sheet 3: Patient Demographics  (age + first vs. follow-up)
# ---------------------------------------------------------------------------

FOLLOWUP_KWS = [
    "befundänderung", "voraufnahme", "im vergleich", "unverändert",
    "voruntersuchung", "vorbefund", "vergleichsaufnahme", "vgl.",
]
FIRST_KWS = ["liegen nicht vor", "liegt nicht vor", "keine voraufnahmen"]


def build_demographics(splits_csv_or_df) -> pd.DataFrame:
    if isinstance(splits_csv_or_df, pd.DataFrame):
        df = splits_csv_or_df.copy()
    else:
        df = pd.read_csv(splits_csv_or_df)
    df["Untersuchungsdatum"] = pd.to_datetime(df["Untersuchungsdatum"], dayfirst=True)

    # Join Geburtsdatum + Dokumentinhalt from Befundtext via Anforderungsnummer
    befund = pd.read_excel(
        BEFUND_FILE, header=8,
        usecols=["Anforderungsnummer", "Geburtsdatum", "Dokumentinhalt"],
    )
    befund["Anforderungsnummer"] = pd.to_numeric(befund["Anforderungsnummer"], errors="coerce")
    befund["Geburtsdatum"] = pd.to_datetime(befund["Geburtsdatum"], errors="coerce")
    befund = befund.dropna(subset=["Anforderungsnummer"]).drop_duplicates(subset="Anforderungsnummer")

    df = df.merge(befund, on="Anforderungsnummer", how="left")
    valid = df.dropna(subset=["Geburtsdatum", "Untersuchungsdatum"]).copy()

    # Age at examination: floor years
    valid["age"] = (
        (valid["Untersuchungsdatum"] - valid["Geburtsdatum"]).dt.days / 365.25
    ).astype(int)

    # Date-based first vs. follow-up
    first_date = valid.groupby("patientID")["Untersuchungsdatum"].transform("min")
    valid["date_based"] = (valid["Untersuchungsdatum"] == first_date).map(
        {True: "first", False: "followup"}
    )

    # Text-based signal from Dokumentinhalt
    text = valid["Dokumentinhalt"].fillna("").str.lower()
    sig_fu    = text.apply(lambda t: any(k in t for k in FOLLOWUP_KWS))
    sig_first = text.apply(lambda t: any(k in t for k in FIRST_KWS))
    valid["text_signal"] = "unclear"
    valid.loc[sig_fu  & ~sig_first, "text_signal"] = "followup"
    valid.loc[sig_first & ~sig_fu,  "text_signal"] = "first"

    n_total   = len(valid)
    n_no_dob  = len(df) - n_total
    n_first   = (valid["date_based"] == "first").sum()
    n_followup = (valid["date_based"] == "followup").sum()

    # Validation cross-counts
    clear            = valid["text_signal"] != "unclear"
    n_clear          = int(clear.sum())
    agree = (
        ((valid["date_based"] == "first")    & (valid["text_signal"] == "first")) |
        ((valid["date_based"] == "followup") & (valid["text_signal"] == "followup"))
    )
    n_agree          = int(agree[clear].sum())
    n_date1_text_fu  = int(((valid["date_based"] == "first") & (valid["text_signal"] == "followup")).sum())
    n_date_fu_text1  = int(((valid["date_based"] == "followup") & (valid["text_signal"] == "first")).sum())
    # Corrected first: date=first rows that text confirms are actually follow-ups are subtracted
    n_first_corrected   = int(n_first - n_date1_text_fu)
    n_followup_corrected = int(n_followup + n_date1_text_fu)

    rows = [
        {"Metric": "Total rows (training data)",                    "Value": n_total},
        {"Metric": "Rows without Geburtsdatum (no Befundtext match)", "Value": n_no_dob},
        {"Metric": "",                                              "Value": ""},
        # Age
        {"Metric": "── Age at examination ──",                     "Value": ""},
        {"Metric": "Age — Mean",                                    "Value": round(valid["age"].mean(), 1)},
        {"Metric": "Age — Median",                                  "Value": round(valid["age"].median(), 1)},
        {"Metric": "Age — Min",                                     "Value": int(valid["age"].min())},
        {"Metric": "Age — Max",                                     "Value": int(valid["age"].max())},
        {"Metric": "",                                              "Value": ""},
        # Date-based classification
        {"Metric": "── First vs. Follow-up (date-based) ──",       "Value": ""},
        {"Metric": "First examination",                             "Value": int(n_first)},
        {"Metric": "Follow-up",                                     "Value": int(n_followup)},
        {"Metric": "First examination [%]",                         "Value": round(100 * n_first / n_total, 1)},
        {"Metric": "Follow-up [%]",                                 "Value": round(100 * n_followup / n_total, 1)},
        {"Metric": "",                                              "Value": ""},
        # Text-based validation
        {"Metric": "── Text-based validation (Befundtext keywords) ──", "Value": ""},
        {"Metric": "Follow-up keywords",                            "Value": ", ".join(FOLLOWUP_KWS)},
        {"Metric": "First-exam keywords",                           "Value": ", ".join(FIRST_KWS)},
        {"Metric": "Rows with clear text signal",                   "Value": f"{n_clear} / {n_total} ({100*n_clear/n_total:.1f}%)"},
        {"Metric": "Agreement (date vs. text, where text clear)",   "Value": f"{n_agree} / {n_clear} ({100*n_agree/n_clear:.1f}%)"},
        {"Metric": "Date=first but text=follow-up",                 "Value": f"{n_date1_text_fu}  → prior exam existed outside dataset"},
        {"Metric": "Date=follow-up but text=first",                 "Value": n_date_fu_text1},
        {"Metric": "",                                              "Value": ""},
        # Corrected counts
        {"Metric": "── Corrected counts (subtracting text-confirmed misclassifications) ──", "Value": ""},
        {"Metric": "First examination (corrected)",                 "Value": f"{n_first_corrected} ({round(100*n_first_corrected/n_total,1)}%)"},
        {"Metric": "Follow-up (corrected)",                         "Value": f"{n_followup_corrected} ({round(100*n_followup_corrected/n_total,1)}%)"},
    ]
    return pd.DataFrame(rows)


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

    print("Building demographics sheet...")
    demo_source = splits_csv if os.path.isfile(splits_csv) else df_training
    df_demo = build_demographics(demo_source)

    sheets: list[tuple[str, pd.DataFrame]] = [
        ("Label Distribution", df_labels),
        ("Data Audit", df_audit),
        ("Patient Demographics", df_demo),
    ]

    print(f"Writing {args.output}...")
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        for sheet_name, df in sheets:
            ws = writer.sheets[sheet_name]
            for col_idx, col in enumerate(df.columns, 1):
                max_len = max(len(str(col)), df[col].astype(str).str.len().max())
                ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 4, 60)

    print(f"Done: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
