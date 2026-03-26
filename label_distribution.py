#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_LABELS = [
    "mixed_shapes",
    "diffuse_pleural_location",
    "local_pleural_location",
    "pleural_calcification_location",
    "occupational_disease",
]

DEFAULT_FOLDS_CSV = os.path.join("splits", "dichotome_data_pseudonym_stratified_folds.csv")


def _is_missing_label_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        v = value.strip()
        if v == "" or v.lower() in {"nan", "none"}:
            return True
        if v in {"-1", "-1.0"}:
            return True
    return value == -1


def _coerce_file_id(value: object) -> Optional[str]:
    """
    Best-effort conversion of fileID to the canonical string used in filenames.
    Input is typically a CSV string (e.g. "3768" or "3768.0").
    """
    if _is_missing_label_value(value):
        return None
    s = str(value).strip()
    try:
        fv = float(s)
        if fv != fv:  # NaN
            return None
        if float(fv).is_integer():
            return str(int(fv))
    except Exception:
        pass
    # Fallback: accept raw string if it looks like an integer id
    if s.isdigit():
        return s
    return None


def _zip_has_files(path: str) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    return True
    except Exception:
        return False
    return False


def _has_image_file(image_dir: str, file_id: str) -> bool:
    # Mirror main.py candidate roots: image_dir/png (preferred), image_dir, image_dir/anon (optional)
    candidate_roots = []
    png_dir = os.path.join(image_dir, "png")
    anon_dir = os.path.join(image_dir, "anon")
    if os.path.isdir(png_dir):
        candidate_roots.append(png_dir)
    candidate_roots.append(image_dir)
    if os.path.isdir(anon_dir):
        candidate_roots.append(anon_dir)
    candidate_roots = [d for d in candidate_roots if os.path.isdir(d)]

    for root in candidate_roots:
        for ext in ("png", "jpg", "jpeg"):
            p = os.path.join(root, f"{file_id}.{ext}")
            if os.path.exists(p):
                return True
            # Minimal glob emulation: scan directory for prefix match (keeps stdlib-only).
            prefix = f"{file_id}-"
            try:
                for name in os.listdir(root):
                    if name.startswith(prefix) and name.lower().endswith(("." + ext,)):
                        return True
            except Exception:
                pass

        zp = os.path.join(root, f"{file_id}.zip")
        if os.path.exists(zp) and _zip_has_files(zp):
            return True

    return False


def _print_dataset_mapping_stats(rows: List[Dict[str, str]], image_dir: Optional[str]) -> None:
    total = len(rows)
    if total == 0:
        return
    file_ids = []
    fileid_missing = 0
    for r in rows:
        fid = _coerce_file_id(r.get("fileID"))
        if fid is None:
            fileid_missing += 1
            continue
        file_ids.append(fid)
    mapped = len(file_ids)
    unique = len(set(file_ids))
    print(f"fileID mapped: {mapped}/{total} (unique={unique}, missing={fileid_missing})")

    medico_missing = sum(1 for r in rows if _is_missing_label_value(r.get("medicoID")))
    print(f"medicoID missing: {medico_missing}/{total}")

    if image_dir:
        ok = 0
        for fid in file_ids:
            if _has_image_file(image_dir, fid):
                ok += 1
        print(f"image files found: {ok}/{mapped} (checked in {image_dir})")


def _count_missing(rows: Iterable[Dict[str, str]], col: str) -> int:
    return sum(1 for r in rows if _is_missing_label_value(r.get(col)))


def _count_fileid_mappable(rows: Iterable[Dict[str, str]]) -> int:
    n = 0
    for r in rows:
        if _coerce_file_id(r.get("fileID")) is not None:
            n += 1
    return n


def _count_any_label_present(rows: Iterable[Dict[str, str]], labels: List[str]) -> int:
    n = 0
    for r in rows:
        for lab in labels:
            if not _is_missing_label_value(r.get(lab)):
                n += 1
                break
    return n


def _track_filtering(
    raw_rows: List[Dict[str, str]],
    folds_rows: List[Dict[str, str]],
    *,
    id_col: str,
    fold: int,
    split: str,
    split_label: str,
    labels: List[str],
    image_dir: Optional[str],
) -> None:
    """
    Track how many rows remain after the main filters used in this project.

    This is meant to explain the reduction from the original metadata CSV to a
    particular fold/split (e.g. Fold0=train).
    """
    raw_total = len(raw_rows)
    if raw_total == 0:
        print("\n-- Filter tracking --")
        print("raw rows: 0")
        return

    print("\n-- Filter tracking --")
    print(f"raw rows: {raw_total}")
    print(f"raw fileID mappable: {_count_fileid_mappable(raw_rows)}/{raw_total}")
    print(f"raw medicoID missing: {_count_missing(raw_rows, 'medicoID')}/{raw_total}")
    print(f"folds rows (reference): {len(folds_rows)}")

    # 1) main.py: metadata = metadata[metadata["fileID"] != -1]
    step1 = [r for r in raw_rows if _coerce_file_id(r.get("fileID")) is not None]
    print(f"after fileID filter: {len(step1)} (dropped {raw_total - len(step1)})")

    # 2) Preprocessor_Metadata.create_splits: metadata = metadata[metadata[training_label] != -1]
    step2 = [r for r in step1 if not _is_missing_label_value(r.get(split_label))]
    print(f"after split-label '{split_label}' present: {len(step2)} (dropped {len(step1) - len(step2)})")

    # 3) rows that appear in the folds CSV (i.e., got a Fold assignment)
    fold_col = f"Fold{int(fold)}"
    fold_by_id: Dict[str, str] = {}
    for rr in folds_rows:
        k = str(rr.get(id_col, "")).strip()
        if k:
            fold_by_id[k] = str(rr.get(fold_col, "")).strip().lower()
    step3 = [r for r in step2 if str(r.get(id_col, "")).strip() in fold_by_id]
    print(f"after fold join on '{id_col}': {len(step3)} (dropped {len(step2) - len(step3)})")

    # 4) select split for this fold
    split = str(split).strip().lower()
    split_counts = Counter(fold_by_id.get(str(r.get(id_col, "")).strip(), "") for r in step3)
    split_counts = Counter({k: v for k, v in split_counts.items() if k})
    if split_counts:
        parts = ", ".join([f"{k}={v}" for k, v in split_counts.items()])
        print(f"{fold_col} distribution (joined): {parts}")
    step4 = [r for r in step3 if fold_by_id.get(str(r.get(id_col, "")).strip(), "") == split]
    print(f"{fold_col} == '{split}': {len(step4)}")

    # 5) main.py:_filter_any_label (keeps rows with at least one label present)
    any_present = _count_any_label_present(step4, labels)
    print(f"after any-label-present (labels={len(labels)}): {any_present} (dropped {len(step4) - any_present})")

    # 6) optional: approximate image existence filtering
    if image_dir:
        ok = 0
        for r in step4:
            fid = _coerce_file_id(r.get("fileID"))
            if fid is None:
                continue
            if _has_image_file(image_dir, fid):
                ok += 1
        print(f"after image-found (approx): {ok} (dropped {len(step4) - ok})")


def _canonical_label_value(value: object):
    if _is_missing_label_value(value):
        return None
    if isinstance(value, (int, bool)):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, str):
        s = value.strip()
        try:
            fv = float(s)
            if fv.is_integer():
                return int(fv)
            return fv
        except Exception:
            return s
    return str(value).strip()


def _coerce_binary_label(value: object) -> Optional[int]:
    if _is_missing_label_value(value):
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, float):
        if value.is_integer() and int(value) in (0, 1):
            return int(value)
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"0", "0.0", "false", "no"}:
            return 0
        if v in {"1", "1.0", "true", "yes"}:
            return 1
        try:
            fv = float(v)
            if fv.is_integer() and int(fv) in (0, 1):
                return int(fv)
        except Exception:
            pass
    return None


def _read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


@dataclass
class LabelStats:
    total: int
    pos: int
    neg: int
    missing: int
    other: int
    uniq_present: int
    top_values: List[Tuple[str, int]]
    mode: str  # "binary" or "present_vs_missing"


def _binary_value_map_from_rows(rows: Iterable[Dict[str, str]], label: str) -> Tuple[Dict[object, int], List[str]]:
    """
    Mirror main.py::_binary_value_map_from_series but for CSV rows.
    - missing values are ignored when building the category set
    - if values are true 0/1 encodings, preserve that mapping
    - otherwise map by stable string ordering (0=sorted[0], 1=sorted[1])
    """
    canon = []
    for r in rows:
        raw = r.get(label, None)
        c = _canonical_label_value(raw)
        if c is None:
            continue
        canon.append(c)

    uniq = list(dict.fromkeys(canon))
    if len(uniq) == 0:
        return {}, ["0", "1"]
    if len(uniq) > 2:
        raise SystemExit(
            f"Label '{label}' has >2 unique non-missing values ({len(uniq)}): {uniq[:5]} ..."
        )
    if len(uniq) == 1:
        v = uniq[0]
        return {v: 1}, ["0", str(v)]

    a, b = uniq[0], uniq[1]
    ca, cb = _coerce_binary_label(a), _coerce_binary_label(b)
    if ca is not None and cb is not None and set([ca, cb]).issubset({0, 1}):
        inv = {a: int(ca), b: int(cb)}
        return inv, ["0", "1"]

    ordered = sorted(uniq, key=lambda x: str(x))
    mapping = {ordered[0]: 0, ordered[1]: 1}
    idx_to_label = [str(ordered[0]), str(ordered[1])]
    return mapping, idx_to_label


def _label_stats(rows: Iterable[Dict[str, str]], label: str, mapping: Dict[object, int], idx_to_label: List[str], top_k: int) -> LabelStats:
    total = 0
    pos = 0
    neg = 0
    missing = 0
    other = 0

    present_values = Counter()
    for r in rows:
        total += 1
        raw = r.get(label, None)
        if _is_missing_label_value(raw):
            missing += 1
            continue
        key = _canonical_label_value(raw)
        if key is None or key not in mapping:
            other += 1
            continue
        y = int(mapping[key])
        if y == 1:
            pos += 1
        else:
            neg += 1
        present_values[str(key)] += 1

    uniq_present = len(present_values)
    top_values = present_values.most_common(top_k)
    mode = "binary" if idx_to_label == ["0", "1"] else "categorical_2class"
    return LabelStats(
        total=total,
        pos=pos,
        neg=neg,
        missing=missing,
        other=other,
        uniq_present=uniq_present,
        top_values=top_values,
        mode=mode,
    )


def _fmt_pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{(100.0 * n / d):5.1f}%"


def _print_stats(split_name: str, rows: List[Dict[str, str]], labels: List[str], top_k: int) -> None:
    total = len(rows)
    print(f"\n== {split_name} (rows={total}) ==")
    if total == 0:
        return

    _print_dataset_mapping_stats(rows, image_dir=_PRINT_CTX_IMAGE_DIR)

    header = ["label", "pos", "neg", "missing", "pos% (known)", "neg_label", "pos_label", "top"]
    label_w = max(10, max([len("label")] + [len(l) for l in labels]))
    widths = [label_w, 7, 7, 9, 13, 14, 14, 0]

    def print_row(cols: List[str]) -> None:
        out = []
        for i, c in enumerate(cols[:-1]):
            out.append(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]))
        out.append(cols[-1])
        print("  ".join(out))

    print_row(header)
    print_row(["-" * len(h) for h in header[:-1]] + ["-" * 3])

    for lab in labels:
        if not rows or (lab not in rows[0] and all(lab not in r for r in rows[:5])):
            print_row([lab, "n/a", "n/a", "n/a", "n/a", "missing_col", "", ""])
            continue
        # mapping is always built from TRAIN split in main.py; caller passes the correct mapping dicts.
        mapping = _PRINT_CTX_VALUE_MAPS.get(lab, {})
        idx_to_label = _PRINT_CTX_IDX_TO_LABEL.get(lab, ["0", "1"])
        st = _label_stats(rows, lab, mapping=mapping, idx_to_label=idx_to_label, top_k=top_k)
        known = st.total - st.missing - st.other
        top = ", ".join([f"{k}={v}" for k, v in st.top_values])
        neg_label = idx_to_label[0] if len(idx_to_label) > 0 else "0"
        pos_label = idx_to_label[1] if len(idx_to_label) > 1 else "1"
        print_row(
            [
                lab,
                str(st.pos),
                str(st.neg),
                str(st.missing),
                _fmt_pct(st.pos, known),
                neg_label,
                pos_label,
                top,
            ]
        )


def _load_split_csvs(split_dir: str, fold: int) -> Dict[str, str]:
    out = {}
    for split in ("train", "val", "test"):
        p = os.path.join(split_dir, f"stratified_{split}_set-f{fold}.csv")
        if not os.path.isfile(p):
            raise SystemExit(f"Missing split CSV: {p}")
        out[split] = p
    return out


_PRINT_CTX_VALUE_MAPS: Dict[str, Dict[object, int]] = {}
_PRINT_CTX_IDX_TO_LABEL: Dict[str, List[str]] = {}
_PRINT_CTX_IMAGE_DIR: Optional[str] = None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print pos/neg distribution for the label columns used by the model (missing labels are excluded from pos/neg)."
    )
    ap.add_argument(
        "--csv",
        default=DEFAULT_FOLDS_CSV if os.path.isfile(DEFAULT_FOLDS_CSV) else None,
        help=f"Single CSV to analyze (mapping is built from this CSV). Default: {DEFAULT_FOLDS_CSV}",
    )
    ap.add_argument("--split-dir", default="splits", help="Directory containing stratified_{train,val,test}_set-f*.csv")
    ap.add_argument("--fold", type=int, default=0, help="Fold index used with --split-dir.")
    ap.add_argument(
        "--image-dir",
        default=None,
        help="Optional directory to verify if mapped fileIDs have an image file (png/jpg/jpeg/zip).",
    )
    ap.add_argument(
        "--raw-csv",
        default=None,
        help="Optional original metadata CSV to track how rows reduce to a given fold/split (prints a filter-by-filter summary).",
    )
    ap.add_argument(
        "--folds-csv",
        default=DEFAULT_FOLDS_CSV if os.path.isfile(DEFAULT_FOLDS_CSV) else None,
        help=f"Fold assignment CSV used for tracking (default: {DEFAULT_FOLDS_CSV}).",
    )
    ap.add_argument(
        "--id-col",
        default="id",
        help="Join key column name to link raw-csv rows to folds-csv rows (default: id).",
    )
    ap.add_argument(
        "--track-split",
        default="train",
        choices=["train", "val", "test"],
        help="Which split to track when --raw-csv is provided (default: train).",
    )
    ap.add_argument(
        "--track-split-label",
        default="mixed_shapes",
        help="Split label used for the split-label-present filter in tracking (default: mixed_shapes).",
    )
    ap.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label columns to report.",
    )
    ap.add_argument("--top-k", type=int, default=5, help="Show top-K values for non-missing entries.")
    ap.add_argument("--splits", default="all", help="Comma-separated: train,val,test,all (default: all). Ignored with --csv.")
    ap.add_argument("--by-split", action="store_true", help="Also print per-split tables (train/val/test).")
    args = ap.parse_args()

    labels = [s.strip() for s in str(args.labels).split(",") if s.strip()]
    if not labels:
        raise SystemExit("No labels given. Use --labels a,b,c")

    global _PRINT_CTX_IMAGE_DIR
    _PRINT_CTX_IMAGE_DIR = str(args.image_dir) if args.image_dir else None

    # Precedence:
    # - If the user explicitly passes --csv, always use CSV mode.
    # - Otherwise, if the user passes any split-related flags, use split-dir mode.
    # - Otherwise, fall back to the default folds CSV (if present).
    csv_explicit = "--csv" in sys.argv
    split_flags_explicit = any(
        f in sys.argv for f in ("--split-dir", "--fold", "--splits", "--by-split")
    )
    use_csv_mode = bool(args.csv) and (csv_explicit or not split_flags_explicit)

    if use_csv_mode:
        _, rows = _read_csv_rows(str(args.csv))
        for lab in labels:
            m, idx = _binary_value_map_from_rows(rows, lab)
            _PRINT_CTX_VALUE_MAPS[lab] = m
            _PRINT_CTX_IDX_TO_LABEL[lab] = idx
        if args.raw_csv:
            if not args.folds_csv:
                raise SystemExit("--folds-csv is required when using --raw-csv.")
            _, raw_rows = _read_csv_rows(str(args.raw_csv))
            _, folds_rows = _read_csv_rows(str(args.folds_csv))
            _track_filtering(
                raw_rows=raw_rows,
                folds_rows=folds_rows,
                id_col=str(args.id_col),
                fold=int(args.fold),
                split=str(args.track_split),
                split_label=str(args.track_split_label),
                labels=labels,
                image_dir=_PRINT_CTX_IMAGE_DIR,
            )
        _print_stats(os.path.basename(str(args.csv)), rows, labels, top_k=int(args.top_k))
        return

    split_paths = _load_split_csvs(str(args.split_dir), int(args.fold))
    split_sel_raw = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if not split_sel_raw or (len(split_sel_raw) == 1 and split_sel_raw[0].lower() == "all"):
        split_sel = ["train", "val", "test"]
    else:
        split_sel = [s.lower() for s in split_sel_raw]

    # Build mappings from TRAIN split (matches main.py behavior).
    _, train_rows = _read_csv_rows(split_paths["train"])
    for lab in labels:
        m, idx = _binary_value_map_from_rows(train_rows, lab)
        _PRINT_CTX_VALUE_MAPS[lab] = m
        _PRINT_CTX_IDX_TO_LABEL[lab] = idx

    if args.raw_csv:
        if not args.folds_csv:
            raise SystemExit("--folds-csv is required when using --raw-csv.")
        _, raw_rows = _read_csv_rows(str(args.raw_csv))
        _, folds_rows = _read_csv_rows(str(args.folds_csv))
        _track_filtering(
            raw_rows=raw_rows,
            folds_rows=folds_rows,
            id_col=str(args.id_col),
            fold=int(args.fold),
            split=str(args.track_split),
            split_label=str(args.track_split_label),
            labels=labels,
            image_dir=_PRINT_CTX_IMAGE_DIR,
        )

    selected_rows: List[Dict[str, str]] = []
    for split in split_sel:
        if split not in split_paths:
            raise SystemExit(f"Unknown split '{split}'. Use train,val,test,all.")
        _, rows = _read_csv_rows(split_paths[split])
        selected_rows.extend(rows)
        if bool(args.by_split):
            _print_stats(split, rows, labels, top_k=int(args.top_k))

    name = "overall" if len(split_sel) > 1 else split_sel[0]
    _print_stats(name, selected_rows, labels, top_k=int(args.top_k))


if __name__ == "__main__":
    main()
