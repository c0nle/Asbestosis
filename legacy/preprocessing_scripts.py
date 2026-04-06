"""
preprocessing_scripts.py
------------------------
One-time preprocessing scripts for the asbestosis ILO metadata pipeline.

These functions are **not** called during model training.  They were used to
generate the already-existing ``dichotome_data_anonymized_with_patientID.csv``
from the raw multi-sheet clinical export.

Functions
---------
create_dichotome_metadata       : Derive binary/categorical labels from raw ILO fields.
prepare_metadata                : Merge metadata sources and clean NaN-heavy columns.
get_label_encoding              : Encode raw ILO profusion/location/shape fields to integers.
encode_profusions               : Encode a single slash-separated profusion grade.
get_Subjects                    : (Legacy) Build TorchIO Subject list from DICOM zip files.
split_dash_containing_columns   : Split "/" columns into two numeric columns.
split_at_dash                   : Parse a single "/" delimited value into a pair.
shorten_name                    : Shorten a column name to fit Excel sheet limits.
create_individual_metadata_files: Export per-pleura column subsets to Excel.
"""

import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# Label-encoding helpers
# ---------------------------------------------------------------------------

def encode_profusions(row, first_col: str, second_col: str) -> int:
    """
    Encode a slash-separated ILO profusion rating into a unique integer.

    ILO profusion grades are written as "X/Y" (e.g. "1/2"), split into two
    separate integer columns.  This function maps the pair to a single compact
    integer so it can be used as a classification target.

    Args:
        row:        DataFrame row containing the two profusion columns.
        first_col:  Column name for the left-hand side of the grade.
        second_col: Column name for the right-hand side of the grade.

    Returns:
        Integer encoding of the combined profusion grade.
    """
    first, second = row[first_col], row[second_col]
    return first * 4 + (second - (first - 1))


def _location_bits(df: pd.DataFrame, prefix: str) -> list:
    """
    Pack six binary lung-zone flags (UR, MR, LR, UL, ML, LL) into a single
    6-bit integer per row, yielding values in [0, 63].

    The column names are constructed as ``{prefix}{zone}``, e.g.
    ``"small_rounded_opacities_upper_right"``.

    Args:
        df:     DataFrame containing the six zone columns.
        prefix: Column-name prefix, e.g. ``"small_rounded_opacities_"``.

    Returns:
        List of integers, one per row.
    """
    zones = ["upper_right", "middle_right", "lower_right", "upper_left", "middle_left", "lower_left"]
    cols = [f"{prefix}{z}" for z in zones]
    return [
        (int(ur) << 5) | (int(mr) << 4) | (int(lr) << 3) | (int(ul) << 2) | (int(ml) << 1) | int(ll)
        for ur, mr, lr, ul, ml, ll in zip(*(df[c] for c in cols))
    ]


def get_label_encoding(label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode complex ILO radiology labels into compact integer representations.

    Handles two source-data variants:

    - **Full label set** (identified by column ``small_rounded_opacities_size_p``):
      encodes size, profusion, and 6-zone location for rounded/irregular/mixed
      opacities and large opacities.
    - **Pleural-only label set** (identified by ``costophrenic_angle_obliteration_nad``):
      returns a placeholder DataFrame of NaN values — lung opacity labels are absent.

    Zone locations are packed into 6-bit integers via :func:`_location_bits`.
    Profusion grades use :func:`encode_profusions`.

    Args:
        label_df: DataFrame with raw ILO label columns.

    Returns:
        DataFrame with encoded integer columns, indexed identically to ``label_df``.

    Raises:
        ValueError: If ``label_df`` does not match either known variant.
    """
    if "small_rounded_opacities_size_p" in label_df.columns:
        label_encoding = pd.DataFrame(
            columns=[
                "small_rounded_size",       # 0 = all absent … 7 = p+q+r all set
                "small_rounded_profusion",  # encoded via encode_profusions
                "small_rounded_location",   # 0–63 (6-bit zone bitmask)
                "small_irregular_size",     # 0–7
                "small_irregular_profusion",
                "small_irregular_location", # 0–63
                "mixed_shape_1",            # 0=p, 1=q, 2=s, 3=t, -1=unknown
                "mixed_shape_2",            # 0=r, 1=u, 2=s, 3=t, -1=unknown
                "mixed_shape_profusion",
                "mixed_shape_location",     # 0–63
                "large_location",           # 0–63
            ]
        )

        # --- Small rounded opacities ---
        label_encoding["small_rounded_size"] = [
            0 if p == 0 and q == 0 and r == 0
            else 1 if p == 1 and q == 0 and r == 0
            else 2 if p == 0 and q == 1 and r == 0
            else 3 if p == 0 and q == 0 and r == 1
            else 4 if p == 1 and q == 1 and r == 0
            else 5 if p == 1 and q == 0 and r == 1
            else 6 if p == 0 and q == 1 and r == 1
            else 7
            for p, q, r in zip(
                label_df["small_rounded_opacities_size_p"],
                label_df["small_rounded_opacities_size_q"],
                label_df["small_rounded_opacities_size_r"],
            )
        ]
        label_encoding["small_rounded_profusion"] = label_df.apply(
            encode_profusions, axis=1,
            args=("small_rounded_opacities_profusion_first", "small_rounded_opacities_profusion_second"),
        )
        label_encoding["small_rounded_location"] = _location_bits(label_df, "small_rounded_opacities_")

        # --- Small irregular opacities ---
        label_encoding["small_irregular_size"] = [
            0 if s == 0 and t == 0 and u == 0
            else 1 if s == 1 and t == 0 and u == 0
            else 2 if s == 0 and t == 1 and u == 0
            else 3 if s == 0 and t == 0 and u == 1
            else 4 if s == 1 and t == 1 and u == 0
            else 5 if s == 1 and t == 0 and u == 1
            else 6 if s == 0 and t == 1 and u == 1
            else 7
            for s, t, u in zip(
                label_df["small_irregular_opacities_size_s"],
                label_df["small_irregular_opacities_size_t"],
                label_df["small_irregular_opacities_size_u"],
            )
        ]
        label_encoding["small_irregular_profusion"] = label_df.apply(
            encode_profusions, axis=1,
            args=("small_irregular_opacities_profusion_first", "small_irregular_opacities_profusion_second"),
        )
        label_encoding["small_irregular_location"] = _location_bits(label_df, "small_irregular_opacities_")

        # --- Mixed shapes ---
        label_encoding["mixed_shape_1"] = [
            0 if shape == "p" else 1 if shape == "q" else 2 if shape == "s" else 3 if shape == "t" else -1
            for shape in label_df["mixed_shapes_1"]
        ]
        label_encoding["mixed_shape_2"] = [
            0 if shape == "r" else 1 if shape == "u" else 2 if shape == "s" else 3 if shape == "t" else -1
            for shape in label_df["mixed_shapes_2"]
        ]
        label_encoding["mixed_shape_profusion"] = label_df.apply(
            encode_profusions, axis=1,
            args=("mixed_shapes_profusion_first", "mixed_shapes_profusion_second"),
        )
        label_encoding["mixed_shape_location"] = _location_bits(label_df, "mixed_shapes_")

        # --- Large opacities ---
        label_encoding["large_location"] = _location_bits(label_df, "large_opacities_")

        return label_encoding.set_index(label_df.index)

    elif "costophrenic_angle_obliteration_nad" in label_df.columns:
        cols = [
            "small_rounded_size", "small_rounded_profusion", "small_rounded_location",
            "small_irregular_size", "small_irregular_profusion", "small_irregular_location",
            "mixed_shape_1", "mixed_shape_2", "mixed_shape_profusion", "mixed_shape_location",
            "large_location",
        ]
        return pd.DataFrame(np.nan, index=label_df.index, columns=cols)

    else:
        raise ValueError(
            "label_df does not contain the expected ILO label columns. "
            "Expected either 'small_rounded_opacities_size_p' (full set) or "
            "'costophrenic_angle_obliteration_nad' (pleural-only set)."
        )


# ---------------------------------------------------------------------------
# Legacy subject builder
# ---------------------------------------------------------------------------

def get_Subjects(path_root: str, feature_tensor: pd.DataFrame) -> list:
    """
    Build a list of TorchIO Subject objects from DICOM zip files.

    .. note::
        This function is **not** called by the current training pipeline.
        It is retained for backward compatibility only.
        Requires ``torchio`` (``pip install torchio``).

    Args:
        path_root:      Root directory containing zip files (DICOM format).
        feature_tensor: DataFrame with a ``fileID`` column and label columns.

    Returns:
        List of ``tio.Subject`` objects.
    """
    import torchio as tio  # local import – torchio may not be installed in all envs

    data = []
    if "fileID" in feature_tensor.columns:
        feature_by_file = feature_tensor.set_index("fileID", drop=False)
    else:
        feature_by_file = feature_tensor
    label_encoding = get_label_encoding(feature_by_file)

    for root, _, files in os.walk(path_root):
        for file in files:
            if file.endswith(".zip"):
                file_index = int(file.replace(".zip", ""))
                if file_index not in feature_by_file.index:
                    continue

                output_folder = root.replace("anon", "png")
                os.makedirs(output_folder, exist_ok=True)
                extract_folder = os.path.join(root, file.split(".zip")[0])
                os.makedirs(extract_folder, exist_ok=True)

                zip_path = os.path.join(root, file)
                with zipfile.ZipFile(zip_path, "r") as xray:
                    xray.extractall(extract_folder)
                    dicom_files = [f for f in xray.namelist() if "IM_" in f]
                    for dicom_file in dicom_files:
                        with xray.open(dicom_file) as dcm:
                            dicom_data = pydicom.dcmread(dcm)
                            pixel_array = dicom_data.pixel_array.astype(np.float32)
                            pixel_array -= np.min(pixel_array)
                            pixel_array /= np.max(pixel_array) if np.max(pixel_array) != 0 else 1.0
                            pixel_array = (pixel_array * 255).astype(np.uint8)

                        out_file_path = os.path.join(
                            output_folder,
                            f"{file.split('.zip')[0]}-{dicom_file.split(os.sep)[-1]}.png",
                        )
                        Image.fromarray(pixel_array).save(out_file_path)
                        data.append(
                            tio.Subject(
                                image=tio.ScalarImage(out_file_path),
                                label=label_encoding.loc[file_index].to_dict(),
                            )
                        )
                        print(f"Saved {out_file_path}")

                shutil.rmtree(extract_folder)

            elif file.endswith(".png"):
                file_index = int(file.split("-")[0])
                if file_index not in feature_by_file.index:
                    continue
                image_path = os.path.join(root, file)
                data.append(
                    tio.Subject(
                        image=tio.ScalarImage(image_path),
                        label=label_encoding.loc[file_index],
                    )
                )
    return data


# ---------------------------------------------------------------------------
# Slash-value splitting utilities
# ---------------------------------------------------------------------------

def split_dash_containing_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Split every column that contains "/" values into two numeric columns
    (``<col>_first`` and ``<col>_second``) and remove the original.

    Args:
        dataframe: Input DataFrame (modified in place).

    Returns:
        The modified DataFrame.
    """
    for column in dataframe.columns:
        if dataframe[column].astype(str).str.contains("/").any():
            dataframe[column + "_first"], dataframe[column + "_second"] = dataframe[column].apply(
                lambda x: pd.Series(split_at_dash(x))
            )
            dataframe.drop(column, axis=1, inplace=True)
    return dataframe


def split_at_dash(value) -> tuple:
    """
    Split a "/" delimited string (e.g. ``"1/2"``) into a pair of integers.

    Args:
        value: String value containing exactly one "/", or any other type.

    Returns:
        ``(first_int, second_int)`` on success, or ``(nan, nan)`` otherwise.
    """
    if isinstance(value, str) and "/" in value:
        first, second = value.split("/")
        try:
            first_val = int(first)
        except ValueError:
            first_val = np.nan
        try:
            second_val = int(second)
        except ValueError:
            second_val = np.nan
        return first_val, second_val
    return np.nan, np.nan


# ---------------------------------------------------------------------------
# Per-sheet Excel export
# ---------------------------------------------------------------------------

def shorten_name(sheet_name: str) -> str:
    """Shorten a column name to fit within Excel's 31-character sheet-name limit."""
    return (
        sheet_name
        .replace("thickening",    "thick")
        .replace("right",         "r")
        .replace("left",          "l")
        .replace("localized",     "loc")
        .replace("calcification", "calcif")
    )


def create_individual_metadata_files(overall_metadata_file: str) -> None:
    """
    Export per-pleura-column subsets of the metadata to a multi-sheet Excel file.

    For each column whose name contains "pleura", a sheet is created containing
    only the rows that have a non-``-1`` value.  A summary sheet lists sample
    counts per sheet.

    Args:
        overall_metadata_file: Path to the prepared metadata CSV.
    """
    metadata = pd.read_csv(overall_metadata_file)
    metadata["Geburtsdatum"]      = pd.to_datetime(metadata["Geburtsdatum"],      format="%Y-%m-%d")
    metadata["Untersuchungsdatum"] = pd.to_datetime(metadata["Untersuchungsdatum"], format="%Y-%m-%d")

    sub_metadatas    = {}
    metadata_lengths = []
    new_filename = overall_metadata_file.replace(
        overall_metadata_file.split(os.sep)[-1], "metadata_subset.xlsx"
    )

    for column_name in metadata.columns:
        if "pleura" not in column_name:
            continue
        column_metadata = metadata[metadata[column_name] != -1]
        sheet_name = shorten_name(column_name)
        sub_metadatas[sheet_name] = column_metadata

        if column_name in (
            "localized_pleural_thickening_width_left",
            "localized_pleural_thickening_width_right",
        ):
            c_value = "c" if column_name.endswith("_right") else "C"
            metadata_lengths.append({
                "Sheet": sheet_name, "Column name": column_name,
                "Number of entries": len(column_metadata),
                "Positive samples": 0, "Negative samples": 0,
                "a/1": len(column_metadata[column_metadata[column_name] == "a"]),
                "b/2": len(column_metadata[column_metadata[column_name] == "b"]),
                "c/3": len(column_metadata[column_metadata[column_name] == c_value]),
            })
        elif column_name in (
            "localized_pleural_thickening_extent_left",
            "localized_pleural_thickening_extent_right",
        ):
            metadata_lengths.append({
                "Sheet": sheet_name, "Column name": column_name,
                "Number of entries": len(column_metadata),
                "Positive samples": 0, "Negative samples": 0,
                "a/1": len(column_metadata[column_metadata[column_name] == 1]),
                "b/2": len(column_metadata[column_metadata[column_name] == 2]),
                "c/3": len(column_metadata[column_metadata[column_name] == 3]),
            })
        elif column_name in (
            "diffuse_pleural_thickening_width_right",
            "diffuse_pleural_thickening_width_left",
        ):
            metadata_lengths.append({
                "Sheet": sheet_name, "Column name": column_name,
                "Number of entries": len(column_metadata),
                "Positive samples": 0, "Negative samples": 0,
                "a/1": len(column_metadata[column_metadata[column_name] == "a"]),
                "b/2": len(column_metadata[column_metadata[column_name] == "b"]),
                "c/3": "-",
            })
        else:
            metadata_lengths.append({
                "Sheet": sheet_name, "Column name": column_name,
                "Number of entries": len(column_metadata),
                "Positive samples": len(column_metadata[column_metadata[column_name] == 1]),
                "Negative samples": len(column_metadata[column_metadata[column_name] == 0]),
                "a/1": 0, "b/2": 0, "c/3": 0,
            })

    with pd.ExcelWriter(new_filename) as writer:
        for sheet_name, dataframe in sub_metadatas.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
        pd.DataFrame(metadata_lengths).to_excel(
            writer, sheet_name="Number of entries per sheet", index=False
        )


# ---------------------------------------------------------------------------
# Metadata preparation
# ---------------------------------------------------------------------------

def prepare_metadata(
    metadata_file: str,
    anforderungsnr_file: str,
    mapping_file: str,
    maximum_occurance_of_nans_per_col: int,
    dichotome_metadata: bool = False,
) -> pd.DataFrame:
    """
    Merge metadata sources, fill missing values, and drop high-NaN columns.

    Steps:
    1. Read and inner-join ``metadata_file`` with ``anforderungsnr_file`` on
       name + date fields to obtain ``Anforderungsnummer``.
    2. Inner-join with ``mapping_file`` to obtain DICOM file IDs.
    3. Replace ``NaN`` with ``-1``.
    4. (Non-dichotome only) Map profusion slash-notation to numeric scalars
       and drop the original slash columns.
    5. Drop columns with more than ``maximum_occurance_of_nans_per_col`` NaN
       values.
    6. Save the result next to ``metadata_file`` (``*_prepared.csv``).

    Args:
        metadata_file:                   Path to the primary metadata CSV.
        anforderungsnr_file:             Path to the examination-number CSV.
        mapping_file:                    Path to the medicoID → fileID mapping CSV.
        maximum_occurance_of_nans_per_col: Drop columns that exceed this many NaN
                                           entries.
        dichotome_metadata:              Skip the profusion-value mapping step when
                                         working with already-binarised labels.

    Returns:
        Prepared metadata DataFrame.
    """
    metadata = pd.read_csv(metadata_file)
    metadata["Geburtsdatum"]      = pd.to_datetime(metadata["Geburtsdatum"],      format="%d.%m.%Y")
    metadata["Untersuchungsdatum"] = pd.to_datetime(metadata["Untersuchungsdatum"], format="%d.%m.%Y")

    anford_nr = pd.read_csv(
        anforderungsnr_file,
        usecols=["Name", "Vorname", "Geburtsdatum", "Anforderungsnummer", "Untersuchungsdatum"],
    )
    anford_nr.rename(columns={"Name": "Nachname"}, inplace=True)
    anford_nr["Geburtsdatum"]      = pd.to_datetime(anford_nr["Geburtsdatum"],      format="%m/%d/%Y")
    anford_nr["Untersuchungsdatum"] = pd.to_datetime(anford_nr["Untersuchungsdatum"], format="%m/%d/%Y")

    metadata = metadata.merge(
        anford_nr, "inner", ["Nachname", "Vorname", "Geburtsdatum", "Untersuchungsdatum"]
    )
    mapping = pd.read_csv(mapping_file)
    mapping["medicoID"] = mapping["medicoID"].astype(str).apply(lambda x: x[:-4])
    metadata["Anforderungsnummer_y"] = metadata["Anforderungsnummer_y"].astype(str)
    metadata = metadata.merge(mapping, "inner", left_on="Anforderungsnummer_y", right_on="medicoID")
    metadata.rename(
        columns={"Anforderungsnummer_y": "Anforderungsnummer", "medicoID_x": "medicoID"}, inplace=True
    )

    metadata.fillna(-1, inplace=True)

    col_to_drop = []
    if not dichotome_metadata:
        slash_value_mapping = {
            "0/-": -0.25, "1/-": 0.25,  "2/-": 1.25,  "3/-": 2.25,
            "0/0": 0,     "0/1": 0.5,
            "1/0": 0.75,  "1/1": 1,     "1/2": 1.5,
            "2/1": 1.75,  "2/2": 2,     "2/3": 2.5,
            "3/2": 2.75,  "3/3": 3,     "3/4": 3.5,
            "4/3": 3.75,  "4/4": 4,
        }
        metadata["small_rounded_opacities_profusion_map"] = metadata[
            "small_rounded_opacities_profusion"].map(slash_value_mapping)
        metadata["small_irregular_opacities_profusion_map"] = metadata[
            "small_irregular_opacities_profusion"].map(slash_value_mapping)
        metadata["mixed_shapes_profusion_map"] = metadata["mixed_shapes_profusion"].map(slash_value_mapping)
        col_to_drop = [
            "medicoID_y", "mixed_shapes_profusion",
            "small_irregular_opacities_profusion", "small_rounded_opacities_profusion",
        ]

    for col_name in metadata.columns:
        nan_count = len(metadata[metadata[col_name] != metadata[col_name]])
        if nan_count > maximum_occurance_of_nans_per_col:
            col_to_drop.append(col_name)

    metadata.drop(columns=col_to_drop, inplace=True)
    nan_pct = int(maximum_occurance_of_nans_per_col / len(metadata) * 100)
    print(f"Dropped columns with ≥{nan_pct}% NaN values: {col_to_drop}")

    out_path = metadata_file.replace(".csv", "_prepared.csv")
    metadata.to_csv(out_path, index=False)
    return metadata


# ---------------------------------------------------------------------------
# Binary (dichotome) label derivation
# ---------------------------------------------------------------------------

def create_dichotome_metadata(original_metadata: pd.DataFrame, output_path: str) -> None:
    """
    Derive simplified binary / categorical labels from detailed ILO fields and
    save the result to ``output_path``.

    The function maps fine-grained ILO classification fields (6-zone location,
    profusion grades, size categories) to a reduced set of binary/categorical
    labels suitable for machine-learning classification:

    - **Small rounded opacities**: location per side (``small_rounded_right/left``),
      size class (``small_rounded_size``).
    - **Small irregular opacities**: location per side, size class.
    - **Mixed shapes**: binary present/absent (``mixed_shapes``).
    - **Large opacities**: simplified location category (``large_opacities``).
    - **Diffuse pleural thickening**: width class, extent class, location
      (``diffuse_pleural_location``).
    - **Localized pleural thickening**: width class, extent class, location
      (``local_pleural_location``).
    - **Pleural calcification**: location (``pleural_calcification_location``),
      side (``pleural_calcification_side``).
    - **Occupational disease**: single binary indicator collapsed from nine
      sub-type columns.

    All raw source columns are dropped after derivation.

    Args:
        original_metadata: Full ILO metadata DataFrame.
        output_path:       File path for the output CSV.
    """
    dichotome_metadata = original_metadata.copy()

    # --- 1. Drop unused columns ---
    unused_columns = ["technical_quality_t", "lateral_exposure"] + [
        col for col in dichotome_metadata.columns if col.startswith("symbol_")
    ]
    dichotome_metadata.drop(columns=unused_columns, errors="ignore", inplace=True)

    # --- 2. Technical quality: grades 1 or 2 → acceptable (1), else 0 ---
    dichotome_metadata["technical_quality"] = (
        dichotome_metadata["technical_quality"].isin([1, 2]).astype(int)
    )

    # --- 3. Validity masks: rows with any positive profusion rating ---
    valid_rounded = (
        dichotome_metadata["small_rounded_opacities_profusion"].notna()
        & ~dichotome_metadata["small_rounded_opacities_profusion"].isin(["0/-", "0/0", "0/1"])
    )
    valid_irregular = (
        dichotome_metadata["small_irregular_opacities_profusion"].notna()
        & ~dichotome_metadata["small_irregular_opacities_profusion"].isin(["0/-", "0/0", "0/1"])
    )

    # ### LUNG COLUMNS

    # --- 4. Small rounded opacities: location (lower vs. middle_upper) + size class ---
    for side in ["right", "left"]:
        side_cols = [
            f"small_rounded_opacities_upper_{side}",
            f"small_rounded_opacities_middle_{side}",
            f"small_rounded_opacities_lower_{side}",
        ]
        any_present = dichotome_metadata[side_cols].eq(1).any(axis=1)
        valid_side  = any_present & valid_rounded
        dichotome_metadata.loc[valid_side, f"small_rounded_{side}"] = np.where(
            dichotome_metadata.loc[valid_side, f"small_rounded_opacities_lower_{side}"] == 1,
            "lower", "middle_upper",
        )

    size_mask = (
        dichotome_metadata["small_rounded_right"].notna()
        | dichotome_metadata["small_rounded_left"].notna()
    )
    dichotome_metadata.loc[size_mask, "small_rounded_size"] = np.where(
        dichotome_metadata.loc[size_mask, "small_rounded_opacities_size_p"] == 1,
        "<= 1.5mm", "1.5-10mm",
    )

    # --- 5. Small irregular opacities: location + size class ---
    for side in ["right", "left"]:
        side_cols = [
            f"small_irregular_opacities_upper_{side}",
            f"small_irregular_opacities_middle_{side}",
            f"small_irregular_opacities_lower_{side}",
        ]
        any_present = dichotome_metadata[side_cols].eq(1).any(axis=1)
        valid_side  = any_present & valid_irregular
        dichotome_metadata.loc[valid_side, f"small_irregular_{side}"] = np.where(
            dichotome_metadata.loc[valid_side, f"small_irregular_opacities_lower_{side}"] == 1,
            "lower", "middle_upper",
        )

    size_mask = (
        dichotome_metadata["small_irregular_right"].notna()
        | dichotome_metadata["small_irregular_left"].notna()
    )
    dichotome_metadata.loc[size_mask, "small_irregular_size"] = np.where(
        dichotome_metadata.loc[size_mask, "small_irregular_opacities_size_u"] == 1,
        "3-10mm", "<= 3mm",
    )

    # --- 6. Mixed shapes: present if either shape field is not NaN ---
    dichotome_metadata["mixed_shapes"] = (
        dichotome_metadata[["mixed_shapes_1", "mixed_shapes_2"]].notna().any(axis=1).astype(int)
    )

    # --- 7. Large opacities location: lower vs. middle_upper ---
    loc_mask = dichotome_metadata["large_opacities"].notna()
    dichotome_metadata.loc[loc_mask, "large_opacities"] = np.where(
        (dichotome_metadata.loc[loc_mask, "large_opacities"] == "A")
        & dichotome_metadata.loc[
            loc_mask, ["large_opacities_lower_right", "large_opacities_lower_left"]
        ].eq(1).any(axis=1),
        "lower", "middle_upper",
    )

    # ### PLEURA COLUMNS

    # --- 8. Diffuse pleural thickening: width class + extent class + location ---
    diff_mask = (
        dichotome_metadata["diffuse_pleural_thickening_width_right"].notna()
        | dichotome_metadata["diffuse_pleural_thickening_width_left"].notna()
    )
    dichotome_metadata.loc[diff_mask, "diffuse_pleural_thickening_width"] = np.where(
        (dichotome_metadata.loc[diff_mask, "diffuse_pleural_thickening_width_right"] == "c")
        | (dichotome_metadata.loc[diff_mask, "diffuse_pleural_thickening_width_left"] == "c"),
        "> 10mm", "<= 10mm",
    )

    extend_mask = (
        dichotome_metadata["diffuse_pleural_thickening_extent_right"].notna()
        | dichotome_metadata["diffuse_pleural_thickening_extent_left"].notna()
    )
    dichotome_metadata.loc[extend_mask, "diffuse_pleural_thickening_extend"] = np.where(
        (dichotome_metadata.loc[extend_mask, "diffuse_pleural_thickening_extent_right"] == 3)
        | (dichotome_metadata.loc[extend_mask, "diffuse_pleural_thickening_extent_left"] == 3),
        "> 1/2", "<= 1/2",
    )

    dichotome_metadata.loc[diff_mask | extend_mask, "diffuse_pleural_location"] = np.where(
        (dichotome_metadata.loc[diff_mask | extend_mask, "diffuse_pleural_thickening_lower_right"] == 1)
        | (dichotome_metadata.loc[diff_mask | extend_mask, "diffuse_pleural_thickening_lower_left"] == 1),
        "lower", "middle_upper",
    )

    # --- 9. Localized pleural thickening: width class + extent class + location ---
    local_mask = (
        dichotome_metadata["localized_pleural_thickening_width_right"].notna()
        | dichotome_metadata["localized_pleural_thickening_width_left"].notna()
    )
    dichotome_metadata.loc[local_mask, "localized_pleural_thickening_width"] = np.where(
        (dichotome_metadata.loc[local_mask, "localized_pleural_thickening_width_right"] == "c")
        | (dichotome_metadata.loc[local_mask, "localized_pleural_thickening_width_left"] == "c"),
        "> 10mm", "<= 10mm",
    )

    local_extend_mask = (
        dichotome_metadata["localized_pleural_thickening_extent_right"].notna()
        | dichotome_metadata["localized_pleural_thickening_extent_left"].notna()
    )
    dichotome_metadata.loc[local_extend_mask, "localized_pleural_thickening_extend"] = np.where(
        (dichotome_metadata.loc[local_extend_mask, "localized_pleural_thickening_extent_right"] == 3)
        | (dichotome_metadata.loc[local_extend_mask, "localized_pleural_thickening_extent_left"] == 3),
        "> 1/2", "<= 1/2",
    )

    loc_mask = (
        dichotome_metadata["localized_pleural_thickening_diaphragm_right"].notna()
        | dichotome_metadata["localized_pleural_thickening_diaphragm_left"].notna()
        | dichotome_metadata["localized_pleural_thickening_chest_wall_right"].notna()
        | dichotome_metadata["localized_pleural_thickening_chest_wall_left"].notna()
    )
    dichotome_metadata.loc[loc_mask, "local_pleural_location"] = np.where(
        (dichotome_metadata.loc[loc_mask, "localized_pleural_thickening_diaphragm_right"] == 1)
        | (dichotome_metadata.loc[loc_mask, "localized_pleural_thickening_diaphragm_left"] == 1),
        "diaphragm", "chest_wall",
    )

    # --- 10. Pleural calcification: location + side ---
    pleural_mask = (
        dichotome_metadata["pleural_calcification_diaphragm_right"].notna()
        | dichotome_metadata["pleural_calcification_diaphragm_left"].notna()
        | dichotome_metadata["pleural_calcification_chest_wall_right"].notna()
        | dichotome_metadata["pleural_calcification_chest_wall_left"].notna()
    )
    dichotome_metadata.loc[pleural_mask, "pleural_calcification_location"] = np.where(
        (dichotome_metadata.loc[pleural_mask, "pleural_calcification_diaphragm_right"] == 1)
        | (dichotome_metadata.loc[pleural_mask, "pleural_calcification_diaphragm_left"] == 1),
        "diaphragm", "chest_wall",
    )
    dichotome_metadata.loc[pleural_mask, "pleural_calcification_side"] = np.select(
        [
            (dichotome_metadata.loc[pleural_mask, "pleural_calcification_diaphragm_right"] == 1)
            | (dichotome_metadata.loc[pleural_mask, "pleural_calcification_chest_wall_right"] == 1),
            (dichotome_metadata.loc[pleural_mask, "pleural_calcification_diaphragm_left"] == 1)
            | (dichotome_metadata.loc[pleural_mask, "pleural_calcification_chest_wall_left"] == 1),
        ],
        ["right", "left"],
        default="",
    )

    # --- 11. Occupational disease: single binary indicator from nine sub-types ---
    occ_disease_cols = [
        "occupational_disease_silicosis",
        "occupational_disease_silicotuberculosis",
        "occupational_disease_quartz_dust_lung_cancer",
        "occupational_disease_asbestosis",
        "occupational_disease_asbestos_pleura",
        "occupational_disease_asbestos_lung_cancer",
        "occupational_disease_asbestos_larynx_cancer",
        "occupational_disease_asbestos_mesothelioma",
        "occupational_disease_ionized_radiation",
    ]
    occupational_mask = dichotome_metadata[occ_disease_cols].notna().any(axis=1)
    dichotome_metadata.loc[occupational_mask, "occupational_disease"] = (
        dichotome_metadata.loc[occupational_mask, occ_disease_cols].eq(1).any(axis=1).astype(int)
    )

    # --- 12. Drop all raw source columns ---
    _cols_to_drop = [
        "small_rounded_opacities_profusion",
        "small_rounded_opacities_size_p", "small_rounded_opacities_size_q", "small_rounded_opacities_size_r",
        "small_rounded_opacities_upper_right", "small_rounded_opacities_middle_right", "small_rounded_opacities_lower_right",
        "small_rounded_opacities_upper_left",  "small_rounded_opacities_middle_left",  "small_rounded_opacities_lower_left",
        "small_irregular_opacities_size_s", "small_irregular_opacities_size_t", "small_irregular_opacities_size_u",
        "small_irregular_opacities_profusion",
        "small_irregular_opacities_upper_right", "small_irregular_opacities_middle_right", "small_irregular_opacities_lower_right",
        "small_irregular_opacities_upper_left",  "small_irregular_opacities_middle_left",  "small_irregular_opacities_lower_left",
        "mixed_shapes_1", "mixed_shapes_2", "mixed_shapes_profusion",
        "mixed_shapes_upper_right", "mixed_shapes_middle_right", "mixed_shapes_lower_right",
        "mixed_shapes_upper_left",  "mixed_shapes_middle_left",  "mixed_shapes_lower_left",
        "large_opacities_nad",
        "large_opacities_upper_right", "large_opacities_middle_right", "large_opacities_lower_right",
        "large_opacities_upper_left",  "large_opacities_middle_left",  "large_opacities_lower_left",
        "costophrenic_angle_obliteration_nad",
        "costophrenic_angle_obliteration_right", "costophrenic_angle_obliteration_left",
        "diffuse_pleural_thickening_nad",
        "diffuse_pleural_thickening_width_right",  "diffuse_pleural_thickening_width_left",
        "diffuse_pleural_thickening_extent_right", "diffuse_pleural_thickening_extent_left",
        "diffuse_pleural_thickening_small_right",  "diffuse_pleural_thickening_face_on_right",
        "diffuse_pleural_thickening_small_left",   "diffuse_pleural_thickening_face_on_left",
        "diffuse_pleural_thickening_upper_right", "diffuse_pleural_thickening_middle_right", "diffuse_pleural_thickening_lower_right",
        "diffuse_pleural_thickening_upper_left",  "diffuse_pleural_thickening_middle_left",  "diffuse_pleural_thickening_lower_left",
        "localized_pleural_thickening_nad",
        "localized_pleural_thickening_width_right",  "localized_pleural_thickening_width_left",
        "localized_pleural_thickening_extent_right", "localized_pleural_thickening_extent_left",
        "localized_pleural_thickening_small_right",  "localized_pleural_thickening_small_left",
        "localized_pleural_thickening_face_on_right", "localized_pleural_thickening_face_on_left",
        "localized_pleural_thickening_diaphragm_right", "localized_pleural_thickening_diaphragm_left",
        "localized_pleural_thickening_chest_wall_right", "localized_pleural_thickening_chest_wall_left",
        "pleural_calcification_nad",
        "pleural_calcification_diaphragm_right", "pleural_calcification_diaphragm_left",
        "pleural_calcification_chest_wall_right", "pleural_calcification_chest_wall_left",
        "pleural_calcification_other_right",     "pleural_calcification_other_left",
        *occ_disease_cols,
        "occupational_disease_nad",
        "comments",
    ]
    dichotome_metadata.drop(columns=_cols_to_drop, errors="ignore", inplace=True)
    dichotome_metadata.to_csv(output_path, index=False)


if __name__ == "__main__":
    overall_filename = "D:\\Projects\\Thorax\\merged_data_original.csv"
    metadata = pd.read_csv(overall_filename)
    create_dichotome_metadata(metadata, "D:\\Projects\\Thorax\\dichotome_data.csv")
