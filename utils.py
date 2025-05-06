import os
import zipfile
import cv2
import numpy as np

import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def get_feature_tensor(path: str, train_fraction=1, evaluation_fraction=0) -> (
pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
pd.DataFrame):
    """
    Get train, evaluation (optional), and test tensor in the form of DataFrames.
    DataFrames are divided in three subgroups: Symbol, Lung, and Pleura features.
    :param path: root path to the data to be split into training, (evaluation, ) and test set
    :param train_fraction: fraction of the whole data found under <path> to be used for the training set.
        Must be a floating point between 0 and 1 (both inclusive).
    :param evaluation_fraction: fraction of the (1-train_fraction)*100 % of the data under <path> to be used for the
    evaluation set. Must be a floating point between 0 and 1 (both inclusive).
    :return: train, evaluation and test tensors for lung, pleura and symbol features each in the following order:
    Training feature Tensor for Lung features
    Training feature Tensor for Pleura features
    Training feature Tensor for Symbol features

    Evaluation feature Tensor for Lung features
    Evaluation feature Tensor for Pleura features
    Evaluation feature Tensor for Symbol features

    Testing feature Tensor for Lung features
    Testing feature Tensor for Pleura features
    Testing feature Tensor for Symbol features
    """
    features = pd.read_excel(path)
    features = features.fillna(-1)
    features["grouping"] = features["medicoID"]
    features.loc[features["grouping"].map(features["grouping"].value_counts()) == 1, "grouping"] = 0

    general_columns = features.loc[:,
                      ["Nachname", "Vorname", "Geburtsdatum", "medicoID", "grouping", "Untersuchungsdatum",
                       "technical_quality", "lateral_exposure"]].columns
    symbol_columns = features.filter(like="symbol_", axis=1).columns
    lung_columns = features.loc[:, "small_rounded_opacities_size_p":"large_opacities_lower_left"].columns
    pleura_columns = features.loc[:, "costophrenic_angle_obliteration_nad":"pleural_calcification_other_left"].columns

    symbol_train = features[general_columns.append(symbol_columns)]
    lung_train = features[general_columns.append(lung_columns)]
    pleura_train = features[general_columns.append(pleura_columns)]
    symbol_eval, lung_eval, pleura_eval = None, None, None
    symbol_test, lung_test, pleura_test = None, None, None
    if train_fraction < 1:
        # set medicoIDs of values being present only once, to 0 so that those patients appear in one group

        symbol_train, symbol_test = train_test_split(symbol_train, test_size=1 - train_fraction,
                                                     stratify=symbol_train["grouping"])
        lung_train, lung_test = train_test_split(lung_train, test_size=1 - train_fraction,
                                                 stratify=lung_train["grouping"])
        pleura_train, pleura_test = train_test_split(pleura_train, test_size=1 - train_fraction,
                                                     stratify=pleura_train["grouping"])

        if evaluation_fraction > 0:
            # split test in half for evaluation and test set:
            symbol_eval, symbol_test = train_test_split(symbol_test, test_size=1 - evaluation_fraction)
            lung_eval, lung_test = train_test_split(lung_test, test_size=1 - evaluation_fraction)
            pleura_eval, pleura_test = train_test_split(pleura_test, test_size=1 - evaluation_fraction)

    return lung_train, pleura_train, symbol_train, lung_eval, pleura_eval, symbol_eval, lung_test, pleura_test, symbol_test


def zip_to_img(root_folder_zip, root_folder_png):
    for zip_folder in os.listdir(root_folder_zip):
        zip_path = os.path.join(root_folder_zip, zip_folder)
        if not zip_folder.endswith(".zip"):
            continue

        with zipfile.ZipFile(zip_path, 'r') as zip_dir:
            dicom = [f for f in zip_dir.namelist() if f.endswith("IM_0001")]

            if not dicom:
                print(f"No dicom image found in {zip_folder}")
                continue

            with zip_dir.open(dicom[0]) as dicom_file:
                dicom_ds = pydicom.dcmread(dicom_file, force=True)
                new_path = os.path.join(root_folder_png, zip_folder.replace(".zip", ".jpg"))

                # normalize to [0, 255]
                pixel_array = dicom_ds.pixel_array
                min_val = np.percentile(pixel_array, 0.5)
                max_val = np.percentile(pixel_array, 99.5)
                normalized_pixel_array = np.clip((pixel_array - min_val) / (max_val - min_val), 0, 1)
                normalized_pixel_array = (normalized_pixel_array * 255).astype(np.uint8)

                cv2.imwrite(new_path, normalized_pixel_array)


zip_to_img("D:\\Projects\\Thorax\\DeboraThorax\\anon\\", "D:\\Projects\\Thorax\\DeboraThorax\\jpg\\")