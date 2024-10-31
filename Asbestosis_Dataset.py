import os
import shutil

import numpy as np
import zipfile
from pathlib import Path

import pandas as pd
import torchio as tio
from PIL import Image
import pydicom


def get_Subjects(path_root, feature_tensor):
    data = []
    for root, _, files in os.walk(path_root):
        for file in files:
            if file.endswith('.zip'):
                file_index = int(file.replace('.zip', ''))
                if file_index in feature_tensor.index:
                    # extract files to read them
                    output_folder = root.replace("anon", "png")
                    os.makedirs(output_folder, exist_ok=True)
                    extract_folder = root + file.split(".zip")[0]
                    os.makedirs(extract_folder, exist_ok=True)
                    with zipfile.ZipFile(root + file, 'r') as xray:
                        xray.extractall(extract_folder)

                        dicom_files = [dicom_file for dicom_file in xray.namelist() if 'IM_' in dicom_file]
                        for dicom_file in dicom_files:
                            with xray.open(dicom_file) as dcm:
                                dicom_data = pydicom.dcmread(dcm)
                                dicom_data = dicom_data.pixel_array.astype(np.float32)

                                # Normalize pixel values to range [0, 255]
                                pixel_array = dicom_data - np.min(dicom_data)
                                pixel_array /= np.max(pixel_array)
                                pixel_array *= 255

                                # Convert to uint8 type
                                pixel_array = pixel_array.astype(np.uint8)

                                dicom_image = Image.fromarray(pixel_array)

                                out_file_path = output_folder + file.split(".zip")[0] + "-" + dicom_file.split(os.sep)[
                                    -1] + ".png"
                                dicom_image.save(out_file_path)

                                label = feature_tensor.loc[file_index]
                                subject = tio.Subject(
                                    image=tio.ScalarImage(out_file_path),
                                    label=label.iloc[6:].to_dict()
                                )
                                data.append(subject)
                                print(f"Saved {out_file_path}")
                    shutil.rmtree(extract_folder)
            elif file.endswith('.png'):
                file_index = int(file.split("-")[0])
                if file_index in feature_tensor.index:
                    image_path = os.path.join(root + file)
                    label = feature_tensor.loc[file_index]

                    subject = tio.Subject(
                        image=tio.ScalarImage(image_path),
                        label=label.iloc[6:].to_dict()
                    )
                    data.append(subject)
    return data


def split_dash_containing_columns(dataframe):
    for column in dataframe.columns:
        if dataframe[column].astype(str).str.contains("/").any():
            dataframe[column + "_first"], dataframe[column + "_second"] = dataframe[column].apply(lambda x: pd.Series(split_at_dash(x)))
            dataframe.drop(column, axis=1, inplace=True)
    return dataframe


def split_at_dash(value):
    if type(value) == str and "/" in value:
        first, second = value.split('/')
        try:
            first_val = int(first)
        except ValueError:
            first_val = np.nan
        try:
            second_val = int(second)
        except ValueError:
            second_val = np.nan
        return first_val, second_val
    else:
        return np.nan, np.nan
