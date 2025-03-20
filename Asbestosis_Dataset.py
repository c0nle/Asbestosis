import os
import shutil
from cProfile import label

import numpy as np
import zipfile
from pathlib import Path

import pandas as pd
import torchio as tio
from PIL import Image
import pydicom

def encode_profusions(row, first_col, second_col):
    first, second = row[first_col], row[second_col]
    return first * 4 + (second - (first - 1))  # Mapping to unique integers

def get_label_encoding(label_df):
    if "small_rounded_opacities_size_p" in label_df.keys():
        label_encoding = pd.DataFrame(columns=["small_rounded_size",        # 0 - p=0,q=0; 1 - p=1,q=0; 2 - p=0,q=1; 3 - p=1,q=1
                                              "small_rounded_profusion",    # 0 - 3 for 0/- til 1/0
                                              "small_rounded_location",     # 0 - 63 for 64 different options of combinations
                                              "small_irregular_size",       # 0 - 6 (s=1, t=0, u=1 never observed)
                                              "small_irregular_profusion",  # 0 - 5 for 0/- til 1/2
                                              "small_irregular_location",   # 0 - 63
                                              "mixed_shape_1",              # 0 - p, 1 - q, 2 - s, 3 - t
                                              "mixed_shape_2",              # 0 - r, 1 - u, 2 - s, 3 - t
                                              "mixed_shape_profusion",      # 0 - 4 for 0/0 til 1/2
                                              "mixed_shape_location",       # 0 - 63
                                              "large_location"              # 0 - 63 (right middle, and left lower never observed)
                                              ])

        label_encoding['small_rounded_size'] = [
            0 if p == 0 and q == 0 and r == 0
            else 1 if p == 1 and q == 0 and r == 0
            else 2 if p == 0 and q == 1 and r == 0
            else 3 if p == 0 and q == 0 and r == 1
            else 4 if p == 1 and q == 1 and r == 0
            else 5 if p == 1 and q == 0 and r == 1
            else 6 if p == 0 and q == 1 and r == 1
            else 7
            for p, q, r in zip(label_df["small_rounded_opacities_size_p"],
                               label_df["small_rounded_opacities_size_q"],
                               label_df["small_rounded_opacities_size_r"])
        ]
        label_encoding['small_rounded_profusion'] = label_df.apply(encode_profusions, axis=1, args=("small_rounded_opacities_profusion_first", "small_rounded_opacities_profusion_second"))
        label_encoding['small_rounded_location'] = [
            # use bit represenation to encode all different options
            (int(ur) << 5) | (int(mr) << 4) | (int(lr) << 3) | (int(ul) << 2) | (int(ml) << 1) | int(ll)
            for ur, mr, lr, ul, ml, ll in zip(
                label_df['small_rounded_opacities_upper_right'],
                label_df['small_rounded_opacities_middle_right'],
                label_df['small_rounded_opacities_lower_right'],
                label_df['small_rounded_opacities_upper_left'],
                label_df['small_rounded_opacities_middle_left'],
                label_df['small_rounded_opacities_lower_left']
            )
        ]

        label_encoding['small_irregular_size'] = [
            0 if s == 0 and t == 0 and u == 0
            else 1 if s == 1 and t == 0 and u == 0
            else 2 if s == 0 and t == 1 and u == 0
            else 3 if s == 0 and t == 0 and u == 1
            else 4 if s == 1 and t == 1 and u == 0
            else 5 if s == 1 and t == 0 and u == 1
            else 6 if s == 0 and t == 1 and u == 1
            else 7
            for s, t, u in zip(label_df["small_irregular_opacities_size_s"],
                               label_df["small_irregular_opacities_size_t"],
                               label_df["small_irregular_opacities_size_u"])
        ]
        label_encoding['small_irregular_profusion'] = label_df.apply(encode_profusions, axis=1, args=(
        "small_irregular_opacities_profusion_first", "small_irregular_opacities_profusion_second"))
        label_encoding['small_irregular_location'] = [
            # use bit represenation to encode all different options
            (int(ur) << 5) | (int(mr) << 4) | (int(lr) << 3) | (int(ul) << 2) | (int(ml) << 1) | int(ll)
            for ur, mr, lr, ul, ml, ll in zip(
                label_df['small_irregular_opacities_upper_right'],
                label_df['small_irregular_opacities_middle_right'],
                label_df['small_irregular_opacities_lower_right'],
                label_df['small_irregular_opacities_upper_left'],
                label_df['small_irregular_opacities_middle_left'],
                label_df['small_irregular_opacities_lower_left']
            )
        ]
        label_encoding['small_irregular_location'] = [
            # use bit represenation to encode all different options
            (int(ur) << 5) | (int(mr) << 4) | (int(lr) << 3) | (int(ul) << 2) | (int(ml) << 1) | int(ll)
            for ur, mr, lr, ul, ml, ll in zip(
                label_df['small_irregular_opacities_upper_right'],
                label_df['small_irregular_opacities_middle_right'],
                label_df['small_irregular_opacities_lower_right'],
                label_df['small_irregular_opacities_upper_left'],
                label_df['small_irregular_opacities_middle_left'],
                label_df['small_irregular_opacities_lower_left']
            )
        ]

        label_encoding['mixed_shape_1'] = [
            0 if shape == "p"
            else 1 if shape == "q"
            else 2 if shape == "s"
            else 3 if shape == "t"
            else -1
            for shape in label_df["mixed_shapes_1"]
        ]
        label_encoding['mixed_shape_2'] = [
            0 if shape == "r"
            else 1 if shape == "u"
            else 2 if shape == "s"
            else 3 if shape == "t"
            else -1
            for shape in label_df["mixed_shapes_2"]
        ]
        label_encoding['mixed_shape_profusion'] = label_df.apply(encode_profusions, axis=1, args=(
            "mixed_shapes_profusion_first", "mixed_shapes_profusion_second"))
        label_encoding['mixed_shape_location'] = [
            # use bit represenation to encode all different options
            (int(ur) << 5) | (int(mr) << 4) | (int(lr) << 3) | (int(ul) << 2) | (int(ml) << 1) | int(ll)
            for ur, mr, lr, ul, ml, ll in zip(
                label_df['mixed_shapes_upper_right'],
                label_df['mixed_shapes_middle_right'],
                label_df['mixed_shapes_lower_right'],
                label_df['mixed_shapes_upper_left'],
                label_df['mixed_shapes_middle_left'],
                label_df['mixed_shapes_lower_left']
            )
        ]

        label_encoding['large_location'] = [
            # use bit represenation to encode all different options
            (int(ur) << 5) | (int(mr) << 4) | (int(lr) << 3) | (int(ul) << 2) | (int(ml) << 1) | int(ll)
            for ur, mr, lr, ul, ml, ll in zip(
                label_df['large_opacities_upper_right'],
                label_df['large_opacities_middle_right'],
                label_df['large_opacities_lower_right'],
                label_df['large_opacities_upper_left'],
                label_df['large_opacities_middle_left'],
                label_df['large_opacities_lower_left']
            )
        ]

    elif "costophrenic_angle_obliteration_nad" in label_df.keys():
        label_encoding = pd.DataFrame(["small_rounded_size",  # 0 - p=0,q=0; 1 - p=1,q=0; 2 - p=0,q=1; 3 - p=1,q=1
                                      "small_rounded_profusion",  # 0 - 3 for 0/- til 1/0
                                      "small_rounded_location",  # 0 - 63 for 64 different options of combinations
                                      "small_irregular_size",  # 0 - 6 (s=1, t=0, u=1 never observed)
                                      "small_irregular_profusion",  # 0 - 5 for 0/- til 1/2
                                      "small_irregular_location",  # 0 - 63
                                      "mixed_shape_1",  # 0 - p, 1 - q, 2 - s, 3 - t
                                      "mixed_shape_2",  # 0 - r, 1 - u, 2 - s, 3 - t
                                      "mixed_shape_profusion",  # 0 - 4 for 0/0 til 1/2
                                      "mixed_shape_location",  # 0 - 63
                                      "large_location",  # 0 - 63 (right middle, and left lower never observed)
                                      ])

    return label_encoding.set_index(label_df.index)


def get_Subjects(path_root, feature_tensor):
    data = []
    label_encoding = get_label_encoding(feature_tensor)
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

                    subject = tio.Subject(
                        image=tio.ScalarImage(image_path),
                        label=label_encoding.loc[file_index]
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
