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
from sklearn.model_selection import StratifiedGroupKFold


def encode_profusions(row, first_col, second_col):
    first, second = row[first_col], row[second_col]
    return first * 4 + (second - (first - 1))  # Mapping to unique integers


def get_label_encoding(label_df):
    if "small_rounded_opacities_size_p" in label_df.keys():
        label_encoding = pd.DataFrame(
            columns=["small_rounded_size",  # 0 - p=0,q=0; 1 - p=1,q=0; 2 - p=0,q=1; 3 - p=1,q=1
                     "small_rounded_profusion",  # 0 - 3 for 0/- til 1/0
                     "small_rounded_location",  # 0 - 63 for 64 different options of combinations
                     "small_irregular_size",  # 0 - 6 (s=1, t=0, u=1 never observed)
                     "small_irregular_profusion",  # 0 - 5 for 0/- til 1/2
                     "small_irregular_location",  # 0 - 63
                     "mixed_shape_1",  # 0 - p, 1 - q, 2 - s, 3 - t
                     "mixed_shape_2",  # 0 - r, 1 - u, 2 - s, 3 - t
                     "mixed_shape_profusion",  # 0 - 4 for 0/0 til 1/2
                     "mixed_shape_location",  # 0 - 63
                     "large_location"  # 0 - 63 (right middle, and left lower never observed)
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
        label_encoding['small_rounded_profusion'] = label_df.apply(encode_profusions, axis=1, args=(
        "small_rounded_opacities_profusion_first", "small_rounded_opacities_profusion_second"))
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
            dataframe[column + "_first"], dataframe[column + "_second"] = dataframe[column].apply(
                lambda x: pd.Series(split_at_dash(x)))
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


def create_splits(metadata: pd.DataFrame, n_testfolds: int, output_folder: str, output_filename: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for fold in range(n_testfolds):
        metadata[f'Fold{fold}'] = ''
    strat = StratifiedGroupKFold(n_splits=n_testfolds, shuffle=True, random_state=0)
    for n_fold, test_fold in enumerate(
            strat.split(metadata, y=metadata['technical_quality'], groups=metadata['medicoID'])):
        train_split = metadata.loc[test_fold[0]]
        test_split = metadata.loc[test_fold[1]]

        test_split.to_csv(output_folder + "stratified_test_set-f{}.csv".format(n_fold), index=False)
        train_split.to_csv(output_folder + "stratified_train_set-f{}.csv".format(n_fold), index=False)

        metadata.loc[metadata['medicoID'].isin(test_split['medicoID']), f'Fold{n_fold}'] = 'test'
        metadata.loc[metadata['medicoID'].isin(train_split['medicoID']), f'Fold{n_fold}'] = 'train'
    metadata.to_csv(output_filename)
    return metadata


def get_column_name_groups(metadata: pd.DataFrame):
    general_columns = ['Nachname', 'Vorname', 'Geburtsdatum', 'medicoID', 'Untersuchungsdatum',
                       'Untersuchung_Ort', 'Untersuchung_Art', 'id', 'Untersuchung_id',
                       'technical_quality', "medicoID_y", "fileID"]
    symbol_columns = [col for col in metadata.columns if col.startswith('symbol_')]
    symbol_columns.extend(general_columns)

    rounded_columns = [col for col in metadata.columns if col.startswith('small_rounded_')]
    rounded_columns.extend(general_columns)

    irregular_columns = [col for col in metadata.columns if col.startswith('small_irregular_')]
    irregular_columns.extend(general_columns)

    mixed_columns = [col for col in metadata.columns if col.startswith('mixed_')]
    mixed_columns.extend(general_columns)

    large_columns = [col for col in metadata.columns if col.startswith('large_')]
    large_columns.extend(general_columns)
    large_columns.extend(["costophrenic_angle_obliteration_nad", "costophrenic_angle_obliteration_right",
                          "costophrenic_angle_obliteration_left"])

    pleural_columns = [col for col in metadata.columns if 'pleural' in col]
    pleural_columns.extend(general_columns)

    occupation_columns = [col for col in metadata.columns if col.startswith('occupational_disease_')]
    occupation_columns.extend(general_columns)

    return {"general": general_columns,
            "symbol": symbol_columns,
            "rounded": rounded_columns,
            "irregular": irregular_columns,
            "mixed": mixed_columns,
            "large": large_columns,
            "pleural": pleural_columns,
            "occupational": occupation_columns}


def prepare_metadata(metadata_file: str, anforderungsnr_file: str, mapping_file: str,
                     maximum_occurance_of_nans_per_col: int):
    # Step1: Read files and merge to get correct AnforderungsNummern
    metadata = pd.read_csv(metadata_file)
    metadata['Geburtsdatum'] = pd.to_datetime(metadata['Geburtsdatum'], format='%d.%m.%Y')
    metadata['Untersuchungsdatum'] = pd.to_datetime(metadata['Untersuchungsdatum'], format='%d.%m.%Y')

    anford_nr = pd.read_csv(anforderungsnr_file,
                            usecols=["Name", "Vorname", "Geburtsdatum", "Anforderungsnummer", "Untersuchungsdatum"])
    anford_nr.rename(columns={"Name": "Nachname"}, inplace=True)
    anford_nr['Geburtsdatum'] = pd.to_datetime(anford_nr['Geburtsdatum'], format='%m/%d/%Y')  # '%Y-%m-%d'
    anford_nr['Untersuchungsdatum'] = pd.to_datetime(anford_nr['Untersuchungsdatum'], format='%m/%d/%Y')

    metadata = metadata.merge(anford_nr, "inner", ["Nachname", "Vorname", "Geburtsdatum", "Untersuchungsdatum"])
    mapping = pd.read_csv(mapping_file)
    mapping["medicoID"] = mapping["medicoID"].astype(str).apply(lambda x: x[:-4])
    metadata["Anforderungsnummer_y"] = metadata["Anforderungsnummer_y"].astype(str)  # .apply(lambda x: x[:-2])
    metadata = metadata.merge(mapping, "inner", left_on="Anforderungsnummer_y", right_on="medicoID")
    metadata.rename(columns={"Anforderungsnummer_y": "Anforderungsnummer", "medicoID_x": "medicoID"}, inplace=True)

    # Step2: replace nan entries with -1
    metadata.fillna(-1, inplace=True)

    # Step3: map values for columns containing values like 0/0
    slash_value_mapping = {"0/-": -0.25, "1/-": 0.25, "2/-": 1.25, "3/-": 2.25,
                           "0/0": 0, "0/1": 0.5,
                           "1/0": 0.75, "1/1": 1, "1/2": 1.5,
                           "2/1": 1.75, "2/2": 2, "2/3": 2.5,
                           "3/2": 2.75, "3/3": 3, "3/4": 3.5,
                           "4/3": 3.75, "4/4": 4}
    metadata['small_rounded_opacities_profusion_map'] = metadata[
        'small_rounded_opacities_profusion'].map(slash_value_mapping)
    metadata['small_irregular_opacities_profusion_map'] = metadata[
        'small_irregular_opacities_profusion'].map(slash_value_mapping)
    metadata['mixed_shapes_profusion_map'] = metadata['mixed_shapes_profusion'].map(
        slash_value_mapping)

    # Step4: drop all columns that are not needed for training
    col_to_drop = ["medicoID_y", "mixed_shapes_profusion", "small_irregular_opacities_profusion", "small_rounded_opacities_profusion"]
    # find other columns to drop: all that have more nan entries than nan_thresh
    nan_per_column_count = [(col_name, len(metadata[metadata[col_name] != metadata[col_name]])) for col_name in
                            metadata.columns]
    for col_name, nan_count in nan_per_column_count:
        if nan_count > maximum_occurance_of_nans_per_col:
            col_to_drop.append(col_name)
    metadata.drop(columns=col_to_drop, inplace=True)
    print("Dropped following columns because they contained at least " + str(
        int(maximum_occurance_of_nans_per_col / len(metadata) * 100)) + " % nan-values.")
    print(col_to_drop[1:])
    metadata.to_csv(metadata_file.replace('.csv', '_prepared.csv'), index=False)
    return metadata


def shorten_name(sheet_name):
    shortended_name = sheet_name.replace("thickening", "thick")
    shortended_name = shortended_name.replace("right", "r")
    shortended_name = shortended_name.replace("left", "l")
    shortended_name = shortended_name.replace("localized", "loc")
    shortended_name = shortended_name.replace("calcification", "calcif")
    return shortended_name


def create_individual_metadata_files(overall_metadata_file: str):
    metadata = pd.read_csv(overall_metadata_file)
    metadata['Geburtsdatum'] = pd.to_datetime(metadata['Geburtsdatum'], format='%Y-%m-%d')
    metadata['Untersuchungsdatum'] = pd.to_datetime(metadata['Untersuchungsdatum'], format='%Y-%m-%d')

    sub_metadatas = {}
    metadata_lengths = []
    new_filename = overall_metadata_file.replace(overall_metadata_file.split(os.sep)[-1], 'metadata_subset.xlsx')
    for column_name in metadata.columns:
        if "pleura" in column_name:
            column_metadata = metadata[metadata[column_name] != -1]
            sheet_name = shorten_name(column_name)
            sub_metadatas[sheet_name] = column_metadata
            if column_name == "localized_pleural_thickening_width_left" or column_name == "localized_pleural_thickening_width_right":
                c_value = "c" if column_name.endswith("_right") else "C"
                # options are a, b, C
                metadata_lengths.append({'Sheet': sheet_name,
                                         "Column name": column_name,
                                         'Number of entries': len(column_metadata),
                                         'Positive samples': 0, 'Negative samples': 0,
                                         "a/1": len(column_metadata[column_metadata[column_name] == "a"]),
                                         "b/2": len(column_metadata[column_metadata[column_name] == "b"]),
                                         "c/3": len(column_metadata[column_metadata[column_name] == c_value])})

            elif column_name == "localized_pleural_thickening_extent_left" or column_name == "localized_pleural_thickening_extent_right":
                # options are 1, 2, 3
                metadata_lengths.append({'Sheet': sheet_name,
                                         "Column name": column_name,
                                         'Number of entries': len(column_metadata),
                                         'Positive samples': 0, 'Negative samples': 0,
                                         "a/1": len(column_metadata[column_metadata[column_name] == 1]),
                                         "b/2": len(column_metadata[column_metadata[column_name] == 2]),
                                         "c/3": len(column_metadata[column_metadata[column_name] == 3])})
            elif column_name == "diffuse_pleural_thickening_width_right" or column_name == "diffuse_pleural_thickening_width_left":
                metadata_lengths.append({'Sheet': sheet_name,
                                         "Column name": column_name,
                                         'Number of entries': len(column_metadata),
                                         'Positive samples': 0, 'Negative samples': 0,
                                         "a/1": len(column_metadata[column_metadata[column_name] == "a"]),
                                         "b/2": len(column_metadata[column_metadata[column_name] == "b"]),
                                         "c/3": "-"})
            else:
                metadata_lengths.append({'Sheet': sheet_name,
                                         "Column name": column_name,
                                         'Number of entries': len(column_metadata),
                                         'Positive samples': len(column_metadata[column_metadata[column_name] == 1]),
                                         'Negative samples': len(column_metadata[column_metadata[column_name] == 0]),
                                         "a/1": 0, "b/2": 0, "c/3": 0})

    with pd.ExcelWriter(new_filename) as writer:
        for sheet_name, dataframe in sub_metadatas.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
        pd.DataFrame(metadata_lengths).to_excel(writer, sheet_name="Number of entries per sheet", index=False)

if __name__ == "__main__":
    overall_filename = "D:\\Projects\\Thorax\\DeboraThorax\\split_folds\\merged_data_stratified_folds.csv"
    create_individual_metadata_files(overall_filename)

