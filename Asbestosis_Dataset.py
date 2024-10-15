import os
import shutil

import numpy as np
import zipfile
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import pydicom



class Asbestosis_Dataset(Dataset):

    def __init__(self, path_root: str, feature_tensor: pd.DataFrame):
        if feature_tensor is None:
            print("Feature tensor is None")
            return
        self.feature_tensor = feature_tensor

        if not Path(path_root).is_dir():
            print(f"Path {path_root} does not exist")
            return
        self.path_root = path_root
        self.data = []

        for root, _, files in os.walk(self.path_root):
            self.get_images(root, files)


    def __len__(self):
        return len(self.feature_tensor)

    def __getitem__(self, index: int):
        image_path = self.data[index]
        image = Image.open(image_path)
        # TODO augmentation ??
        return image

    def get_images(self, root, files):
        for file in files:
            if file.endswith('.zip'):
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

                            out_file_path = output_folder + file.split(".zip")[0] + "-" + dicom_file.split(os.sep)[-1] + ".png"
                            dicom_image.save(out_file_path)

                            self.data.append(out_file_path)
                            print(f"Saved {out_file_path}")
                shutil.rmtree(extract_folder)
            elif file.endswith('.png'):
                self.data.append(file)
