import gc

import PIL.Image
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import roc_curve, auc

import wandb
from torch.utils.data import Dataset, RandomSampler, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
import torch.multiprocessing as mp
from torch import nn, optim, GradScaler
import pandas
import os
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt


class MultiOutputResNet(nn.Module):
    def __init__(self, number_of_classes: list):  # Define number of outputs per task
        super(MultiOutputResNet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove original FC layer

        # Create multiple output heads dynamically
        self.output_heads = nn.ModuleList([
            nn.Linear(in_features, num_classes) for num_classes in number_of_classes
        ])

    def forward(self, x):
        x = self.resnet(x)
        return [head(x) for head in self.output_heads]  # Return multiple outputs

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

class X_rayImageDataset(Dataset):
    def __init__(self, annotations, img_dir, label_column: str, transform=None):
        self.img_dir = img_dir
        self.img_labels = self.get_available_data(annotations)
        self.transform = transform
        self.label_column = label_column

    def get_available_data(self, annotations):
        available_annotations = []
        for filename in annotations["fileID"]:
            if os.path.exists(self.img_dir + str(int(filename)) + '-IM_0001.png'):
                available_annotations.append(int(filename))
        return annotations[annotations['fileID'].isin(available_annotations)]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir + str(int(self.img_labels.iloc[idx]['fileID'])) + '-IM_0001.png'
        image = PIL.Image.open(img_path)
        label = self.img_labels.iloc[idx][self.label_column]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


def run_epoch(model, optimizer, criterion, scaler, data_loader, train=True, show_figs=True):
    if train:
        model.train()
    else:
        model.eval()

    torch.set_grad_enabled(train)
    epoch_loss = 0
    counter = -1
    set_length = len(data_loader)
    y_true = torch.zeros((set_length, batch_size), dtype=torch.int)  # true labels
    y_probs = torch.zeros((set_length, batch_size), dtype=torch.float)  # predicted probabilities for cancer
    for batch_id, (data, ground_truth, path) in enumerate(data_loader):
        counter += 1
        if counter % 100 == 0:
            gc.collect()
            if show_figs:
                # Create a figure and axis objects for depth dimension (images of size 256x256)
                plt.imshow(torch.clip(data[0, 0], 0, 100), cmap='gray', interpolation=None)
                plt.title(path[0].split(os.sep)[-1])
                plt.show()
                plt.close()

        data = data.to(device)
        ground_truth = ground_truth.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.float16):
            pred_probs = model(data.float())
            loss = criterion(pred_probs[:, 0].float(), ground_truth.float())
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        y_true[batch_id][0:len(ground_truth)] = ground_truth
        y_probs[batch_id][0:len(ground_truth)] = pred_probs[:, 0]

        wandb.log({f"loss/train": loss.item()}, commit=False)
        epoch_loss += loss.item()
        print('Train step loss: {}'.format(loss.item()))
    avg_epoch_loss = epoch_loss / len(data_loader)
    print('Train {}: \tAverage Loss: {:.6f}'.format(
        "Epoch " + str(epoch),
        avg_epoch_loss))
    return y_true, y_probs

def compute_roc_curve(y_true, y_scores, plot=False, title_suffix=''):
    """
    Compute the ROC curve and AUC (Area Under the Curve).

    Parameters:
    - y_true: Ground truth labels (true binary labels).
    - y_scores: Predicted scores or probabilities for positive class.
    - plot: Boolean to define if computed ROC AUC shall be plotted or not. Default is False.
    - title_suffix: string to be appended to the plot title in case plot=True.
    If plot is False, this parameter is ignored.

    Returns:
    - fpr: False Positive Rate (1 - Specificity).
    - tpr: True Positive Rate (Sensitivity).
    - roc_auc: Area Under the ROC Curve (AUC).
    """
    fpr, tpr, thresholds = roc_curve(y_true.flatten().detach().numpy(), y_scores.flatten().detach().numpy())
    best_index = np.argmax(tpr - fpr)
    print("Youden Index: ", thresholds[best_index])
    print(f"FPR: {fpr[best_index]}, TPR: {tpr[best_index]}")
    print(f"Specificity: {1 - fpr[best_index]}, Sensitivity: {tpr[best_index]}")
    roc_auc = auc(fpr, tpr)

    if plot:
        # ROC plot
        if title_suffix is None:
            title_suffix = ''
        plt.title('Receiver Operating Characteristic: ' + title_suffix)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        #plt.show()

        title_suffix = title_suffix.replace(' ', '_')
        title_suffix = title_suffix.replace('resnet50_', '')
        title_suffix = title_suffix.replace('resnet18_', '')
        roc_key = "roc_auc/" + title_suffix
        spec_key = "Specificity (Youden)/" + title_suffix
        sens_key = "Sensitivity (Youden)/" + title_suffix
        wandb.log({
            roc_key: wandb.Image(plt),
            spec_key: 1 - fpr[best_index],
            sens_key: tpr[best_index]
        })
        plt.close()

    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "best_index_val": best_index, "roc_auc_val": roc_auc}


if __name__ == '__main__':
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(0.5, 0.5),
    ])

    # Datenvorverarbeitung: fehlende Werte handeln; einlesen
    root_folder = "D:\\Projects\\Thorax\\DeboraThorax\\png\\"
    parent_folder = "D:\\Projects\\Thorax\\ArbeitsRadio\\ArbeitsRadio\\"
    mapping_file = "D:\\Projects\\Thorax\\mapping.csv"
    fold_folder = parent_folder + "split_folds\\"

    metadata_file = parent_folder + "merged_data.csv"
    anford_nr_file = parent_folder + "table.csv"
    nan_thresh = 999
    batch_size = 16
    learning_rate = 0.0001
    number_of_epochs = 20
    fold = 0
    # column_groups keys are general, symbol, rounded, irregular, mixed, large, pleura, occupational
    column_group = "symbol"

    metadata = pandas.read_csv(metadata_file)
    metadata['Geburtsdatum'] = pandas.to_datetime(metadata['Geburtsdatum'], format='%d.%m.%Y')
    metadata['Untersuchungsdatum'] = pandas.to_datetime(metadata['Untersuchungsdatum'], format='%d.%m.%Y')

    anford_nr = pandas.read_csv(anford_nr_file,
                                usecols=["Name", "Vorname", "Geburtsdatum", "Anforderungsnummer", "Untersuchungsdatum"])
    anford_nr.rename(columns={"Name": "Nachname"}, inplace=True)
    anford_nr['Geburtsdatum'] = pandas.to_datetime(anford_nr['Geburtsdatum'], format='%Y-%m-%d')
    anford_nr['Untersuchungsdatum'] = pandas.to_datetime(anford_nr['Untersuchungsdatum'], format='%Y-%m-%d')

    metadata = metadata.merge(anford_nr, "inner", ["Nachname", "Vorname", "Geburtsdatum", "Untersuchungsdatum"])
    mapping = pandas.read_csv(mapping_file)
    mapping["medicoID"] = mapping["medicoID"].astype(str).apply(lambda x: x[:-4])
    metadata["Anforderungsnummer_y"] = metadata["Anforderungsnummer_y"].astype(str).apply(lambda x: x[:-2])
    metadata = metadata.merge(mapping, "inner", left_on="Anforderungsnummer_y", right_on="medicoID")
    metadata.rename(columns={"Anforderungsnummer_y": "Anforderungsnummer", "medicoID_x": "medicoID"}, inplace=True)

    column_groups = get_column_name_groups(metadata)  # group columns into logical

    col_to_drop = ["medicoID_y"]
    # find other columns to drop: all that have more nan entries than nan_thresh
    nan_per_column_count = [(col_name, len(metadata[metadata[col_name] != metadata[col_name]])) for col_name in metadata.columns]
    for col_name, nan_count in nan_per_column_count:
        if nan_count > nan_thresh:
            col_to_drop.append(col_name)
    metadata.drop(columns=col_to_drop, inplace=True)
    print("Dropped following columns because they contained at least " + str(int(nan_thresh/len(metadata) * 100)) + " % nan-values.")
    print(col_to_drop[1:])

    # replace nan entries with -1
    metadata.fillna(-1, inplace=True)

    # Split Train-Test:
    fold_splitted_metadata_filename = fold_folder + metadata_file.split(os.sep)[-1].replace('.csv', '_stratified_folds.csv')
    if not os.path.isfile(fold_splitted_metadata_filename):
        metadata = create_splits(metadata,
                                 5,
                                 fold_folder,
                                 fold_splitted_metadata_filename)
    else:
        metadata = pandas.read_csv(fold_splitted_metadata_filename)
    test_metadata = metadata[metadata[f'Fold{fold}'] == 'test']
    train_metadata = metadata[metadata[f'Fold{fold}'] == 'train']

    current_train_metadata = train_metadata[[col for col in column_groups[column_group] if col in metadata.columns]]
    current_test_metadata = test_metadata[[col for col in column_groups[column_group] if col in metadata.columns]]

    model = resnet50(weights=ResNet50_Weights)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    num_classes = len(current_train_metadata.columns) - len(column_groups["general"])
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # dataloader
    n_workers = mp.cpu_count() if mp.cpu_count() < 25 else 24
    gen = torch.Generator()
    train_dataset = X_rayImageDataset(current_train_metadata, root_folder, label_column=column_groups[column_group][0], transform=preprocess)
    train_sampler = RandomSampler(train_dataset, replacement=False, generator=gen)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, sampler=train_sampler,
                        generator=gen, drop_last=True, pin_memory=False)

    test_dataset = X_rayImageDataset(current_test_metadata, root_folder, label_column=column_groups[column_group][0], transform=preprocess)
    test_sampler = RandomSampler(test_dataset, replacement=False, generator=gen)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, sampler=test_sampler,
                        generator=gen, drop_last=True, pin_memory=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    wandb.init(project="Asbest", config={
        "learning-rate": learning_rate,
        "dataset:": root_folder,
        "split folder": fold_folder,
        "number of training samples": len(train_loader.dataset),
        "number of test samples": len(test_loader.dataset),
        "epochs": number_of_epochs,
        "batch size": batch_size,
        "optimizer": str(optimizer),
        "Augmentation": str(preprocess),
        "Machine": "Local",
        "Pretrained": str(ResNet50_Weights),
        "Fold": fold
    }, name="b={}_l={}_n={}_fold={}".format(batch_size, learning_rate, number_of_epochs, fold))

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    for epoch in range(number_of_epochs):
        y_true, y_probs = run_epoch(model, optimizer, criterion, scaler, train_loader, train=True)
        if epoch % 10 == 0:
            compute_roc_curve(y_true, y_probs, plot=False)

    y_true_eval, y_probs_eval = run_epoch(model, optimizer, criterion, scaler, test_loader, train=False)
    compute_roc_curve(y_true_eval, y_probs_eval, plot=True)



