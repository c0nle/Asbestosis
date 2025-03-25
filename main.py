import gc

import pandas as pd
import torch
import torchvision
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
                       'technical_quality']
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
        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.label_column = label_column

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # TODO match metadata file with image name
        image = torchvision.io.read_image(img_path)
        label = self.img_labels.iloc[idx][self.label_column]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datenvorverarbeitung: fehlende Werte handeln; einlesen
    parent_folder = "/hpcwork/it336446/Data/DeboraThorax/" # "D:\\Projects\\Thorax\\ArbeitsRadio\\ArbeitsRadio\\"
    root_folder = parent_folder + "png/" # "\\wsl.localhost\\Ubuntu\\home\\debora\\DeboraThorax\\png\\"
    fold_folder = parent_folder + "split_folds\\"

    metadata_file = parent_folder + "merged_data.csv"
    nan_thresh = 999
    batch_size = 16
    learning_rate = 0.0001
    number_of_epochs = 20
    fold = 0
    # column_groups keys are general, symbol, rounded, irregular, mixed, large, pleural, occupational
    column_group = "pleural"

    metadata = pandas.read_csv(metadata_file)
    column_groups = get_column_name_groups(metadata) # group columns into logical

    # generate medicoIDs to replace nan-values
    unique_patients_with_nan_medicoID = metadata[['Nachname', 'Vorname', 'Geburtsdatum']].drop_duplicates()
    unique_patients_with_nan_medicoID['medicoID_created'] = range(1, len(unique_patients_with_nan_medicoID) + 1)
    metadata = metadata.merge(unique_patients_with_nan_medicoID, on=['Nachname', 'Vorname', 'Geburtsdatum'], how='left')
    metadata['medicoID'] = metadata['medicoID_created']

    col_to_drop = ['medicoID_created']
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        "train criteria": column_group,
        "Fold": fold
    }, name="b={}_l={}_n={}_fold={}".format(batch_size, learning_rate, number_of_epochs, fold))
    # perform training
    for epoch in range(number_of_epochs):
        show_figs = True

        # train
        model.train()
        torch.set_grad_enabled(True)
        epoch_loss = 0
        counter = -1

        for data, ground_truth, path in train_loader:
            counter += 1
            if counter % 100 == 0:
                gc.collect()
                if show_figs:
                    # Create a figure and axis objects for depth dimension (images of size 256x256)
                    fig, axs = plt.subplots(4, 8, figsize=(12, 6))

                    # Plot each slice of the tensor
                    for i in range(4):
                        for j in range(8):
                            index = i * 8 + j
                            axs[i, j].imshow(torch.clip(data[0, 0, index], 0, 100), cmap='gray', interpolation=None)
                            axs[i, j].axis('off')
                            title = path[0] if i == 0 and j == 0 else f"Slice {index}"
                            axs[i, j].set_title(title)

                    # Adjust layout and display the plot
                    plt.tight_layout()
                    plt.show()
                    plt.close()

            data = data.to(device)
            ground_truth = ground_truth.to(device)
            optimizer.zero_grad()
            criterion = nn.BCEWithLogitsLoss()
            scaler = GradScaler()

            with torch.autocast(device_type=device, dtype=torch.float16):
                pred_probs = model(data.float())
                loss = criterion(pred_probs[:, 0].float(), ground_truth.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            wandb.log({f"loss/train": loss.item()}, commit=False)
            epoch_loss += loss.item()
            print('Train step loss: {}'.format(loss.item()))

        avg_epoch_loss = epoch_loss / len(train_loader)
        print('Train {}: \tAverage Loss: {:.6f}'.format(
            "Epoch " + str(epoch),
            avg_epoch_loss))

    # Evaluation
    model.eval()
    torch.set_grad_enabled(False)
    epoch_loss = 0
    counter = -1

    for data, ground_truth, path in test_loader:
        counter += 1
        data = data.to(device)
        ground_truth = ground_truth.to(device)
        optimizer.zero_grad()
        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler()

        with torch.autocast(device_type=device, dtype=torch.float16):
            pred_probs = model(data.float())
            loss = criterion(pred_probs[:, 0].float(), ground_truth.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        wandb.log({f"loss/test": loss.item()}, commit=False)
        epoch_loss += loss.item()
        print('Test step loss: {}'.format(loss.item()))

    avg_epoch_loss = epoch_loss / len(test_loader)
    print('Test {}: \tAverage Loss: {:.6f}'.format(
        "Epoch " + str(epoch),
        avg_epoch_loss))


