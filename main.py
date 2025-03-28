import gc
import os
import numpy as np
import pandas as pd

import PIL.Image
import torch
from torch import nn, optim, GradScaler
from torch.utils.data import Dataset, RandomSampler, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
import torch.multiprocessing as mp
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
import matplotlib.pyplot as plt

import Preprocessor_Metadata
import wandb


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
        label = float(self.img_labels.iloc[idx][self.label_column])
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


def run_epoch(model, optimizer, criterion, scaler, data_loader, train=True, show_figs=True):
    if train:
        model.train()
        suffix = "train"
    else:
        model.eval()
        suffix = "eval"

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
            pred_probs = model(data)
            loss = criterion(pred_probs, ground_truth)  #[:, 0]
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        y_true[batch_id][0:len(ground_truth)] = ground_truth
        y_probs[batch_id][0:len(ground_truth)] = pred_probs[:, 0]

        wandb.log({f"loss/{suffix}": loss.item()}, commit=False)
        epoch_loss += loss.item()
        print('{} step loss: {}'.format(suffix, loss.item()))
    avg_epoch_loss = epoch_loss / len(data_loader)
    print('{} {}: \tAverage Loss: {:.6f}'.format(
        suffix,
        "Epoch " + str(epoch),
        avg_epoch_loss))
    return y_true, y_probs

def compute_roc_one_vs_rest(y_true, y_scores, classes, category_label):
    label_binarizer = LabelBinarizer().fit(y_scores)
    y_onehot_test = label_binarizer.transform(y_true)
    fpr, tpr, _ = roc_curve(y_onehot_test.ravel(), y_scores.ravel())
    roc_auc_value = auc(fpr, tpr)
    wandb.log({"multi_roc_auc/" + category_label: roc_auc_value})

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(classes, colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_scores[:, class_id],
            name=f"ROC curve for {str(class_id)}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
            despine=True,
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Receiver Operating Characteristic (One-vs-Rest multiclass) for " + category_label,
    )


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
    auc_key = "AUC/" + title_suffix
    wandb.log({auc_key: roc_auc})

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
            sens_key: tpr[best_index],
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
    base_folder = "/hpcwork/it336446/Data/DeboraThorax/"  # "D:\\Projects\\Thorax\\DeboraThorax\\"
    root_folder = base_folder + "png/" 
    mapping_file = base_folder + "mapping.csv"
    fold_folder = base_folder + "split_folds/"

    metadata_file = base_folder + "merged_data_prepared.csv"
    anford_nr_file = base_folder + "table.csv"
    nan_thresh = 999
    batch_size = 16
    learning_rate = 0.0001
    number_of_epochs = 1 # 40
    fold = 0
    # column_groups keys are general, symbol, rounded, irregular, mixed, large, pleural, occupational
    column_group = "pleural"

    # Split Train-Test:
    fold_splitted_metadata_filename = fold_folder + metadata_file.split(os.sep)[-1].replace('.csv', '_stratified_folds.csv')
    if not os.path.isfile(fold_splitted_metadata_filename):
        metadata = Preprocessor_Metadata.prepare_metadata(metadata_file, anford_nr_file, mapping_file, nan_thresh)
        metadata = Preprocessor_Metadata.create_splits(metadata,
                                 5,
                                 fold_folder,
                                 fold_splitted_metadata_filename)
    else:
        metadata = pd.read_csv(fold_splitted_metadata_filename)

    prepared_metadata = metadata
    column_groups = Preprocessor_Metadata.get_column_name_groups(metadata)  # group columns into logical

    for criterion_label in column_groups[column_group]:
        #criterion_label = "diffuse_pleural_thickening_extent_right"  #  diffuse_pleural_thickening_nad
        if criterion_label in metadata.keys() and criterion_label not in column_groups["general"]:
            metadata = prepared_metadata[prepared_metadata[criterion_label] != -1]
            test_metadata = metadata[metadata[f'Fold{fold}'] == 'test']
            train_metadata = metadata[metadata[f'Fold{fold}'] == 'train']
            #print(criterion_label + ":\n\ttrain --- pos: " + str(len(train_metadata[train_metadata[criterion_label] == 1])) + " neg: " + str(len(train_metadata[train_metadata[criterion_label] == 0])) +
            #      "\n\ttest --- pos: " + str(len(test_metadata[test_metadata[criterion_label] == 1])) + " neg: " + str(len(test_metadata[test_metadata[criterion_label] == 0])))

            current_train_metadata = train_metadata[[col for col in column_groups[column_group] if col in metadata.columns]]
            current_test_metadata = test_metadata[[col for col in column_groups[column_group] if col in metadata.columns]]
            num_classes = len(prepared_metadata[criterion_label].unique())
            is_binary_classification = num_classes == 2
            model = resnet50() #weights=ResNet50_Weights)

            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            #num_classes = len(current_train_metadata.columns) - len(column_groups["general"])
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

            # dataloader
            n_workers = mp.cpu_count() if mp.cpu_count() < 25 else 24
            gen = torch.Generator()
            train_dataset = X_rayImageDataset(current_train_metadata, root_folder, label_column=criterion_label, transform=preprocess)
            train_sampler = RandomSampler(train_dataset, replacement=False, generator=gen)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, sampler=train_sampler,
                               generator=gen, drop_last=True, pin_memory=False)

            test_dataset = X_rayImageDataset(current_test_metadata, root_folder, label_column=column_groups[column_group][0], transform=preprocess)
            test_sampler = RandomSampler(test_dataset, replacement=False, generator=gen)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, sampler=test_sampler,
                               generator=gen, drop_last=True, pin_memory=False)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

            wandb.init(project="Asbestosis_test", config={
                "learning-rate": learning_rate,
                "dataset:": root_folder,
                "split folder": fold_folder,
                "number of training samples": len(train_loader.dataset),
                "number of test samples": len(test_loader.dataset),
                "epochs": number_of_epochs,
                "batch size": batch_size,
                "optimizer": str(optimizer),
                "Augmentation": str(preprocess),
                "Machine": "HPC",
                "Pretrained": "No", # str(ResNet50_Weights),
                "train criteria": criterion_label,
                "Fold": fold
            }, name="b={}_l={}_n={}_fold={}_{}_prepMeta".format(batch_size, learning_rate, number_of_epochs, fold, criterion_label))

            criterion = nn.BCEWithLogitsLoss() if is_binary_classification else nn.CrossEntropyLoss()
            scaler = GradScaler()
            model = model.to(device)
            for epoch in range(number_of_epochs):
                y_true, y_probs = run_epoch(model, optimizer, criterion, scaler, train_loader, train=True)
                if epoch % 10 == 0 and is_binary_classification:
                    compute_roc_curve(y_true, y_probs, plot=False, title_suffix="train")

            y_true_eval, y_probs_eval = run_epoch(model, optimizer, criterion, scaler, test_loader, train=False)
            if is_binary_classification:
                compute_roc_curve(y_true_eval, y_probs_eval, plot=True, title_suffix="test")

            torch.save(model.state_dict(), os.path.join(base_folder, f'asbestosis_n{number_of_epochs}_b{batch_size}_label={criterion_label}.pth'))
            wandb.finish()
