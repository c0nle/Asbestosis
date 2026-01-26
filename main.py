import gc
import os
import numpy as np
import pandas as pd
import re

from pathlib import Path
import PIL.Image
import torch

from torch import nn, optim
from torch.cuda.amp import GradScaler

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms.v2 import (
    ColorJitter, RandomResizedCrop, RandomRotation, ToTensor, Compose, Normalize, Resize, CenterCrop, ToImage, ToDtype
)

from torchvision.models import resnet50, ResNet50_Weights

from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler
import torch.multiprocessing as mp
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import Preprocessor_Metadata
import wandb

# -------------------------
# Helpers
# -------------------------
def gray_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """(1,H,W) -> (3,H,W) for pretrained ViT"""
    return x.repeat(3, 1, 1)


def log_auc_per_task(y_true: torch.Tensor, y_prob: torch.Tensor, mask: torch.Tensor,
                     criteria_cols: list, split: str = "test"):
    """
    y_true: (N,T) values {0,1,-1}
    y_prob: (N,T) values [0,1]
    mask:   (N,T) values {0,1} where 1 means label exists
    """
    y_true_np = y_true.numpy()
    y_prob_np = y_prob.numpy()
    mask_np = mask.numpy()

    for t, name in enumerate(criteria_cols):
        valid = mask_np[:, t] > 0.5
        if valid.sum() < 10:
            continue

        yt = y_true_np[valid, t].astype(int)
        yp = y_prob_np[valid, t].astype(float)

        # AUC needs both classes
        if len(np.unique(yt)) < 2:
            continue

        try:
            auc_val = roc_auc_score(yt, yp)
        except Exception:
            continue

        wandb.log({f"AUC/{split}/{name}": auc_val})

def log_task_counts(y_true: torch.Tensor, mask: torch.Tensor, criteria_cols: list, split: str = "train"):
    """
    Logs how many valid pos/neg labels exist per task.
    y_true: (N,T) {0,1,-1}
    mask:   (N,T) {0,1}
    """
    y = y_true.numpy()
    m = mask.numpy()

    for t, name in enumerate(criteria_cols):
        valid = m[:, t] > 0.5
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        pos = int((y[valid, t] == 1).sum())
        neg = int((y[valid, t] == 0).sum())

        wandb.log({
            f"count/{split}/{name}/valid": n_valid,
            f"count/{split}/{name}/pos": pos,
            f"count/{split}/{name}/neg": neg,
        })
# -------------------------
# Model
# -------------------------
class MultiOutputViT(nn.Module):
    """
    Multi-task binary classification:
    - ViT backbone (pretrained)
    - one head per criterion, each outputs 1 logit
    Output: logits (B, T)
    """
    def __init__(self, n_tasks: int):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        in_features = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()  # remove original classifier head

        self.output_heads = nn.ModuleList([nn.Linear(in_features, 1) for _ in range(n_tasks)])

    def forward(self, x):
        feats = self.vit(x)  # (B, F)
        logits = [head(feats) for head in self.output_heads]  # list of (B,1)
        return torch.cat(logits, dim=1)  # (B, T)


class MultiOutputResNet(nn.Module):
    """
    Multi-task binary classification:
    - shared ResNet50 backbone
    - one head per criterion, each outputs 1 logit
    Output shape: (B, T)
    """
    def __init__(self, n_tasks: int):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.output_heads = nn.ModuleList([nn.Linear(in_features, 1) for _ in range(n_tasks)])

    def forward(self, x):
        feats = self.resnet(x)                   # (B, F)
        logits = [head(feats) for head in self.output_heads]  # list of (B,1)
        return torch.cat(logits, dim=1)          # (B, T)


# -------------------------
# Dataset
# -------------------------
class X_rayImageDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, img_dir: str, criteria_cols: list,
                 img_id_col: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.criteria_cols = criteria_cols
        self.img_id_col = img_id_col

        # Build mapping: extracted_key -> filepath
        self.file_map = {}
        for f in os.listdir(img_dir):
            if not f.endswith(".png"):
                continue
            stem = Path(f).stem  # filename without .png

            # key = part before '-' if present, else leading digits
            key = stem.split("-")[0]
            m = re.match(r"^\d+", key)
            if m:
                key = m.group(0)

            # store first occurrence
            if key not in self.file_map:
                self.file_map[key] = os.path.join(img_dir, f)

        ann = annotations.copy()
        ann.columns = ann.columns.str.strip()
        ann[self.img_id_col] = ann[self.img_id_col].astype(str)

        # keep only rows with an existing image key
        self.img_labels = ann[ann[self.img_id_col].isin(self.file_map.keys())].reset_index(drop=True)

        print(f"[Dataset] Using img_id_col='{img_id_col}', matched {len(self.img_labels)}/{len(ann)} rows to PNGs.")
        print(f"[Dataset] n_tasks={len(self.criteria_cols)}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_key = self.img_labels.iloc[idx][self.img_id_col]
        img_path = self.file_map[img_key]

        image = PIL.Image.open(img_path).convert("L")

        # labels vector (T,)
        y = self.img_labels.iloc[idx][self.criteria_cols].values.astype(np.float32)
        y = torch.from_numpy(y)  # (T,)

        if self.transform:
            image = self.transform(image)

        return image, y, img_path


# -------------------------
# Training / Eval
# -------------------------
def run_epoch(model, optimizer, bce, scaler, data_loader, train=True, show_figs=False):
    if train:
        model.train()
        suffix = "train"
    else:
        model.eval()
        suffix = "eval"

    torch.set_grad_enabled(train)
    epoch_loss = 0.0

    y_true_list, y_prob_list, mask_list = [], [], []

    for batch_id, (data, targets, path) in enumerate(data_loader):
        if show_figs and batch_id % 200 == 0:
            gc.collect()

        data = data.to(device, non_blocking=True)              # (B,3,H,W)
        targets = targets.to(device, non_blocking=True)        # (B,T)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(data)                               # (B,T)

            # mask out missing labels (-1)
            mask = (targets != -1).float()                     # (B,T)
            targets_clean = torch.clamp(targets, 0, 1)         # -1 -> 0 dummy

            loss_mat = bce(logits, targets_clean)              # (B,T)
            loss = (loss_mat * mask).sum() / mask.sum().clamp_min(1.0)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

        probs = torch.sigmoid(logits)                          # (B,T)

        y_true_list.append(targets.detach().cpu())
        y_prob_list.append(probs.detach().cpu())
        mask_list.append(mask.detach().cpu())

        wandb.log({f"loss/{suffix}": float(loss.item())}, commit=False)
        epoch_loss += float(loss.item())

    avg_epoch_loss = epoch_loss / max(len(data_loader), 1)
    wandb.log({f"average_loss/{suffix}": avg_epoch_loss})

    y_true = torch.cat(y_true_list, dim=0)   # (N,T)
    y_prob = torch.cat(y_prob_list, dim=0)   # (N,T)
    mask = torch.cat(mask_list, dim=0)       # (N,T)
    return y_true, y_prob, mask

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
    # ViT pretrained expects normalized RGB-like input
    train_tf = Compose([
        RandomResizedCrop(224),
        RandomRotation(5),
        ColorJitter(0.3),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        gray_to_rgb,  # oder später entfernen, siehe 1.4
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_tf = Compose([
        Resize(256),
        CenterCrop(224),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        gray_to_rgb,
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Paths / config
    base_folder = "/hpcwork/rwth1954/Asbestosis_Data/"
    root_folder = os.path.join(base_folder, "png")
    mapping_file = os.path.join(base_folder, "mapping.csv")
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    fold_folder = os.path.join(PROJECT_DIR, "splits")
    os.makedirs(fold_folder, exist_ok=True)

    metadata_file = os.path.join(base_folder, "dichotome_data_pseudonym.csv")
    anford_nr_file = os.path.join(base_folder, "table_pseudonym.csv")

    nan_thresh = 999
    batch_size = 8
    learning_rate = 3e-4  # IMPORTANT for ViT
    number_of_epochs = 30
    fold = 0

    # choose which group of labels to train together
    column_group = "mixed"

    # stratification label for folds (single label just to build folds)
    # keep as one stable binary label; training itself is multi-task
    training_label_for_split = "mixed_shapes"

    # Split Train-Test:
    fold_splitted_metadata_filename = os.path.join(
        fold_folder,
        Path(metadata_file).name.replace('.csv', '_stratified_folds.csv')
    )

    if not os.path.isfile(fold_splitted_metadata_filename):
        metadata = Preprocessor_Metadata.prepare_metadata(
            metadata_file, anford_nr_file, mapping_file, nan_thresh, True
        )
        metadata = Preprocessor_Metadata.create_splits(
            metadata,
            5,
            fold_folder,
            fold_splitted_metadata_filename,
            training_label_for_split
        )
    else:
        metadata = pd.read_csv(fold_splitted_metadata_filename)

    prepared_metadata = metadata
    column_groups = Preprocessor_Metadata.get_column_name_groups(metadata, True)

    # Build multi-task criteria list from the chosen group
    candidate_cols = [c for c in column_groups[column_group]
                      if c in prepared_metadata.columns and c not in column_groups["general"]]

    def is_binary_series(s: pd.Series) -> bool:
        vals = set(pd.unique(s.dropna()))
        return vals.issubset({-1, 0, 1})

    criteria_cols = [c for c in candidate_cols if is_binary_series(prepared_metadata[c])]
    print(f"[Main] Using {len(criteria_cols)} binary criteria from group '{column_group}'")

    # Keep only rows where at least one task label exists
    metadata = prepared_metadata.copy()
    if len(criteria_cols) == 0:
        raise RuntimeError(f"No binary criteria found for group '{column_group}'")

    metadata = metadata[(metadata[criteria_cols] != -1).any(axis=1)]
    test_metadata = metadata[metadata[f'Fold{fold}'] == 'test']
    train_metadata = metadata[metadata[f'Fold{fold}'] == 'train']

    needed_cols = ["id"] + criteria_cols
    current_train_metadata = train_metadata[needed_cols]
    current_test_metadata = test_metadata[needed_cols]

    # Model
    model = MultiOutputViT(n_tasks=len(criteria_cols))

    # Dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_workers = mp.cpu_count() if mp.cpu_count() < 25 else 24
    gen = torch.Generator()

    train_dataset = X_rayImageDataset(
        current_train_metadata, root_folder, criteria_cols=criteria_cols, img_id_col="id", transform=train_tf
    )
    train_sampler = RandomSampler(train_dataset, replacement=False, generator=gen)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        sampler=train_sampler,
        generator=gen,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    test_dataset = X_rayImageDataset(
        current_test_metadata, root_folder, criteria_cols=criteria_cols, img_id_col="id", transform=test_tf
    )
    test_sampler_rnd = RandomSampler(test_dataset, replacement=False, generator=gen)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        sampler=test_sampler,
        generator=gen,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=number_of_epochs * len(train_loader)
    )

    wandb.init(
        project="Asbestosis",
        config={
            "learning_rate": learning_rate,
            "dataset": root_folder,
            "split_folder": fold_folder,
            "n_train": len(train_loader.dataset),
            "n_test": len(test_loader.dataset),
            "epochs": number_of_epochs,
            "batch_size": batch_size,
            "optimizer": str(optimizer),
            "augmentation_train": str(train_tf),
            "augmentation_test": str(test_tf),
            "machine": "HPC",
            "pretrained": str(ViT_B_16_Weights.DEFAULT),
            "column_group": column_group,
            "criteria_cols": criteria_cols,
            "n_tasks": len(criteria_cols),
            "fold": fold,
            "metadata": metadata_file,
        },
        name=f"vit_b={batch_size}_lr={learning_rate}_n={number_of_epochs}_fold={fold}_group={column_group}",
    )

    # Loss + AMP
    bce = nn.BCEWithLogitsLoss(reduction="none")
    scaler = GradScaler(enabled=(device.type == "cuda"))

    model = model.to(device)

    for epoch in range(number_of_epochs):
        y_true_tr, y_prob_tr, mask_tr = run_epoch(
            model, optimizer, bce, scaler, train_loader, train=True, show_figs=False
        )
        log_auc_per_task(y_true_tr, y_prob_tr, mask_tr, criteria_cols, split="train")
        log_task_counts(y_true_tr, mask_tr, criteria_cols, split="train")

    y_true_ev, y_prob_ev, mask_ev = run_epoch(
        model, optimizer, bce, scaler, test_loader, train=False, show_figs=False
    )
    log_auc_per_task(y_true_ev, y_prob_ev, mask_ev, criteria_cols, split="test")
    log_task_counts(y_true_ev, mask_ev, criteria_cols, split="test")
    torch.save(
        model.state_dict(),
        os.path.join(base_folder, f"asbestosis_vit_multitask_{column_group}_n{number_of_epochs}_b{batch_size}.pth")
    )
    wandb.finish()
