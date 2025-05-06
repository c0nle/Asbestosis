import datetime
import os
import time

import numpy as np
import torch
from monai.losses.perceptual import torchvision

import wandb
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.amp import GradScaler
from torchmetrics import AUROC, Accuracy, Precision, Specificity, ROC
import torchio as tio
import torch.multiprocessing as mp

from torchvision.models import resnet101

import utils
from Preprocessor_Metadata import split_dash_containing_columns, get_Subjects

CPU = "cpu"
GPU = "cuda"


def get_logging_metrices():
    auc_roc = nn.ModuleDict(
        {state: AUROC(**{"task": "binary"}).to(DEVICE) for state in ["train_", "eval_", "Test_"]})
    roc = nn.ModuleDict({state: ROC(**{"task": "binary"}).to(DEVICE) for state in ["Test_"]})
    acc = nn.ModuleDict(
        {state: Accuracy(**{"task": "binary"}).to(DEVICE) for state in ["train_", "eval_", "Test_"]})
    sens = nn.ModuleDict(
        {state: Precision(**{"task": "binary"}).to(DEVICE) for state in ["train_", "eval_", "Test_"]})
    spec = nn.ModuleDict(
        {state: Specificity(**{"task": "binary"}).to(DEVICE) for state in ["train_", "eval_", "Test_"]})
    return [auc_roc, roc, acc, sens, spec]


def train_epoch(data_loader, model, optimizer, criterion, DEVICE, logging_metrices, epoch_number=0):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    start = time.time()
    model.train()
    torch.set_grad_enabled(True)
    epoch_loss = 0
    for batch in data_loader:
        data = batch["image"][tio.DATA]
        data = data.squeeze(-1)
        data = data.to(DEVICE)
        ground_truth = batch["label"]
        ground_truth = ground_truth #.to(DEVICE)
        optimizer.zero_grad()

        pred_probs = model(data)
        loss = criterion(pred_probs[:, 0].float(), ground_truth.float())
        loss.backward()
        optimizer.step()

        for metric in logging_metrices:
            metric["train_"].update(pred_probs[:, 0], ground_truth)

        wandb.log({"loss/train": loss.item()}, commit=False)
        epoch_loss += loss.item()
        print('Train step loss: {}'.format(loss.item()))

    avg_epoch_loss = epoch_loss / len(data_loader)
    end = time.time()
    print('Train Epoch {}: Duration = {} sec\n\tAverage Loss: {:.6f}'.format(
        epoch_number,
        end - start,
        avg_epoch_loss))

    wandb_dict = {}
    for metric in logging_metrices:
        wandb_dict[f"{str(metric)}/train"] = metric.compute()
        metric.reset()
    wandb.log(wandb_dict)
    return avg_epoch_loss


def eval_epoch(data_loader, model, criterion, DEVICE, logging_metrices, epoch_number, suffix: str, label, prediction):
    start = time.time()
    model.eval()
    torch.set_grad_enabled(False)
    epoch_loss = 0
    for batch_id, (data, ground_truth, path) in enumerate(data_loader):
        data = data.to(DEVICE)
        ground_truth = ground_truth.to(DEVICE)

        pred_probs = model(data.float())
        loss = criterion(pred_probs[:, 0].float(), ground_truth.float())
        label.extend(ground_truth.flatten().tolist())
        prediction.extend(pred_probs.flatten().tolist())

        for metric in logging_metrices:
            metric[suffix + "_"].update(pred_probs[:, 0], ground_truth)

        wandb.log({"loss/ " +suffix: loss.item()}, commit=False)

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(data_loader)
    end = time.time()
    print('Validation ({}) Epoch {}: Duration = {} sec\n\tAverage Loss: {:.6f}'.format(
        suffix,
        epoch_number,
        end - start,
        avg_epoch_loss))
    wandb_dict = {}
    results = {'label': label,
               'prediction': prediction}
    auc = -1
    for metric in logging_metrices:
        wandb_dict[f"{str(metric)}/train"] = metric.compute()
        if suffix == "Test" and str(metric).startswith("roc"):
            fpr, tpr, thresholds = metric[suffix + "_"].to(CPU).compute()
            results['tpr'] = tpr
            results['fpr'] = fpr
            results['thresholds'] = thresholds
            results['auc'] = auc

            # Plot tpr vs 1-fpr
            fig, ax = plt.subplots()
            plt.plot(fpr, tpr, 'b', label="AUC = %0.3f" % auc)
            plt.legend(loc='lower right')
            plt.plot()
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.title('Receiver Operating Characteristic - Test ')
            plt.show()
            best_index = np.argmax(tpr - fpr)
            wandb.log({
                "ROC/Test": fig,
                "Specificity (Youden)/Test": 1 - fpr[best_index],
                "Sensitivity (Youden)/Test": tpr[best_index]
            })
            results['Spec [i]'] = fpr[best_index]
            results['Sens [i]'] = tpr[best_index]
            results['i'] = best_index
            plt.close(fig)

        metric.reset()
    wandb.log(wandb_dict)
    return avg_epoch_loss, results


def get_dataloader(training_feature, lung_train, pleura_train, symbol_train, lung_eval, pleura_eval, symbol_eval, lung_test, pleura_test, symbol_test):
    n_workers = 8 # mp.cpu_count()
    generator = torch.Generator()
    generator.manual_seed(0)
    train_set, eval_set, test_set = None, None, None

    if training_feature.lower().strip() == "lung":
        train_set = lung_train
        eval_set = lung_eval
        test_set = lung_test
    elif training_feature.lower().strip() == "pleura":
        train_set = pleura_train
        eval_set = pleura_eval
        test_set = pleura_test
    elif training_feature.lower().strip() == "symbol":
        train_set = symbol_train
        eval_set = symbol_eval
        test_set = symbol_test
    else:
        print("Unknown training key word: " + training_feature)
    train_subjects = get_Subjects(path_root, train_set)
    dataset_train = tio.SubjectsDataset(train_subjects, transform)
    loader_train = tio.SubjectsLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                                   generator=generator,
                                   drop_last=False, pin_memory=False, prefetch_factor=15)

    eval_subjects = get_Subjects(path_root, eval_set)
    dataset_eval = tio.SubjectsDataset(eval_subjects, transform)
    loader_eval = tio.SubjectsLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                                  generator=generator,
                                  drop_last=False, pin_memory=False, prefetch_factor=15)

    test_subjects = get_Subjects(path_root, test_set)
    dataset_test = tio.SubjectsDataset(test_subjects, transform)
    loader_test = tio.SubjectsLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                                  generator=generator,
                                  drop_last=False, pin_memory=False, prefetch_factor=15)
    return loader_train, loader_eval, loader_test


if __name__ == "__main__":
    path_root = "/hpcwork/p0020933/workspace_debora/Data/Thorax_data/"
    feature_file_path = "/hpcwork/p0020933/workspace_debora/Data/dichotome_data.csv"  # "/hpcwork/it336446/Data/found_merged_data.xlsx"
    training_feature = "lung"
    batch_size = 4
    number_of_epochs = 80
    learning_rate = 0.001
    weight_decay = 1e-2

    torch.manual_seed(31)
    DEVICE = GPU
    if not torch.cuda.is_available():
        DEVICE = CPU

    model = resnet101(pretrained=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
    # input shall accept one input channel instead of 3.
    new_conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv1.weight[:] = model.conv1.weight.mean(dim=1, keepdim=True)
    model.conv1 = new_conv1
    model.to(DEVICE)

    output_size = (3000, 3000, 1)


    transform = tio.Compose([
        tio.CropOrPad(output_size),
        tio.RandomAffine(scales=0.2, degrees=0, center='image', default_pad_value='minimum', check_shape=True),
        tio.RandomFlip((0,1), flip_probability=0.5),
        tio.RandomMotion(),
        tio.RandomGhosting(),
        tio.RandomSpike(),
        tio.RandomBiasField(),
        tio.RandomBlur(),
        tio.RandomNoise(std=(0.25, 0.5))
    ])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # evaluation metrics:
    logging_metrices = get_logging_metrices()

    lung_train, pleura_train, symbol_train, lung_eval, pleura_eval, symbol_eval, lung_test, pleura_test, symbol_test = utils.get_feature_tensor(feature_file_path, train_fraction=0.7, evaluation_fraction=0.5)

    for data_set in [lung_train, pleura_train, symbol_train, lung_eval, pleura_eval, symbol_eval, lung_test, pleura_test, symbol_test]:
        split_dash_containing_columns(data_set)

    train_loader, eval_loader, test_loader = get_dataloader(training_feature, lung_train, pleura_train, symbol_train, lung_eval, pleura_eval, symbol_eval, lung_test, pleura_test, symbol_test)

    wandb.init(project="Asbestose", config={
        "learning-rate": learning_rate,
        "architecture": model._get_name(),
        "dataset:": path_root,
        "number of training samples": len(train_loader),
        "number of evaluation samples": len(eval_loader),
        "number of test samples": len(test_loader),
        "epochs": number_of_epochs,
        "batch size": batch_size,
        "optimizer": str(optimizer),
        "weight decay": weight_decay,
        "Augmentation": str(transform),
        "Machine": "Local"
    }, name="b={}_l={}_n={}".format(batch_size, learning_rate, number_of_epochs))

    for epoch in range(number_of_epochs):
        train_loss = train_epoch(train_loader, model, optimizer, criterion, DEVICE, logging_metrices, epoch)

        eval_loss = eval_epoch(eval_loader, model, criterion, DEVICE, logging_metrices, epoch, "eval", [], [])

    test_loss = eval_epoch(test_loader, model, criterion, DEVICE, logging_metrices, number_of_epochs, "Test", [], [])

    # save the final model
    data_path_parts = path_root.split(os.sep)
    final_path_part = data_path_parts[-1] if len(data_path_parts[-1]) > 1 else data_path_parts[-2] + os.sep
    output_dir = path_root.replace(final_path_part, "")

    # Get timestamp
    now = datetime.datetime.now()
    timestamp = now.strftime("%y-%m-%d_%H-%M-%S")

    torch.save(model.state_dict(), output_dir + f'asbestose_e{number_of_epochs}_l{learning_rate}_b{batch_size}_{timestamp}.pth')
