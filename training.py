import time

import numpy as np
import torch
import wandb
import pylab
from torch import optim, nn
from torch.amp import GradScaler
from torchmetrics import AUROC, Accuracy, Precision, Specificity, ROC

from monai.networks.nets import resnet101

import utils
from Asbestosis_Dataset import Asbestosis_Dataset

CPU = "cpu"
GPU = "cuda"


def train_epoch(data_loader, model, optimizer, criterion, DEVICE, logging_metrices, epoch_number=0):
    start = time.time()
    model.train()
    torch.set_grad_enabled(True)
    epoch_loss = 0
    for data, ground_truth, path in data_loader:


        data = data.to(DEVICE)
        ground_truth = ground_truth.to(DEVICE)
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


def eval_epoch(data_loader, model, criterion, DEVICE, logging_metrices, suffix: str, epoch_number, label, prediction):
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
            fig, ax = pylab.subplots()
            pylab.plot(fpr, tpr, 'b', label="AUC = %0.3f" % auc)
            pylab.legend(loc='lower right')
            pylab.plot()
            pylab.plot([0, 1], [0, 1], 'r--')
            pylab.xlim([0, 1])
            pylab.ylim([0, 1])
            pylab.ylabel('True Positive Rate')
            pylab.xlabel('False Positive Rate')
            pylab.title('Receiver Operating Characteristic - Test ')
            pylab.show()
            best_index = np.argmax(tpr - fpr)
            wandb.log({
                "ROC/Test": fig,
                "Specificity (Youden)/Test": 1 - fpr[best_index],
                "Sensitivity (Youden)/Test": tpr[best_index]
            })
            results['Spec [i]'] = fpr[best_index]
            results['Sens [i]'] = tpr[best_index]
            results['i'] = best_index
            pylab.close(fig)

        metric.reset()
    wandb.log(wandb_dict)
    return avg_epoch_loss, results


if __name__ == "__main__":
    path_root = "/home/debora/Documents/Data/DeboraThorax/anon/"

    torch.manual_seed(31)
    DEVICE = GPU
    if not torch.cuda.is_available():
        DEVICE = CPU

    # evaluation metrics:
    auc_roc = nn.ModuleDict(
        {state: AUROC(**{"task": "binary"}).to(DEVICE) for state in ["train_", "eval_", "Test_"]})
    roc = nn.ModuleDict({state: ROC(**{"task": "binary"}).to(DEVICE) for state in ["Test_"]})
    acc = nn.ModuleDict(
        {state: Accuracy(**{"task": "binary"}).to(DEVICE) for state in ["train_", "eval_", "Test_"]})
    sens = nn.ModuleDict(
        {state: Precision(**{"task": "binary"}).to(DEVICE) for state in ["train_", "eval_", "Test_"]})
    spec = nn.ModuleDict(
        {state: Specificity(**{"task": "binary"}).to(DEVICE) for state in ["train_", "eval_", "Test_"]})

    model = resnet101(pretrained=False, spatial_dims=2)
    lung_train, pleura_train, _, lung_test, pleura_test, _ = utils.get_feature_tensor("/home/debora/Documents/Projects/Thorax/found_merged_data.xlsx", train_fraction=0.7)
    lung_dataset = Asbestosis_Dataset(path_root, lung_train)

