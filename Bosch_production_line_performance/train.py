# %% imports
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
)

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

import mlflow.pytorch

import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.dataloading import GraphDataLoader
from dgl.data import utils
from data_preparation import MyDataset
from model import HeteroClassifier

from tqdm import tqdm

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

th.manual_seed(42)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Specify MLFlow tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Data preparation
raw_dir = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/bosch_production_line_performance/"  # Raw data
save_dir = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/bosch_production_line_performance/processed_data"  # Processed data

# Load train set (oversampled)
graph_data = MyDataset(
    data_Name="product_8463_train_new",
    data_name_state="_new_graph_dataset",
    chunksize=4000,
    break_early=False,
    raw_dir=raw_dir,
    save_dir=save_dir,
    force_reload=False,
    verbose=False,
    oversampling=False,
)

# Load test set
graph_data_test = MyDataset(
    data_Name="product_8463_test_new",
    data_name_state="_new_graph_dataset",
    chunksize=4000,
    break_early=False,
    raw_dir=raw_dir,
    save_dir=save_dir,
    force_reload=False,
    verbose=False,
    oversampling=False,
)


def calculate_metrics(y_true, y_pred, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"MCC: {mcc}")
    print(f"F1 Score: {f1}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    mlflow.log_metric(key=f"MCC-{type}", value=float(mcc), step=epoch)
    mlflow.log_metric(key=f"F1 Score-{type}", value=float(f1), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")


def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap="Blues", fmt="g")
    cfm_plot.figure.savefig(f"data/images/cm_{epoch}.png")
    mlflow.log_artifact(f"data/images/cm_{epoch}.png")


# Training
# Returns the total number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(epoch, model, train_dataloader, optimizer, loss_fn, batch_size):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batched_graph, labels in tqdm(train_dataloader):
        # Use GPU
        batched_graph = batched_graph.to(device)
        # labels = labels.type(th.LongTensor)
        labels = labels.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batched_graph)
        # print(pred)
        # print(labels)
        # Calculating the loss and gradients
        loss = loss_fn(
            th.reshape(pred, [1, batch_size]), labels.unsqueeze(0).float()
        )  # th.squeeze(pred, dim=1)
        # print(loss)
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(th.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(labels.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    # all_preds = np.argmax(all_preds, axis=1)
    print(all_preds)
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_labels, all_preds, epoch, "train")
    log_conf_matrix(all_labels, all_preds, epoch)
    return running_loss / step


# Testing
def test(epoch, model, test_dataloader, loss_fn, batch_size):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batched_graph, labels in tqdm(test_dataloader):
        batched_graph = batched_graph.to(device)
        # labels = labels.type(th.LongTensor)
        labels = labels.to(device)
        pred = model(batched_graph)
        loss = loss_fn(
            th.reshape(pred, [1, batch_size]), labels.unsqueeze(0).float()
        )  # th.squeeze(pred, dim=1)
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(th.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(pred.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    # all_preds = np.argmax(all_preds, axis=0)
    all_labels = np.concatenate(all_labels).ravel()
    # print(all_preds_raw[0])
    # print(all_preds)
    # print(all_labels)
    calculate_metrics(all_labels, all_preds, epoch, "test")
    log_conf_matrix(all_labels, all_preds, epoch)

    return running_loss / step


# %% Run the training
from mango import scheduler, Tuner
from config import HYPERPARAMETERS, BEST_PARAMETERS, SIGNATURE


def run_one_training(params):
    params = params[0]
    print(params)
    with mlflow.start_run() as run:
        # Log parameters used in this experiment
        for key in params.keys():
            mlflow.log_param(key, params[key])

        # Prepare training
        # Train sampler
        class_ratio = sum(graph_data.labels) / graph_data.__len__()
        train_sample_weights = np.zeros_like(graph_data.labels, dtype=np.float32)
        train_sample_weights[np.array(graph_data.labels) == 0] = class_ratio
        train_sample_weights[np.array(graph_data.labels) == 1] = 1 - class_ratio
        train_sample_weights = np.squeeze(train_sample_weights)
        train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=graph_data.__len__(),
            replacement=True,
        )

        train_dataloader = GraphDataLoader(
            graph_data,
            sampler=train_sampler,
            batch_size=params["batch_size"],
            drop_last=True,
        )

        # Test sampler
        test_idx = graph_data_test.__len__()

        test_sampler = SubsetRandomSampler(th.arange(test_idx))

        test_dataloader = GraphDataLoader(
            graph_data_test,
            sampler=test_sampler,
            batch_size=params["batch_size"],
            drop_last=True,
        )

        # Loading the model
        print("Loading model...")
        # model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = HeteroClassifier(
            hid_feats=params["hid_feats"],
            out_feats=params["out_feats"],
            aggregator=params["aggregator"],
            feat_drop=params["feat_drop"],
            activation=params["activation"],
        )
        model = model.to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        # < 1 increases precision, > 1 recall
        weight = th.tensor([params["pos_weight"]], dtype=th.float32).to(device)
        loss_fn = th.nn.BCEWithLogitsLoss()  # pos_weight=weight
        # optimizer = th.optim.SGD(model.parameters(),
        #     lr=params["learning_rate"],
        #     momentum=params["sgd_momentum"],
        #     weight_decay=params["weight_decay"]
        # )
        optimizer = th.optim.Adam(model.parameters(), lr=params["learning_rate"])
        scheduler = th.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=params["scheduler_gamma"]
        )

        # Start training
        best_loss = 1000
        early_stopping_counter = 0
        for epoch in range(30):
            if early_stopping_counter <= 10:  # = x * 5
                # Training
                print(epoch)
                model.train()
                loss = train_one_epoch(
                    epoch,
                    model,
                    train_dataloader,
                    optimizer,
                    loss_fn,
                    batch_size=params["batch_size"],
                )
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

                # Testing
                model.eval()
                if epoch % 5 == 0:
                    loss = test(
                        epoch,
                        model,
                        test_dataloader,
                        loss_fn,
                        batch_size=params["batch_size"],
                    )
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)

                    # Update best loss
                    if float(loss) < best_loss:
                        best_loss = loss
                        # Save the currently best model
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                return [best_loss]
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]


# %% Hyperparameter search
import traceback

try:
    print("Running hyperparameter search...")
    config = dict()
    config["optimizer"] = "Random"
    config[
        "num_iteration"
    ] = 0  # Hyperparameter tunning not being conducted for this experiment

    tuner = Tuner(HYPERPARAMETERS, objective=run_one_training, conf_dict=config)
    results = tuner.minimize()
except:
    traceback.print_exc()

# %%
