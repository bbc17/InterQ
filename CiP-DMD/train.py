#%%
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
)
import torch as th
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import mlflow.pytorch
from config import train_test_size
from dgl.dataloading import GraphDataLoader
from data_preparation import CiP_DMGD, Dataset
from model import HeteroClassifier
from tqdm import tqdm
import warnings
import logging

# Set logging level to DEBUG for the mlflow logger
logging.getLogger("mlflow").setLevel(logging.DEBUG)

warnings.filterwarnings("ignore", category=FutureWarning)

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

random.seed(42)
th.manual_seed(42)

# Specify MLFlow tracking server
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "GNN_n_features_5_17"

# Directories
raw_dir = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/"  # Raw data
save_dir = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/cip_dmgd_n5/"  # Processed data

# Load graph dataset
graph_data = CiP_DMGD(
    data_Name="qc_data_labels",
    data_name_state="_graph_5",
    raw_dir=raw_dir,
    save_dir=save_dir,
    force_reload=False,
    verbose=False,
    oversampling=False,
    n_aggregated_features=5,
)
graph_data.graphs.pop(414)
graph_data.labels.pop(414)

graph_data.graphs.pop(522)
graph_data.labels.pop(522)

graph_data.graphs.pop(821)
graph_data.labels.pop(821)

# Create train and test sets based on the original graph dataset and using the class Dataset
all_graphs = graph_data.graphs
all_labels = graph_data.labels

ind_list = np.arange(len(all_graphs))
np.random.shuffle(ind_list)
train_size = np.floor(train_test_size * len(all_graphs)).astype(np.int32)
train_inds = ind_list[:train_size]
test_inds = ind_list[train_size:]

train_graphs = [graph for i, graph in enumerate(all_graphs) if i in train_inds]
test_graphs = [graph for i, graph in enumerate(all_graphs) if i not in train_inds]

train_labels = [graph for i, graph in enumerate(all_labels) if i in train_inds]
test_labels = [graph for i, graph in enumerate(all_labels) if i not in train_inds]

train_dataset = Dataset(graphs=train_graphs, labels=train_labels)
test_dataset = Dataset(graphs=test_graphs, labels=test_labels)


def calculate_metrics(y_true, y_pred, epoch, type):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    mlflow.log_metric(key=f"MCC-{type}", value=float(mcc), step=epoch)
    mlflow.log_metric(key=f"F1 Score-{type}", value=float(f1), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)


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
    for batched_graph, labels in train_dataloader:
        # Use GPU
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Passing the node features and the connection info
        pred = model(batched_graph)

        # Calculating the loss and gradients
        loss = loss_fn(th.reshape(pred, [1, batch_size]), labels.unsqueeze(0).float())
        loss.backward()
        optimizer.step()

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(th.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(labels.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_labels, all_preds, epoch, "train")
    return running_loss / step


# Testing
def test(epoch, model, test_dataloader, loss_fn, batch_size):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batched_graph, labels in test_dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        pred = model(batched_graph)
        loss = loss_fn(th.reshape(pred, [1, batch_size]), labels.unsqueeze(0).float())

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(th.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(pred.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_labels, all_preds, epoch, "test")
    return running_loss / step


# Hyperparameter tunning
from mango import scheduler, Tuner
from config import HYPERPARAMETERS, SIGNATURE
import json


def run_one_training(params):
    params = params[0]

    # Start the run in the specified experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(nested=True) as run:
        # Log parameters used in this experiment
        for key in params.keys():
            mlflow.log_param(key, params[key])

        # Prepare training
        # Train sampler
        # Calculate the class weights based on the number of positive (label=1) and negative (label=0) samples
        num_positive_samples = sum(train_dataset.labels)
        num_negative_samples = len(train_dataset.labels) - num_positive_samples

        # Calculate class weights inversely proportional to class frequencies
        class_weight_positive = 1.0 / num_positive_samples
        class_weight_negative = 1.0 / num_negative_samples

        # Convert class weights to tensors
        class_weight_positive_tensor = th.tensor(class_weight_positive)
        class_weight_negative_tensor = th.tensor(class_weight_negative)

        # Create a tensor of labels
        labels_tensor = th.tensor(train_dataset.labels)

        # Use th.where() to calculate sample weights
        sample_weights = th.where(
            labels_tensor == 1,
            class_weight_positive_tensor,
            class_weight_negative_tensor,
        )

        # Convert the sample weights to a NumPy array
        sample_weights = sample_weights.numpy()

        # Create a WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=th.Generator().manual_seed(42),  # Random seed for reproducibility
        )

        train_dataloader = GraphDataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=params["batch_size"],
            drop_last=True,
        )

        # Test sampler
        test_idx = test_dataset.__len__()
        test_sampler = SubsetRandomSampler(th.arange(test_idx))

        test_dataloader = GraphDataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=params["batch_size"],
            drop_last=True,
        )

        # Loading the model
        # print("Loading model...")
        # model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = HeteroClassifier(
            n_processes=params["n_processes"],
            n_aggregated_features=params["n_aggregated_features"],
            hid_feats=params["hid_feats"],
            out_feats=params["out_feats"],
            aggregator=params["aggregator"],
            feat_drop=params["feat_drop"],
            activation=params["activation"],
            num_layers=params["num_layers"],
            conv_type=params["conv_type"],
        )
        model = model.to(device)
        # print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        loss_fn = th.nn.BCEWithLogitsLoss()

        optimizer = th.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        scheduler = th.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=params["scheduler_gamma"]
        )

        # Start training
        best_loss = 1000
        early_stopping_counter = 0
        for epoch in range(100):
            if early_stopping_counter <= 10:
                # Training
                # print(epoch)
                model.train()
                loss = train_one_epoch(
                    epoch,
                    model,
                    train_dataloader,
                    optimizer,
                    loss_fn,
                    batch_size=params["batch_size"],
                )
                # print(f"Epoch {epoch} | Train Loss {loss}")
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
                    # print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)

                    # Update best loss
                    if float(loss) < best_loss:
                        best_loss = loss
                        # Save the currently best model
                        mlflow.pytorch.log_model(model, "model")  # signature=SIGNATURE)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                # mlflow.end_run()
                return [best_loss]
    # print(f"Finishing training with best test loss: {best_loss}")
    # Return the path to the best model
    # best_model_uri = mlflow.get_artifact_uri("best_model")
    # print(best_model_uri)

    return [best_loss]


# Hyperparameter search
def main():
    print("Running hyperparameter search...")
    config = dict()
    config["optimizer"] = "Random"
    config["num_iteration"] = 20

    tuner = Tuner(HYPERPARAMETERS, objective=run_one_training, conf_dict=config)
    results = tuner.minimize()

    # Find the best hyperparameters and specify the folder to save them
    best_hyperparameters = results["best_params"]

    # Filter out non-serializable objects from best_hyperparameters
    serializable_hyperparameters = {
        key: value
        for key, value in best_hyperparameters.items()
        if isinstance(value, (int, float, str, list, dict, bool, type(None)))
    }

    file_path = os.path.join(save_dir, "best_hyperparameters.json")

    with open(file_path, "w") as config_file:
        json.dump(serializable_hyperparameters, config_file)


if __name__ == "__main__":
    main()

# %%
