#%%
import mlflow
import mlflow.pytorch
import torch as th
import numpy as np
from data_preparation import CiP_DMGD
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
import mlflow.pytorch

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

# Specify MLFlow tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Directory
raw_dir = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/"  # Raw data
save_dir = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/cip_dmgd_n5/"  # Processed data, change when number of features changes

# Load test graph set
graph_data_test = CiP_DMGD(
    data_Name="qc_data_17_test",
    data_name_state="_graph_5",
    raw_dir=raw_dir,
    save_dir=save_dir,
    force_reload=False,
    verbose=False,
    oversampling=False,
    n_aggregated_features=5,  # number of aggregated features (number of features per process)
)

# for i in range(0, len(graph_data_test.graphs)):
#     node_features = (graph_data_test.__getitem__(i)[0]).ndata["h"]
#     has_nan = False
#     for key, value in node_features.items():
#         if np.isnan(value).any():
#             has_nan = True
#             print(f"NaN values found in key '{i}'.")

# if has_nan:
#     print("The dictionary contains NaN values.")
# else:
#     print("The dictionary does not contain NaN values.")

graph_data_test.graphs.pop(108)
graph_data_test.labels.pop(108)

# Test sampler
test_idx = graph_data_test.__len__()

test_sampler = SubsetRandomSampler(th.arange(test_idx))

test_dataloader = GraphDataLoader(
    graph_data_test,
    sampler=test_sampler,
    batch_size=8,
    drop_last=True,
)


def predict_with_model(model, dataloader, device):
    all_preds = []
    all_labels = []

    # Set the model to evaluation mode
    model.eval()

    for batched_graph, labels in dataloader:
        # Move tensors to the same device (GPU 'cuda:0')
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)

        with th.no_grad():
            # Assuming your model expects input in the form of batched_graph
            pred = model(batched_graph)

        all_preds.append(np.rint(th.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(labels.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    return all_preds, all_labels


def calculate_metrics(ground_truth, predictions):

    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    mcc = matthews_corrcoef(ground_truth, predictions)
    roc_auc = roc_auc_score(ground_truth, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
    }


if __name__ == "__main__":
    # Get the path to the best model from the last run
    # logged_model = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/artifacts/1/c4a640f7c7d348e4b15f6d6d191bfdef/artifacts/model/"
    # # popular-bass-664

    logged_model = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/artifacts/3/afb0e79e8d4f40f2ad3a8660ab6e7f54/artifacts/model/"

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pytorch.load_model(logged_model)

    # Make predictions on the unseen dataset
    predictions, ground_truth = predict_with_model(
        loaded_model, test_dataloader, device
    )

    # Calculate evaluation metrics
    metrics = calculate_metrics(ground_truth, predictions)

    # Log the evaluation metrics in MLflow
    # for metric_name, metric_value in metrics.items():
    #     mlflow.log_metric(metric_name, metric_value)

    # Print or log the evaluation metrics
    print("Evaluation Metrics:")
    print(metrics)

    # mlflow.end_run()

# %%
