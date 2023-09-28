import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import torch.nn as nn
import torch.nn.functional as F


HYPERPARAMETERS = {
    "batch_size": [24, 32, 64, 128, 256],
    "learning_rate": [0.00001, 0.0001, 0.001, 0.01, 0, 1],
    "weight_decay": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001],
    "sgd_momentum": [0.9],
    "scheduler_gamma": [0.995],
    "pos_weight": [1.0],
    "hid_feats": [4, 8, 16, 128, 256],
    "out_feats": [1],
    "aggregator": ["mean", "gcn", "pool", "lstm"],
    "feat_drop": [0.3],
    "activation": [F.relu],
}

BEST_PARAMETERS = {
    "batch_size": [24],
    "learning_rate": [0.00001],
    "weight_decay": [0.0001],
    "sgd_momentum": [0.8],
    "scheduler_gamma": [0.8],
    "pos_weight": [1.3],
}

input_schema = Schema(
    [
        TensorSpec(np.dtype(np.float32), (-1, 1), name="Line"),
        TensorSpec(np.dtype(np.float32), (-1, 1), name="Station"),
        TensorSpec(np.dtype(np.float32), (-1, 1), name="Feature"),
    ]
)

output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])

SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)
