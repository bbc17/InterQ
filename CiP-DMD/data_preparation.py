import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch as th
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs

from process.sawing import SawingProcessData
from process.milling_subprocesses import MillingProcessData

from config import Processes


def edges_and_nodes_from_df(Processes, Aggregated_Features):
    # Creates separate dataframes for each node and edge types.
    # Input:
    # Processes = list of processes

    ################################################
    # Node types dataframes

    # Processes
    df_processes = pd.DataFrame(
        {
            "Index": np.arange(0, len(Processes)),
            "Processes": Processes,
            "Values": [
                [1 if j == i else 0 for j in range(len(Processes))]
                for i in range(len(Processes))
            ],  # creates an array of one-hot-encoded processes
        }
    )

    # Aggregated features
    # The aggregated features repeat for each process.
    # The IDs are updated, but the number of aggregated features remain the same.
    df_aggregated_features = pd.DataFrame(
        {
            "Index": np.arange(0, len(Processes) * len(Aggregated_Features)),
            "Aggregated_Features": np.arange(
                0, len(Processes) * len(Aggregated_Features)
            ),
            "Values": [
                [1] for i in range(len(Processes) * len(Aggregated_Features))
            ],  # creates an array of 1s
        }
    )

    ################################################
    # Edge types dataframes

    # Aggregated Feature to Process
    num_processes = np.arange(0, len(Processes))
    processes_ids = [
        item for item in num_processes for _ in range(len(Aggregated_Features))
    ]  # repeats the process ID according to the number of aggregated features to create the edges between aggreated feature nodes and process nodes.

    df_edges_AFtoP = pd.DataFrame(
        {
            "From_ID_AF": np.arange(0, len(Processes) * len(Aggregated_Features)),
            "To_ID_P": processes_ids,
        }
    )

    # Process to Process
    df_edges_PtoP = pd.DataFrame(
        {
            "From_ID_P": np.arange(0, len(Processes) - 1),
            "To_ID_P": np.arange(1, len(Processes)),
        }
    )

    edges_and_nodes_dict = {
        1: df_processes,
        2: df_aggregated_features,
        3: df_edges_AFtoP,
        4: df_edges_PtoP,
    }

    return edges_and_nodes_dict


def load_data_to_graph(edges_and_nodes_dict, path_data, part_ID):
    # Create graph edges
    graph_data = {
        # Aggregated Feature to Process
        ("Aggregated_Feature", "BELONGS_TO", "Process"): (
            th.tensor(edges_and_nodes_dict[3]["From_ID_AF"]),
            th.tensor(edges_and_nodes_dict[3]["To_ID_P"]),
        ),
        # Process to Process
        ("Process", "TRANSFERS_TO", "Process"): (
            th.tensor(edges_and_nodes_dict[4]["From_ID_P"]),
            th.tensor(edges_and_nodes_dict[4]["To_ID_P"]),
        ),
    }

    g = dgl.heterograph(graph_data)

    # Implement node features
    # Process values
    g.nodes["Process"].data["h"] = th.tensor(
        edges_and_nodes_dict[1]["Values"], dtype=th.float32
    )

    # Aggregated Feature values
    g.nodes["Aggregated_Feature"].data["h"] = th.tensor(
        edges_and_nodes_dict[2]["Values"], dtype=th.float32
    )
    # Get sawing process features
    reader = SawingProcessData(path_data=path_data)
    saw_features_raw = reader.get_process_QH_id(part_ID)
    saw_features = np.array(saw_features_raw).reshape(-1)

    # Get milling process features
    reader = MillingProcessData(path_data=path_data + "/milling_raw_data/")
    milling_features = np.array(reader.get_process_QH_id(part_ID))

    aggregated_feature_values = th.from_numpy(
        np.concatenate((saw_features, milling_features))
    )

    # Perform element-wise multiplication
    g.nodes["Aggregated_Feature"].data["h"] = (
        g.nodes["Aggregated_Feature"].data["h"]
        * aggregated_feature_values[:, np.newaxis]
    )

    g.nodes["Aggregated_Feature"].data["h"] = (
        g.nodes["Aggregated_Feature"]
        .data["h"]
        .clone()
        .detach()
        .requires_grad_(True)
        .to(th.float32)
    )

    return g


class CiP_DMGD(DGLDataset):  # CiP - Discrete Manufacturing Graph Dataset
    """

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    def __init__(
        self,
        data_Name,
        data_name_state="_raw",
        url=None,
        raw_dir=None,
        save_dir=None,
        force_reload=False,  # Overwrites existing dataset
        verbose=False,
        oversampling=False,
        n_aggregated_features=None,
    ):
        self.data_name_state = data_name_state
        self.n_aggregated_features = n_aggregated_features

        super(CiP_DMGD, self).__init__(
            name=data_Name,  # Super passes all arguments in () to DGLDataset class
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

        if oversampling == True:
            self.oversampling()
            self.data_name_state = "_oversampled"
            self.save()

    def download(self):
        # Download raw data to local disk
        pass

    def process(self):
        """Process raw data to graphs and labels by using DGL_functions"""
        self.graphs = []
        self.part_IDs = []
        self.labels = []
        ctr = 0
        labels_df = pd.read_csv(self.raw_dir + self.name + ".csv", sep=";")
        part_IDs = labels_df.iloc[:, 0].tolist()
        response = labels_df.iloc[:, 1].tolist()  # surface_roughness_label

        # Prepare list with names based on the number of aggregated features
        Aggregated_Features = []

        for i in range(self.n_aggregated_features):
            Aggregated_Features.append(f"Aggregated_Feature_{i}")

        for id in tqdm(range(0, len(part_IDs))):
            part_ID = part_IDs[id]
            print(part_ID)

            try:
                edges_and_nodes_dict = edges_and_nodes_from_df(
                    Processes,
                    Aggregated_Features,
                )
                graph = load_data_to_graph(
                    edges_and_nodes_dict, self.raw_dir, str(part_ID)
                )
                self.graphs.append(graph)
                self.labels.append(response[id])
                ctr += 1
            except:
                print("Graph number", id, part_ID, "has nan-values.")

        print(ctr, " graphs were created.")

    def oversampling(self):
        graphs = self.graphs
        labels = self.labels

        # Check how many additional samples are needed
        pos_class = sum(labels)
        neg_class = len(labels) - pos_class

        multiplier = int(neg_class / pos_class) - 1

        # Replicate the dataset for the positive class
        true_graphs, true_labels = self._get_true_graphs()

        self._append_graph_data(true_graphs, true_labels, multiplier)

    def _get_true_graphs(self):
        """
        :param
        :return:
        """
        true_graph_data = []
        true_label_data = []
        label_list = self.labels
        graph_list = self.graphs

        for i in range(0, len(label_list)):
            if label_list[i] == 1:
                true_graph = graph_list[i]
                true_list = label_list[i]
                true_graph_data.append(true_graph)
                true_label_data.append(true_list)
        return true_graph_data, true_label_data

    def _append_graph_data(
        self, graphs_to_append, labels_to_append, oversample_multiplier
    ):
        for j in range(0, oversample_multiplier):
            for i in range(0, len(labels_to_append)):
                self.graphs.append(graphs_to_append[i])
                self.labels.append(1)

    def __getitem__(self, idx):
        """Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    def save(self):
        """Save graphs and labels"""
        graph_path = os.path.join(
            self.save_path, self.name + self.data_name_state + ".bin"
        )
        save_graphs(graph_path, self.graphs, {"labels": th.tensor(self.labels)})

    def load(self):
        """Load processed data from directory `self.save_path`"""
        graph_path = os.path.join(
            self.save_path, self.name + self.data_name_state + ".bin"
        )
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict["labels"].detach().numpy().tolist()

    def has_cache(self):
        """Check whether there are processed data in `self.save_path`"""
        try:
            os.path.join(self.save_path, self.name + self.data_name_state + ".bin")
            return True
        except:
            return False

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 1


class Dataset(
    CiP_DMGD
):  # class for creating train and test sets based on the original graph dataset
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
