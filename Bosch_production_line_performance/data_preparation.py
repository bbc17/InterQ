import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

import random

import dgl
import torch as th
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs


def edges_and_nodes_from_DF(df, with_chunk=True):
    if with_chunk == True:
        df_short = df.dropna(axis=0).reset_index(level=0)
    elif with_chunk == False:
        df_short = df.dropna(axis=1).T.reset_index(level=0)

    Graph_ID = df_short.iloc[0, 1]
    Response = df_short.iloc[df_short.shape[0] - 1, 1]
    df_short = df_short.drop(index=[0], axis=0).reset_index(drop=True)[:-1]

    df_split = df_short["index"].str.split("_", 2, expand=True)
    df_split = df_split.join(df_short, lsuffix="_caller", rsuffix="_other").drop(
        ["index"], axis=1
    )
    df_split = df_split.rename(
        columns={
            df_split.columns[0]: "Line",
            df_split.columns[1]: "Station",
            df_split.columns[2]: "Feature",
            df_split.columns[3]: "Feature_Value",
        }
    )

    ################################################
    # Nodes dataframes
    Lines = df_split["Line"].unique()  # [:-1]
    ID_Lines = np.arange(0, len(Lines))

    Stations = df_split["Station"].unique()
    ID_Stations = np.arange(0, len(Stations))

    Features = df_split["Feature"].unique()
    ID_Features = np.arange(0, len(Features))
    Value_Features = df_split["Feature_Value"]

    df_nodes_L = pd.DataFrame(data={"ID_L": ID_Lines, "Line": Lines})
    df_nodes_S = pd.DataFrame(data={"ID_S": ID_Stations, "Station": Stations})
    df_nodes_F = pd.DataFrame(
        data={"ID_F": ID_Features, "Feature": Features, "Value": Value_Features}
    )

    ################################################
    # Edges dataframes
    df_edges = (
        df_split.set_index("Line")
        .join(df_nodes_L.set_index("Line"))
        .reset_index()
        .drop("Feature_Value", axis=1)
    )
    df_edges = (
        df_edges.set_index("Station")
        .join(df_nodes_S.set_index("Station"))
        .reset_index()
    )
    df_edges = (
        df_edges.set_index("Feature")
        .join(df_nodes_F.set_index("Feature"))
        .reset_index()
    )
    df_edges = df_edges.sort_values(by=["ID_F"]).reset_index(drop=True)

    df_edges_IDs = df_edges.drop(["Feature", "Station", "Line", "Value"], axis=1)

    df_edges_LtoS = (
        df_edges_IDs.drop_duplicates(subset=["ID_S"])
        .drop("ID_F", axis=1)
        .reset_index(drop=True)
        .rename(columns={"ID_L": "From_ID_L", "ID_S": "To_ID_S"})[:-1]
    )
    df_edges_StoF = (
        df_edges_IDs.drop("ID_L", axis=1)
        .reset_index(drop=True)
        .rename(columns={"ID_S": "From_ID_S", "ID_F": "To_ID_F"})
    )
    df_edges_StoS = pd.DataFrame(
        data={
            "From_ID_S": df_edges_IDs["ID_S"].unique()[:-1],
            "To_ID_S": df_edges_IDs["ID_S"].unique()[1:],
        }
    )

    edges_and_nodes_dict = {
        1: df_nodes_L,
        2: df_nodes_S,
        3: df_nodes_F,
        4: df_edges_LtoS,
        5: df_edges_StoF,
        6: df_edges_StoS,
    }

    return edges_and_nodes_dict, Response, Lines, Stations, Features


def Load_Data_to_Graph(
    edges_and_nodes_dict,
    line_nodes,
    station_nodes,
    feature_nodes,
    all_lines,
    all_stations,
    all_features,
):
    # Implement all edges
    graph_data = {
        ("Station", "TRANSFERS_TO", "Station"): (
            th.tensor(edges_and_nodes_dict[6]["From_ID_S"]),
            th.tensor(edges_and_nodes_dict[6]["To_ID_S"]),
        ),
        ("Station", "HAS_FEATURE", "Feature"): (
            th.tensor(edges_and_nodes_dict[5]["From_ID_S"]),
            th.tensor(edges_and_nodes_dict[5]["To_ID_F"]),
        ),
    }

    g = dgl.heterograph(graph_data)

    # Implement all features of all nodes
    g.nodes["Station"].data["h"] = th.zeros(len(station_nodes), len(all_stations))

    g.nodes["Feature"].data["h"] = th.zeros(len(feature_nodes), len(all_features))

    for i, station_node in enumerate(station_nodes):
        g.nodes["Station"].data["h"][i][np.where(all_stations == station_node)[0]] = 1

    for i, feature_node in enumerate(feature_nodes):
        g.nodes["Feature"].data["h"][i][
            np.where(all_features == feature_node)[0]
        ] = edges_and_nodes_dict[3]["Value"][i]

    # for i in range(len(feature_nodes)):
    #     g.nodes["Station"].data["h"][i][
    #         edges_and_nodes_dict[2]["Station"].str.replace(r"\D", "").astype(int)
    #     ] = th.tensor(edges_and_nodes_dict[3]["Value"], dtype=th.float32)

    ###
    # g.nodes["Line"].data["h"] = th.tensor(
    #     edges_and_nodes_dict[1]["Line"].str.replace(r"\D", "").astype(int),
    #     dtype=th.float32,
    # ).reshape(-1, 1)

    # g.nodes["Station"].data["h"] = th.tensor(
    #     edges_and_nodes_dict[2]["Station"].str.replace(r"\D", "").astype(int),
    #     dtype=th.float32,
    # ).reshape(-1, 1)

    # g.nodes["Feature"].data["h"] = th.tensor(
    #     edges_and_nodes_dict[3]["Feature"].str.replace(r"\D", "").astype(int)
    # ).reshape(-1, 1)

    # g.nodes["Feature"].data["h"] = th.tensor(
    #     edges_and_nodes_dict[3]["Value"], dtype=th.float32
    # ).reshape(-1, 1)

    return g


class MyDataset(DGLDataset):
    """Template for customizing graph datasets in DGL.

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
        force_reload=False,
        verbose=False,
        chunksize=None,
        break_early=False,  # Interrupts after processing first chunck
        oversampling=False,  # Overwrites existing file
    ):
        self.chunksize = chunksize
        self.break_early = break_early
        self.data_name_state = data_name_state
        super(MyDataset, self).__init__(
            name=data_Name,
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
        # download raw data to local disk
        pass

    def process(self):
        """process raw data to graphs and labels by using DGL_functions and iterating through chunks"""
        self.graphs = []
        self.labels = []
        ctr = 0

        num_chunks = (
            sum(1 for row in open(self.raw_dir + self.name + ".csv", "r"))
            // self.chunksize
        )

        df = pd.read_csv(self.raw_dir + self.name + ".csv", chunksize=self.chunksize)

        df_header = pd.read_csv(self.raw_dir + self.name + ".csv", nrows=1).drop(
            columns=["Id", "Response"]
        )

        all_lines = np.unique(
            [df_header.columns[i].split("_")[0] for i in range(len(df_header.columns))]
        )
        all_stations = np.unique(
            [df_header.columns[i].split("_")[1] for i in range(len(df_header.columns))]
        )
        all_features = np.unique(
            [df_header.columns[i].split("_")[2] for i in range(len(df_header.columns))]
        )

        for ii, data in tqdm(enumerate(df), total=num_chunks):
            print("aktuell wurden", ctr, "Graphen erstellt")
            for jj in range(0, len(data)):
                df_graph = data.iloc[jj]

                try:
                    (
                        edges_and_nodes_dict,
                        Response,
                        line_nodes,
                        station_nodes,
                        feature_nodes,
                    ) = edges_and_nodes_from_DF(df_graph, with_chunk=True)

                    df_graph = Load_Data_to_Graph(
                        edges_and_nodes_dict,
                        line_nodes,
                        station_nodes,
                        feature_nodes,
                        all_lines,
                        all_stations,
                        all_features,
                    )
                    self.graphs.append(df_graph)
                    self.labels.append(Response)
                    ctr += 1
                except:
                    print(
                        "Graph number", jj + ii * self.chunksize, "has only NaN-values"
                    )

            if self.break_early:
                print("In total, ", ctr, "graphs were created")
                break

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
        """save graphs and labels"""
        graph_path = os.path.join(
            self.save_path, self.name + self.data_name_state + ".bin"
        )
        save_graphs(graph_path, self.graphs, {"labels": th.tensor(self.labels)})
        # Save additional information in python dict
        # info_path = os.path.join(self.save_path, self.name + '_info.pkl')
        # save_info(info_path, {'num_labels': self.num_labels})

    def load(self):
        """load processed data from directory `self.save_path`"""
        graph_path = os.path.join(
            self.save_path, self.name + self.data_name_state + ".bin"
        )
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict["labels"].detach().numpy().tolist()
        # info_path = os.path.join(self.save_path, self.name + '_info.pkl')
        # self.num_labels = load_info(info_path)['num_labels']

    def has_cache(self):
        """check whether there are processed data in `self.save_path`"""
        try:
            os.path.join(self.save_path, self.name + self.data_name_state + ".bin")
            return True
        except:
            return False

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 1
