import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


class HGCN(nn.Module):  # Heterogeneous Graph Convolutional Network
    def __init__(self, hid_feats, aggregator, feat_drop, activation):
        super().__init__()
        # Graph convolutional layers with SAGEConv or GraphConv
        self.conv1 = dglnn.HeteroGraphConv(
            {
                "BELONGS_TO": dglnn.SAGEConv(
                    in_feats=hid_feats,
                    out_feats=hid_feats,
                    aggregator_type=aggregator,
                    feat_drop=feat_drop,
                    activation=activation,
                ),
                "TRANSFERS_TO": dglnn.SAGEConv(
                    in_feats=hid_feats,
                    out_feats=hid_feats,
                    aggregator_type=aggregator,
                    feat_drop=feat_drop,
                    activation=activation,
                ),
            },
            aggregate="mean",
        )

        self.conv2 = dglnn.HeteroGraphConv(
            {
                "BELONGS_TO": dglnn.SAGEConv(
                    in_feats=hid_feats,
                    out_feats=hid_feats,
                    aggregator_type=aggregator,
                    feat_drop=feat_drop,
                    activation=activation,
                ),
                "TRANSFERS_TO": dglnn.SAGEConv(
                    in_feats=hid_feats,
                    out_feats=hid_feats,
                    aggregator_type=aggregator,
                    feat_drop=feat_drop,
                    activation=activation,
                ),
            },
            aggregate="mean",
        )

        self.conv3 = dglnn.HeteroGraphConv(
            {
                "BELONGS_TO": dglnn.SAGEConv(
                    in_feats=hid_feats,
                    out_feats=hid_feats,
                    aggregator_type=aggregator,
                    feat_drop=feat_drop,
                    activation=activation,
                ),
                "TRANSFERS_TO": dglnn.SAGEConv(
                    in_feats=hid_feats,
                    out_feats=hid_feats,
                    aggregator_type=aggregator,
                    feat_drop=feat_drop,
                    activation=activation,
                ),
            },
            aggregate="mean",
        )

        self.conv4 = dglnn.HeteroGraphConv(
            {
                "BELONGS_TO": dglnn.SAGEConv(
                    in_feats=hid_feats,
                    out_feats=hid_feats,
                    aggregator_type=aggregator,
                    feat_drop=feat_drop,
                    activation=activation,
                ),
                "TRANSFERS_TO": dglnn.SAGEConv(
                    in_feats=hid_feats,
                    out_feats=hid_feats,
                    aggregator_type=aggregator,
                    feat_drop=feat_drop,
                    activation=activation,
                ),
            },
            aggregate="mean",
        )

        # self.conv1 = dglnn.HeteroGraphConv(
        #     {
        #         "BELONGS_TO": dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats),
        #         "TRANSFERS_TO": dglnn.GraphConv(
        #             in_feats=hid_feats, out_feats=hid_feats
        #         ),
        #     },
        #     aggregate="sum",
        # )

        # self.conv2 = dglnn.HeteroGraphConv(
        #     {
        #         "BELONGS_TO": dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats),
        #         "TRANSFERS_TO": dglnn.GraphConv(
        #             in_feats=hid_feats, out_feats=hid_feats
        #         ),
        #     },
        #     aggregate="sum",
        # )

        # self.conv3 = dglnn.HeteroGraphConv(
        #     {
        #         "BELONGS_TO": dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats),
        #         "TRANSFERS_TO": dglnn.GraphConv(
        #             in_feats=hid_feats, out_feats=hid_feats
        #         ),
        #     },
        #     aggregate="sum",
        # )

        # self.conv4 = dglnn.HeteroGraphConv(
        #     {
        #         "BELONGS_TO": dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats),
        #         "TRANSFERS_TO": dglnn.GraphConv(
        #             in_feats=hid_feats, out_feats=hid_feats
        #         ),
        #     },
        #     aggregate="sum",
        # )

    def forward(self, g, h):
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        h = self.conv3(g, h)
        # h = self.conv4(g, h)

        return h


class HeteroClassifier(nn.Module):
    def __init__(self, hid_feats, out_feats, aggregator, feat_drop, activation):
        super().__init__()
        # Node feature size
        n_processes = 14
        n_aggregated_features = 1  # number of features for each aggregated feature

        self.linear1 = dglnn.HeteroLinear(
            {
                "Process": n_processes,
                "Aggregated_Feature": n_aggregated_features,
            },
            hid_feats,
        )
        self.linear2 = dglnn.HeteroLinear(
            {
                "Process": hid_feats,
                "Aggregated_Feature": hid_feats,
            },
            out_feats,
        )

        self.hgcn = HGCN(hid_feats, aggregator, feat_drop, activation)

        self.linear = nn.Linear(2, 1)  # 2 = number of node classes

    def forward(self, g):
        # TO DO: compare g.ndata["h"] with g.data["h"]
        with g.local_scope():
            g.ndata["h"] = self.linear1(g.ndata["h"])
            g.ndata["h"] = self.hgcn(g, g.ndata["h"])
            g.ndata["h"] = self.linear2(g.ndata["h"])

            # Readout with sum nodes
            hg = (
                th.stack(
                    (
                        dgl.readout_nodes(g, "h", ntype="Process"),
                        dgl.readout_nodes(g, "h", ntype="Aggregated_Feature"),
                    ),
                    -1,
                )
                .squeeze(dim=1)
                .to(device)
            )

            # hg = (th.Tensor(dgl.readout_nodes(g, 'h', ntype='Station'))).to(device)

            output = self.linear(hg)

            return output
