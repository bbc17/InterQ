import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


class HGCN(nn.Module):  # Heterogeneous Graph Convolutional Network
    def __init__(self, hid_feats, out_feats, aggregator, feat_drop, activation):
        super().__init__()

        n_stations = 8
        n_features = 210

        # Graph convolutional layers with GraphConv
        self.conv1 = dglnn.HeteroGraphConv(
            {
                "HAS_FEATURE": dglnn.SAGEConv(
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
            aggregate="sum",
        )

        self.conv2 = dglnn.HeteroGraphConv(
            {
                "HAS_FEATURE": dglnn.SAGEConv(
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
            aggregate="sum",
        )

        self.conv3 = dglnn.HeteroGraphConv(
            {
                "HAS_FEATURE": dglnn.SAGEConv(
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
            aggregate="sum",
        )

    def forward(self, g, h):

        h = self.conv1(g, h)
        h = self.conv2(g, h)
        h = self.conv3(g, h)

        return h


class HeteroClassifier(nn.Module):
    def __init__(self, hid_feats, out_feats, aggregator, feat_drop, activation):
        super().__init__()

        n_stations = 8
        n_features = 210

        self.linear1 = dglnn.HeteroLinear(
            {"Station": n_stations, "Feature": n_features}, hid_feats
        )
        self.linear2 = dglnn.HeteroLinear(
            {"Station": hid_feats, "Feature": hid_feats}, out_feats
        )

        self.hgcn = HGCN(hid_feats, out_feats, aggregator, feat_drop, activation)

        self.linear = nn.Linear(2, 1)

    def forward(self, g):
        with g.local_scope():

            g.ndata["h"] = self.linear1(g.ndata["h"])
            g.ndata["h"] = self.hgcn(g, g.ndata["h"])
            g.ndata["h"] = self.linear2(g.ndata["h"])

            # Readout with sum nodes
            hg = (
                th.stack(
                    (
                        dgl.readout_nodes(g, "h", ntype="Station"),
                        dgl.readout_nodes(g, "h", ntype="Feature"),
                    ),
                    -1,
                )
                .squeeze(dim=1)
                .to(device)
            )

            # hg = (th.Tensor(dgl.readout_nodes(g, 'h', ntype='Station'))).to(device)

            output = self.linear(hg)

            return output
