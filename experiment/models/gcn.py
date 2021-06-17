import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, args, activation):
        super().__init__()
        self.activation = activation

        self.conv1 = GCNConv(args.feats_per_node, args.layer_1_feats, cached=False)
        self.conv2 = GCNConv(args.layer_1_feats, args.layer_2_feats, cached=False)

    def forward(self, edge_index, node_feats, edge_feats, nodes_mask_list):
        # Take the last element, which should be the only element if the framework does its job right
        node_feats = node_feats[-1]
        edge_index = edge_index[-1]
        edge_feats = edge_feats[-1]

        x = self.activation(self.conv1(node_feats, edge_index, edge_feats))
        x = self.activation(self.conv2(x, edge_index, edge_feats))

        return x
