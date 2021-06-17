import torch
from torch_geometric.nn import GATConv
from torch.nn import functional as F

class GAT(torch.nn.Module):
    def __init__(self, args, activation):
        super().__init__()
        self.activation = activation
        self.dropout = args.dropout

        self.conv1 = GATConv(args.feats_per_node, args.layer_1_feats, heads=args.attention_heads,
                             dropout=args.dropout)
        self.conv2 = GATConv(args.layer_1_feats*args.attention_heads, args.layer_2_feats, heads=1,
                             dropout=args.dropout)
                             #concat=False, dropout=args.dropout) #To concat or not to concat?

    def forward(self, edge_index, node_feats, edge_feats, nodes_mask_list):
        # Take the last element, which should be the only element if the framework does its job right
        node_feats = node_feats[-1]
        edge_index = edge_index[-1]
        edge_feats = edge_feats[-1]

        #x = F.dropout(node_feats, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(node_feats, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv2(x, edge_index))

        return x
