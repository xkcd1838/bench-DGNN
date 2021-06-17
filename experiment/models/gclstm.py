import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM


class GCLSTM(torch.nn.Module):
    def __init__(self, args, activation):
        super(GCLSTM, self).__init__()
        self.activation = activation
        self.K = args.K

        # Note, when using one layer the out size is layer_2 size to fit the decoder.
        self.recurrent_1 = GConvLSTM(args.feats_per_node, args.layer_2_feats, args.K)
        # K = 3 in the original paper

        # Original seem to use only one layer
        #self.recurrent_1 = GConvLSTM(args.feats_per_node, args.layer_1_feats, args.K)
        #self.recurrent_2 = GConvLSTM(args.layer_1_feats, args.layer_2_feats, args.K)

    def forward(self, edge_index_list, node_feats_list, edge_feats_list, nodes_mask_list):
        #node_feats = node_feats[-1].to_dense() #Can't be sparse for some reason.
        #edge_index = edge_index[-1]
        #edge_feats = edge_feats[-1]

        for t, edge_index in enumerate(edge_index_list):
            node_feats = node_feats_list[t].to_dense()
            edge_feats = edge_feats_list[t]

            x, _ = self.recurrent_1(node_feats, edge_index, edge_feats)
            # Original seem to use only one layer
            #x, _ = self.recurrent_2(x, edge_index, edge_weight)
            x = F.relu(x)

        #x = self.activation(x)
        return x
