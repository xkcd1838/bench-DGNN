import torch
import utils as u
import numpy as np
import pandas as pd

from datareaders.dataset import Dataset

def normalize_time(time_vector):
    return time_vector - time_vector.min()

# Consider renaming to snapshotify for fun.
def aggregate_by_time(time_vector, time_win_aggr):
    return time_vector // time_win_aggr


# Not used? Remove?
def edges_to_dataset(edges, ecols):
    idx = edges[:,[ecols.source,
                    ecols.target,
                    ecols.time,
                    ecols.snapshot]]

    vals = edges[:,ecols.weight]
    return Dataset(idx, vals)

def tgat_preprocess(data_name):
    # Wikipedia and Reddit - Preprocessing code kindly borrowed from TGAT.
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                        'i': i_list,
                        'ts': ts_list,
                        'label': label_list,
                        'idx': idx_list}), np.array(feat_l)

def tgat_reindex(df, bipartite=True):
    # Improved TGAT reindexing borrowed from TGN
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df

def continuous2discrete(continuous: Dataset) -> Dataset:
    # Remove time col
    discrete_edgeidx = continuous.edges['idx'].detach().clone()
    discrete_edgeidx = discrete_edgeidx[:, [continuous.cols.source,
                                            continuous.cols.target,
                                            continuous.cols.snapshot]]
    #add the reversed link to make the graph undirected. This is also required for the
    #  uniqueness check to make sense. Because it would count a,b as a different edge to b,a.
    #  for an edge between node a and b.
    cols = u.Namespace({'source': 0,
                        'target': 1,
                        'snapshot': 2})
    discrete_edgeidx = torch.cat([discrete_edgeidx, discrete_edgeidx[:, [cols.target,
                                                                            cols.source,
                                                                            cols.snapshot]]])
    # Returns unique (source, target, snapshot) triplets.
    # We therefore get one unique edge per snapshot
    # The count is therefore the number of times an edge occur in each snapshot
    discrete_edgeidx, discrete_vals = discrete_edgeidx.unique(sorted=False, return_counts=True, dim=0)

    # Duplicate the snapshot index, thus the snapshot index is now in the time AND snapshot col
    discrete_edgeidx = torch.cat([discrete_edgeidx, discrete_edgeidx[:, 2].view(-1, 1)], dim=1)
    return Dataset(discrete_edgeidx, discrete_vals)

# Makes the number of occurrences of a link in a snapshot the weight of that link
# Coalesce sums the values of identical indexes together
# Thus make sure that time col is not included in index, if not this has no effect.
def weight_by_occurrece(index, occurences, num_nodes, max_time):
    return torch.sparse.LongTensor(index,
                                    occurences,
                                    torch.Size([num_nodes,
                                                num_nodes,
                                                max_time+1])).coalesce()

def make_contiguous_node_ids(edges, ecols):
    new_edges = edges[:,[ecols.source, ecols.target]]
    _, new_edges = new_edges.unique(return_inverse=True)
    edges[:,[ecols.source,ecols.target]] = new_edges
    return edges

def load_edges_as_tensor(filepath):
    with open(filepath) as f:
        lines = f.read().splitlines()
    edges = [[float(r) for r in row.split(',')] for row in lines]
    edges = torch.tensor(edges,dtype = torch.long)
    return edges
