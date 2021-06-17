import torch
import utils as u
import numpy as np

ECOLS = u.Namespace({'source': 0,
                     'target': 1,
                     'time': 2,
                     'snapshot': 3,
                     'label':4}) #--> added for edge_cls (only called in edge_cls tasker, thus useless?)

# def get_2_hot_deg_feats(adj,max_deg_out,max_deg_in,num_nodes):
#     #For now it'll just return a 2-hot vector
#     adj['vals'] = torch.ones(adj['idx'].size(0))
#     degs_out, degs_in = get_degree_vects(adj,num_nodes)

#     degs_out = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
#                                   degs_out.view(-1,1)],dim=1),
#                 'vals': torch.ones(num_nodes)}

#     # print ('XXX degs_out',degs_out['idx'].size(),degs_out['vals'].size())
#     degs_out = u.make_sparse_tensor(degs_out,'long',[num_nodes,max_deg_out])

#     degs_in = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
#                                   degs_in.view(-1,1)],dim=1),
#                 'vals': torch.ones(num_nodes)}
#     degs_in = u.make_sparse_tensor(degs_in,'long',[num_nodes,max_deg_in])

#     hot_2 = torch.cat([degs_out,degs_in],dim = 1)
#     hot_2 = {'idx': hot_2._indices().t(),
#              'vals': hot_2._values()}

#     return hot_2

def get_1_hot_deg_feats(adj, max_deg, num_nodes):
    new_vals = torch.ones(adj['idx'].size(0))
    new_adj = {'idx':adj['idx'], 'vals': new_vals}
    degs_out, _ = get_degree_vects(new_adj,num_nodes)

    degs_out = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
                                  degs_out.view(-1,1)],dim=1),
                'vals': torch.ones(num_nodes)}

    degs_out = u.make_sparse_tensor(degs_out,'long',[num_nodes, max_deg])

    hot_1 = {'idx': degs_out._indices().t(),
             'vals': degs_out._values()}
    return hot_1

def get_random_features(adj, max_deg, num_nodes):
    # The framework expects a sparse tensor of dimensions num_nodes x max_deg.
    # That works well if they are one-hot encoded, however, if they are randomly encoded,
    # a dense tensor would have made more sense.
    raise NotImplementedError

    return {'idx': adj['idx'], 'vals': new_vals}

'''
Get the max node degree for the entire network, including validation and test.
One alternative is to get the node degree snapshot by snapshot. This is possible for a discrete model, but in the interest of a fair comparison we want the node feature dimensions to be the same across models. Therefore we pick maximum node degree in the same way for all models.
'''
def get_max_degs(args, dataset):
    # Selects the entire dataset (including validation and test) as one snapshot
    cur_adj = get_sp_adj(edges = dataset.edges,
                        snapshot = dataset.max_time,
                        weighted = False,
                        time_window = dataset.max_time+1)
    cur_out, cur_in = get_degree_vects(cur_adj, dataset.num_nodes)
    max_deg_out = int(cur_out.max().item()) + 1
    max_deg_in = int(cur_in.max().item()) + 1

    return max_deg_out, max_deg_in

def get_max_degs_static(num_nodes, adj_matrix):
    cur_out, cur_in = get_degree_vects(adj_matrix, num_nodes)
    max_deg_out = int(cur_out.max().item()) + 1
    max_deg_in = int(cur_in.max().item()) + 1

    return max_deg_out, max_deg_in


def get_degree_vects(adj,num_nodes):
    adj = u.make_sparse_tensor(adj,'long',[num_nodes])
    degs_out = adj.matmul(torch.ones(num_nodes,1,dtype = torch.long))
    degs_in = adj.t().matmul(torch.ones(num_nodes,1,dtype = torch.long))
    return degs_out, degs_in

def get_sp_adj(edges, snapshot: int, weighted: bool, time_window, temporal_granularity='static'):
    idx = edges['idx']
    subset = idx[:,ECOLS.snapshot] <= snapshot
    if time_window is not None:
        subset = subset * (idx[:,ECOLS.snapshot] > (snapshot - time_window))

    idx = edges['idx'][subset][:,[ECOLS.source, ECOLS.target]]
    time = edges['idx'][subset][:,[ECOLS.time]]
    vals = edges['vals'][subset]
    if temporal_granularity != 'continuous':
        # Duplicates in the static and discrete cases should be removed during data loading, if not this will remove the duplicates BUT will also sum the vals of duplicate edges
        # Even though duplicates were removed in the data preprocessing stage, if multiple snapshots are selected we'll remove duplicates again here.
        out = torch.sparse.FloatTensor(idx.t(),vals).coalesce()
        idx = out._indices().t()

        if weighted:
            vals = out._values()
        else:
            vals = torch.ones(idx.size(0), dtype=torch.long)
    else:
        if weighted:
            vals = vals
        else:
            vals = torch.ones_like(vals)

    return {'idx': idx, 'vals': vals, 'time': time}

# Deprecated? Seems to only be used in edge_cls_tasker
def get_edge_labels(edges, snapshot):
    idx = edges['idx']
    subset = idx[:,ECOLS.snapshot] == snapshot
    idx = edges['idx'][subset][:,[ECOLS.source, ECOLS.target]]
    vals = edges['idx'][subset][:,ECOLS.label]

    return {'idx': idx, 'vals': vals}


def get_node_mask(cur_adj,num_nodes):
    mask = torch.zeros(num_nodes) - float("Inf")
    non_zero = cur_adj['idx'].unique()

    mask[non_zero] = 0

    return mask

def get_static_sp_adj(edges, weighted):
    idx = edges['idx']
    #subset = idx[:,ECOLS.snapshot] <= snapshot
    #subset = subset * (idx[:,ECOLS.snapshot] > (snapshot - time_window))

    #idx = edges['idx'][subset][:,[ECOLS.source, ECOLS.target]]
    if weighted:
        vals = edges['vals'][subset]
    else:
        vals = torch.ones(idx.size(0),dtype = torch.long)

    return {'idx': idx, 'vals': vals}

def get_sp_adj_only_new(edges, snapshot ,weighted):
    return get_sp_adj(edges, snapshot, weighted, time_window=1)

def normalize_adj(adj,num_nodes):
    '''
    takes an adj matrix as a dict with idx and vals and normalize it by:
        - adding an identity matrix,
        - computing the degree vector
        - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2
    '''
    idx = adj['idx']
    vals = adj['vals']


    sp_tensor = torch.sparse.FloatTensor(idx.t(),vals.type(torch.float),torch.Size([num_nodes,num_nodes]))

    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = sparse_eye + sp_tensor

    idx = sp_tensor._indices()
    vals = sp_tensor._values()

    degree = torch.sparse.sum(sp_tensor,dim=1).to_dense()
    di = degree[idx[0]]
    dj = degree[idx[1]]

    vals = vals * ((di * dj) ** -0.5)

    return {'idx': idx.t(), 'vals': vals}

def make_sparse_eye(size):
    eye_idx = torch.arange(size)
    eye_idx = torch.stack([eye_idx,eye_idx],dim=1).t()
    vals = torch.ones(size)
    eye = torch.sparse.FloatTensor(eye_idx,vals,torch.Size([size,size]))
    return eye

def get_all_non_existing_edges(adj,tot_nodes):
    true_ids = adj['idx'].t().numpy()
    true_ids = get_edges_ids(true_ids,tot_nodes)

    all_edges_idx = np.arange(tot_nodes)
    all_edges_idx = np.array(np.meshgrid(all_edges_idx,
                                         all_edges_idx)).reshape(2,-1)

    all_edges_ids = get_edges_ids(all_edges_idx,tot_nodes)

    #only edges that are not in the true_ids should keep here
    mask = np.logical_not(np.isin(all_edges_ids, true_ids))

    non_existing_edges_idx = all_edges_idx[:,mask]
    edges = torch.tensor(non_existing_edges_idx).t()
    vals = torch.zeros(edges.size(0), dtype = torch.long)
    return {'idx': edges, 'vals': vals}

def get_non_existing_edges(adj, num_edges, tot_nodes, smart_sampling, existing_nodes=None):
    oversampling_factor = 4

    idx = adj['idx'].t().numpy()
    true_ids = get_edges_ids(idx, tot_nodes)

    true_ids = set(true_ids)
    num_non_existing_edges = tot_nodes * (tot_nodes-1) - len(true_ids)
    if num_edges > num_non_existing_edges:
        # If we can't sample enough edges might just as well grab them all without the sampling
        return get_all_non_existing_edges(adj, tot_nodes)

    #existing_nodes = existing_nodes.numpy()
    def sample_edges_smart(num_edges):
        from_id = np.random.choice(idx[0], size = num_edges, replace = True)
        to_id = np.random.choice(existing_nodes, size = num_edges, replace = True)

        if num_edges>1:
            edges = np.stack([from_id,to_id])
        else:
            edges = np.concatenate([from_id,to_id])
        return edges

    def sample_edges_simple(num_edges):
        if num_edges > 1:
            edges = np.random.randint(0,tot_nodes,(2,num_edges))
        else:
            edges = np.random.randint(0,tot_nodes,(2,))
        return edges

    if smart_sampling:
        sample_edges = sample_edges_smart
    else:
        sample_edges = sample_edges_simple

    edges = sample_edges(num_edges*oversampling_factor)
    edge_ids = get_edges_ids(edges, tot_nodes)

    out_ids = set()
    num_sampled = 0
    sampled_indices = []
    for i in range(num_edges*oversampling_factor):
        eid = edge_ids[i]
        if eid in out_ids or edges[0,i] == edges[1,i] or eid in true_ids:
            continue

        #add the eid and the index to a list
        out_ids.add(eid)
        sampled_indices.append(i)
        num_sampled += 1

        #if we have sampled enough edges break
        if num_sampled >= num_edges:
            break

    edges = edges[:,sampled_indices]

    # Fix issue where smart sampling doesn't deliver enough negative samples
    # Fill in with simple sampling
    missing_links = num_edges - num_sampled
    if missing_links > 0:
        medges = sample_edges_simple(missing_links*oversampling_factor)
        medge_ids = get_edges_ids(medges, tot_nodes)
        msampled_indices = []
        for i in range(missing_links*oversampling_factor):
            eid = medge_ids[i]
            if eid in out_ids or medges[0,i] == medges[1,i] or eid in true_ids:
                continue

            #add the eid and the index to a list
            out_ids.add(eid)
            msampled_indices.append(i)
            num_sampled += 1

            #if we have sampled enough edges break
            if num_sampled >= num_edges:
                break
        medges = medges[:, msampled_indices]
        edges = np.append(edges, medges, axis=1)
        #print(edges.shape, medges.shape, missing_links, i/(missing_links*oversampling_factor))

    edges = torch.tensor(edges).t()
    vals = torch.zeros(edges.size(0),dtype = torch.long)

    missing_links = num_edges - num_sampled
    if missing_links > 0:
        print("aim", num_edges, 'sampled', num_sampled)
        print("WARNING: Negative sampling not equal to requested edges. The negative sampler have failed to supply the right amount of samples.")
    return {'idx': edges, 'vals': vals}

def get_edges_ids(sp_idx, tot_nodes):
    # print(sp_idx)
    # print(tot_nodes)
    # print(sp_idx[0]*tot_nodes)
    return sp_idx[0]*tot_nodes + sp_idx[1]

def sample_labels(label_adj, sample_rate):
    # Sample links from label_adj according to sample rate
    # Keep the ratio of existing links and non-existing links the same
    def sample(label_adj):
        num_preds = label_adj['vals'].size()[0]
        num_samples = int(num_preds*sample_rate)
        sample_idx = np.random.choice(num_preds, num_samples, replace=False)
        label_adj['idx'] = label_adj['idx'][sample_idx]
        label_adj['vals'] = label_adj['vals'][sample_idx]
        return label_adj

    link_mask = label_adj['vals'] == 1
    nolink_mask = ~link_mask

    label_adj_link = {'idx': label_adj['idx'][link_mask], 'vals': label_adj['vals'][link_mask]}
    label_adj_nolink = {'idx': label_adj['idx'][nolink_mask], 'vals': label_adj['vals'][nolink_mask]}

    sampled_la_link = sample(label_adj_link)
    sampled_la_nolink = sample(label_adj_nolink)

    idx = torch.cat([sampled_la_link['idx'], sampled_la_nolink['idx']], axis=0)
    vals = torch.cat([sampled_la_link['vals'], sampled_la_nolink['vals']], axis=0)

    return {'idx': idx, 'vals': vals}
