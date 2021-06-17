import argparse
import socket
import datetime
import yaml
import torch
import numpy as np
import time
import random
import math
import os
import getpass
import glob
from functools import reduce
import operator

def pad_with_last_col(matrix,cols):
    out = [matrix]
    pad = [matrix[:,[-1]]] * (cols - matrix.size(1))
    out.extend(pad)
    return torch.cat(out,dim=1)

def pad_with_last_val(vect,k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect


def sparse_prepare_tensor(tensor, torch_size, ignore_batch_dim = True):
    if ignore_batch_dim:
        tensor = sp_ignore_batch_dim(tensor)
    tensor = make_sparse_tensor(tensor,
                                tensor_type = 'float',
                                torch_size = torch_size)
    return tensor

def sp_ignore_batch_dim(tensor_dict):
    tensor_dict['idx'] = tensor_dict['idx'][0]
    tensor_dict['vals'] = tensor_dict['vals'][0]
    return tensor_dict

def sort_by_time(data,time_col):
        _, sort = torch.sort(data[:,time_col])
        data = data[sort]
        return data

def print_sp_tensor(sp_tensor,size):
    print(torch.sparse.FloatTensor(sp_tensor['idx'].t(),sp_tensor['vals'],torch.Size([size,size])).to_dense())

def reset_param(t):
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv,stdv)

# Takes an edge list and turns it into an adjacency matrix
def make_sparse_tensor(adj, tensor_type, torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def sp_to_dict(sp_tensor):
    return  {'idx': sp_tensor._indices().t(),
             'vals': sp_tensor._values()}

class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

def set_seeds(rank):
    seed = int(time.time())+rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_param_value(param, param_min, param_max, type='int'):
    if str(param) is None or str(param).lower()=='none':
        if type=='int':
            return random.randrange(param_min, param_max+1)
        elif type=='logscale':
            interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval,1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param

def load_data(file):
    with open(file) as file:
        file = file.read().splitlines()
    data = torch.tensor([[float(r) for r in row.split(',')] for row in file[1:]])
    return data

def load_data_from_tar(file, tar_archive, replace_unknow=False, starting_line=1, sep=',', type_fn = float, tensor_const = torch.DoubleTensor):
    f = tar_archive.extractfile(file)
    lines = f.read()
    lines=lines.decode('utf-8')
    if replace_unknow:
        lines=lines.replace('unknow', '-1')
        lines=lines.replace('-1n', '-1')

    lines=lines.splitlines()

    data = [[type_fn(r) for r in row.split(sep)] for row in lines[starting_line:]]
    data = tensor_const(data)
    return data

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file', default='experiments/parameters_example.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    parser.add_argument('--one_cell', action='store_true', help='optional, indicate whether to search just one grid cell in the grid search or all')
    #parser.add_argument('--ncores', type=int, help='optional, indicate how many threads pytorch should spawn, note that dataloader has a num workers and joblib also spawn parallel processes')
    return parser

def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file, Loader=yaml.FullLoader)
        #delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value
        arg_dict['config_file'] = arg_dict['config_file'].name
    return arg_dict

def read_master_args(yaml_file):
    try:
        with open(yaml_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            arg_dict = {}
            for key, value in data.items():
                arg_dict[key] = value
            return arg_dict
    except FileNotFoundError as e:
        raise type(e)(str(e) + ' Master file not found in config folder')


def get_log_folder(args):
    if hasattr(args, 'log_folder'):
        root_log_folder = args.log_folder
    else:
        root_log_folder = 'log'

    log_folder = root_log_folder + '/'+args.data+'-'+args.model+'/'
    os.makedirs(log_folder, exist_ok=True)
    return log_folder

def get_log_name(args, classifier_name='decoder', train_encoder=True):
    log_folder = get_log_folder(args)
    hostname = socket.gethostname()
    if args.temporal_granularity == 'continuous' and train_encoder==True:
        gridcell = str(args.learning_rate) #Only learning rate is significant for this log, also used to distinguish between encoder and decoder logs
    else:
        gridcell = get_gridcell(args)
    currdate=str(datetime.datetime.today().strftime('%Y%m%d%H%M%S'))
    log_name = log_folder+'log_'+args.data+'_'+args.task+'_'+currdate+'_'+args.model+'_'+classifier_name+'_r'+str(args.rank)+'_'+hostname+"__grid_"+gridcell+'.log'
    return log_name

def get_experiment_notification(args):
    hostname = socket.gethostname()
    username = getpass.getuser()
    return args.data+'_'+args.model+'_'+hostname+'_'+username

def get_gridcell(args):
    grid = args.grid.items()
    if len(grid) <= 0:
        grid_str = 'nogrid'
    else:
        grid_str = "_".join(["{}:{}".format(key, value) for key, value in args.grid.items()])
    return grid_str

# Returns bool whether to skip a grid cell or not
def skip_cell(args):
    log_folder = get_log_folder(args)
    gridcell = get_gridcell(args)

    for filename in glob.glob(log_folder+'*'):
        if gridcell in filename:
            return args.skip_computed_grid_cells == True

def add_log_lock(args):
    if args.use_logfile:
        open(get_log_folder(args)+'/'+get_gridcell(args)+'_lock.log', 'a').close()

def remove_log_lock(args):
    gridcell = get_gridcell(args)
    for filename in glob.glob(get_log_folder(args)+'*'):
        if gridcell in filename and '_lock' in filename:
            os.remove(filename)

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def get_initial_features_continuous(args, gcn_args, dataset, tasker):
    ## TGAT requires that node and edge features are the same size.
    #if args.random_feats == True:
    #    num_feats = gcn_args.layer_2_feats
    #    edge_features = np.random.rand(dataset.num_edges, num_feats)
    #    node_features = np.zeros((dataset.num_nodes, edge_features.shape[1]))

    #elif type(dataset.edge_features) == type(None):
    #    # Edge features don't exist, make random edge features, size defined by node features
    #    start_idx = tasker.data.min_time + args.num_hist_steps

    #    # Get initial node features
    #    s = tasker.get_sample(start_idx, partition='TRAIN', test = False, snapshot_based=False, split_start=tasker.data.min_time)
    #    assert(len(s['hist_ndFeats']) == 1)
    #    node_features = s['hist_ndFeats'][0]
    #    node_features = make_sparse_tensor(node_features, tensor_type='float',
    #                        torch_size=[dataset.num_nodes, tasker.feats_per_node]).to_dense().cpu().numpy()
    #    # Random init of node features. The same number as the output layer should have.. no change of features through the model.
    #    gcn_args.layer_2_feats = tasker.feats_per_node
    #    #features_per_node = gcn_args.layer_2_feats
    #    #node_features = np.random.rand(dataset.num_nodes, features_per_node)

    #    # Random initiation of (all) edge features
    #    # Use same dimensions as node features, if not, the attention model breaks...
    #    features_per_edge = node_features.shape[1]
    #    num_edges = len(dataset.edges['vals'])
    #    edge_features = np.random.rand(num_edges, features_per_edge)
    #else:
    # Edge features exist, make zero node features, size defined by edge features

    if type(dataset.edge_features) == type(None):
        num_feats = gcn_args.layer_2_feats
        features_per_edge = num_feats
        num_edges = len(dataset.edges['vals'])
        edge_features = np.random.rand(num_edges, features_per_edge)
    else:
        edge_features = dataset.edge_features

    node_features = np.zeros((dataset.num_nodes, edge_features.shape[1]))
    return edge_features, node_features
