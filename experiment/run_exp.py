import sys
import requests
import os
# Used for debugging gpu errors
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
import utils as u
import torch
import torch.distributed as dist
import numpy as np
import time
import datetime
import random
from copy import deepcopy

from datareaders.datareader import Datareader

#taskers
import link_pred_tasker as lpt

#models
from models import egcn_components as mls
from models.gcn import GCN
from models.gat import GAT
from models import egcn_h_old
from models import egcn_o_old
from models.gcn import GCN
from models.gat import GAT
from models.gclstm import GCLSTM
from models.tgat import TGAT
from models.tgatneighborfinder import NeighborFinder as TGAT_NeighborFinder
from models.tgn import TGN
from models.tgn_utils.utils import get_neighbor_finder as TGN_get_neighbor_finder
from models.tgn_utils.utils import compute_time_statistics as TGN_compute_time_statistics



import splitter as sp
import Cross_Entropy as ce

import trainer as tr

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

def prepare_args(args):
    static_models = ['gcn', 'gat'] # seal implementation cancelled due to scaling issues
    discrete_models = ['egcn_o', 'egcn_h', 'gclstm', 'egcn_h_old', 'egcn_o_old']
    continuous_models = ['tgat', 'tgn']

    if not args.model in static_models + discrete_models + continuous_models:
        raise NotImplementedError('Model {} not found'.format(args.model))
    elif args.model in static_models:
        args.temporal_granularity = 'static'
    elif args.model in discrete_models:
        args.temporal_granularity = 'discrete'
    elif args.model in continuous_models:
        args.temporal_granularity = 'continuous'

    if args.num_hist_steps in ['expanding', 'static'] and args.temporal_granularity != 'static':
        raise ValueError('An expanding or static time window can only be used with static models')

    if args.gcn_parameters['layer_2_feats_same_as_l1']:
        args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
    if ('lstm_l2_feats_name_as_l1' in args.gcn_parameters.keys()) and args.gcn_parameters['layer_2_feats_same_as_l1']:
        args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']

    return args

def build_tasker(args, dataset, temporal_granularity):
    if args.task == 'link_pred':
        return lpt.Link_Pred_Tasker(args, dataset, temporal_granularity)
    elif args.task == 'edge_cls':
        return ect.Edge_Cls_Tasker(args,dataset)
    elif args.task == 'node_cls':
        return nct.Node_Cls_Tasker(args,dataset)
    elif args.task == 'static_node_cls':
        return nct.Static_Node_Cls_Tasker(args,dataset)

    else:
        raise NotImplementedError('still need to implement the other tasks')

def build_gcn(args, tasker, dataset, splitter, feats_per_node):
    gcn_args = u.Namespace(args.gcn_parameters)
    gcn_args.feats_per_node = feats_per_node

    if args.model == 'simplegcn': # Same as 'gcn' only manually implemented
        gcn = mls.Sp_GCN(gcn_args, activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'gcn': #GCN but the PyGeometric implementation
        gcn = GCN(gcn_args, activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'gat':
        gcn = GAT(gcn_args, activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'seal':
        gcn = SEAL(gcn_args)
    elif args.model == 'gclstm':
        gcn = GCLSTM(gcn_args, activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'skipgcn':
        gcn = mls.Sp_Skip_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'skipfeatsgcn':
        gcn = mls.Sp_Skip_NodeFeats_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'tgat':
        neighborhood_finder = TGAT_NeighborFinder(dataset)
        edge_features, node_features = u.get_initial_features_continuous(args, gcn_args, dataset, tasker)
        print("edge feature and node features size", edge_features.shape, node_features.shape)
        args.gcn_parameters['layer_2_feats'] = node_features.shape[1]

        gcn = TGAT(gcn_args, neighborhood_finder, node_features, edge_features,
                    num_layers=gcn_args.num_layers, n_head=gcn_args.attention_heads,
                    drop_out=gcn_args.dropout, device=args.device).to(args.device)
    elif args.model == 'tgn':
        # Default values
        n_neighbors = 20
        uniform = False # Uniform_neighborhood_finder_sampling
        # What are these things?
        message_dim = 100
        memory_update_at_end = False #Update memory at the beginning or at the end of the batch
        embedding_module = 'graph_attention' #choices=["graph_attention", "graph_sum", "identity", "time"]
        message_function = 'identity' #choices=['identity', 'mlp']
        aggregator = 'last' #What is?
        memory_updater = 'gru' #choices=['gru', 'rnn']
        use_destination_embedding_in_message = False #What is?
        use_source_embedding_in_message = False #What is?

        neighborhood_finder = TGN_get_neighbor_finder(dataset, uniform)
        edge_features, node_features = u.get_initial_features_continuous(args, gcn_args, dataset, tasker)
        args.gcn_parameters['layer_2_feats'] = node_features.shape[1]
        memory_dim = node_features.shape[1]

        # Compute time statistics
        sources = dataset.edges['idx'][:, dataset.cols.source]
        destinations = dataset.edges['idx'][:, dataset.cols.target]
        timestamps = dataset.edges['idx'][:, dataset.cols.time]

        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        TGN_compute_time_statistics(sources, destinations, timestamps)

        gcn = TGN(neighbor_finder=neighborhood_finder, node_features=node_features,
                   edge_features=edge_features, device=args.device,
                   n_layers=gcn_args.num_layers, n_heads=gcn_args.attention_heads,
                   dropout=gcn_args.dropout, use_memory=gcn_args.use_memory,
                   message_dimension=message_dim, memory_dimension=memory_dim,
                   memory_update_at_start=not memory_update_at_end,
                   embedding_module_type=embedding_module,
                   message_function=message_function,
                   aggregator_type=aggregator,
                   memory_updater_type=memory_updater,
                   n_neighbors=n_neighbors,
                   mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                   mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                   use_destination_embedding_in_message=use_destination_embedding_in_message,
                   use_source_embedding_in_message=use_source_embedding_in_message).to(args.device)
    else:
        assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
        if args.model == 'lstmA':
            gcn = mls.Sp_GCN_LSTM_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
        elif args.model == 'gruA':
            gcn = mls.Sp_GCN_GRU_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
        elif args.model == 'lstmB':
            gcn = mls.Sp_GCN_LSTM_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
        elif args.model == 'gruB':
            gcn = mls.Sp_GCN_GRU_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
        elif args.model == 'egcn_h_old':
            gcn = egcn_h_old.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
        elif args.model == 'egcn_o_old':
            gcn = egcn_o_old.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
        elif args.model == 'skipfeatsegcn_h':
            gcn = egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device, skipfeats=True)
        else:
            raise NotImplementedError('Model not found')
    return gcn, args

def build_classifier(args,tasker):
    if 'node_cls' == args.task or 'static_node_cls' == args.task:
        mult = 1
    else:
        mult = 2
    if 'gru' in args.model or ('lstm' in args.model and not args.model=='gclstm'):
        in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
    elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
        in_feats = (args.gcn_parameters['layer_2_feats'] + args.gcn_parameters['feats_per_node']) * mult
    else:
        in_feats = args.gcn_parameters['layer_2_feats'] * mult

    return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)


# Return list of args ready for use that the framework iterates through for the grid search
# Each args in the list is the args used for each run of the grid search
def build_grid(all_args):
    lists_not_included_in_grid_search = ['class_weights', 'comments']
    args_dict = vars(all_args)

    # Gather parameters for permutation
    for_permutation = {}
    for key in args_dict:
        if type(args_dict[key]) is list and not key in lists_not_included_in_grid_search:
            for_permutation[key] = args_dict[key]
        elif type(args_dict[key]) is dict:
            d = args_dict[key]
            for inner_key in d:
                if type(d[inner_key]) is list:
                    for_permutation["{}.{}".format(key, inner_key)] = d[inner_key]

    # Convenience
    # Putting learning rate at the end, it will be ordered by permutate to be the outermost parameter in the grid search
    # Thus for continuous models, the training of the encoder happens intermittently, rather than all at once in the beginning
    if all_args.model in ['tgn', 'tgat']:
        lr = for_permutation.pop('learning_rate')
        for_permutation['learning_rate'] = lr

    args_list = []
    def permutate(for_permutation, args_dict, permutated):
        if for_permutation == {}:
            # Add grid arg to show which grid cell this is
            args_dict['grid'] = permutated
            args_list.append(args_dict)
        else:
            new_for_permutation = deepcopy(for_permutation)
            param_name, param_values = new_for_permutation.popitem()
            for param in param_values:
                new_args_dict = deepcopy(args_dict)
                new_permutated = deepcopy(permutated)
                new_permutated[param_name] = param
                if "." in param_name:
                    key, inner_key = param_name.split(".")
                    new_args_dict[key][inner_key] = param
                else:
                    new_args_dict[param_name] = param
                permutate(new_for_permutation, new_args_dict, new_permutated)

    permutate(for_permutation, args_dict, {})
    assert(len(args_list) == u.prod([len(param_list) for param_list in for_permutation.values()]))

    return [u.Namespace(args) for args in args_list]

def read_data_master(args, dataset_name=None):
    if not dataset_name: dataset_name = args.data
    master = pd.read_csv(os.path.join('config', 'data_master.csv'), index_col = 0)
    if not dataset_name in master:
            error_mssg = 'Dataset not found in data master. Dataset name {}.\n'.format(dataset_name)
            error_mssg += 'Available datasets are as follows:\n'
            error_mssg += '\n'.join(master.keys())
            raise ValueError(error_mssg)
    meta_info = master[dataset_name]

    args.data_filepath = os.path.join('data', meta_info['filename'])
    args.snapshot_size = float(meta_info['snapshot size'])
    args.train_proportion = float(meta_info['train proportion'])
    args.val_proportion = float(meta_info['val proportion'])

    steps_acc = meta_info['steps accounted']
    try:
        args.steps_accounted = int(steps_acc)
    except ValueError:
        args.steps_accounted = None

    if meta_info['node encoding'] == '2 hot':
        args.use_2_hot_node_feats = True
        args.use_1_hot_node_feats = False
    else:
        args.use_2_hot_node_feats = False
        args.use_1_hot_node_feats = True

    return args

def run_experiment(args):
    ### Seed, rank and cuda
    global rank, wsize, use_cuda
    #if hasattr(args, 'ncores') and type(args.ncores) == type(1) and args.ncores >= 1:
    #    print(args.ncores)
    torch.set_num_threads(16)
    args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
    args.device = 'cpu'
    if args.use_cuda:
        args.device = 'cuda'
    print ("use CUDA:", args.use_cuda, "- device:", args.device)
    try:
        dist.init_process_group(backend='mpi') #, world_size=4 # Consider changing to NCCL
        rank = dist.get_rank()
        wsize = dist.get_world_size()
        print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
        if args.use_cuda:
            torch.cuda.set_device(rank )  # are we sure of the rank+1????
            print('using the device {}'.format(torch.cuda.current_device()))
    except:
        rank = 0
        wsize = 1
        print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank, wsize)))

    if args.seed is None and args.seed != 'None':
        seed = 123+rank#int(time.time())+rank
    else:
        seed = args.seed#+rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.seed = seed
    args.rank = rank
    args.wsize = wsize

    ### Load datasets and build everything
    args = read_data_master(args)
    args = prepare_args(args)

    # If static/discrete, just train as normal as before
    # If continuous or SEAL, get both discrete AND static
    #build the dataset

    if args.temporal_granularity == 'continuous' or args.model == 'seal':
        cont_dataset, disc_dataset = Datareader(args).dataset
        dataset = cont_dataset
        #build the tasker
        tasker = build_tasker(args, cont_dataset, 'continuous')
        #build the splitters
        #all_in_one_snapshot = args.temporal_granularity == 'continuous' or args.model == 'seal'
        #eval_all_edges = not all_in_one_snapshot #args.model != 'seal'
        splitter = sp.splitter(args, tasker, 'continuous', all_in_one_snapshot=True, eval_all_edges=False)

        #test_dataset = Datareader(args, return_continuous=False).dataset
        #build the tasker. This needs to be done in a static way because it is for testing.
        test_tasker = build_tasker(args, disc_dataset, 'static')
        frozen_encoder_splitter = sp.splitter(args, tasker, 'static', test_tasker=test_tasker, frozen_encoder=True) # The tasker is passed here to enable encoding for the continuous model
        downstream_splitter = sp.splitter(args, tasker, 'static', test_tasker=test_tasker, downstream=True)

        feats_per_node = tasker.feats_per_node
        if args.model == 'seal':
            seal_splitter, seal_test_splitter, num_features = preprocess_seal(
                args, splitter, frozen_encoder_splitter, tasker, test_tasker)
            splitter = seal_splitter
            frozen_encoder_splitter = seal_test_splitter
            feats_per_node = num_features

    else: # Discrete and static
        _, dataset = Datareader(args).dataset
        #build the tasker
        tasker = build_tasker(args, dataset, args.temporal_granularity)
        #build the splitters
        splitter = sp.splitter(args, tasker, args.temporal_granularity,
                               all_in_one_snapshot=False, eval_all_edges=True)
        frozen_encoder_splitter = None
        downstream_splitter = sp.splitter(args, tasker, args.temporal_granularity, downstream=True)

        feats_per_node = tasker.feats_per_node

    #build the models
    gcn, args = build_gcn(args, tasker, dataset, splitter, feats_per_node)
    classifier = build_classifier(args, tasker)
    #build a loss
    cross_entropy = ce.Cross_Entropy(args,dataset).to(args.device)

    #trainer
    if args.temporal_granularity != 'continuous':
        trainer = tr.Trainer(args,
                            splitter = splitter,
                            frozen_encoder_splitter = frozen_encoder_splitter,
                            downstream_splitter = downstream_splitter,
                            gcn = gcn,
                            classifier = classifier,
                            comp_loss = cross_entropy,
                            dataset = dataset)
        ### Train
        trainer.train()
    else: #Continuous
        # Encoder training
        decoder_num_epochs = args.num_epochs
        args.num_epochs = args.num_epochs_continuous
        print("Num epochs ", args.num_epochs)
        trainer = tr.Trainer(args,
                            splitter = splitter,
                            frozen_encoder_splitter = frozen_encoder_splitter,
                            downstream_splitter = downstream_splitter,
                            gcn = gcn,
                            classifier = classifier,
                            comp_loss = cross_entropy,
                             dataset = dataset,
                             train_encoder = True)
        print("checkpoint exists: " + str(trainer.checkpoint_exists()))
        if (hasattr(args, 'force_encode') and args.force_encode == True) or not trainer.checkpoint_exists(): #Only train encoder if it has not been done before
            # checkpoint_filename_prefix includes the parameters which are used to consider whether it exists or not.
            print("checkpoint not here")
            trainer.train()
        # Decoder training
        args.num_epochs = decoder_num_epochs
        decoder_trainer = tr.Trainer(args,
                            splitter = splitter,
                            frozen_encoder_splitter = frozen_encoder_splitter,
                            downstream_splitter = downstream_splitter,
                            gcn = gcn,
                            classifier = classifier,
                            comp_loss = cross_entropy,
                            dataset = dataset,
                            train_encoder = False)
        decoder_trainer.train()

def notify(args, text):
    if False: #args.notify: #Add a slack hook for this notification to work.
        print('notify ' + text)
        notify_text = u.get_experiment_notification(args)+' '+text
        print('Notifying of completion')
        with open('../slack_incoming_hook.txt','r') as f:
            slack_hook = f.read()

        try:
            # Send a notification to a slack channel.
            r = requests.post(slack_hook, json={'text': notify_text})
            if (r.status_code != requests.codes.ok):
                print('Notification failed, status code {}'.format(r.status_code))
        except requests.exceptions.ConnectionError as e:
            print('could not connect, is there any internet?')

if __name__ == '__main__':
    parser = u.create_parser()
    args = u.parse_args(parser)
    config_folder = "/".join(args['config_file'].split('/')[:-1])
    master_args = u.read_master_args(config_folder+'/master.yaml')
    all_args = u.Namespace({**master_args, **args})
    args = all_args

    # Assign the requested random hyper parameters
    #args = build_random_hyper_params(args) #Replaced by grid

    exp_args_list = build_grid(all_args)
    exp_durations = []
    start_time_tot = time.time()
    remove_microsec = lambda x: str(x).split(".")[0]
    cell_tot = len(exp_args_list)
    cell_num = 0
    start_notify_done = False
    print('Total number of runs', cell_tot)
    try:
        for i, exp_args in enumerate(exp_args_list):
            cell_num = i+1
            print('Grid cell {}/{} args {}'.format(cell_num, cell_tot, exp_args.grid))
            if not u.skip_cell(exp_args):
                if not start_notify_done:
                    notify(args, '{}/{} started'.format(cell_num, cell_tot))
                    start_notify_done = True

                # Initiate log to signal to other cells that this cell is taken
                # Useful if cells are run as different processes and preprocessing takes time.
                u.add_log_lock(exp_args)

                start = time.time()
                run_experiment(exp_args)
                if exp_args.one_cell:
                    print("Exiting after one cell")
                    break
                end = time.time()
                exp_durations.append(end-start)
            else:
                print("SKIPPING CELL "+ u.get_gridcell(exp_args))
    except Exception as e:
        notify(args, '{}/{} crashed {}'.format(cell_num, cell_tot, str(e)))
        raise
    except:
        notify(args, '{}/{} crashed {}'.format(cell_num, cell_tot, sys.exc_info()[0]))
        raise

    end_time_tot = time.time()
    time_tot = end_time_tot - start_time_tot
    duration = remove_microsec(str(datetime.timedelta(seconds=time_tot)))
    notify(args, '{}/{} complete, duration {}'.format(cell_num, cell_tot, duration))
