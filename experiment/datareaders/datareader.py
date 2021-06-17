import torch
import os, sys
import pandas as pd
import numpy as np
import utils as u
import tarfile
from datetime import datetime
import datareaders.data_utils as du
from datareaders.dataset import Dataset


class Datareader:
    def __init__(self, args):
        self.args = args
        #return_continuous = args.temporal_granularity == 'continuous' #Returns both by default
        dataset_name = args.data
        if dataset_name != 'autonomous-systems':
            self.args.snapshot_size = 60*60*24*self.args.snapshot_size #days to seconds
        self.args.snapshot_size = int(self.args.snapshot_size)

        if dataset_name == 'enron':
            dataset = self.load_enron()
        elif dataset_name == 'bitcoin-otc':
            dataset = self.load_bitcoin_otc()
        elif dataset_name == 'autonomous-systems':
            steps_accounted = self.args.steps_accounted
            dataset = self.load_autosys(steps_accounted)
        elif dataset_name == 'uc':
            edges_file = 'opsahl-ucsocial/out.opsahl-ucsocial'
            dataset = self.load_uc(edges_file)
        elif dataset_name == 'wikipedia':
            dataset = self.load_tgat_data()
        elif dataset_name == 'reddit':
            dataset = self.load_tgat_data()
        else:
            raise ValueError('Dataset {} not found'.format(self.args.data))

        #self.print_all_dataset_info_and_exit()
        self.dataset = dataset

    # Returns a dataframe with columns: {source, target, t, snapshot}
    # First event is always at t=0
    def load_enron(self) -> Dataset:
        snapshot_size = self.args.snapshot_size
        filepath = self.args.data_filepath
        #The "whatisthis" column is just filled with 1 so it has no impact and can safely be removed.
        df = pd.read_csv(filepath, header=None, names=['source', 'target', 'ones', 't'], sep=' ')
        df = df.drop('ones', axis=1)

        # Make t relative. I.e. set the first event to be at t=0
        df['t'] = df['t'] - df['t'].min()
        # Snapshot by day (divide by seconds in a day)
        df['snapshot'] = (df['t'] / snapshot_size).apply(int)

        # First node should be node 0, not node 1.
        df[['source', 'target']] = df[['source', 'target']] - 1

        edgesidx = torch.from_numpy(df.to_numpy())
        continuous = Dataset(edgesidx, torch.ones(edgesidx.size(0)))
        return continuous, du.continuous2discrete(continuous)

    def load_bitcoin_otc(self) -> Dataset:
        snapshot_size = self.args.snapshot_size
        filepath = self.args.data_filepath
        ecols = u.Namespace({'source': 0,
                             'target': 1,
                             'weight': 2,
                             'time': 3,
                             'snapshot': 4
        })
        def cluster_negs_and_positives(ratings):
            pos_indices = ratings > 0
            neg_indices = ratings <= 0
            ratings[pos_indices] = 1
            ratings[neg_indices] = -1
            return ratings

        #build edge data structure
        edges = du.load_edges_as_tensor(filepath)

        edges = du.make_contiguous_node_ids(edges, ecols)

        edges[:,ecols.time] = du.normalize_time(edges[:, ecols.time])
        snapshots = du.aggregate_by_time(edges[:,ecols.time], snapshot_size)
        edges = torch.cat([edges, snapshots.view(-1, 1)], dim=1)

        edges[:,ecols.weight] = cluster_negs_and_positives(edges[:,ecols.weight])

        continuous = Dataset(
            edges[:, [ecols.source,
                      ecols.target,
                      ecols.time,
                      ecols.snapshot]],
            edges[:, ecols.weight])
        return continuous, du.continuous2discrete(continuous)

    def load_autosys(self, steps_accounted: int) -> Dataset:
        snapshot_size = self.args.snapshot_size
        tar_file = self.args.data_filepath
        def times_from_names(files):
            files2times = {}
            times2files = {}

            base = datetime.strptime("19800101", '%Y%m%d')
            for file in files:
                delta =  (datetime.strptime(file[2:-4], '%Y%m%d') - base).days

                files2times[file] = delta
                times2files[delta] = file


            cont_files2times = {}

            sorted_times = sorted(files2times.values())
            new_t = 0

            for t in sorted_times:

                file = times2files[t]

                cont_files2times[file] = new_t

                new_t += 1
            return cont_files2times

        tar_archive = tarfile.open(tar_file, 'r:gz')
        files = tar_archive.getnames()

        cont_files2times = times_from_names(files)

        edges = []
        cols = u.Namespace({'source': 0,
                            'target': 1,
                            'time': 2,
                            'snapshot': 3})
        for file in files:
            data = u.load_data_from_tar(file,
                                        tar_archive,
                                        starting_line=4,
                                        sep='\t',
                                        type_fn = int,
                                        tensor_const = torch.LongTensor)

            # Is this really the correct way to turn a dict into a tensor?
            time_col = torch.zeros(data.size(0), 1 , dtype=torch.long) + cont_files2times[file]

            # This double time col thing is not pretty, but it kind of works.
            data = torch.cat([data, time_col, time_col], dim = 1)
            data = torch.cat([data, data[:,[cols.target,
                                            cols.source,
                                            cols.time,
                                            cols.snapshot]]])

            edges.append(data)

        edges.reverse()
        edges = torch.cat(edges)

        edges = du.make_contiguous_node_ids(edges, cols)

        #use only first X time steps
        # NOTE: This filters away some nodes and the node ids are no longer contiguous.
        # Max node counts and such are based on this and the decoder will now train on nodes which are not in the selected snapshots.
        # Chose to leave du.make_contiguous_node_ids above this step to remain comparable to EvolveGCN results. However, consider moving it below in the future to save time.

        indices = edges[:,cols.snapshot] < steps_accounted
        edges = edges[indices,:]

        #Snapshot col already added and time already normalized by previous code
        edges[:, cols.snapshot] = du.aggregate_by_time(edges[:, cols.time], snapshot_size)

        continuous = Dataset(edges, torch.ones(edges.size(0)))

        return continuous, du.continuous2discrete(continuous)

    def load_uc(self, edges_file: str) -> Dataset:
        snapshot_size = self.args.snapshot_size
        tar_file = self.args.data_filepath

        with tarfile.open(tar_file, 'r:bz2') as tar_archive:
            data = u.load_data_from_tar(edges_file,
                                        tar_archive,
                                        starting_line=2,
                                        sep=' ')
        edges = data.long()

        cols = u.Namespace({'source': 0,
                            'target': 1,
                            'weight': 2,
                            'time': 3,
                            'snapshot': 4})
        #first id should be 0 (they are already contiguous)
        edges[:,[cols.source, cols.target]] -= 1

        edges[:,cols.time] = du.normalize_time(edges[:, cols.time])
        snapshots = du.aggregate_by_time(edges[:, cols.time], snapshot_size)
        edges = torch.cat([edges, snapshots.view(-1, 1)], dim=1)

        continuous = Dataset(
            edges[:, [cols.source,
                      cols.target,
                      cols.time,
                      cols.snapshot]],
            edges[:, cols.weight])

        return continuous, du.continuous2discrete(continuous)

    def load_tgat_data(self) -> Dataset:
        snapshot_size = self.args.snapshot_size
        df, feat = du.tgat_preprocess(self.args.data_filepath)
        df = du.tgat_reindex(df, bipartite=True)
        df['ss'] = (df['ts'] / snapshot_size).apply(int)
        df = df[['u', 'i', 'ts', 'ss']]
        # First node should be node 0, not node 1.
        df[['u', 'i']] = df[['u', 'i']] - 1

        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        edge_features = np.vstack([empty, feat])

        edgesidx = torch.from_numpy(df.to_numpy()).long()
        continuous = Dataset(edgesidx, torch.zeros(edgesidx.size(0)), edge_features)#, dtype=torch.long))

        return continuous, du.continuous2discrete(continuous)

    def print_all_dataset_info_and_exit(self):
        dataset_names = ['enron', 'bitcoin-otc', 'autonomous-systems', 'uc', 'wikipedia', 'reddit']
        for dataset_name in dataset_names:
            from run_exp import read_data_master
            self.args = read_data_master(self.args, dataset_name=dataset_name)
            if dataset_name != 'autonomous-systems':
                self.args.snapshot_size = 60*60*24*self.args.snapshot_size #days to seconds
            self.args.snapshot_size = int(self.args.snapshot_size)
            if dataset_name == 'enron':
                cdataset, dataset = self.load_enron()
            elif dataset_name == 'bitcoin-otc':
                cdataset, dataset = self.load_bitcoin_otc()
            elif dataset_name == 'autonomous-systems':
                steps_accounted = self.args.steps_accounted
                cdataset, dataset = self.load_autosys(steps_accounted)
            elif dataset_name == 'uc':
                edges_file = 'opsahl-ucsocial/out.opsahl-ucsocial'
                cdataset, dataset = self.load_uc(edges_file)
            elif dataset_name == 'wikipedia':
                cdataset, dataset = self.load_tgat_data()
            elif dataset_name == 'reddit':
                cdataset, dataset = self.load_tgat_data()
            else:
                raise ValueError('Dataset {} not found'.format(self.args.data))

            if dataset_name != 'autonomous-systems':
                duration_in_days = int(cdataset.duration / (60.0*60*24))
            else:
                duration_in_days = cdataset.duration

            print('|{}|{}|{}|{}|{:.4f}|{}|{}|{}|{:.4f}|{}|{:d}|'.format(
                dataset_name,
                dataset.num_nodes,
                dataset.mean_snapshot_density,
                int(dataset.num_edges/2), #Divided by two because these are directed edges that have reciprocial edges added.
                int(dataset.num_unique_edges/2), #Divided by two because these are directed edges that have reciprocial edges added.
                dataset.density,
                dataset.num_snapshots,
                cdataset.num_edges,
                cdataset.num_unique_edges,
                cdataset.density,
                duration_in_days,
                round(cdataset.num_edges / cdataset.num_nodes)
            ))
        sys.exit(0)
