import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as pygeomDataLoader
from torch_geometric.data import InMemoryDataset

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import numpy as np
import random
import utils as u


class splitter:
    """
    creates 3 splits as data_split classes
    train
    val
    test

    """

    def __init__(
        self,
        args,
        tasker,
        temporal_granularity,
        test_tasker=None,
        all_in_one_snapshot=False,
        downstream=False,
        eval_all_edges=True,
        frozen_encoder=False,
        train_all_edges=False,
    ):
        self.temporal_granularity = temporal_granularity
        assert (
            args.train_proportion + args.val_proportion < 1
        ), "there's no space for test samples"

        if downstream:
            data_split = Static_train__double_tasker_data_split
        elif frozen_encoder:
            data_split = Double_tasker_data_split
        else:
            if all_in_one_snapshot:
                # These should not be used for the final evaluation
                # Since they have no snapshots in val and test and thus the evaluation is not easily comparable
                if self.temporal_granularity == "continuous":
                    data_split = Continuous_data_split
                else:
                    data_split = Static_data_split
            elif hasattr(args, "num_hist_steps") and args.num_hist_steps == "static":
                if self.temporal_granularity != "static":
                    raise (
                        "num_hist_steps set to static, but temporal granularity is not static."
                    )
                data_split = Static_train_data_split
            else:
                data_split = Data_split

        ## Train
        # only the training one requires special handling on start, the others are fine with the split IDX.
        if (
            args.num_hist_steps in ["expanding", "static"]
            or self.temporal_granularity == "continuous"
            or downstream
        ):
            start_offset = 0
        else:
            start_offset = args.num_hist_steps
        start = tasker.data.min_time + start_offset
        end = args.train_proportion

        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        train = data_split(
            tasker,
            start,
            end,
            partition="TRAIN",
            test_tasker=test_tasker,
            all_edges=train_all_edges,
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        train = DataLoader(
            train, worker_init_fn=seed_worker, **args.data_loading_params
        )  # Why no num workers here? - Less data here

        start = end
        end = args.val_proportion + args.train_proportion
        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        if args.task == "link_pred":
            val = data_split(
                tasker,
                start,
                end,
                partition="VALID",
                test_tasker=test_tasker,
                all_edges=eval_all_edges,
            )
        else:
            val = data_split(tasker, start, end, partition="VALID")

        val = DataLoader(val, num_workers=args.data_loading_params["num_workers"])

        ## Test
        start = end
        # the +1 is because I assume that max_time exists in the dataset
        end = int(tasker.max_time) + 1
        if args.task == "link_pred":
            test = data_split(
                tasker,
                start,
                end,
                partition="TEST",
                test_tasker=test_tasker,
                all_edges=eval_all_edges,
            )
        else:
            test = data_split(tasker, start, end, partition="TEST")

        test = DataLoader(test, num_workers=args.data_loading_params["num_workers"])

        print("Start offset:", start_offset)
        print(
            "Dataset splits sizes:  train",
            len(train),
            "val",
            len(val),
            "test",
            len(test),
        )

        self.tasker = tasker
        self.train = train
        self.val = val
        self.test = test


class Data_split(Dataset):
    def __init__(self, tasker, start, end, partition, **kwargs):
        """
        start and end are indices indicating what items belong to this split
        """
        self.tasker = tasker
        self.start = start
        self.end = end
        self.partition = partition
        self.kwargs = kwargs

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        idx = self.start + idx
        t = self.tasker.get_sample(
            idx, partition=self.partition, snapshot_based=True, **self.kwargs
        )
        return t


"""
The continuous datasplit is "snapshotless" this is implemented as it containing only one snapshot.
In practice the continuous training is then done on 3 snapshots, training, validation and test.
"""


class Continuous_data_split(Dataset):
    def __init__(self, tasker, start, end, partition, **kwargs):
        """
        start and end are indices indicating what items belong to this split
        """
        self.tasker = tasker
        self.start = start
        self.end = end
        self.partition = partition
        self.kwargs = kwargs

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        t = self.tasker.get_sample(
            self.end - 1,
            partition=self.partition,
            split_start=self.start,
            **self.kwargs
        )
        return t


# Serve one sample for train. (Downstream learners can only fit once)
# Otherwise act as a normal datasplit and serve snapshots
# Thus train become on snapshot, while val and test are snapshots like the default datasplitter
class Static_train_data_split(Dataset):
    def __init__(self, tasker, start, end, partition, **kwargs):
        """
        start and end are indices indicating what items belong to this split
        """
        self.tasker = tasker
        self.start = start
        self.end = end
        self.partition = partition
        self.kwargs = kwargs

    def __len__(self):
        if self.partition == "TRAIN":
            return 1
        else:
            return self.end - self.start

    def __getitem__(self, idx):
        if self.partition == "TRAIN":
            t = self.tasker.get_sample(
                self.end - 1,
                partition=self.partition,
                split_start=self.start,
                snapshot_based=False,
                **self.kwargs
            )
        else:
            idx = self.start + idx
            t = self.tasker.get_sample(
                idx,
                partition=self.partition,
                split_start=self.start,
                snapshot_based=True,
                **self.kwargs
            )

        return t


# Split with no snapshots
class Static_data_split(Dataset):
    def __init__(self, tasker, start, end, partition, **kwargs):
        """
        start and end are indices indicating what items belong to this split
        """
        self.tasker = tasker
        self.start = start
        self.end = end
        self.partition = partition
        self.kwargs = kwargs

    def __len__(self):
        return 1
        # if self.partition == 'TRAIN':
        #    return 1
        # else:
        #    return self.end - self.start

    def __getitem__(self, idx):
        idx = self.start + idx
        t = self.tasker.get_sample(
            self.end - 1,
            partition=self.partition,
            split_start=self.start,
            snapshot_based=False,
            **self.kwargs
        )
        return t


# Same as data_split except that it takes two taskers and returns one sample for each
# This allows it to return one for a continuous dataset and one for a discrete dataset
class Double_tasker_data_split(Dataset):
    def __init__(self, tasker, start, end, partition, **kwargs):
        """
        start and end are indices indicating what items belong to this split
        """
        self.tasker = tasker
        self.start = start
        self.end = end
        self.partition = partition
        self.test_tasker = kwargs["test_tasker"]
        self.kwargs = kwargs

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        idx = self.start + idx

        # Get the continuous one for training
        t1 = self.tasker.get_sample(
            idx, partition=self.partition, snapshot_based=True, **self.kwargs
        )

        # Get the discrete one for validation & testing
        t2 = self.test_tasker.get_sample(
            idx, partition=self.partition, snapshot_based=True, **self.kwargs
        )
        return t1, t2


class Static_train__double_tasker_data_split(Dataset):
    def __init__(self, tasker, start, end, partition, **kwargs):
        """
        start and end are indices indicating what items belong to this split
        """
        self.tasker = tasker
        self.start = start
        self.end = end
        self.partition = partition
        self.test_tasker = kwargs["test_tasker"]
        self.kwargs = kwargs

    def __len__(self):
        if self.partition == "TRAIN":
            return 1
        else:
            return self.end - self.start

    def __getitem__(self, idx):
        if self.partition == "TRAIN":
            t1 = self.tasker.get_sample(
                self.end - 1,
                partition=self.partition,
                split_start=self.start,
                snapshot_based=False,
                **self.kwargs
            )
            t2 = self.test_tasker.get_sample(
                self.end - 1,
                partition=self.partition,
                split_start=self.start,
                snapshot_based=False,
                **self.kwargs
            )
        else:
            idx = self.start + idx
            t1 = self.tasker.get_sample(
                idx,
                partition=self.partition,
                split_start=self.start,
                snapshot_based=True,
                **self.kwargs
            )
            t2 = self.tasker.get_sample(
                idx,
                partition=self.partition,
                split_start=self.start,
                snapshot_based=True,
                **self.kwargs
            )

        return t1, t2
