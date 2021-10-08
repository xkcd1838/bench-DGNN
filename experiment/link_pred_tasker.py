import torch
import taskers_utils as tu
import utils as u


class Link_Pred_Tasker:
    """
    Creates a tasker object which computes the required inputs for training on a link prediction
    task. It receives a dataset object which should have two attributes: nodes_feats and edges, this
    makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
    structure).

    Based on the dataset it implements the get_sample function required by edge_cls_trainer.
    This is a dictionary with:
        - time_step: the snapshot(time_step) of the prediction
        - hist_adj: the input adjacency matrices (actually an edge list) until t, each element of the list
                         is a sparse tensor with the current edges. For link_pred they're
                         unweighted. For static and continuous, this is a list of one element which is the one snapshot,
                         or the event list we learn from. For discrete this is a list of snapshots.
        - nodes_feats: the input nodes for the GCN models, each element of the list is a tensor
                          two dimmensions: node_idx and node_feats
        - label_adj: a sparse representation of the target edges. A dict with two keys: idx: M by 2
                     matrix with the indices of the nodes conforming each edge, vals: 1 if the node exists
                     , 0 if it doesn't

    There's a test difference in the behaviour, on test (or development), the number of sampled non existing
    edges should be higher.
    """

    def __init__(self, args, dataset, temporal_granularity):
        self.data = dataset
        self.temporal_granularity = temporal_granularity
        # max_time for link pred should be one before
        self.max_time = dataset.max_time - 1
        self.args = args
        self.num_classes = 2

        # build_get_node_feats does this check, what is this for?
        if not (args.use_2_hot_node_feats or args.use_1_hot_node_feats):
            self.feats_per_node = dataset.feats_per_node

        self.get_node_feats = self.build_get_node_feats(args, dataset)
        self.prepare_node_feats = self.build_prepare_node_feats(args, dataset)
        self.is_static = False

    def build_prepare_node_feats(self, args, dataset):
        if args.use_2_hot_node_feats or args.use_1_hot_node_feats:

            def prepare_node_feats(node_feats):
                return u.sparse_prepare_tensor(
                    node_feats, torch_size=[dataset.num_nodes, self.feats_per_node]
                )

        else:
            prepare_node_feats = self.data.prepare_node_feats

        return prepare_node_feats

    def build_get_node_feats(self, args, dataset):
        max_deg_out, max_deg_in = tu.get_max_degs(args, dataset)
        if args.use_2_hot_node_feats:
            self.feats_per_node = max_deg_out + max_deg_in

            def get_node_feats(adj):
                return tu.get_2_hot_deg_feats(
                    adj, max_deg_out, max_deg_in, dataset.num_nodes
                )

        elif args.use_1_hot_node_feats:
            max_deg = max_deg_out
            self.feats_per_node = max_deg

            def get_node_feats(adj):
                return tu.get_1_hot_deg_feats(adj, max_deg, dataset.num_nodes)

        elif args.use_random_node_feats:
            num_features = max_deg_out
            self.feats_per_node = num_features

            def get_node_feats(adj):
                return tu.get_random_features(adj, num_features, dataset.num_nodes)

        else:

            def get_node_feats(adj):
                return dataset.nodes_feats

        return get_node_feats

    def get_sample(self, idx, partition, **kwargs):

        snapshot_based = (
            False  # Default is not snapshot based, since we want continuous training
        )
        if "snapshot_based" in kwargs.keys() and kwargs["snapshot_based"] == True:
            snapshot_based = True
        if self.temporal_granularity == "static":
            (
                hist_adj,
                hist_ndFeats,
                hist_node_mask,
                existing_nodes,
            ) = self.get_static_hist(idx, snapshot_based)
        elif self.temporal_granularity == "discrete":
            (
                hist_adj,
                hist_ndFeats,
                hist_node_mask,
                existing_nodes,
            ) = self.get_dicrete_hist(idx)
        elif self.temporal_granularity == "continuous":
            split_start = None
            if not snapshot_based:
                if "split_start" in kwargs.keys():
                    split_start = kwargs["split_start"]
                else:
                    raise ValueError(
                        "split start required for non-snapshot based sample."
                    )
            (
                hist_adj,
                hist_ndFeats,
                hist_node_mask,
                existing_nodes,
            ) = self.get_continuous_hist(idx, snapshot_based, split_start)
        else:
            raise ValueError(
                "Temporal granularity can be either: static, discrete or continuous, but was {}".format(
                    self.temporal_granularity
                )
            )

        # label_adj is an edge list of the edges we test on.
        label_adj = tu.get_sp_adj(
            edges=self.data.edges,
            snapshot=idx + 1,
            weighted=False,
            time_window=1,
            temporal_granularity="static",  # This is for testing, not for training. Must therefore be the same for all models
        )

        if partition == "TEST":
            neg_mult = self.args.negative_mult_test
        else:
            neg_mult = self.args.negative_mult_training

        if self.args.smart_neg_sampling:
            existing_nodes = torch.cat(existing_nodes)

        add_all_edges = "all_edges" in kwargs.keys() and kwargs["all_edges"] == True
        if add_all_edges:  # This is the case for link pred
            non_existing_adj = tu.get_all_non_existing_edges(
                adj=label_adj, tot_nodes=self.data.num_nodes
            )
        else:
            non_existing_adj = tu.get_non_existing_edges(
                adj=label_adj,
                num_edges=label_adj["vals"].size(0) * neg_mult,
                tot_nodes=self.data.num_nodes,
                smart_sampling=self.args.smart_neg_sampling,
                existing_nodes=existing_nodes,
            )

        if self.args.model == "seal":  # Include labels to support building SEAL Dataset
            label_exist = label_adj["idx"].detach().clone()
            label_non_exist = non_existing_adj["idx"].detach().clone()

        label_adj["idx"] = torch.cat([label_adj["idx"], non_existing_adj["idx"]])
        label_adj["vals"] = torch.cat([label_adj["vals"], non_existing_adj["vals"]])

        # Extra tensors needed for special cases
        # 1. Some metrics require that we distinguish between new and reappearing edges
        hist_link_probs = (
            hasattr(self.args, "include_existing_edges")
            and self.args.include_existing_edges == "adaptive"
        )
        if partition == "TEST" or (
            partition == "VALID"
            and self.args.target_measure.lower() in ["gmauc", "lp_map", "lp_auc"]
            or hist_link_probs
        ):
            # All edges that appeared up until this point, used to calculate some metrics
            prev_adj = tu.get_sp_adj(
                edges=self.data.edges,
                snapshot=idx,
                weighted=False,
                time_window=None,
                temporal_granularity="static",
            )
        else:
            prev_adj = {"idx": [], "vals": []}

        return {
            "idx": idx,
            "hist_adj": hist_adj,
            "hist_ndFeats": hist_ndFeats,
            "label_sp": label_adj,
            "hist_node_mask": hist_node_mask,
            "prev_adj": prev_adj,
        }

    def get_static_hist(self, idx, snapshot_based):
        use_expanding_time_window = (
            hasattr(self.args, "num_hist_steps")
            and (
                self.args.num_hist_steps in ["expanding", "static"]
                or 0 > self.args.num_hist_steps
            )
        ) or not snapshot_based

        if use_expanding_time_window:
            time_window = None
        else:
            time_window = self.args.num_hist_steps

        adj = tu.get_sp_adj(
            edges=self.data.edges, snapshot=idx, weighted=True, time_window=time_window
        )

        if self.args.smart_neg_sampling:
            existing_nodes = adj["idx"].unique()
        else:
            existing_nodes = None
        node_mask = tu.get_node_mask(adj, self.data.num_nodes)
        node_features = self.get_node_feats(adj)
        adj = tu.normalize_adj(adj=adj, num_nodes=self.data.num_nodes)

        # print('existing nodes', len(existing_nodes)) #Verify node dynamics
        return (
            [adj],
            [node_features],
            [node_mask],
            [existing_nodes],
        )  # Lists of one element to fit the rest of the framework

    def get_continuous_hist(self, idx, snapshot_based=False, split_start=None):
        if not snapshot_based:
            # All preceding edges presented at once. Used for contrastive training of the DGNNs
            assert split_start is not None
            adj = tu.get_sp_adj(
                edges=self.data.edges,
                snapshot=idx,
                weighted=True,
                time_window=idx - split_start,
                temporal_granularity="continuous",
            )
        else:
            # Snapshot based. Deliver one snapshot at a time, used for downstream learning
            # Time window is 1, because the continuous model has already seen the data leading up to this point. And only needs the next snapshot.
            adj = tu.get_sp_adj(
                edges=self.data.edges,
                snapshot=idx,
                weighted=True,
                time_window=1,
                temporal_granularity="continuous",
            )

        if self.args.smart_neg_sampling:
            existing_nodes = adj["idx"].unique()
        else:
            existing_nodes = None
        node_mask = tu.get_node_mask(adj, self.data.num_nodes)

        # Use a strictly evolving network to get one hot encoded node features
        static_adj = tu.get_sp_adj(
            edges=self.data.edges,
            snapshot=idx,
            weighted=False,
            time_window=None,
            temporal_granularity="static",
        )
        node_features = self.get_node_feats(static_adj)
        return (
            [adj],
            [node_features],
            [node_mask],
            [existing_nodes],
        )  # Lists of one element to fit the rest of the framework

    def get_dicrete_hist(self, idx):
        hist_adj_list = []
        hist_ndFeats_list = []
        hist_mask_list = []
        existing_nodes = []

        for i in range(idx - self.args.num_hist_steps + 1, idx + 1):
            cur_adj = tu.get_sp_adj(
                edges=self.data.edges, snapshot=i, weighted=True, time_window=1
            )

            if self.args.smart_neg_sampling:
                existing_nodes.append(cur_adj["idx"].unique())
            else:
                existing_nodes = None

            node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

            node_feats = self.get_node_feats(cur_adj)

            cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)

            hist_adj_list.append(cur_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)
        return hist_adj_list, hist_ndFeats_list, hist_mask_list, existing_nodes
