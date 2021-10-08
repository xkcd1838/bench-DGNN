import os
import functools
import utils as u
import logger
import pandas as pd
import numpy as np
import networkx as nx

import torch
import math
import os.path

from itertools import chain


class Trainer:
    def __init__(self, args, splitter, hist_splitter, heuristic):
        self.args = args
        self.splitter = splitter
        self.hist_splitter = hist_splitter
        self.tasker = splitter.tasker
        self.dataset = self.tasker.data
        self.heuristic = heuristic

        self.hist_probs = [
            {
                "new_app": 0,
                "new_tot": 0,
                "rec_app": 0,
                "rec_tot": 0,
            }
            for i in range(self.dataset.num_snapshots)
        ]

        self.num_nodes = self.tasker.data.num_nodes
        self.num_classes = self.tasker.num_classes

    def train(self):
        self.logger = logger.Logger(
            self.args, self.num_classes, self.num_nodes, train_encoder=False
        )

        # Calculate historical probabilities of links appearing
        if self.args.include_existing_edges == "adaptive":
            self.hist_link_probs(self.hist_splitter)

        # Heuristics don't learn anything so we run it straight on the test set
        if "random" not in self.args.model:
            eval_test = self.run_epoch(self.splitter.test, 1, "TEST")
        else:
            eval_test = self.run_epoch_random(self.splitter.test, 1, "TEST")

        self.logger.close()

    def reappearing_mask(self, adj, prev_adj):
        adj = adj.cpu().numpy()
        prev_adj = prev_adj.squeeze().cpu().numpy()

        prev_edge_set = set(tuple(edge) for edge in prev_adj)

        def prev_edge(n1, n2):
            return (n1, n2) in prev_edge_set or (n2, n1) in prev_edge_set

        vprev_edge = np.vectorize(prev_edge)
        mask = vprev_edge(adj[0], adj[1])
        return mask, prev_edge_set

    def hist_link_probs(self, splitter):
        for s in chain(splitter.train, splitter.val, splitter.test):
            s = self.prepare_sample(
                s, self.args.temporal_granularity, only_label_sp=True
            )
            adj = s.label_sp["idx"]
            prev_adj = s.prev_adj["idx"]
            mask, prev_edge_set = self.reappearing_mask(adj, prev_adj)

            # Discrete network with undirected edges. One edge is counted twice, a->b and b->a.
            rec_app = (
                s.label_sp["vals"][mask].sum().item()
            )  # Actually reoccurring links
            rec_tot = (
                len(prev_edge_set) * 2
            )  # Possible reoccurring links (x2 cuz undirected)
            new_app = s.label_sp["vals"].sum().item() - rec_app  # Actually new links
            tot_edges = s.label_sp["vals"].size()[0]
            new_tot = tot_edges - rec_tot  # Total possible new links

            # print("rec", rec_app/rec_tot, "app", new_app/new_tot)
            if s.idx > 0:
                prev_s = self.hist_probs[s.idx - 1]
                rec_app += prev_s["rec_app"]
                rec_tot += prev_s["rec_tot"]
                new_app += prev_s["new_app"]
                new_tot += prev_s["new_tot"]
            hp = self.hist_probs[s.idx]
            hp["rec_app"] = rec_app
            hp["rec_tot"] = rec_tot
            hp["new_app"] = new_app
            hp["new_tot"] = new_tot

    def _epoch_decorator(run_epoch_func):
        @functools.wraps(run_epoch_func)
        def wrapper(*args, **kwards):
            self = args[0]
            split = args[1]
            epoch = args[2]
            set_name = args[3]

            log_interval = 999
            if set_name == "TEST":
                log_interval = 1

            self.logger.log_epoch_start(
                epoch, len(split), set_name, minibatch_log_interval=log_interval
            )
            run_epoch_func(*args, **kwards)
            eval_measure = self.logger.log_epoch_done()
            return eval_measure

        return wrapper

    @_epoch_decorator
    def run_epoch(self, split, epoch, set_name):
        # Epoch
        j = 0
        for s in split:
            s = self.prepare_sample(
                s, self.args.temporal_granularity, only_label_sp=True
            )
            j = j + 1
            # print("ss", j, 'set name', set_name)

            fake_loss = torch.tensor(0.0)  # No loss since we only encode
            hist_adj = s.hist_adj[0].t().tolist()

            # Calculate scores for Networkx heuristic
            G = nx.Graph(hist_adj)
            G.remove_edges_from(nx.selfloop_edges(G))

            if (
                hasattr(self.args, "include_existing_edges")
                and self.args.include_existing_edges == True
            ):
                ebunch = G.edges
            else:
                ebunch = nx.non_edges(G)
            pred_generator = self.heuristic(G, ebunch=ebunch)
            pred_list = []
            c = 0
            max_links = self.num_nodes ** 2 / 2
            for edge in pred_generator:
                # if c % 100000 == 0:
                #     print(self.args.model, " edge nr", c, "-", (100 * c) / float(max_links), "%",)
                # c += 1
                pred_list.append(edge)
                # Needed since the an undirected link is represented as two links in the matrices
                pred_list.append((edge[1], edge[0], edge[2]))

            scores_unordered = np.array(pred_list)

            # Normalize
            log_norm = np.log(1 + scores_unordered[:, 2])
            norm_scores_unordered = log_norm / log_norm.max()

            # Used for sorting the scores returned by the heuristic
            # A better solution surely exists. This is inefficient in both time and space
            translator = {}
            for i in range(len(s.label_sp["vals"])):
                link = tuple(s.label_sp["idx"][:, i].tolist())
                translator[link] = i

            # Link heuristics don't give a score for existing links.
            # Here we choose the probability for an existing link to appear in the next snapshot.
            if self.args.include_existing_edges == "adaptive":
                hist_probs = self.hist_probs[s.idx]
                alpha = hist_probs["rec_app"] / hist_probs["rec_tot"]
            else:
                alpha = 1
            scores = np.ones(s.label_sp["vals"].size()) * alpha

            # Sort unordered scores
            for i in range(len(scores_unordered)):
                row = scores_unordered[i]
                score = norm_scores_unordered[i]
                link = tuple(row[:2].tolist())
                try:
                    j = translator[link]
                except:
                    # Will happen if the link is predicted by the heuristic but not in label_sp.
                    print("Link {} not in translator. Skipping".format(link))
                    continue
                scores[j] = score

            probs = torch.tensor(scores)

            # Calculate the probability of a link not existing. Different thresholds can be used for this.
            # Default = 0.5, which turns the 2*threshold- scores into 1-scores.
            # The threshold only affects precision, recall, error and confmap metrics.
            threshold = 0.5
            no_link_probs = np.clip(2 * threshold - scores, 0, 1)
            predictions = torch.tensor(np.stack([no_link_probs, scores], axis=1))

            self.logger.log_minibatch(
                fake_loss.detach(),
                predictions,
                probs,
                s.label_sp["vals"],
                adj=s.label_sp["idx"],
                prev_adj=s.prev_adj["idx"],
            )

    @_epoch_decorator
    def run_epoch_random(self, split, epoch, set_name):
        # Epoch
        j = 0
        for s in split:
            s = self.prepare_sample(
                s, self.args.temporal_granularity, only_label_sp=True
            )
            j = j + 1

            fake_loss = torch.tensor(0.0)  # No loss since we only encode

            if self.args.model == "random_heuristic":
                scores = np.random.rand(s.label_sp["vals"].size()[0])
            elif self.args.model == "random_adaptive":
                assert self.args.include_existing_edges == "adaptive"

                adj = s.label_sp["idx"]
                prev_adj = s.prev_adj["idx"]

                mask, prev_edge_set = self.reappearing_mask(adj, prev_adj)

                hist_probs = self.hist_probs[s.idx]
                rec_prob = hist_probs["rec_app"] / hist_probs["rec_tot"]
                new_prob = hist_probs["new_app"] / hist_probs["new_tot"]

                # print(hist_probs["new_app"], hist_probs["new_tot"])
                # print("reoccurring", rec_prob, "new", new_prob)
                scores = (
                    np.ones(s.label_sp["vals"].size()) * new_prob
                )  # Set to prob for new links
                scores[mask] = rec_prob  # Set to prob for reoccurring links
            else:
                raise ValueError(
                    "Random heuristic (model) {} not found".format(self.args.model)
                )

            probs = torch.tensor(scores)
            predictions = torch.tensor(np.stack([1 - scores, scores], axis=1))

            self.logger.log_minibatch(
                fake_loss.detach(),
                predictions,
                probs,
                s.label_sp["vals"],
                adj=s.label_sp["idx"],
                prev_adj=s.prev_adj["idx"],
            )

    def prepare_sample(
        self, sample, temporal_granularity="static", only_label_sp=False
    ):
        sample = u.Namespace(sample)
        sample.hist_vals, sample.hist_time = [], []
        # For the static and continuous case there will be only one iteration
        for i, adj in enumerate(sample.hist_adj):
            # Prepares an edge index (edge list) as expected by PyTorch Geometric
            # Squeeze removes dimensions of size 1
            vals = adj["vals"].squeeze().t()
            sample.hist_vals.append(vals.to(self.args.device))
            if temporal_granularity == "continuous":
                hist_time = adj["time"].squeeze().t()
                sample.hist_time.append(hist_time.to(self.args.device))

            if hasattr(self.args, "pygeom") and self.args.pygeom == False:
                # Only used for the original implementation of EGCN
                adj_idx = u.sparse_prepare_tensor(adj, torch_size=[self.num_nodes])
            else:
                adj_idx = adj["idx"].squeeze().t()
            sample.hist_adj[i] = adj_idx.to(self.args.device)

            if (
                not only_label_sp
            ):  # Created some problems for reddit_tgn, we don't use this there anyways.
                nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats[i])

                sample.hist_ndFeats[i] = nodes.to(self.args.device)
                hist_node_mask = sample.hist_node_mask[i]
                sample.hist_node_mask[i] = hist_node_mask.to(
                    self.args.device
                ).t()  # transposed to have same dimensions as scorer

        label_sp = self.ignore_batch_dim(sample.label_sp)

        if self.args.task in ["link_pred", "edge_cls"]:
            label_sp["idx"] = label_sp["idx"].to(self.args.device).t()
        else:
            label_sp["idx"] = label_sp["idx"].to(self.args.device)

        label_sp["vals"] = label_sp["vals"].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    def ignore_batch_dim(self, adj):
        if self.args.task in ["link_pred", "edge_cls"]:
            adj["idx"] = adj["idx"][0]
        adj["vals"] = adj["vals"][0]
        return adj

    def random(self, model):
        if model == "random_heuristic":
            return np.random.rand()
        pass
