import logging
import pprint
import sys
import math
import torch
import time
import random
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.sparse import coo_matrix
import numpy as np
import utils
from joblib import Parallel, delayed


class Logger:
    def __init__(
        self,
        args,
        num_classes,
        num_nodes,
        classifier_name="decoder",
        minibatch_log_interval=10,
        train_encoder=True,
    ):

        if args is not None:
            self.log_name = utils.get_log_name(args, classifier_name, train_encoder)

            if args.use_logfile:
                print(classifier_name, " Log file:", self.log_name)
                handler = logging.FileHandler(self.log_name)
                utils.remove_log_lock(args)
            else:
                print("Log: STDOUT")
                handler = logging.StreamHandler(sys.stdout)
        else:
            print("Log: STDOUT")
            handler = logging.StreamHandler(sys.stdout)

        self.logger = logging.getLogger(classifier_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        self.logger.info("*** PARAMETERS ***")
        self.logger.info(pprint.pformat(args.__dict__))  # displays the string
        self.logger.info("")

        self.classifier_name = classifier_name
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.minibatch_log_interval = minibatch_log_interval
        self.eval_k_list = [10, 100, 1000]
        self.args = args

    def close(self):
        self.logger.info("##### FINISHED")
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def get_log_file_name(self):
        return self.log_name

    def log_epoch_start(
        self, epoch, num_minibatches, partition, minibatch_log_interval=None
    ):
        self.epoch = epoch
        self.partition = partition
        self.info_preamble = self.partition
        self.losses = []
        self.errors = []
        self.MRRs = []
        self.GMAUCs = []
        self.LP_MAPs = []
        self.LP_AUCs = []
        self.MAPs = []
        self.AUCs = []

        self.neg_samples = []
        self.conf_mat_tp = {}
        self.conf_mat_fn = {}
        self.conf_mat_fp = {}
        self.conf_mat_tp_at_k = {}
        self.conf_mat_fn_at_k = {}
        self.conf_mat_fp_at_k = {}
        for k in self.eval_k_list:
            self.conf_mat_tp_at_k[k] = {}
            self.conf_mat_fn_at_k[k] = {}
            self.conf_mat_fp_at_k[k] = {}

        for cl in range(self.num_classes):
            self.conf_mat_tp[cl] = 0
            self.conf_mat_fn[cl] = 0
            self.conf_mat_fp[cl] = 0
            for k in self.eval_k_list:
                self.conf_mat_tp_at_k[k][cl] = 0
                self.conf_mat_fn_at_k[k][cl] = 0
                self.conf_mat_fp_at_k[k][cl] = 0

        if self.partition == "TEST":
            self.conf_mat_tp_list = {}
            self.conf_mat_fn_list = {}
            self.conf_mat_fp_list = {}
            for cl in range(self.num_classes):
                self.conf_mat_tp_list[cl] = []
                self.conf_mat_fn_list[cl] = []
                self.conf_mat_fp_list[cl] = []

        self.batch_sizes = []
        self.minibatch_done = 0
        self.num_minibatches = num_minibatches
        if minibatch_log_interval is not None:
            self.minibatch_log_interval = minibatch_log_interval
        self.logger.info(
            "################ "
            + self.info_preamble
            + " epoch "
            + str(epoch)
            + " ###################"
            + " time "
            + str(int(time.time()))
        )
        self.lasttime = time.monotonic()
        self.ep_time = self.lasttime

    def log_minibatch(self, loss, predictions, probs, true_classes, **kwargs):
        loss = loss.cpu()
        predictions = predictions.cpu()
        probs = probs.cpu()
        true_classes = true_classes.cpu()

        probs_np = probs.numpy()
        true_classes_np = true_classes.numpy()

        # Initialize everything
        MRR = torch.tensor(0.0)
        GMAUC = torch.tensor(0.0)
        LP_MAP = torch.tensor(0.0)
        LP_AUC = torch.tensor(0.0)
        MAP = torch.tensor(0.0)
        AUC = torch.tensor(0.0)
        error = torch.tensor(0.0)
        conf_mat_per_class = {}
        for cl in range(self.num_classes):
            self.conf_mat_tp[cl] += torch.tensor(0.0)
            self.conf_mat_fn[cl] += torch.tensor(0.0)
            self.conf_mat_fp[cl] += torch.tensor(0.0)
            for k in self.eval_k_list:
                self.conf_mat_tp_at_k[k][cl] += torch.tensor(0.0)
                self.conf_mat_fn_at_k[k][cl] += torch.tensor(0.0)
                self.conf_mat_fp_at_k[k][cl] += torch.tensor(0.0)
        adj, adj_np, prev_adj, new_probs, new_tc, rec_probs, rec_tc = [None] * 7

        # Ensure that continuous-DGNN link interpolation training
        # doesn't waste time calculating metrics
        calc_lp_metrics = True
        if "calc_lp_metrics" in kwargs and kwargs["calc_lp_metrics"] == False:
            calc_lp_metrics = False

        if self.partition == "VALID":
            target_measure = self.args.target_measure.lower()

            if target_measure == "mrr":
                adj = kwargs["adj"].cpu()
                adj_np = adj.numpy()
            elif target_measure in ["gmauc", "lp_map", "lp_auc"]:
                adj = kwargs["adj"].cpu()
                prev_adj = kwargs["prev_adj"]
                new_probs, new_tc, rec_probs, rec_tc = self.new_reappear_split_tensor(
                    probs, true_classes, adj, prev_adj
                )
            metrics_to_calc = [target_measure]

        elif (
            calc_lp_metrics
            and self.partition in ["TEST", "VALID"]
            and self.args.task == "link_pred"
        ):
            adj = kwargs["adj"].cpu()
            adj_np = adj.numpy()
            prev_adj = kwargs["prev_adj"]
            new_probs, new_tc, rec_probs, rec_tc = self.new_reappear_split_tensor(
                probs, true_classes, adj, prev_adj
            )
            metrics_to_calc = [
                "mrr",
                "gmauc",
                "lp_map",
                "lp_auc",
                "map",
                "auc",
                "confmat",
            ]
        else:
            metrics_to_calc = ["map", "auc", "confmat"]

        def calculate_metric(
            metric_name,
            probs,
            true_classes,
            predictions_torch,
            true_classes_torch,
            new_probs,
            new_tc,
            rec_probs,
            rec_tc,
            adj,
        ):
            metric = torch.tensor(0.0)
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    if metric_name == "mrr":
                        metric = self.get_MRR(probs, true_classes, adj)
                    elif metric_name == "gmauc":
                        metric = torch.tensor(
                            self.get_GMAUC(new_probs, new_tc, rec_probs, rec_tc)
                        )
                    elif metric_name == "lp_map":
                        metric = torch.tensor(self.get_MAP(new_probs, new_tc))
                    elif metric_name == "lp_auc":
                        metric = torch.tensor(self.get_AUC(new_probs, new_tc))
                    elif metric_name == "map":
                        metric = torch.tensor(self.get_MAP(probs, true_classes))
                    elif metric_name == "auc":
                        metric = torch.tensor(self.get_AUC(probs, true_classes))
                    elif metric_name == "confmat":
                        metric = self.eval_predicitions(
                            predictions_torch, true_classes_torch, self.num_classes
                        )
                    else:
                        raise Exception("Metric {}, not found".format(metric_name))
            except ValueError as e:
                self.logger.info(
                    "Encountered value error in calculating metric {}, most likely there are no new links or no reoccurring links. Error: {}".format(
                        metric_name, e
                    )
                )
            if type(metric) == type(torch.tensor(0.0)) and torch.isnan(metric):
                self.logger.info("Metric {} was nan setting to 0".format(metric_name))
                metric[
                    torch.isnan(metric)
                ] = 0  # Set metric to 0 if nan, may happen with GMAUC or LP_MAP
            return (metric_name, metric)

        if len(metrics_to_calc) == 1:
            results = [
                calculate_metric(
                    metrics_to_calc[0],
                    probs_np,
                    true_classes_np,
                    predictions,
                    true_classes,
                    new_probs,
                    new_tc,
                    rec_probs,
                    rec_tc,
                    adj_np,
                )
            ]
        else:
            parallel = Parallel(n_jobs=len(metrics_to_calc))
            results = parallel(
                delayed(calculate_metric)(
                    metric_name,
                    probs_np,
                    true_classes_np,
                    predictions,
                    true_classes,
                    new_probs,
                    new_tc,
                    rec_probs,
                    rec_tc,
                    adj_np,
                )
                for metric_name in metrics_to_calc
            )

        for metric_name, metric in results:
            if metric_name == "mrr":
                MRR = metric
            elif metric_name == "gmauc":
                GMAUC = metric
            elif metric_name == "lp_map":
                LP_MAP = metric
            elif metric_name == "lp_auc":
                LP_AUC = metric
            elif metric_name == "map":
                MAP = metric
            elif metric_name == "auc":
                AUC = metric
            elif metric_name == "confmat":
                error, conf_mat_per_class = metric
            else:
                raise Exception("Metric {}, not found".format(metric_name))

        batch_size = predictions.size(0)

        self.batch_sizes.append(batch_size)
        self.losses.append(loss)
        self.errors.append(error)
        self.MRRs.append(MRR)
        self.GMAUCs.append(GMAUC)
        self.LP_MAPs.append(LP_MAP)
        self.LP_AUCs.append(LP_AUC)
        self.MAPs.append(MAP)
        self.AUCs.append(AUC)

        if self.partition == "VALID":
            # Return early to save time
            self.lasttime = time.monotonic()
            return

        if self.partition == "TEST":
            # Negative sampling metrics
            if (
                hasattr(self.args, "log_negative_sample_range")
                and self.args.log_negative_sample_range == True
            ):
                negative_sampling = self.get_negative_sample_metrics(
                    probs_np, true_classes_np
                )
            else:
                ks = [1000, 100, 10, 1]
                empty_neg_samples = {}
                for k in ks:
                    empty_neg_samples["{}_{}".format(k, "map")] = torch.tensor(0.0)
                    empty_neg_samples["{}_{}".format(k, "auc")] = torch.tensor(0.0)
                negative_sampling = empty_neg_samples
            self.neg_samples.append(negative_sampling)
        conf_mat_per_class_at_k = {}
        for k in self.eval_k_list:
            conf_mat_per_class_at_k[k] = self.eval_predicitions_at_k(
                predictions, true_classes, self.num_classes, k
            )
        for cl in range(self.num_classes):
            self.conf_mat_tp[cl] += conf_mat_per_class.true_positives[cl]
            self.conf_mat_fn[cl] += conf_mat_per_class.false_negatives[cl]
            self.conf_mat_fp[cl] += conf_mat_per_class.false_positives[cl]
            for k in self.eval_k_list:
                self.conf_mat_tp_at_k[k][cl] += conf_mat_per_class_at_k[
                    k
                ].true_positives[cl]
                self.conf_mat_fn_at_k[k][cl] += conf_mat_per_class_at_k[
                    k
                ].false_negatives[cl]
                self.conf_mat_fp_at_k[k][cl] += conf_mat_per_class_at_k[
                    k
                ].false_positives[cl]
            if self.partition == "TEST":
                self.conf_mat_tp_list[cl].append(conf_mat_per_class.true_positives[cl])
                self.conf_mat_fn_list[cl].append(conf_mat_per_class.false_negatives[cl])
                self.conf_mat_fp_list[cl].append(conf_mat_per_class.false_positives[cl])

        self.minibatch_done += 1
        if self.minibatch_done % self.minibatch_log_interval == 0:
            mb_error = self.calc_epoch_metric(self.batch_sizes, self.errors)
            mb_MRR = self.calc_epoch_metric(self.batch_sizes, self.MRRs)
            mb_GMAUC = self.calc_epoch_metric(self.batch_sizes, self.GMAUCs)
            mb_LP_MAP = self.calc_epoch_metric(self.batch_sizes, self.LP_MAPs)
            mb_LP_AUC = self.calc_epoch_metric(self.batch_sizes, self.LP_AUCs)
            mb_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
            mb_AUC = self.calc_epoch_metric(self.batch_sizes, self.AUCs)
            partial_losses = torch.stack(self.losses)
            self.logger.info(
                self.info_preamble
                + " batch %d / %d - partial error %0.4f - partial loss %0.4f - partial MRR  %0.4f - partial GMAUC %0.4f - partial MAP %0.4f - partial AUC %0.4f"
                % (
                    self.minibatch_done,
                    self.num_minibatches,
                    mb_error,
                    partial_losses.mean(),
                    mb_MRR,
                    mb_GMAUC,
                    mb_MAP,
                    mb_AUC,
                )
            )
            self.logger.info(
                self.info_preamble
                + "LP partial MAP %0.4f - AUC %0.4f" % (mb_LP_MAP, mb_LP_AUC)
            )

            tp = conf_mat_per_class.true_positives
            fn = conf_mat_per_class.false_negatives
            fp = conf_mat_per_class.false_positives
            self.logger.info(
                self.info_preamble
                + " batch %d / %d -  partial tp %s,fn %s,fp %s"
                % (self.minibatch_done, self.num_minibatches, tp, fn, fp)
            )
            precision, recall, f1 = self.calc_microavg_eval_measures(tp, fn, fp)
            self.logger.info(
                self.info_preamble
                + " batch %d / %d - measures partial microavg - precision %0.4f - recall %0.4f - f1 %0.4f "
                % (self.minibatch_done, self.num_minibatches, precision, recall, f1)
            )
            for cl in range(self.num_classes):
                cl_precision, cl_recall, cl_f1 = self.calc_eval_measures_per_class(
                    tp, fn, fp, cl
                )
                self.logger.info(
                    self.info_preamble
                    + " batch %d / %d - measures partial for class %d - precision %0.4f - recall %0.4f - f1 %0.4f "
                    % (
                        self.minibatch_done,
                        self.num_minibatches,
                        cl,
                        cl_precision,
                        cl_recall,
                        cl_f1,
                    )
                )

            self.logger.info(
                self.info_preamble
                + " batch %d / %d - Batch time %d "
                % (
                    self.minibatch_done,
                    self.num_minibatches,
                    (time.monotonic() - self.lasttime),
                )
            )

        self.lasttime = time.monotonic()

    def log_epoch_done(self):
        epoch_metrics = {}

        self.losses = torch.stack(self.losses)
        epoch_metrics["loss"] = self.losses.mean()

        epoch_metrics["error"] = self.calc_epoch_metric(self.batch_sizes, self.errors)

        epoch_metrics["mrr"] = self.calc_epoch_metric(self.batch_sizes, self.MRRs)
        epoch_metrics["gmauc"] = self.calc_epoch_metric(self.batch_sizes, self.GMAUCs)
        epoch_metrics["lp_map"] = self.calc_epoch_metric(self.batch_sizes, self.LP_MAPs)
        epoch_metrics["lp_auc"] = self.calc_epoch_metric(self.batch_sizes, self.LP_AUCs)
        epoch_metrics["map"] = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
        epoch_metrics["auc"] = self.calc_epoch_metric(self.batch_sizes, self.AUCs)

        self.logger.info(
            "{} {}".format(
                self.info_preamble,
                " - ".join(
                    "mean {} {}".format(metric, value)
                    for metric, value in epoch_metrics.items()
                ),
            )
        )

        if self.partition == "TEST":
            # Reshape neg_sampling metrics for calc_epoch_metric
            neg_sampling_lists = {}
            for neg_sampling_batch in self.neg_samples:
                for key in neg_sampling_batch.keys():
                    value = neg_sampling_batch[key]
                    if key not in neg_sampling_lists:
                        neg_sampling_lists[key] = []
                    neg_sampling_lists[key].append(value)

            neg_sampling_aggregated = {}
            for metric, value_list in neg_sampling_lists.items():
                if len(self.batch_sizes) == len(value_list):
                    neg_sampling_aggregated[metric] = self.calc_epoch_metric(
                        self.batch_sizes, value_list
                    )
                else:
                    # This may happen if the metric is not calculated in all snapshots, which may happen if the negative samples is too high
                    neg_sampling_aggregated[metric] = "nan"

            self.logger.info(
                "{} {}".format(
                    self.info_preamble,
                    " - ".join(
                        "mean {} {}".format(metric, value)
                        for metric, value in neg_sampling_aggregated.items()
                    ),
                )
            )

        self.logger.info(
            self.info_preamble
            + " tp %s,fn %s,fp %s"
            % (self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp)
        )
        precision, recall, f1 = self.calc_microavg_eval_measures(
            self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp
        )
        self.logger.info(
            self.info_preamble
            + " measures microavg - precision %0.4f - recall %0.4f - f1 %0.4f "
            % (precision, recall, f1)
        )
        epoch_metrics["avg_precision"] = precision
        epoch_metrics["avg_recall"] = recall
        epoch_metrics["avg_f1"] = f1

        for cl in range(self.num_classes):
            cl_precision, cl_recall, cl_f1 = self.calc_eval_measures_per_class(
                self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp, cl
            )
            self.logger.info(
                self.info_preamble
                + " measures for class %d - precision %0.4f - recall %0.4f - f1 %0.4f "
                % (cl, cl_precision, cl_recall, cl_f1)
            )
            epoch_metrics["{}_{}".format(cl, "precision")] = cl_precision
            epoch_metrics["{}_{}".format(cl, "recall")] = cl_recall
            epoch_metrics["{}_{}".format(cl, "f1")] = cl_f1

        for k in self.eval_k_list:
            precision, recall, f1 = self.calc_microavg_eval_measures(
                self.conf_mat_tp_at_k[k],
                self.conf_mat_fn_at_k[k],
                self.conf_mat_fp_at_k[k],
            )
            self.logger.info(
                self.info_preamble
                + " measures@%d microavg - precision %0.4f - recall %0.4f - f1 %0.4f "
                % (k, precision, recall, f1)
            )

            for cl in range(self.num_classes):
                cl_precision, cl_recall, cl_f1 = self.calc_eval_measures_per_class(
                    self.conf_mat_tp_at_k[k],
                    self.conf_mat_fn_at_k[k],
                    self.conf_mat_fp_at_k[k],
                    cl,
                )
                self.logger.info(
                    self.info_preamble
                    + " measures@%d for class %d - precision %0.4f - recall %0.4f - f1 %0.4f "
                    % (k, cl, cl_precision, cl_recall, cl_f1)
                )

        self.logger.info(
            self.info_preamble
            + " Total epoch time: "
            + str(((time.monotonic() - self.ep_time)))
        )

        target_measure = self.args.target_measure.lower()
        if target_measure in epoch_metrics.keys():
            return epoch_metrics[target_measure]
        else:
            self.logger.warning("Target measure not found, using MAP")
            return epoch_metrics["map"]

    def get_negative_sample_metrics(self, probs, true_classes):
        # Assumes that true_classes is split such that 1s come first and 0s after. probs_np
        # Return dict of map and auc at different ratios of negative sampling
        # Keys are on the form "multiplier_metric", e.g. 1000_map
        num_edges = int(true_classes.sum())
        link_probs = probs[:num_edges]
        no_link_probs = probs[num_edges:]
        ratio = float(num_edges) / len(no_link_probs)
        ks = [1000, 100, 10, 1]
        ks_to_sample = [k for k in ks if 1 / k > ratio]
        k_last = ks[0]
        first_sample = True
        neg_edges = {}
        for k in ks:
            if k in ks_to_sample:
                # Sample indicies
                # Simple time estimates this random sample to take ~3 seconds even on autonomous. Hopefully this is not too bad.
                if first_sample:
                    neg_edges[k] = random.sample(
                        range(len(no_link_probs)), k=num_edges * k
                    )
                    first_sample = False
                else:
                    neg_edges[k] = random.sample(neg_edges[k_last], k=num_edges * k)
            else:
                neg_edges[k] = range(len(no_link_probs))
            k_last = k

        neg_samples = {}
        for k in neg_edges.keys():
            if k not in ks_to_sample:
                continue
            neg_samples_indicies = neg_edges[k]
            probs_k = np.concatenate(
                (link_probs, no_link_probs[neg_samples_indicies]), axis=None
            )
            true_classes_k = np.concatenate(
                (np.ones(num_edges), np.zeros(len(neg_samples_indicies))), axis=None
            )
            map_metric = torch.tensor(self.get_MAP(probs_k, true_classes_k))
            auc_metric = torch.tensor(self.get_AUC(probs_k, true_classes_k))
            neg_samples["{}_{}".format(k, "map")] = map_metric
            neg_samples["{}_{}".format(k, "auc")] = auc_metric
        return neg_samples

    # adj is an edge list
    def get_MRR(self, probs, true_classes, adj):
        pred_matrix = coo_matrix((probs, (adj[0], adj[1]))).toarray()
        true_matrix = coo_matrix((true_classes, (adj[0], adj[1]))).toarray()

        row_MRRs = []
        for i, pred_row in enumerate(pred_matrix):
            # check if there are any existing edges
            if np.isin(1, true_matrix[i]):
                row_MRRs.append(self.get_row_MRR(pred_row, true_matrix[i]))

        avg_MRR = torch.tensor(row_MRRs).mean()
        return avg_MRR

    """
    GMAUC metric is the harmonic mean of 1. the PRAUC of new edges and 2. AUC of recurring edges
    Source: https://arxiv.org/pdf/1607.07330.pdf
    The calculation of density depends on the number of negative samples that the model is given.
    If all_edges=True then it is calculated from all edges
    """

    def get_GMAUC(self, new_probs, new_tc, rec_probs, rec_tc):

        # Figure out which edges in adj are new and which are reoccuring
        # Calculate PRAUC for new edges
        # Calculate AUC for recurring edges
        # Calculate ratio P/(P+N) aka density (but for new links only)

        prauc_new = average_precision_score(new_tc, new_probs)
        auc_prev = roc_auc_score(rec_tc, rec_probs)
        density = np.sum(new_tc) / new_tc.shape[0]
        under_root = ((prauc_new - density) / (1 - density)) * 2 * (auc_prev - 0.5)
        GMAUC = math.sqrt(max(under_root, 0))

        return GMAUC

    def get_row_MRR(self, probs, true_classes):
        existing_mask = true_classes == 1
        # descending in probability
        ordered_indices = np.flip(probs.argsort())

        ordered_existing_mask = existing_mask[ordered_indices]

        existing_ranks = np.arange(1, true_classes.shape[0] + 1, dtype=np.float)[
            ordered_existing_mask
        ]

        MRR = (1 / existing_ranks).sum() / existing_ranks.shape[0]
        return MRR

    def get_MAP(self, probs, true_classes):
        predictions_np = probs
        true_classes_np = true_classes

        return average_precision_score(true_classes_np, predictions_np)

    def get_AUC(self, probs, true_classes):
        predictions_np = probs
        true_classes_np = true_classes

        return roc_auc_score(true_classes_np, predictions_np)

    def eval_predicitions(self, predictions, true_classes, num_classes):
        predicted_classes = predictions.argmax(dim=1)
        failures = (predicted_classes != true_classes).sum(dtype=torch.float)
        error = failures / predictions.size(0)

        conf_mat_per_class = utils.Namespace({})
        conf_mat_per_class.true_positives = {}
        conf_mat_per_class.false_negatives = {}
        conf_mat_per_class.false_positives = {}

        for cl in range(num_classes):
            cl_indices = true_classes == cl

            pos = predicted_classes == cl
            hits = predicted_classes[cl_indices] == true_classes[cl_indices]

            tp = hits.sum()
            fn = hits.size(0) - tp
            fp = pos.sum() - tp

            conf_mat_per_class.true_positives[cl] = tp
            conf_mat_per_class.false_negatives[cl] = fn
            conf_mat_per_class.false_positives[cl] = fp
        return error, conf_mat_per_class

    def eval_predicitions_at_k(self, predictions, true_classes, num_classes, k):
        conf_mat_per_class = utils.Namespace({})
        conf_mat_per_class.true_positives = {}
        conf_mat_per_class.false_negatives = {}
        conf_mat_per_class.false_positives = {}

        if predictions.size(0) < k:
            k = predictions.size(0)

        for cl in range(num_classes):
            # sort for prediction with higher score for target class (cl)
            _, idx_preds_at_k = torch.topk(
                predictions[:, cl], k, dim=0, largest=True, sorted=True
            )
            predictions_at_k = predictions[idx_preds_at_k]
            predicted_classes = predictions_at_k.argmax(dim=1)

            cl_indices_at_k = true_classes[idx_preds_at_k] == cl
            cl_indices = true_classes == cl

            pos = predicted_classes == cl
            hits = (
                predicted_classes[cl_indices_at_k]
                == true_classes[idx_preds_at_k][cl_indices_at_k]
            )

            tp = hits.sum()
            fn = (
                true_classes[cl_indices].size(0) - tp
            )  # This only if we want to consider the size at K -> hits.size(0) - tp
            fp = pos.sum() - tp

            conf_mat_per_class.true_positives[cl] = tp
            conf_mat_per_class.false_negatives[cl] = fn
            conf_mat_per_class.false_positives[cl] = fp
        return conf_mat_per_class

    def calc_microavg_eval_measures(self, tp, fn, fp):
        tp_sum = sum(tp.values()).item()
        fn_sum = sum(fn.values()).item()
        fp_sum = sum(fp.values()).item()

        p = tp_sum * 1.0 / (tp_sum + fp_sum) if tp_sum + fp_sum != 0 else 0
        r = tp_sum * 1.0 / (tp_sum + fn_sum) if tp_sum + fn_sum != 0 else 0
        if (p + r) > 0:
            f1 = 2.0 * (p * r) / (p + r)
        else:
            f1 = 0
        return p, r, f1

    def calc_eval_measures_per_class(self, tp, fn, fp, class_id):
        if type(tp) is dict:
            tp_sum = tp[class_id].item()
            fn_sum = fn[class_id].item()
            fp_sum = fp[class_id].item()
        else:
            tp_sum = tp.item()
            fn_sum = fn.item()
            fp_sum = fp.item()
        if tp_sum == 0:
            return 0, 0, 0

        p = tp_sum * 1.0 / (tp_sum + fp_sum)
        r = tp_sum * 1.0 / (tp_sum + fn_sum)
        if (p + r) > 0:
            f1 = 2.0 * (p * r) / (p + r)
        else:
            f1 = 0
        return p, r, f1

    def calc_epoch_metric(self, batch_sizes, metric_val):
        batch_sizes = torch.tensor(batch_sizes, dtype=torch.float)
        epoch_metric_val = torch.stack(metric_val).cpu() * batch_sizes
        epoch_metric_val = epoch_metric_val.sum() / batch_sizes.sum()

        return epoch_metric_val.detach().item()

    def new_reappear_split_tensor(self, probs, true_classes, adj, prev_adj):
        adj = adj.numpy()

        prev_edge_list = set(tuple(edge) for edge in prev_adj.squeeze().numpy())

        def prev_edge(n1, n2):
            return (n1, n2) in prev_edge_list or (n2, n1) in prev_edge_list

        vprev_edge = np.vectorize(prev_edge)
        mask = vprev_edge(adj[0], adj[1])
        # mask = [((n1, n2) in prev_edge_list or (n2, n1) in prev_edge_list) for (n1, n2) in adj]
        invmask = np.invert(mask)

        new_probs = probs[invmask].numpy()
        new_tc = true_classes[invmask].numpy()
        rec_probs = probs[mask].numpy()
        rec_tc = true_classes[mask].numpy()

        return new_probs, new_tc, rec_probs, rec_tc

    """
    Split the label adjacency matrix between new links and reappearing links
    Return probs and true classes for both kinds of links
    """

    def new_reappear_split(self, probs, true_classes, adj, prev_adj):
        # True classes is the combination of existing and non-existing requires filtering to get P and N
        adj = adj.T.tolist()
        prev_edge_list = set(tuple(edge) for edge in prev_adj.squeeze().tolist())

        new_probs = []
        new_tc = []
        rec_probs = []
        rec_tc = []
        for i, e in enumerate(adj):
            prob = probs[i]
            tc = true_classes[i]
            edge = tuple(e)
            n1 = edge[0]
            n2 = edge[1]

            if (n1, n2) in prev_edge_list or (n2, n1) in prev_edge_list:
                rec_probs.append(prob)
                rec_tc.append(tc)
            else:
                new_probs.append(prob)
                new_tc.append(tc)
        return new_probs, new_tc, rec_probs, rec_tc
