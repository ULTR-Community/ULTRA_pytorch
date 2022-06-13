# Copyright 2018 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Revised a little bit by Qingyao Ai in order to be compatible with other
# programs.

"""Defines ranking metrics as TF ops.

The metrics here are meant to be used during the TF training. That is, a batch
of instances in the Tensor format are evaluated by ops. It works with listwise
Tensors only.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ultra.utils import metric_utils as utils
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RankingMetricKey(object):
    """Ranking metric key strings."""
    # Mean Receiprocal Rank. For binary relevance.
    MRR = 'mrr'

    # Expected Reciprocal Rank
    ERR = 'err'
    MAX_LABEL = None

    # Average Relvance Position.
    ARP = 'arp'

    # Normalized Discounted Culmulative Gain.
    NDCG = 'ndcg'

    # Discounted Culmulative Gain.
    DCG = 'dcg'

    # Precision. For binary relevance.
    PRECISION = 'precision'

    # Mean Average Precision. For binary relevance.
    MAP = 'map'

    # Ordered Pair Accuracy.
    ORDERED_PAIR_ACCURACY = 'ordered_pair_accuracy'


def make_ranking_metric_fn(metric_key,
                           topn=None,
                           name=None):
    """Factory method to create a ranking metric function.

    Args:
      metric_key: A key in `RankingMetricKey`.
      topn: An `integer` specifying the cutoff of how many items are considered in
        the metric.
      name: A `string` used as the name for this metric.

    Returns:
      A metric fn with the following Args:
      * `labels`: A `Tensor` of the same shape as `predictions` representing
      graded
          relevance.
      * `predictions`: A `Tensor` with shape [batch_size, list_size]. Each value
      is
          the ranking score of the corresponding example.
      * `weights`: A `Tensor` of weights (read more from each metric function).
    """

    def _average_relevance_position_fn(labels, predictions, weights):
        """Returns average relevance position as the metric."""
        return average_relevance_position(
            labels, predictions, weights=weights, topn=topn, name=name)

    def _mean_reciprocal_rank_fn(labels, predictions, weights):
        """Returns mean reciprocal rank as the metric."""
        return mean_reciprocal_rank(
            labels, predictions, weights=weights, topn=topn, name=name)

    def _expected_reciprocal_rank_fn(labels, predictions, weights):
        """Returns expected reciprocal rank as the metric."""
        return expected_reciprocal_rank(
            labels, predictions, weights=weights, topn=topn, name=name)

    def _normalized_discounted_cumulative_gain_fn(
            labels, predictions, weights):
        """Returns normalized discounted cumulative gain as the metric."""
        return normalized_discounted_cumulative_gain(
            labels,
            predictions,
            weights=weights,
            topn=topn,
            name=name)

    def _discounted_cumulative_gain_fn(labels, predictions, weights):
        """Returns discounted cumulative gain as the metric."""
        return discounted_cumulative_gain(
            labels,
            predictions,
            weights=weights,
            topn=topn,
            name=name)

    def _precision_fn(labels, predictions, weights):
        """Returns precision as the metric."""
        return precision(
            labels,
            predictions,
            weights=weights,
            topn=topn,
            name=name)

    def _mean_average_precision_fn(labels, predictions, weights):
        """Returns mean average precision as the metric."""
        return mean_average_precision(
            labels,
            predictions,
            weights=weights,
            topn=topn,
            name=name)

    def _ordered_pair_accuracy_fn(labels, predictions, weights):
        """Returns ordered pair accuracy as the metric."""
        return ordered_pair_accuracy(
            labels, predictions, weights=weights, topn=topn, name=name)

    metric_fn_dict = {
        RankingMetricKey.ARP: _average_relevance_position_fn,
        RankingMetricKey.MRR: _mean_reciprocal_rank_fn,
        RankingMetricKey.ERR: _expected_reciprocal_rank_fn,
        RankingMetricKey.NDCG: _normalized_discounted_cumulative_gain_fn,
        RankingMetricKey.DCG: _discounted_cumulative_gain_fn,
        RankingMetricKey.PRECISION: _precision_fn,
        RankingMetricKey.MAP: _mean_average_precision_fn,
        RankingMetricKey.ORDERED_PAIR_ACCURACY: _ordered_pair_accuracy_fn,
    }
    assert metric_key in metric_fn_dict, (
        'metric_key %s not supported.' % metric_key)
    return metric_fn_dict[metric_key]


def _safe_div(numerator, denominator, name='safe_div'):
    """Computes a safe divide which returns 0 if the denominator is zero.

    Args:
      numerator: An arbitrary `Tensor`.
      denominator: `Tensor` whose shape matches `numerator`.
      name: An optional name for the returned op.

    Returns:
      The element-wise value of the numerator divided by the denominator.
    """
    return torch.where(
        torch.eq(denominator, 0),
        torch.zeros_like(numerator),
        torch.div(numerator, denominator))


def _per_example_weights_to_per_list_weights(weights, relevance):
    """Computes per list weight from per example weight.

    Args:
      weights:  The weights `Tensor` of shape [batch_size, list_size].
      relevance:  The relevance `Tensor` of shape [batch_size, list_size].

    Returns:
      The per list `Tensor` of shape [batch_size, 1]
    """
    if torch.cuda.is_available():
        relevance = relevance.to(device=device)
    per_list_weights = _safe_div(
        torch.sum(weights * relevance, 1, keepdim=True),
        torch.sum(relevance, 1, keepdim=True))
    return per_list_weights


def _discounted_cumulative_gain(prediction, labels, weights=None, topn=None):
    """Computes discounted cumulative gain (DCG).

    DCG =  SUM((2^label -1) / (log(1+rank))).

    Args:
     labels: The relevance `Tensor` of shape [batch_size, list_size]. For the
       ideal ranking, the examples are sorted by relevance in reverse order.
      weights: A `Tensor` of the same shape as labels or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
    topn: A list of cutoff for how many examples to consider for this metric.

    Returns:
      A `Tensor` as the weighted discounted cumulative gain per-list. The
      tensor shape is [batch_size, 1].
    """
    list_size = labels.shape[1]
    _, indices = prediction.sort(descending=True, dim=-1)
    sorted_labels = torch.gather(labels, dim=1, index=indices)
    sorted_weights = torch.gather(weights, dim =1, index=indices)
    discounts = (torch.tensor(1) / torch.log2(torch.arange(list_size, dtype=torch.float) + 2.0)).to(
        device=device)
    gains = sorted_weights*torch.pow(torch.tensor(2.0, device=device), sorted_labels.to(torch.float32)) - 1.0
    discounted_gains = (gains * discounts)[:, :np.max(topn)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    topn_tensor = torch.tensor(topn, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, topn_tensor]
    return dcg


def _prepare_and_validate_params(labels, predictions, weights=None, topn=None):
    """Prepares and validates the parameters.

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.

    Returns:
      (labels, predictions, weights, topn) ready to be used for metric
      calculation.
    """
    weights = 1.0 if weights is None else weights
    example_weights = torch.ones_like(labels) * weights
    assert predictions.shape == example_weights.shape
    assert predictions.shape == labels.shape
    assert predictions.dim() == 2
    list_size = predictions.shape[1]
    if topn is None:
        topn = [list_size]

    topn = [min(n, list_size) for n in topn]

    # All labels should be >= 0. Invalid entries are reset.
    is_label_valid = utils.is_label_valid(labels)
    labels = labels
    labels = torch.where(
        is_label_valid,
        labels,
        torch.zeros_like(labels))
    # is_label_valid = is_label_valid.to(device=device)
    # if predictions.is_cuda:
    #     predictions = predictions.cpu()
    predictions = torch.where(
        is_label_valid, predictions,
        -1e-6 * torch.ones_like(predictions) + torch.min(
            input=predictions, dim=1, keepdim=True).values)
    return labels, predictions, example_weights, topn


def mean_reciprocal_rank(labels, predictions, weights=None, topn=None, name=None):
    """Computes mean reciprocal rank (MRR).

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted mean reciprocal rank of the batch.
    """
    list_size = predictions.size()[-1]
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, topn)
    _, indices = predictions.sort(descending=True, dim=-1)
    sorted_labels = torch.gather(labels, dim=1, index=indices)
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = torch.ge(sorted_labels, 1.0).type(torch.float32)
    reciprocal_rank = 1.0 / torch.arange(start=1, end=list_size + 1, device=device,
                                         dtype=torch.float32)
    # MRR has a shape of [batch_size, 1]
    mrr = torch.max(
        relevance * reciprocal_rank, dim=1, keepdim=True).values
    mrr =  torch.mean(
        mrr * torch.ones_like(weights) * weights).repeat(len(topn))
    return mrr

def expected_reciprocal_rank(
        labels, predictions, weights=None, topn=None, name=None):
    """Computes expected reciprocal rank (ERR).

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted expected reciprocal rank of the batch.
    """
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, topn)
    _, indices = predictions.sort(descending=True, dim=-1)
    sorted_labels = torch.gather(labels, dim=1, index=indices)
    sorted_weights = torch.gather(weights, dim=1, index=indices)
    list_size = sorted_labels.size()[-1]
    pow = torch.as_tensor(2.0, device=device)
    relevance = (torch.pow(pow, sorted_labels) - 1) / \
      torch.pow(pow, torch.as_tensor(RankingMetricKey.MAX_LABEL,device=device))
    non_rel = torch.cumprod(1.0 - relevance, dim=1) / (1.0 - relevance)
    reciprocal_rank = 1.0 / \
      torch.arange(start=1, end=list_size + 1,device=device,dtype=torch.float32)
    mask = [torch.ge(reciprocal_rank, 1.0 / n).type(torch.float32) for n in topn]
    reciprocal_rank_topn = [reciprocal_rank * top_n_mask for top_n_mask in mask]
    # ERR has a shape of [batch_size, 1]
    err = [torch.sum(
        relevance * non_rel * reciprocal_rank * sorted_weights, dim=1, keepdim=True) for reciprocal_rank in reciprocal_rank_topn]
    err = torch.stack(err,dim=0)
    # ERR has size [top_N, batch_size, 1]
    return torch.mean(err,dim=1)


def average_relevance_position(labels, predictions, weights=None, topn=None, name=None):
    """Computes average relevance position (ARP).

    This can also be named as average_relevance_rank, but this can be confusing
    with mean_reciprocal_rank in acronyms. This name is more distinguishing and
    has been used historically for binary relevance as average_click_position.

    Args:
      labels: A `Tensor` of the same shape as `predictions`.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted average relevance position.
    """
    list_size = predictions.size()[1]
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights,topn)
    _, indices = predictions.sort(descending=True, dim=-1)
    sorted_labels = torch.gather(labels, dim=1, index=indices)
    sorted_weights = torch.gather(weights, dim=1, index=indices)
    position = torch.arange(1, list_size + 1, dtype=torch.float)
    weighted_labels = sorted_labels * sorted_weights
    per_list_weights = torch.sum(weighted_labels, dim=1, keepdim=True)
    # TODO(xuanhui): Consider to add a cap poistion topn + 1 when there is no
    # relevant examples.
    per_list_arp = _safe_div(torch.sum(position * weighted_labels, dim=1, keepdim=True),
        per_list_weights)
    arp= torch.mean( per_list_arp).repeat(len(topn))
    return arp


def precision(labels, predictions, weights=None, topn=None, name=None):
    """Computes precision as weighted average of relevant examples.

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted precision of the batch.
    """
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, topn)
    _, indices = predictions.sort(descending=True, dim=-1)
    sorted_labels = torch.gather(labels, dim=1, index=indices)
    sorted_weights = torch.gather(weights, dim=1, index=indices)
    # Relevance = 1.0 when labels >= 1.0.
    relevance = torch.ge(sorted_labels, 1.0).to(dtype=torch.float)
    per_list_precision = _safe_div(
        torch.sum(relevance * sorted_weights, 1, keepdim=True),
        torch.sum(torch.ones_like(relevance) * sorted_weights, 1, keepdim=True))
    # per_list_weights are computed from the whole list to avoid the problem of
    # 0 when there is no relevant example in topn.
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights, torch.ge(labels, 1.0).to(dtype=torch.float))
    return torch.mean(per_list_precision * per_list_weights)


def mean_average_precision(labels,
                           predictions,
                           weights=None,
                           topn=None,
                           name=None):
    """Computes mean average precision (MAP).
    The implementation of MAP is based on Equation (1.7) in the following:
    Liu, T-Y "Learning to Rank for Information Retrieval" found at
    https://www.nowpublishers.com/article/DownloadSummary/INR-016

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the mean average precision.
    """
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, topn)
    _, indices = predictions.sort(descending=True, dim=-1)
    sorted_labels = torch.gather(labels, dim=1, index=indices)
    sorted_weights = torch.gather(weights, dim=1, index=indices)
    # Relevance = 1.0 when labels >= 1.0.
    sorted_relevance = torch.ge(sorted_labels, 1.0).to(dtype=torch.float32)
    per_list_relevant_counts = torch.cumsum(sorted_relevance, dim=1)
    per_list_cutoffs = torch.cumsum(torch.ones_like(sorted_relevance), dim=1)
    per_list_precisions = torch.nan_to_num(torch.div(per_list_relevant_counts,
                                                per_list_cutoffs))
    total_precision = torch.sum(
        input=per_list_precisions * sorted_weights * sorted_relevance,
        dim=1,
        keepdim=True)
    total_relevance = torch.sum(
        input=sorted_weights * sorted_relevance, dim=1, keepdim=True)
    per_list_map = torch.nan_to_num(torch.div(total_precision, total_relevance))
    # per_list_weights are computed from the whole list to avoid the problem of
    # 0 when there is no relevant example in topn.
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights, torch.ge(labels, 1.0).to(dtype=torch.float32))
    map= torch.mean(per_list_map* per_list_weights).repeat(len(topn))
    return map

def normalized_discounted_cumulative_gain(labels,
                                          predictions,
                                          weights=None,
                                          topn=None,
                                          name=None):
    """Computes normalized discounted cumulative gain (NDCG).

    Args:
      labels: A `Tensor` of the same shape as `predictions`.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted normalized discounted cumulative gain of the
      batch.
    """
    if weights is not None:
        labels, predictions, weights, topn = _prepare_and_validate_params(
            labels, predictions, weights, topn)
        dcg = _discounted_cumulative_gain(predictions, labels, weights, topn)
        # Sorting over the weighted labels to get ideal ranking.
        ideal_dcg = _discounted_cumulative_gain(labels, labels, weights, topn)
        per_list_ndcg = _safe_div(dcg, ideal_dcg)
        per_list_weights = _per_example_weights_to_per_list_weights(
            weights=weights,
            relevance=torch.pow(torch.tensor(2.0),labels.to(torch.float)) - 1.0)
        ndcg = torch.mean(per_list_ndcg * per_list_weights)
    else:
        labels, predictions, weights, topn = _prepare_and_validate_params(
            labels, predictions, weights, topn)
        dcg = _discounted_cumulative_gain(predictions, labels, weights, topn)
        # Sorting over the weighted labels to get ideal ranking.
        ideal_dcg = _discounted_cumulative_gain(labels, labels, weights, topn)
        per_list_ndcg = _safe_div(dcg, ideal_dcg)
        ndcg = torch.mean(per_list_ndcg,dim=0)
    return ndcg


def discounted_cumulative_gain(labels,
                               predictions,
                               weights=None,
                               topn=None,
                               name=None):
    """Computes discounted cumulative gain (DCG).

    Args:
      labels: A `Tensor` of the same shape as `predictions`.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted discounted cumulative gain of the batch.
    """
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, topn)
    _, indices = predictions.sort(descending=True, dim=-1)
    sorted_labels = torch.gather(labels, dim=1, index=indices)
    sorted_weights = torch.gather(weights, dim=1, index=indices)
    dcg = _discounted_cumulative_gain(sorted_labels,
                                      sorted_weights) * torch.log1p(1.0)
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=torch.pow(2.0, labels.to(dtype=torch.float)) - 1.0)
    return torch.mean(
        _safe_div(dcg, per_list_weights) * per_list_weights,dim=0)


def ordered_pair_accuracy(labels, predictions, weights=None, topn=None, name=None):
    """Computes the percentage of correctedly ordered pair.

    For any pair of examples, we compare their orders determined by `labels` and
    `predictions`. They are correctly ordered if the two orders are compatible.
    That is, labels l_i > l_j and predictions s_i > s_j and the weight for this
    pair is the weight from the l_i.

    Args:
      labels: A `Tensor` of the same shape as `predictions`.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the accuracy or ordered pairs.
    """
    clean_labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, topn)
    label_valid = torch.eq(clean_labels, labels)
    valid_pair = torch.logical_and(
        torch.unsqueeze(label_valid, 2),
        torch.unsqueeze(label_valid, 1))
    pair_label_diff = torch.unsqueeze(
        clean_labels, 2) - torch.unsqueeze(clean_labels, 1)
    pair_pred_diff = torch.unsqueeze(
        predictions, 2) - torch.unsqueeze(predictions, 1)
    # Correct pairs are represented twice in the above pair difference tensors.
    # We only take one copy for each pair.
    correct_pairs = torch.gt(pair_label_diff, 0).to(dtype=torch.float) * \
                    torch.gt(pair_pred_diff, 0).to(dtype=torch.float)
    pair_weights = torch.gt(pair_label_diff, 0).to(dtype=torch.float) * torch.unsqueeze(
            weights, 2) * valid_pair.to(dtype=torch.float)
    opa = torch.mean(correct_pairs * pair_weights).repeat(len(topn))
    return opa
