"""Training and testing the inverse propensity weighting algorithm for unbiased learning to rank.

See the following paper for more information on the inverse propensity weighting algorithm.

    * Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
    * Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils


class PRSrank(BaseAlgorithm):
    """The Lambda Rank algorithm for unbiased learning to rank.

    This class implements the training and testing of theLambda algorithm for unbiased learning to rank. See the following paper for more information on the algorithm.

    From RankNet to LambdaRank to LambdaMART: An Overview
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
    https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf


    """

    def __init__(self, data_set, exp_settings):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """

        self.hparams = ultra.utils.hparams.HParams(
            propensity_estimator_type='ultra.utils.propensity_estimator.RandomizedPropensityEstimator',
            # the setting file for the predefined click models.
            propensity_estimator_json='./example/PropensityEstimator/randomized_pbm_0.1_1.0_4_1.0.json',
            learning_rate=0.05,  # Learning rate.
            max_gradient_norm=5.0,  # Clip gradients to this norm.
            grad_strategy='ada',  # Select gradient strategy
            sigma=1.0
        )
        self.is_cuda_avail = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.is_cuda_avail else torch.device('cpu')
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.is_training = "is_train"
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.feature_size = data_set.feature_size
        self.model = self.create_model(self.feature_size)
        self.model = self.model.to(device=self.device)
        self.propensity_estimator = ultra.utils.find_class(
            self.hparams.propensity_estimator_type)(
            self.hparams.propensity_estimator_json)

        self.max_candidate_num = exp_settings['max_candidate_num']
        self.learning_rate = float(self.hparams.learning_rate)
        self.sigma = self.hparams.sigma
        self.global_step = 0

        # Feeds for inputs.
        # self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))
        self.PAD_embed = torch.zeros(1, self.feature_size)
        self.PAD_embed = self.PAD_embed.to(dtype=torch.float32)
        self.lambda_weights = None
        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)

        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        # Output feed: depends on whether we do a backward step or not.
        # compute propensity weights for the input data.
        self.model.train()
        ipw = []
        for l in range(self.rank_list_size):
            input_feed["propensity_weights{0}".format(l)] = []
        for i in range(len(input_feed[self.labels_name[0]])):
            click_list = [input_feed[self.labels_name[l]][i]
                          for l in range(self.rank_list_size)]
            pw_list = self.propensity_estimator.getPropensityForOneList(
                click_list,use_non_clicked_data=True)
            ipw.append(torch.as_tensor(pw_list))
            for l in range(self.rank_list_size):
                input_feed["propensity_weights{0}".format(l)].append(
                    pw_list[l])
        ipw = torch.stack(ipw)
        pw = ultra.utils.metrics._safe_div(torch.tensor(1.0), ipw)

        self.create_input_feed(input_feed, self.rank_list_size)
        training_output = self.ranking_model(self.model,
                                          self.rank_list_size)
        preds_sorted, preds_sorted_inds = torch.sort(training_output, dim=1, descending=True)
        labels_sorted_via_preds = torch.gather(self.labels, dim=1, index=preds_sorted_inds)
        ipw_sorted_via_preds = torch.gather(ipw, dim=1, index=preds_sorted_inds)
        print(ipw_sorted_via_preds[0])
        pw_sorted_via_preds = torch.gather(pw, dim=1, index=preds_sorted_inds)
        print(pw_sorted_via_preds[0])

        #calculate the prs score using the pw of unclick document and ipw of clicked document
        prs = torch.unsqueeze(ipw_sorted_via_preds, dim=2) * torch.unsqueeze(pw_sorted_via_preds, dim=1)
        prs = torch.triu(prs, diagonal=1)
        print(prs[0])

        std_diffs = torch.unsqueeze(labels_sorted_via_preds, dim=2) - torch.unsqueeze(
            labels_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
        std_Sij = torch.clamp(std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        std_p_ij = 0.5 * (1.0 + std_Sij)
        # s_ij has shape [batch_size, rank_list_size, rank_list_size]
        s_ij = torch.unsqueeze(preds_sorted, dim=2) - torch.unsqueeze(preds_sorted,dim=1)  # computing pairwise differences, i.e., s_i - s_j
        p_ij = 1.0 / (torch.exp(-self.sigma * s_ij) + 1.0)
        ideally_sorted_labels, _ = torch.sort(self.labels, dim =1, descending=True)
        delta_NDCG = self.compute_delta_ndcg(ideally_sorted_labels, labels_sorted_via_preds)
        self.loss = F.binary_cross_entropy(torch.triu(p_ij, diagonal=1),
                                            torch.triu(std_p_ij,diagonal=1),
                                            torch.triu(delta_NDCG,diagonal=1), reduction='none')
        self.loss = self.loss * prs
        self.loss = torch.sum(self.loss)
        params = self.model.parameters()
        self.opt_step(self.optimizer_func, params)
        # self.create_summary('Learning Rate', 'Learning_rate at global step %d' % self.global_step, self.learning_rate,
        #                     True)
        # self.create_summary('Loss', 'Loss at global step %d' % self.global_step, self.learning_rate,True)

        # pad_removed_train_output = self.remove_padding_for_metric_eval(
        #     self.docid_inputs, train_output)
        # for metric in self.exp_settings['metrics']:
        #     for topn in self.exp_settings['metrics_topn']:
        #         list_weights = torch.mean(
        #             train_pw * train_labels, dim=1, keepdim=True)
        #         metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
        #             train_pw, pad_removed_train_output, None)
        #         # self.create_summary('%s_%d' % (metric, topn),
        #         #                     '%s_%d at global step %d' % (metric, topn, self.global_step), metric_value, True)
        #         weighted_metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
        #             train_labels, pad_removed_train_output, list_weights)
        #         # self.create_summary('Weighted_%s_%d' % (metric, topn),
        #         #                     'Weighted_%s_%d at global step %d' %(metric, topn, self.global_step),
        #         #                     weighted_metric_value, True)

        print(" Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        self.global_step += 1
        return self.loss.item(), None, self.train_summary

    def validation(self, input_feed, is_online_simulation=False):
        """Run a step of the model feeding the given inputs for validating process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        self.model.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)
        with torch.no_grad():
            self.output = self.ranking_model(self.model,
                                             self.max_candidate_num)
            if not is_online_simulation:
                pad_removed_output = self.remove_padding_for_metric_eval(
                    self.docid_inputs, self.output)

                for metric in self.exp_settings['metrics']:
                    topn = self.exp_settings['metrics_topn']
                    metric_values = ultra.utils.make_ranking_metric_fn(
                        metric, topn)(self.labels, pad_removed_output, None)
                    for topn, metric_value in zip(topn, metric_values):
                        self.create_summary('%s_%d' % (metric, topn),
                                            '%s_%d' % (metric, topn), metric_value.item(), False)
        return None, self.output, self.eval_summary  # loss, outputs, summary.

    def dcg(self, labels):
        """Computes discounted cumulative gain (DCG).

        DCG =  SUM((2^label -1) / (log(1+rank))).

        Args:
         labels: The relevance `Tensor` of shape [batch_size, list_size]. For the
           ideal ranking, the examples are sorted by relevance in reverse order.
          weights: A `Tensor` of the same shape as labels or [batch_size, 1]. The
            former case is per-example and the latter case is per-list.

        Returns:
          A `Tensor` as the weighted discounted cumulative gain per-list. The
          tensor shape is [batch_size, 1].
        """
        list_size = labels.shape[1]
        position = torch.arange(1, list_size + 1, device=self.device, dtype=torch.float32)
        denominator = torch.log(position + 1)
        numerator = torch.pow(torch.tensor(2.0, device=self.device), labels.to(torch.float32)) - 1.0
        return torch.sum(ultra.utils.metrics._safe_div(numerator, denominator))

    def compute_delta_ndcg(self, ideally_sorted_stds, stds_sorted_via_preds):
        '''
        Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
        :param batch_stds: the standard labels sorted in a descending order
        :param batch_stds_sorted_via_preds: the standard labels sorted based on the corresponding predictions
        :return:
        '''
        # ideal discount cumulative gains
        batch_idcgs = self.dcg(ideally_sorted_stds)

        batch_gains = torch.pow(2.0, stds_sorted_via_preds) - 1.0

        batch_n_gains = batch_gains / batch_idcgs  # normalised gains
        batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

        batch_std_ranks = torch.arange(stds_sorted_via_preds.size(1)).type(torch.cuda.FloatTensor) if self.is_cuda_avail \
            else torch.arange(stds_sorted_via_preds.size(1))
        batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)  # discount co-efficients
        batch_dists = torch.unsqueeze(batch_dists, dim=0)
        batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
        batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(
            batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

        return batch_delta_ndcg
