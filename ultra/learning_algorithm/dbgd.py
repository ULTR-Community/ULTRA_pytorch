"""Training and testing the Dueling Bandit Gradient Descent (DBGD) algorithm for unbiased learning to rank.

See the following paper for more information on the Dueling Bandit Gradient Descent (DBGD) algorithm.

    * Yisong Yue and Thorsten Joachims. 2009. Interactively optimizing information retrieval systems as a dueling bandits problem. In ICML. 1201–1208.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils
import ultra
from ultra.utils.team_draft_interleave import TeamDraftInterleaving
from ultra.utils import click_models as cm
import json

import numpy as np


class DBGD(BaseAlgorithm):
    """The Dueling Bandit Gradient Descent (DBGD) algorithm for unbiased learning to rank.

    This class implements the Dueling Bandit Gradient Descent (DBGD) algorithm based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Yisong Yue and Thorsten Joachims. 2009. Interactively optimizing information retrieval systems as a dueling bandits problem. In ICML. 1201–1208.

    """

    def __init__(self, data_set, exp_settings):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build Dueling Bandit Gradient Descent (DBGD) algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            # the setting file for the predefined click models.
            click_model_json='./example/ClickModel/pbm_0.1_1.0_4_1.0.json',
            # The update rate for randomly sampled weights.
            learning_rate=0.5,         # Learning rate.
            max_gradient_norm=5.0,      # Clip gradients to this norm.
            need_interleave=True,       # Set True to use result interleaving
            interleave_strategy='Stochastic', # Choose interleave strategy
            grad_strategy='sgd',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.cuda = torch.device('cuda')
        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.feature_size = data_set.feature_size

        self.model = self.create_model(self.feature_size)
        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.learning_rate = self.hparams.learning_rate
        self.winners_name = "winners"
        self.winners = None
        self.interleaving_strategy = self.hparams.interleave_strategy

        # Feeds for inputs.
        self.is_training = True
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))

        self.global_step = 0
        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.MAX_SAMPLE_ROUND_NUM = None
        self.interleaving = None
        self.click_model = None

        if self.hparams.need_interleave:
            self.MAX_SAMPLE_ROUND_NUM = 100
            self.interleaving = TeamDraftInterleaving()
            with open(self.hparams.click_model_json) as fin:
                model_desc = json.load(fin)
                self.click_model = cm.loadModelFromJson(model_desc)

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        self.create_input_feed(input_feed, self.max_candidate_num)
        with torch.no_grad():
            self.output = self.ranking_model(self.model, self.max_candidate_num)
        train_output = self.ranking_model(self.model, self.rank_list_size)

        noisy_params = self.create_noisy_param()

        # Compute NDCG for the old ranking scores and new ranking scores
        # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_train_labels = self.labels[:, :self.rank_list_size]
        with torch.no_grad():
            self.new_output = self.create_new_output_list(noisy_params)
        previous_ndcg = ultra.utils.make_ranking_metric_fn(
            'ndcg', self.rank_list_size)(
            reshaped_train_labels, train_output, None)
        self.loss = torch.sub(1,previous_ndcg)
        self.loss.requires_grad=True

        if self.hparams.need_interleave:
            self.output = (self.output, self.new_output)
            self.winners = self.click_simulation_winners(input_feed, self.output, self.interleaving_strategy)
            final_winners = self.winners
        else:
            score_lists = [train_output, self.new_output]
            ndcg_lists = []
            for scores in score_lists:
                ndcg = ultra.utils.make_ranking_metric_fn(
                    'ndcg', self.rank_list_size)(
                    reshaped_train_labels, scores, None)
                ndcg_lists.append(ndcg - previous_ndcg)
            ndcg_gains = torch.ceil(torch.stack(ndcg_lists))
            final_winners = ndcg_gains / \
                            (torch.sum(ndcg_gains, dim=0) + 0.000000001)

        self.optimizer_func.zero_grad()
        self.loss.backward()
        self.compute_gradient(final_winners, noisy_params)

        if self.hparams.max_gradient_norm > 0:
            self.clipped_gradient = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.hparams.max_gradient_norm)
        self.optimizer_func.step()

        # self.create_summary('Learning Rate', 'Learning_rate at global step %d' % self.global_step, self.learning_rate,
        #                     True)
        # self.create_summary('Loss', 'Loss at global step %d' % self.global_step, self.learning_rate, True)
        # pad_removed_train_output = self.remove_padding_for_metric_eval(
        #     self.docid_inputs, train_output)
        # for metric in self.exp_settings['metrics']:
        #     for topn in self.exp_settings['metrics_topn']:
        #         metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
        #             reshaped_train_labels, pad_removed_train_output, None)
        #         self.create_summary('%s_%d' % (metric, topn),
        #                             '%s_%d at global step %d' % (metric, topn, self.global_step), metric_value, True)
        # loss, no outputs, summary.
        print(" Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        self.global_step+=1
        return self.loss.item(), self.output, self.train_summary

    def validation(self, input_feed, is_online_simulation=False):
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

        return None, self.output, self.eval_summary  # no loss, outputs, summary.

    def compute_gradient(self, final_winners, noisy_params):
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for name, parameter in layer.named_parameters():
                    if "linear" in name:
                        noisy_param = noisy_params[name]
                        gradient_matrix = torch.unsqueeze(
                            torch.stack([torch.zeros_like(parameter), noisy_param]), dim=0)
                        expended_winners = torch.tensor(final_winners)
                        for i in range(gradient_matrix.dim() - expended_winners.dim()):
                            expended_winners = torch.unsqueeze(
                                expended_winners, dim=-1)
                        gradient = torch.mean(
                            torch.sum(
                                expended_winners * gradient_matrix,
                                dim=1
                            ),
                            dim=0)
                        if parameter.grad == None:
                            if self.is_cuda_avail:
                                self.model.to('cpu')
                                dummy_loss = 0
                                dummy_loss += torch.mean(parameter)
                                dummy_loss.backward()
                                self.model.to(self.cuda)
                        gradient = gradient.to(dtype=torch.float32)
                        parameter.grad = gradient

    def create_noisy_param(self):
        noisy_params = {}
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for name,param in layer.named_parameters():
                    if "linear" in name:
                        noisy_params[name]= F.normalize(torch.normal(mean=0.0, std=1.0, size=param.shape), dim=0)
        return noisy_params

    def create_new_output_list(self, noisy_params):
        # Apply the noise to get new ranking scores
        model_prime = ultra.utils.find_class(
                self.exp_settings['ranking_model'])(
                self.exp_settings['ranking_model_hparams'], self.feature_size)
        if self.hparams.need_interleave:  # compute scores on whole list if needs interleave
            new_output_list = self.get_ranking_scores(model_prime,
                                                      self.docid_inputs, noisy_params=noisy_params,
                                                      noise_rate=self.hparams.learning_rate)
        else:
            new_output_list = self.get_ranking_scores(model_prime,
                                                      self.docid_inputs[:self.rank_list_size],
                                                      noisy_params=noisy_params, noise_rate=self.hparams.learning_rate)
        return torch.cat(new_output_list, 1)

    def click_simulation_winners(self, input_feed, rank_scores, interleave_strategy):
        # Rerank documents and collect clicks
        letor_features_length = len(input_feed[self.letor_features_name])
        local_batch_size = len(input_feed[self.docid_inputs_name[0]])
        input_feed[self.winners_name] = [
            None for _ in range(local_batch_size)]
        for i in range(local_batch_size):
            # Get valid doc index
            valid_idx = self.max_candidate_num - 1
            while valid_idx > -1:
                if input_feed[self.docid_inputs_name[valid_idx]][i] < letor_features_length:  # a valid doc
                    break
                valid_idx -= 1
            list_len = valid_idx + 1

            if interleave_strategy == 'Stochastic':
                def plackett_luce_sampling(score_list):
                    # Sample document ranking
                    scores = score_list[:list_len]
                    scores = scores - max(scores)
                    exp_scores = np.exp(self.hparams.tau * scores)
                    exp_scores = exp_scores.numpy()
                    probs = exp_scores / np.sum(exp_scores)
                    re_list = np.random.choice(np.arange(list_len),
                                               replace=False,
                                               p=probs,
                                               size=np.count_nonzero(probs))
                    # Append unselected documents to the end
                    used_indexs = set(re_list)
                    unused_indexs = []
                    for tmp_index in range(list_len):
                        if tmp_index not in used_indexs:
                            unused_indexs.append(tmp_index)
                    re_list = np.append(re_list, unused_indexs).astype(int)
                    return re_list

                rank_lists = []

                for j in range(len(rank_scores)):
                    scores = rank_scores[j][i][:list_len]
                    rank_list = plackett_luce_sampling(scores)
                    rank_lists.append(rank_list)

                rerank_list = self.interleaving.interleave(
                    np.asarray(rank_lists))
            else:
                # Rerank documents via interleaving
                rank_lists = []
                for j in range(len(rank_scores)):
                    scores = rank_scores[j][i][:list_len]
                    rank_list = sorted(
                        range(
                            len(scores)),
                        key=lambda k: scores[k],
                        reverse=True)
                    rank_lists.append(rank_list)

                rerank_list = self.interleaving.interleave(
                    np.asarray(rank_lists))

            new_label_list = np.zeros(list_len)
            for j in range(list_len):
                new_label_list[j] = input_feed[self.labels_name[rerank_list[j]]][i]

            # Collect clicks online
            click_list, _, _ = self.click_model.sampleClicksForOneList(
                new_label_list[:self.rank_list_size])
            sample_count = 0
            while sum(click_list) == 0 and sample_count < self.MAX_SAMPLE_ROUND_NUM:
                click_list, _, _ = self.click_model.sampleClicksForOneList(
                    new_label_list[:self.rank_list_size])
                sample_count += 1

            # Infer winner in interleaving
            input_feed[self.winners_name][i] = self.interleaving.infer_winner(click_list)

        return input_feed[self.winners_name]


