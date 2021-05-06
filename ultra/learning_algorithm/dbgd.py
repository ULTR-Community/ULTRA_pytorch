"""Training and testing the Dueling Bandit Gradient Descent (DBGD) algorithm for unbiased learning to rank.

See the following paper for more information on the Dueling Bandit Gradient Descent (DBGD) algorithm.

    * Yisong Yue and Thorsten Joachims. 2009. Interactively optimizing information retrieval systems as a dueling bandits problem. In ICML. 1201–1208.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from six.moves import zip
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils
import ultra


class DBGD(BaseAlgorithm):
    """The Dueling Bandit Gradient Descent (DBGD) algorithm for unbiased learning to rank.

    This class implements the Dueling Bandit Gradient Descent (DBGD) algorithm based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Yisong Yue and Thorsten Joachims. 2009. Interactively optimizing information retrieval systems as a dueling bandits problem. In ICML. 1201–1208.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Dueling Bandit Gradient Descent (DBGD) algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            # The update rate for randomly sampled weights.
            learning_rate=0.5,         # Learning rate.
            max_gradient_norm=5.0,      # Clip gradients to this norm.
            need_interleave=True,       # Set True to use result interleaving
            grad_strategy='sgd',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])

        self.cuda = torch.device('cuda')
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.feature_size = data_set.feature_size
        self.rank_list_size = exp_settings['selection_bias_cutoff']
        self.model = self.create_model(self.feature_size)
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.learning_rate = self.hparams.learning_rate
        self.winners_name = "winners"
        self.winners = None

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


    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        self.create_input_feed(input_feed, self.max_candidate_num)
        self.winners = input_feed[self.winners_name]
        self.output = self.ranking_model(self.model, self.max_candidate_num)
        train_output = self.ranking_model(self.model, self.rank_list_size)

        noisy_params = self.create_noisy_param()
        new_output_list = self.create_new_output_list(noisy_params)

        # Compute NDCG for the old ranking scores and new ranking scores
        # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_train_labels = torch.transpose(self.labels[:self.rank_list_size], 0 ,1)
        self.new_output = new_output_list

        previous_ndcg = ultra.utils.make_ranking_metric_fn(
            'ndcg', self.rank_list_size)(
            reshaped_train_labels, train_output, None)
        self.loss = torch.sub(1,previous_ndcg)
        self.loss.requires_grad=True

        if self.hparams.need_interleave:
            self.output = (self.output, self.new_output)
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


        ranking_model_params = self.model.parameters()
        self.optimizer_func.zero_grad()
        self.loss.backward()
        self.compute_gradient(final_winners, noisy_params)

        if self.hparams.max_gradient_norm > 0:
            self.clipped_gradient = torch.nn.utils.clip_grad_norm_(
                ranking_model_params, self.hparams.max_gradient_norm)

        self.optimizer_func.step()

        self.create_summary('Learning Rate', 'Learning_rate at global step %d' % self.global_step, self.learning_rate,
                            True)
        self.create_summary('Loss', 'Loss at global step %d' % self.global_step, self.learning_rate, True)
        pad_removed_train_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, train_output)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                    reshaped_train_labels, pad_removed_train_output, None)
                self.create_summary('%s_%d' % (metric, topn),
                                    '%s_%d at global step %d' % (metric, topn, self.global_step), metric_value, True)
        # loss, no outputs, summary.
        self.global_step+=1
        return self.loss, self.output, self.train_summary

    def validation(self, input_feed):
        self.model.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)
        self.output = self.ranking_model(self.model,
                                         self.max_candidate_num)
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, self.output)

        # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        reshaped_labels = torch.transpose(self.labels, 0, 1)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(reshaped_labels, pad_removed_output, None)
                self.create_summary('%s_%d' % (metric, topn),
                                    '%s_%d at global step %d' % (metric, topn, self.global_step), metric_value, False)

        noisy_params = self.create_noisy_param()
        # Apply the noise to get new ranking scores
        new_output_list = self.create_new_output_list(noisy_params)
        pair_outputs = (self.output, new_output_list)

        return pair_outputs, self.output, self.eval_summary  # no loss, outputs, summary.

    def compute_gradient(self, final_winners, noisy_params):
        self.model.to('cpu')
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
                            dummy_loss = 0
                            dummy_loss += torch.mean(parameter)
                            dummy_loss.backward()
                            gradient = gradient.to(dtype=torch.float32)
                            parameter.grad = gradient
        self.model.to(self.cuda)

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
        if self.hparams.need_interleave:  # compute scores on whole list if needs interleave
            new_output_list = self.get_ranking_scores(self.model,
                                                      self.docid_inputs, noisy_params=noisy_params,
                                                      noise_rate=self.hparams.learning_rate)
        else:
            new_output_list = self.get_ranking_scores(self.model,
                                                      self.docid_inputs[:self.rank_list_size],
                                                      noisy_params=noisy_params, noise_rate=self.hparams.learning_rate)
        return torch.cat(new_output_list, 1)



