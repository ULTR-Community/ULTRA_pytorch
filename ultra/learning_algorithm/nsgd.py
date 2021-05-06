"""Training and testing the Null Space Gradient Descent (NSGD) algorithm for unbiased learning to rank.

See the following paper for more information on the Null Space Gradient Descent (NSGD) algorithm.

    * Huazheng Wang, Ramsey Langley, Sonwoo Kim, Eric McCord-Snook, Hongning Wang. 2018. Efficient Exploration of Gradient Space for Online Learning to Rank. In SIGIR.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F

from six.moves import zip
from torch.utils.tensorboard import SummaryWriter

from ultra.learning_algorithm.dbgd import DBGD
import ultra.utils
import ultra


class NSGD(DBGD):
    """The Null Space Gradient Descent (NSGD) algorithm for unbiased learning to rank.

    This class implements the Null Space Gradient Descent (NSGD) algorithm based on the input layer feed. See the following paper for more information on the algorithm.

    * Huazheng Wang, Ramsey Langley, Sonwoo Kim, Eric McCord-Snook, Hongning Wang. 2018. Efficient Exploration of Gradient Space for Online Learning to Rank. In SIGIR.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Null Space Gradient Descent (DBGD) algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            # The update rate for randomly sampled weights.
            learning_rate=0.5,         # Learning rate.
            max_gradient_norm=5.0,      # Clip gradients to this norm.
            need_interleave=True,       # Set True to use result interleaving
            grad_strategy='sgd',        # Select gradient strategy
            # Select number of rankers to try in each batch.
            ranker_num=4,
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.cuda = torch.device('cuda')
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.feature_size = data_set.feature_size
        self.model = self.create_model(self.feature_size)
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.learning_rate = self.hparams.learning_rate
        self.ranker_num = self.hparams.ranker_num
        self.winners_name = "winners"

        # Feeds for inputs.
        self.rank_list_size = exp_settings['selection_bias_cutoff']
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

        self.bad_noisy_params = {}
        self.model_params_to_update = {}

        # Create memory to store historical bad noisy parameters
        for sequential in self.model.children():
            if isinstance(sequential, nn.Sequential):
                for name, parameter in sequential.named_parameters():
                    if "linear" in name:
                        self.model_params_to_update[name] = parameter
                        self.bad_noisy_params[name] = []
                        for i in range(self.ranker_num):
                            self.bad_noisy_params[name].append(torch.zeros(parameter.shape))


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
        train_labels = self.labels[:self.rank_list_size]

        null_space_dict = {}
        for noise in self.bad_noisy_params:
            null_space_dict[noise] = self.compute_null_space(self.bad_noisy_params[noise])

        new_output_lists = []
        param_gradient_from_rankers = {}
        # noise_lists = [tf.zeros_like(self.output, tf.float32)]
        for i in range(self.ranker_num):
            # Create unit noise from null space
            noisy_params = {}
            for layer in self.model_params_to_update:
                noisy_params[layer] = self.sample_from_null_space(
                    null_space_dict[layer][0], null_space_dict[layer][1])

            # Apply the noise to get new ranking scores
            new_output_list = self.create_new_output_list(noisy_params)
            new_output_lists.append(new_output_list)

            for noise in noisy_params:
                if noise not in param_gradient_from_rankers:
                    param_gradient_from_rankers[noise] = [
                        torch.zeros_like(self.model_params_to_update[noise])]
                param_gradient_from_rankers[noise].append(noisy_params[noise])

        # Compute NDCG for the old ranking scores.
        # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_train_labels = torch.transpose(train_labels, 0, 1)
        previous_ndcg = ultra.utils.make_ranking_metric_fn('ndcg', self.rank_list_size)(
            reshaped_train_labels, train_output, None)
        self.loss = 1.0 - previous_ndcg

        if self.hparams.need_interleave:  # Use result interleaving
            self.output = [self.output] + new_output_lists
            final_winners = self.winners
        else:  # No result interleaving
            score_lists = [train_output] + new_output_lists
            ndcg_lists = []
            for scores in score_lists:
                ndcg = ultra.utils.make_ranking_metric_fn(
                    'ndcg', self.rank_list_size)(
                    reshaped_train_labels, scores, None)
                ndcg_lists.append(ndcg - previous_ndcg)
            ndcg_gains = torch.ceil(torch.stack(ndcg_lists))
            final_winners = ndcg_gains / \
                            (torch.sum(ndcg_gains, dim=0) + 0.000000001)

        # Compute gradients
        self.update_ops_list = []
        self.compute_gradient(final_winners, param_gradient_from_rankers)

        # Gradients and SGD update operation for training the model.
        if self.hparams.max_gradient_norm > 0:
            self.clipped_gradient = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.hparams.max_gradient_norm)

        self.optimizer_func.step()

        self.create_summary('Learning Rate', 'Learning_rate at global step %d' % self.global_step,
                            self.learning_rate,
                            True)
        self.create_summary('Loss', 'Loss at global step %d' % self.global_step, self.learning_rate, True)
        pad_removed_train_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, train_output)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                    reshaped_train_labels, pad_removed_train_output, None)
                self.create_summary('%s_%d' % (metric, topn),
                                    '%s_%d at global step %d' % (metric, topn, self.global_step), metric_value,
                                    True)
        # loss, no outputs, summary.
        self.global_step += 1
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

        null_space_dict = {}
        new_output_lists = []
        for noise in self.bad_noisy_params:
            null_space_dict[noise] = self.compute_null_space(self.bad_noisy_params[noise])

        for i in range(self.ranker_num):
            # Create unit noise from null space
            noisy_params = {}
            for layer in self.model_params_to_update:
                noisy_params[layer] = self.sample_from_null_space(
                    null_space_dict[layer][0], null_space_dict[layer][1])

            # Apply the noise to get new ranking scores
            new_output_list = self.create_new_output_list(noisy_params)
            new_output_lists.append(new_output_list)

        rankers_output = [self.output] + new_output_lists
        return rankers_output, self.output, self.eval_summary  # no loss, outputs, summary.

    def compute_gradient(self, final_winners, noisy_params):
        self.model.to('cpu')
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for name, parameter in layer.named_parameters():
                    if "linear" in name:
                        for i in range(len(noisy_params[name])):
                            noisy_params[name][i] = noisy_params[name][i].cpu()
                        gradient_matrix = torch.unsqueeze(
                            torch.stack(noisy_params[name]), dim=0)
                        expended_winners =  torch.tensor(final_winners)
                        for i in range(gradient_matrix.dim() - expended_winners.dim()):
                            expended_winners = torch.unsqueeze(expended_winners, dim=-1)
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

                            # Update historical bad nosiy parameters
                            expended_losers = torch.eq(torch.sum(
                                expended_winners,
                                dim=0,
                                keepdim=True), 0).to(dtype=torch.float32)
                            bad_noise_list = torch.unbind(
                                torch.squeeze(
                                    expended_losers *
                                    gradient_matrix), dim=0)[1:]
                            bad_noise_list = list(bad_noise_list)

                            for i in range(self.ranker_num):
                                bad_noise_list[i] = torch.reshape(
                                    bad_noise_list[i], self.bad_noisy_params[name][i].shape)
                                self.bad_noisy_params[name][i] = bad_noise_list[i]
                                self.update_ops_list.append(
                                    bad_noise_list
                                )
        self.model.to(self.cuda)

    def sample_from_null_space(self,null_space_matrix, original_shape):
        if sum([original_shape[i] for i in range(
                len(original_shape))]) > 1:
            sampled_vector = torch.sum(
                null_space_matrix * torch.normal(mean=0.0, std=1.0,size=(1, self.ranker_num)), dim=1)
            return F.normalize(
                torch.reshape(sampled_vector, original_shape))
        else:
            return F.normalize(
                torch.normal(mean=0.0, std=1.0,size=original_shape), dim=0)

    # Compute null space
    def compute_null_space(self, param_list):
        original_shape = param_list[0].shape
        flatten_list = [torch.reshape(param_list[i], (1, -1))
                        for i in range(len(param_list))]
        matrix = torch.stack(flatten_list, dim=1)
        u, s, v = torch.svd(matrix)
        mask = torch.eq(s, 0).to(dtype=torch.float32)
        return (torch.squeeze(v * mask), original_shape)

