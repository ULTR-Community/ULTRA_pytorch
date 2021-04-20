"""Training and testing the Pairwise Debiasing algorithm for unbiased learning to rank.

See the following paper for more information on the Pairwise Debiasing algorithm.

    * Hu, Ziniu, Yang Wang, Qu Peng, and Hang Li. "Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm." In The World Wide Web Conference, pp. 2830-2836. ACM, 2019.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from six.moves import zip
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils


def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.

        Args:
            prob: (tf.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.

        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.

        """
    return torch.ceil(probs - torch.rand(probs.shape).to(device=torch.device('cuda')))


class PairDebias(BaseAlgorithm):
    """The Pairwise Debiasing algorithm for unbiased learning to rank.

    This class implements the Pairwise Debiasing algorithm based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Hu, Ziniu, Yang Wang, Qu Peng, and Hang Li. "Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm." In The World Wide Web Conference, pp. 2830-2836. ACM, 2019.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Pairwise Debiasing algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            EM_step_size=0.05,                  # Step size for EM algorithm.
            learning_rate=0.005,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            # An int specify the regularization term.
            regulation_p=1,
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
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
        self.learning_rate =  float(self.hparams.learning_rate)

        # Feeds for inputs.
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
        self.rank_list_size = exp_settings['selection_bias_cutoff']
        self.t_plus = torch.ones([1, self.rank_list_size]).to(device=self.cuda)
        self.t_plus.requires_grad = False
        self.t_minus = torch.ones([1, self.rank_list_size]).to(device=self.cuda)
        self.t_minus.requires_grad = False

        # Select optimizer
        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate)

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        self.model.train()
        self.create_input_feed(input_feed, self.rank_list_size)
        train_output = self.ranking_model(self.model,
                                          self.rank_list_size)

        self.splitted_t_plus = torch.split(
            self.t_plus, 1, dim=1)
        self.splitted_t_minus = torch.split(
            self.t_minus, 1, dim=1)
        for i in range(self.rank_list_size):
            self.create_summary('t_plus Probability %d' % i,
                                't_plus Probability %d at global step %d' % (i, self.global_step),
                                torch.max(self.splitted_t_plus[i]), True)

            self.create_summary('t_minus Probability %d' % i,
                                't_minus Probability %d at global step %d' % (i, self.global_step),
                                torch.max(self.splitted_t_plus[i]), True)

        # Build pairwise loss based on clicks (0 for unclick, 1 for click)
        split_size = int(train_output.shape[1] / self.rank_list_size)
        output_list = torch.split(train_output, split_size, dim=1)
        t_plus_loss_list = [0.0 for _ in range(self.rank_list_size)]
        t_minus_loss_list = [0.0 for _ in range(self.rank_list_size)]
        self.loss = 0.0
        for i in range(self.rank_list_size):
            for j in range(self.rank_list_size):
                if i == j:
                    continue
                valid_pair_mask = torch.minimum(
                    torch.ones_like(
                        self.labels[i]), F.relu(
                        self.labels[i] - self.labels[j]))
                pair_loss = torch.sum(
                    valid_pair_mask *
                    self.pairwise_cross_entropy_loss(
                        output_list[i], output_list[j])
                )
                t_plus_loss_list[i] = pair_loss / self.splitted_t_minus[j]
                t_minus_loss_list[j] = pair_loss / self.splitted_t_plus[i]
                self.loss += pair_loss / \
                             self.splitted_t_plus[i] / self.splitted_t_minus[j]
        with torch.no_grad():
            self.t_plus = (1 - self.hparams.EM_step_size) * self.t_plus + self.hparams.EM_step_size * torch.pow(
                    torch.cat(t_plus_loss_list, dim=1) / t_plus_loss_list[0], 1 / (self.hparams.regulation_p + 1))
            self.t_minus = (1 - self.hparams.EM_step_size) * self.t_minus + self.hparams.EM_step_size * torch.pow(torch.cat(
                    t_minus_loss_list, dim=1) / t_minus_loss_list[0], 1 / (self.hparams.regulation_p + 1))

        # Add l2 loss
        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * torch.nn.MSELoss(p) * 0.5

        self.opt_step(self.optimizer_func, params)
        self.create_summary('Learning Rate', 'Learning_rate at global step %d' % self.global_step, self.learning_rate,
                            True)
        self.create_summary('Loss', 'Loss at global step %d' % self.global_step, self.learning_rate,True)

        # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_train_labels = torch.transpose(self.labels, 0, 1)
        pad_removed_train_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, train_output)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                    reshaped_train_labels, pad_removed_train_output, None)
                self.create_summary( '%s_%d' % (metric, topn),
                                     '%s_%d at global step %d' % (metric, topn, self.global_step), metric_value, True)

        return self.loss, None, self.train_summary

    def validation(self, input_feed):
        self.model.eval()
        self.letor_features = torch.from_numpy(input_feed["letor_features"])
        self.create_input_feed(input_feed, self.max_candidate_num)
        self.output = self.ranking_model(self.model,
                                         self.max_candidate_num)
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, self.output)

        # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        reshaped_labels = torch.transpose(torch.tensor(self.labels), 0, 1)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(reshaped_labels, pad_removed_output, None)
                self.create_summary('%s_%d' % (metric, topn),
                                    '%s_%d at global step %d' % (metric, topn, self.global_step), metric_value, False)
        return None, self.output, self.eval_summary  # no loss, outputs, summary.