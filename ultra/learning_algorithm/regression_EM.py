"""Training and testing the regression-based EM algorithm for unbiased learning to rank.

See the following paper for more information on the regression-based EM algorithm.

    * Wang, Xuanhui, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. "Position bias estimation for unbiased learning to rank in personal search." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 610-618. ACM, 2018.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

import copy
import itertools
from six.moves import zip
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils


def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.

        Args:
            prob: (torch.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.

        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.

        """
    return torch.ceil(probs - torch.rand(probs.shape).to(device=torch.device('cuda')))


class RegressionEM(BaseAlgorithm):
    """The regression-based EM algorithm for unbiased learning to rank.

    This class implements the regression-based EM algorithm based on the input layer
    feed. See the following paper for more information.

    * Wang, Xuanhui, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. "Position bias estimation for unbiased learning to rank in personal search." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 610-618. ACM, 2018.

    In particular, we use the online EM algorithm for the parameter estimations:

    * Cappé, Olivier, and Eric Moulines. "Online expectation–maximization algorithm for latent data models." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 71.3 (2009): 593-613.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Regression-based EM algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            EM_step_size=0.05,                  # Step size for EM algorithm.
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
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
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.rank_list_size = exp_settings['selection_bias_cutoff']
        self.model = self.create_model(self.feature_size)
        self.rank_list_size = exp_settings['selection_bias_cutoff']
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))
        self.propensity = (torch.ones([1, self.rank_list_size]) * 0.9).to(device=self.cuda).requires_grad=False
        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

    def train(self, input_feed):
        self.model.train()
        self.labels = []
        self.docid_inputs = []
        self.letor_features = torch.from_numpy(input_feed["letor_features"])
        for i in range(self.rank_list_size):
            self.docid_inputs.append(input_feed[self.docid_inputs_name[i]])
            self.labels.append(input_feed[self.labels_name[i]])

        self.labels = torch.tensor(data=self.labels, device=self.cuda)
        self.docid_inputs = torch.tensor(data=self.docid_inputs, dtype=torch.int64)

        sigmoid_prob_b = (torch.ones([1]) - 1.0).to(device=self.cuda)
        train_output = self.ranking_model(self.model,
                                          self.rank_list_size)
        train_output = train_output + sigmoid_prob_b
        train_labels = self.labels
        self.propensity = (torch.ones([1, self.rank_list_size]) * 0.9).to(device=self.cuda)
        split_size = int(self.propensity.shape[1]/self.rank_list_size)
        self.splitted_propensity = torch.split(
            self.propensity, split_size, dim=1)
        for i in range(self.rank_list_size):
            self.writer.add_scalar(
                'Examination Probability %d' %
                i,
                torch.max(
                    self.splitted_propensity[i]))
            self.train_summary['Examination Probability %d' % i] = torch.max(self.splitted_propensity[i])

        # Conduct estimation step.
        gamma = torch.sigmoid(train_output)
        # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_train_labels = torch.transpose(train_labels, 0, 1)
        p_e1_r1_c1 = 1
        p_e1_r0_c0 = self.propensity * \
                     (1 - gamma) / (1 - self.propensity * gamma)
        p_e0_r1_c0 = (1 - self.propensity) * gamma / \
                     (1 - self.propensity * gamma)
        p_e0_r0_c0 = (1 - self.propensity) * (1 - gamma) / \
                     (1 - self.propensity * gamma)
        p_e1 = p_e1_r0_c0 + p_e1_r1_c1
        p_r1 = reshaped_train_labels + \
               (1 - reshaped_train_labels) * p_e0_r1_c0

        # Conduct maximization step
        self.propensity = (1 - self.hparams.EM_step_size) * self.propensity + self.hparams.EM_step_size * torch.mean(
            reshaped_train_labels + (1 - reshaped_train_labels) * p_e1_r0_c0, dim=0, keepdim=True)
        self.update_propensity_op = self.propensity

        # Get Bernoulli samples and compute rank loss
        self.ranker_labels = get_bernoulli_sample(p_r1).to(device=self.cuda)
        self.ranker_labels.to(device=self.cuda)
        # base_loss = self.ranker_labels*-torch.log(torch.sigmoid(train_output)) + (1-self.ranker_labels)*-torch.log(1-torch.sigmoid(train_output))
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.loss = torch.mean(
          torch.sum(
            criterion(train_output,self.ranker_labels), dim =1
          )
        )
        # record additional positive instance from sampling
        labels_split_size = int(self.ranker_labels.shape[1] / self.rank_list_size)
        split_ranker_labels = torch.split(
            self.ranker_labels, labels_split_size, dim=1)
        for i in range(self.rank_list_size):
            additional_postive_instance = (torch.sum(split_ranker_labels[i]) - torch.sum(
                train_labels[i])) / (torch.sum(torch.ones_like(train_labels[i])) - torch.sum(train_labels[i]))
            self.writer.add_scalar(
                'Additional pseudo clicks %d' %
                i, additional_postive_instance)
            self.train_summary['Additional pseudo clicks %d' % i] = additional_postive_instance

        self.propensity_weights = 1.0 / self.propensity
        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * nn.MSELoss(p) * 0.5

        # Select optimizer
        self.optimizer_func = torch.optim.SGD(params, lr=self.hparams.learning_rate)
        # tf.train.AdagradOptimizer
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(params, lr=self.hparams.learning_rate)
            # tf.train.GradientDescentOptimizer

        opt = self.optimizer_func
        # tf.gradients(self.loss, params)
        if self.hparams.max_gradient_norm > 0:
            # tf.clip_by_global_norm(self.gradients,self.hparams.max_gradient_norm)
            opt.zero_grad()
            self.loss.backward()
            self.clipped_gradient = nn.utils.clip_grad_norm_(
                params, self.hparams.max_gradient_norm)
            opt.step()
        else:
            self.norm = None
            opt.zero_grad()
            self.loss.backward()
            opt.step()
        self.writer.add_scalar(
            'Learning Rate',
            self.learning_rate,
            self.global_step)
        self.train_summary['Learning_rate at global step %d' % self.global_step] = self.learning_rate
        self.writer.add_scalar(
            'Loss', torch.mean(
                self.loss), self.global_step)
        self.train_summary['Loss at global step %d' % self.global_step] = self.loss

        nn.utils.clip_grad_value_(reshaped_train_labels, 1)

        pad_removed_train_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, train_output)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                list_weights = torch.mean(
                    self.propensity_weights * reshaped_train_labels, dim=1, keepdim=True)
                metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                    reshaped_train_labels, pad_removed_train_output, None)
                self.writer.add_scalar(
                    '%s_%d' %
                    (metric, topn), metric_value, self.global_step)
                self.train_summary['%s_%d at global step %d' %
                                   (metric, topn, self.global_step)] = metric_value
                weighted_metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                    reshaped_train_labels, pad_removed_train_output, list_weights)
                self.writer.add_scalar(
                    'Weighted_%s_%d' %
                    (metric, topn), weighted_metric_value, self.global_step)

                self.train_summary['Weighted_%s_%d at global step %d' %
                                   (metric, topn, self.global_step)] = weighted_metric_value

            print("Global Step: ", self.global_step)
            self.global_step += 1
            return self.loss, None, self.train_summary

        input_feed[self.is_training.name] = True
        return self.loss, None, self.train_summary

    def validation(self, input_feed):
        self.model.eval()
        self.letor_features = torch.from_numpy(input_feed["letor_features"])
        self.labels = []
        self.docid_inputs = []
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(input_feed[self.docid_inputs_name[i]])
            self.labels.append(input_feed[self.labels_name[i]])
        self.labels = torch.tensor(data=self.labels, device=self.cuda)
        self.docid_inputs = torch.tensor(data=self.docid_inputs, dtype=torch.int64)
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
                self.writer.add_scalar(
                    '%s_%d' %
                    (metric, topn), metric_value)
                self.eval_summary['%s_%d' %
                                  (metric, topn)] = metric_value
        return None, self.output, self.eval_summary  # no loss, outputs, summary.