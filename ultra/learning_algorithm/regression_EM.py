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

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils

def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.

        Args:
            prob: (torch.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.

        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.

        """
    if torch.cuda.is_available():
        bernoulli_sample = torch.ceil(probs - torch.rand(probs.shape, device=torch.device('cuda')))
    else:
        bernoulli_sample = torch.ceil(probs - torch.rand(probs.shape))
    return bernoulli_sample


class RegressionEM(BaseAlgorithm):
    """The regression-based EM algorithm for unbiased learning to rank.

    This class implements the regression-based EM algorithm based on the input layer
    feed. See the following paper for more information.

    * Wang, Xuanhui, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. "Position bias estimation for unbiased learning to rank in personal search." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 610-618. ACM, 2018.

    In particular, we use the online EM algorithm for the parameter estimations:

    * Cappé, Olivier, and Eric Moulines. "Online expectation–maximization algorithm for latent data models." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 71.3 (2009): 593-613.

    """

    def __init__(self, data_set, exp_settings):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
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
        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.model = self.create_model(self.feature_size)
        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))
        with torch.no_grad():
            self.propensity = (torch.ones([1, self.rank_list_size]) * 0.9)
            if self.is_cuda_avail:
                self.propensity = self.propensity.to(device=self.cuda)
        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0
        self.sigmoid_prob_b = (torch.ones([1]) - 1.0)
        if self.is_cuda_avail:
            self.sigmoid_prob_b = self.sigmoid_prob_b.to(device=self.cuda)
        # Select optimizer
        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.hparams.learning_rate)
        # tf.train.AdagradOptimizer
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
        train_output = train_output + self.sigmoid_prob_b
        # self.splitted_propensity = torch.split(
        #     self.propensity, 1, dim=1)
        # for i in range(self.rank_list_size):
        #     self.create_summary('Examination Probability %d' % i,
        #                         'Examination Probability %d at global step %d' % (i, self.global_step),
        #                         torch.max(self.splitted_propensity[i]), True)

        # Conduct estimation step.
        gamma = torch.sigmoid(train_output)
        # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_train_labels = self.labels
        p_e1_r0_c0 = self.propensity * \
                     (1 - gamma) / (1 - self.propensity * gamma)
        p_e0_r1_c0 = (1 - self.propensity) * gamma / \
                     (1 - self.propensity * gamma)
        # p_e0_r0_c0 = (1 - self.propensity) * (1 - gamma) / \
        #              (1 - self.propensity * gamma)
        # p_e1 = p_e1_r0_c0 + p_e1_r1_c1
        p_r1 = reshaped_train_labels + \
               (1 - reshaped_train_labels) * p_e0_r1_c0

        # Get Bernoulli samples and compute rank loss
        self.ranker_labels = get_bernoulli_sample(p_r1)
        if self.is_cuda_avail:
            self.ranker_labels = self.ranker_labels.to(device=self.cuda)
        # criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        criterion = torch.nn.BCEWithLogitsLoss()

        self.loss = criterion(train_output,self.ranker_labels)
        # record additional positive instance from sampling
        # labels_split_size = int(self.ranker_labels.shape[1] / self.rank_list_size)
        # split_ranker_labels = torch.split(
        #     self.ranker_labels, labels_split_size, dim=1)
        # for i in range(self.rank_list_size):
        #     additional_postive_instance = (torch.sum(split_ranker_labels[i]) - torch.sum(
        #         train_labels[i])) / (torch.sum(torch.ones_like(train_labels[i])) - torch.sum(train_labels[i]))
            # self.create_summary('Additional pseudo clicks %d' %i,
            #                     'Additional pseudo clicks %d at global step %d' % (i, self.global_step),
            #                     additional_postive_instance, True)


        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * self.l2_loss(p)

        opt = self.optimizer_func
        opt.zero_grad(set_to_none=True)
        self.loss.backward()
        if self.loss == 0:
            for name, param in self.model.named_parameters():
                print(name, param)
        if self.hparams.max_gradient_norm > 0:
            self.clipped_gradient = nn.utils.clip_grad_norm_(
                params, self.hparams.max_gradient_norm)
        opt.step()
        nn.utils.clip_grad_value_(reshaped_train_labels, 1)

        # Conduct maximization step
        with torch.no_grad():
            self.propensity = (1 - self.hparams.EM_step_size) * self.propensity + self.hparams.EM_step_size * torch.mean(
        reshaped_train_labels + (1 - reshaped_train_labels) * p_e1_r0_c0, dim=0, keepdim=True)
        self.update_propensity_op = self.propensity
        self.propensity_weights = 1.0 / self.propensity


        self.global_step += 1
        print('Loss %f at global step %d' % (self.loss, self.global_step))
        return self.loss.item(), None, self.train_summary

    def validation(self, input_feed, is_online_simulation= False):
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
