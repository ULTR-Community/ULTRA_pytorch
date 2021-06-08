"""Training and testing the inverse propensity weighting algorithm for unbiased learning to rank.

See the following paper for more information on the inverse propensity weighting algorithm.

    * Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
    * Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import time

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils


def selu(x):
    # with tf.name_scope('selu') as scope:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * torch.where(x >= 0.0, x, alpha * F.elu(x))


class IPWrank(BaseAlgorithm):
    """The Inverse Propensity Weighting algorithm for unbiased learning to rank.

    This class implements the training and testing of the Inverse Propensity Weighting algorithm for unbiased learning to rank. See the following paper for more information on the algorithm.

    * Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
    * Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

    """

    def __init__(self, data_set, exp_settings):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """

        self.hparams = ultra.utils.hparams.HParams(
            propensity_estimator_type='ultra.utils.propensity_estimator.RandomizedPropensityEstimator',
            # the setting file for the predefined click models.
            propensity_estimator_json='./example/PropensityEstimator/randomized_pbm_0.1_1.0_4_1.0.json',
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_loss',      # Select Loss function
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )

        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.cuda = torch.device('cuda')
        self.train_summary = {}
        self.eval_summary = {}
        self.is_training = "is_train"
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.feature_size = data_set.feature_size
        self.model = self.create_model(self.feature_size)
        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
        self.propensity_estimator = ultra.utils.find_class(
            self.hparams.propensity_estimator_type)(
            self.hparams.propensity_estimator_json)

        self.max_candidate_num = exp_settings['max_candidate_num']
        with torch.no_grad():
            self.learning_rate = float(self.hparams.learning_rate)
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
        self.PAD_embed = self.PAD_embed.to(dtype = torch.float32)

        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        # tf.train.AdagradOptimizer
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            # tf.train.GradientDescentOptimizer

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
        self.global_step += 1
        pw = []
        self.model.train()
        for l in range(self.rank_list_size):
            input_feed["propensity_weights{0}".format(l)] = []
        for i in range(len(input_feed[self.labels_name[0]])):
            click_list = [input_feed[self.labels_name[l]][i]
                          for l in range(self.rank_list_size)]
            pw_list = self.propensity_estimator.getPropensityForOneList(
                click_list)
            pw.append(pw_list)
            for l in range(self.rank_list_size):
                input_feed["propensity_weights{0}".format(l)].append(
                    pw_list[l])

        self.create_input_feed(input_feed, self.rank_list_size)

        self.propensity_weights = pw
        # Gradients and SGD update operation for training the model.

        train_output = self.ranking_model(self.model,
            self.rank_list_size)
        train_labels = self.labels
        train_pw = torch.as_tensor(self.propensity_weights)
        if self.is_cuda_avail:
            train_pw = train_pw.to(device=self.cuda)
        self.loss = None

        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss = self.sigmoid_loss_on_list(
                train_output, train_labels, train_pw)
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss = self.pairwise_loss_on_list(
                train_output, train_labels, train_pw)
        else:
            self.loss = self.softmax_loss(
                train_output, train_labels, train_pw)

        # params = tf.trainable_variables()
        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * self.l2_loss(p)

        self.opt_step(self.optimizer_func, params)
        # self.create_summary('Learning Rate', 'Learning_rate at global step %d' % self.global_step, self.learning_rate,
        #                     True)
        # self.create_summary('Loss', 'Loss at global step %d' % self.global_step, self.learning_rate,True)

        nn.utils.clip_grad_value_(train_labels, 1)
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

        print(" Loss %f at Global Step %d: " % (self.loss.item(),self.global_step))
        return self.loss.item(), None, self.train_summary

    def validation(self, input_feed):
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

        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, self.output)

        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(self.labels, pad_removed_output, None)
                self.create_summary('%s_%d' % (metric, topn),
                                    '%s_%d' % (metric, topn), metric_value, False)
        return None, self.output, self.eval_summary  # loss, outputs, summary.