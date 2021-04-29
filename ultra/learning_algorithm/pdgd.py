"""Training and testing the Pairwise Differentiable Gradient Descent (PDGD) algorithm for unbiased learning to rank.

See the following paper for more information on the Pairwise Differentiable Gradient Descent (PDGD) algorithm.

    * Oosterhuis, Harrie, and Maarten de Rijke. "Differentiable unbiased online learning to rank." In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 1293-1302. ACM, 2018.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


from six.moves import zip

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils as utils
import ultra


class PDGD(BaseAlgorithm):
    """The Pairwise Differentiable Gradient Descent (PDGD) algorithm for unbiased learning to rank.

    This class implements the Pairwise Differentiable Gradient Descent (PDGD) algorithm based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Oosterhuis, Harrie, and Maarten de Rijke. "Differentiable unbiased online learning to rank." In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 1293-1302. ACM, 2018.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Pairwise Differentiable Gradient Descent (PDGD) algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,                 # Learning rate (\mu).
            # Scalar for the probability distribution.
            tau=1,
            max_gradient_norm=1.0,            # Clip gradients to this norm.
            # Set strength for L2 regularization.
            l2_loss=0.005,
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
        self.learning_rate = self.hparams.learning_rate

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
        self.rank_list_size = exp_settings['selection_bias_cutoff']
        self.positive_docid_inputs = None
        self.negative_docid_inputs = None
        self.pair_weights = None

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
        # Run the model to get ranking scores

        self.model.eval()
        self.create_input_feed(input_feed, self.rank_list_size)

        outputs = self.ranking_model(self.model, self.rank_list_size)
        # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_train_labels = torch.transpose(self.labels,0,1)
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, outputs)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                    reshaped_train_labels, pad_removed_output, None)
                self.writer.add_scalar('train_eval %s_%d' %
                    (metric, topn), metric_value, self.global_step)

        # reduce value to avoid numerical problems
        rank_outputs= outputs.cpu().detach().numpy()
        rank_outputs = rank_outputs - \
            np.amax(rank_outputs, axis=1, keepdims=True)
        exp_ranking_scores = np.exp(self.hparams.tau * rank_outputs)

        # Remove scores for padding documents
        letor_features_length = len(self.letor_features)
        for i in range(len(input_feed[self.labels_name[0]])):
            for j in range(self.rank_list_size):
                # not a valid doc
                if input_feed[self.docid_inputs_name[j]][i] == letor_features_length:
                    exp_ranking_scores[i][j] = 0.0
        # Compute denominator for each position
        denominators = np.cumsum(
            exp_ranking_scores[:, ::-1], axis=1)[:, ::-1]
        sum_log_denominators = np.sum(
            np.log(
                denominators,
                out=np.zeros_like(denominators),
                where=denominators > 0),
            axis=1)
        # Create training pairs based on the ranking scores and the labels
        positive_docids, negative_docids, pair_weights = [], [], []
        for i in range(len(input_feed[self.labels_name[0]])):
            # Generate pairs and compute weights
            for j in range(self.rank_list_size):
                l = self.rank_list_size - 1 - j
                # not a valid doc
                if input_feed[self.docid_inputs_name[l]][i] == letor_features_length:
                    continue
                if input_feed[self.labels_name[l]][i] > 0:  # a clicked doc
                    for k in range(l + 2):
                        # find a negative/unclicked doc
                        if k < self.rank_list_size and \
                                input_feed[self.labels_name[k]][i] < input_feed[self.labels_name[l]][i]:
                            # not a valid doc
                            if input_feed[self.docid_inputs_name[k] ][i] == letor_features_length:
                                continue
                            positive_docids.append(
                                input_feed[self.docid_inputs_name[l]][i])
                            negative_docids.append(
                                input_feed[self.docid_inputs_name[k]][i])
                            flipped_exp_scores = np.copy(
                                exp_ranking_scores[i])
                            flipped_exp_scores[k] = exp_ranking_scores[i][l]
                            flipped_exp_scores[l] = exp_ranking_scores[i][k]
                            flipped_denominator = np.cumsum(
                                flipped_exp_scores[::-1])[::-1]

                            sum_log_flipped_denominator = np.sum(
                                np.log(
                                    flipped_denominator,
                                    out=np.zeros_like(flipped_denominator),
                                    where=flipped_denominator > 0))
                            #p_r = np.prod(rank_prob[i][min_i:max_i+1])
                            #p_rs = np.prod(flipped_rank_prob[min_i:max_i+1])
                            # weight = p_rs / (p_r + p_rs) = 1 / (1 +
                            # (d_rs/d_r)) = 1 / (1 + exp(log_drs - log_dr))
                            weight = 1.0 / \
                                (1.0 +
                                 np.exp(min(sum_log_flipped_denominator -
                                            sum_log_denominators[i], 20)))
                            if np.isnan(weight):
                                print('SOMETHING WRONG!!!!!!!')
                                print(
                                    'sum_log_denominators[i] is nan: ' + str(np.isnan(sum_log_denominators[i])))
                                print('sum_log_flipped_denominator is nan ' +
                                      str(np.isnan(sum_log_flipped_denominator)))
                            pair_weights.append(weight)

        self.positive_docid_inputs = torch.tensor(data=positive_docids, dtype=torch.int64)
        self.negative_docid_inputs = torch.tensor(data=negative_docids, dtype=torch.int64)
        self.pair_weights = torch.tensor(data=pair_weights, device=self.cuda)

        # Train the model
        pair_scores = self.get_ranking_scores(self.model,
            [self.positive_docid_inputs,
             self.negative_docid_inputs])
        # print("pair scores: ", pair_scores)

        self.loss = torch.sum(
            torch.mul(
                # self.pairwise_cross_entropy_loss(pair_scores[0], pair_scores[1]),
                torch.sum(-torch.exp(pair_scores[0]) / (
                        torch.exp(pair_scores[0]) + torch.exp(pair_scores[1])), 1),
                self.pair_weights
            )
        )
        print("first loss: ", self.loss)
        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * self.l2_loss(p)

        # Gradients and SGD update operation for training the model.
        self.opt_step(self.optimizer_func, params)
        self.create_summary('Learning Rate', 'Learning_rate at global step %d' % self.global_step, self.learning_rate,
                            True)
        self.create_summary('Loss', 'Loss at global step %d' % self.global_step, self.learning_rate, True)
        # loss, no outputs, summary.
        self.global_step+=1
        return self.loss, None, self.train_summary

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
        return None, self.output, self.eval_summary  # no loss, outputs, summary.
