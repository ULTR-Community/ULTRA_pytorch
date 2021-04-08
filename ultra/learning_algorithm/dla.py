"""Training and testing the dual learning algorithm for unbiased learning to rank.

See the following paper for more information on the dual learning algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from six.moves import zip
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils


def sigmoid_prob(logits):
    return torch.sigmoid(logits - torch.mean(logits, -1, keepdim=True))

class DenoisingNet(nn.Module):
    def __init__(self, input_vec_size):
        super(DenoisingNet, self).__init__()
        self.linear_layer = nn.Linear(input_vec_size, 1)
        self.elu_layer = nn.ELU()
        self.propensity_net = nn.Sequential(self.linear_layer, self.elu_layer)
        self.list_size = input_vec_size

    def forward(self, input_list):
        output_propensity_list = []
        for i in range(self.list_size):
            # Add position information (one-hot vector)
            click_feature = [
                torch.unsqueeze(
                    torch.zeros_like(
                        input_list[i]), -1) for _ in range(self.list_size)]
            click_feature[i] = torch.unsqueeze(
                torch.ones_like(input_list[i]), -1)
            # Predict propensity with a simple network
            output_propensity_list.append(
                self.propensity_net(
                    torch.cat(
                        click_feature, 1)))

        return torch.cat(output_propensity_list, 1)



class DLA(BaseAlgorithm):
    """The Dual Learning Algorithm for unbiased learning to rank.

    This class implements the Dual Learning Algorithm (DLA) based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build DLA')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_loss',            # Select Loss function
            # the function used to convert logits to probability distributions
            logits_to_prob='softmax',
            # The learning rate for ranker (-1 means same with learning_rate).
            propensity_learning_rate=-1.0,
            ranker_loss_weight=1.0,            # Set the weight of unbiased ranking loss
            # Set strength for L2 regularization.
            l2_loss=0.0,
            max_propensity_weight=-1,      # Set maximum value for propensity weights
            constant_propensity_initialization=False,
            # Set true to initialize propensity with constants.
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
        self.propensity_model = DenoisingNet(self.rank_list_size).to(device=self.cuda)
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

        if self.hparams.propensity_learning_rate < 0:
            self.propensity_learning_rate = float(self.hparams.learning_rate)
        else:
            self.propensity_learning_rate = float(self.hparams.propensity_learning_rate)
        self.learning_rate = float(self.hparams.learning_rate)

        self.global_step = 0

        # Select logits to prob function
        self.logits_to_prob = nn.Softmax()
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob



        print('Loss Function is ' + self.hparams.loss_func)
        # Select loss function
        self.loss_func = None
        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss_func = self.sigmoid_loss_on_list
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss_func = self.pairwise_loss_on_list
        else:  # softmax loss without weighting
            self.loss_func = self.softmax_loss

    def separate_gradient_update(self):
        denoise_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()
        # Select optimizer
        self.optimizer_func = torch.optim.Adagrad
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD


        if self.hparams.l2_loss > 0:
            # for p in denoise_params:
            #    self.exam_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * nn.MSELoss(p) * 0.5
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

        opt_denoise = self.optimizer_func(denoise_params, self.propensity_learning_rate)
        opt_ranker = self.optimizer_func(ranking_model_params, self.learning_rate)

        opt_denoise.zero_grad()
        opt_ranker.zero_grad()

        self.loss.backward()

        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(denoise_params, self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(ranking_model_params, self.hparams.max_gradient_norm)

        opt_denoise.step()
        opt_ranker.step()

        total_norm = 0

        for p in denoise_params:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        for p in ranking_model_params:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.norm = total_norm

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        # Build model
        self.model.train()

        self.labels = []
        self.docid_inputs = []
        self.letor_features = torch.from_numpy(input_feed["letor_features"])
        for i in range(self.rank_list_size):
            self.docid_inputs.append(input_feed[self.docid_inputs_name[i]])
            self.labels.append(input_feed[self.labels_name[i]])

        self.labels = torch.tensor(data=self.labels, device=self.cuda)
        train_labels = self.labels
        self.docid_inputs = torch.tensor(data=self.docid_inputs,dtype=torch.int64)
        train_output = self.ranking_model(self.model,
            self.rank_list_size)
        self.propensity_model.train()
        self.propensity = self.propensity_model(
            self.labels)
        # Compute rank loss
        # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_train_labels = torch.transpose(
            train_labels, 0, 1)
        self.propensity_weights = self.get_normalized_weights(
            self.logits_to_prob(self.propensity))
        self.rank_loss = self.loss_func(
            train_output, reshaped_train_labels, self.propensity_weights)
        pw_list = torch.unbind(
            self.propensity_weights,
            dim=1)  # Compute propensity weights
        for i in range(len(pw_list)):
            self.writer.add_scalar(
                'Inverse Propensity weights %d' %
                i, torch.mean(
                    pw_list[i]))
            self.train_summary['Inverse Propensity weights %d' %i] = torch.mean(pw_list[i])
        self.writer.add_scalar(
            'Rank Loss',
            torch.mean(
                self.rank_loss))
        self.train_summary['Rank Loss'] =  torch.mean(
                self.rank_loss)

        # Compute examination loss
        self.relevance_weights = self.get_normalized_weights(
            self.logits_to_prob(train_output))
        self.exam_loss = self.loss_func(
            self.propensity,
            reshaped_train_labels,
            self.relevance_weights)
        rw_list = torch.unbind(
            self.relevance_weights,
            dim=1)  # Compute propensity weights
        for i in range(len(rw_list)):
            self.writer.add_scalar(
                'Relevance weights %d' %
                i, torch.mean(
                    rw_list[i]))
            self.train_summary['Relevance weights %d' %i] =  torch.mean(rw_list[i])
        self.writer.add_scalar(
            'Exam Loss',
            torch.mean(
                self.exam_loss))
        self.train_summary['Exam Loss'] = torch.mean(self.exam_loss)

        # Gradients and SGD update operation for training the model.
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss
        self.separate_gradient_update()

        self.writer.add_scalar(
            'Gradient Norm',
            self.norm)
        self.train_summary['Gradient Norm'] = self.norm
        self.writer.add_scalar(
            'Learning Rate',
            self.learning_rate)
        self.train_summary['Learning Rate'] = self.learning_rate
        self.writer.add_scalar(
            'Final Loss', torch.mean(self.loss))
        self.train_summary['Final Loss'] = torch.mean(self.loss)

        self.clip_grad_value(reshaped_train_labels, clip_value_min=0, clip_value_max=1)
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
                    (metric, topn), metric_value)
                self.train_summary['%s_%d' %
                    (metric, topn)] = metric_value
                weighted_metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                    reshaped_train_labels, pad_removed_train_output, list_weights)
                self.writer.add_scalar(
                    'Weighted_%s_%d' %
                    (metric, topn), weighted_metric_value)
                self.train_summary['Weighted_%s_%d' %
                                   (metric, topn)] = weighted_metric_value
        # loss, no outputs, summary.
        return self.loss, None, self.train_summary

    def validation(self, input_feed):
        self.model.eval()
        self.propensity_model.eval()
        self.labels = []
        self.docid_inputs = []
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(input_feed[self.docid_inputs_name[i]])
            self.labels.append(input_feed[self.labels_name[i]])
        self.labels = torch.tensor(data=self.labels, device=self.cuda)
        self.docid_inputs = torch.tensor(data=self.docid_inputs,dtype=torch.int64)
        self.output = self.ranking_model(self.model,
                                         self.max_candidate_num)
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, self.output)

        # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        reshaped_labels = torch.transpose(torch.tensor(self.labels),0,1)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(reshaped_labels, pad_removed_output, None)
                self.writer.add_scalar(
                    '%s_%d' %
                    (metric, topn), metric_value)
                self.eval_summary['%s_%d' %
                    (metric, topn)] = metric_value
        return None, self.output, self.eval_summary # no loss, outputs, summary.

    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = torch.unbind(
            propensity, dim=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = torch.stack(pw_list, dim=1)
        if self.hparams.max_propensity_weight > 0:
            self.clip_grad_value(propensity_weights,clip_value_min=0,
                clip_value_max=self.hparams.max_propensity_weight)
        return propensity_weights

    def clip_grad_value(self, parameters, clip_value_min, clip_value_max) -> None:
        r"""Clips gradient of an iterable of parameters at specified value.

        Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            clip_value (float or int): maximum allowed value of the gradients.
                The gradients are clipped in the range
                :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        clip_value_min = float(clip_value_min)
        clip_value_max = float(clip_value_max)
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.clamp_(min=clip_value_min, max=clip_value_max)

    '''
    def click_weighted_softmax_cross_entropy_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_softmax_cross_entropy", [output]):
            label_dis = labels * propensity_weights / \
                tf.reduce_sum(labels * propensity_weights, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=output, labels=label_dis) * tf.reduce_sum(labels * propensity_weights, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(labels * propensity_weights)

    def click_weighted_pairwise_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes pairwise entropy loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
            (tf.Tensor) A tensor containing the propensity weights.
        """
        loss = None
        with tf.name_scope(name, "click_weighted_pairwise_loss", [output]):
            sliced_output = tf.unstack(output, axis=1)
            sliced_label = tf.unstack(labels, axis=1)
            sliced_propensity = tf.unstack(propensity_weights, axis=1)
            for i in range(len(sliced_output)):
                for j in range(i + 1, len(sliced_output)):
                    cur_label_weight = tf.math.sign(
                        sliced_label[i] - sliced_label[j])
                    cur_propensity = sliced_propensity[i] * \
                        sliced_label[i] + \
                        sliced_propensity[j] * sliced_label[j]
                    cur_pair_loss = - \
                        tf.exp(
                            sliced_output[i]) / (tf.exp(sliced_output[i]) + tf.exp(sliced_output[j]))
                    if loss is None:
                        loss = cur_label_weight * cur_pair_loss * cur_propensity
                    loss += cur_label_weight * cur_pair_loss * cur_propensity
        batch_size = tf.shape(labels[0])[0]
        # / (tf.reduce_sum(propensity_weights)+1)
        return tf.reduce_sum(loss) / tf.cast(batch_size, dtypes.float32)

    def click_weighted_log_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes pointwise sigmoid loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_weighted_log_loss", [output]):
            click_prob = tf.sigmoid(output)
            loss = tf.losses.log_loss(labels, click_prob, propensity_weights)
        return loss
        '''
