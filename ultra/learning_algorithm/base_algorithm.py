"""The basic class that contains all the API needed for the implementation of an unbiased learning to rank algorithm.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
import torch
from abc import ABC, abstractmethod

import ultra
import ultra.utils as utils


def softmax_cross_entropy_with_logits(logits, labels):
    """Computes softmax cross entropy between logits and labels.

    Args:
        output: A tensor with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
        labels: A tensor of the same shape as `output`. A value >= 1 means a
        relevant example.
    Returns:
        A single value tensor containing the loss.
    """
    loss = torch.sum(- labels * F.log_softmax(logits, -1), -1)
    return loss

class BaseAlgorithm(ABC):
    """The basic class that contains all the API needed for the
        implementation of an unbiased learning to rank algorithm.

    """
    PADDING_SCORE = -100000

    @abstractmethod
    def __init__(self, data_set, exp_settings):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        self.is_training = None
        self.docid_inputs = None  # a list of top documents
        self.letor_features = None  # the letor features for the documents
        self.labels = None  # the labels for the documents (e.g., clicks)
        self.output = None  # the ranking scores of the inputs
        # the number of documents considered in each rank list.
        self.rank_list_size = None
        # the maximum number of candidates for each query.
        self.max_candidate_num = None
        self.optimizer_func = torch.optim.adagrad()
        pass


    # def step(self, session, input_feed, forward_only):
    #     """Run a step of the model feeding the given inputs.
    #
    #     Args:
    #         session: (tf.Session) tensorflow session to use.
    #         input_feed: (dictionary) A dictionary containing all the input feed data.
    #         forward_only: whether to do the backward step (False) or only forward (True).
    #
    #     Returns:
    #         A triple consisting of the loss, outputs (None if we do backward),
    #         and a tf.summary containing related information about the step.
    #
    #     """
    #     pass
    @abstractmethod
    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        pass

    @abstractmethod
    def validation(self, input_feed):
        """Run a step of the model feeding the given inputs for validating process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        pass

    def remove_padding_for_metric_eval(self, input_id_list, model_output):
        output_scores = torch.unbind(model_output, dim=1)
        if len(output_scores) > len(input_id_list):
            raise AssertionError(
                'Input id list is shorter than output score list when remove padding.')
        # Build mask
        valid_flags = torch.cat((torch.ones(self.letor_features.size()[0]), torch.zeros([1])), dim=0)
        valid_flags = valid_flags.type(torch.bool)
        input_flag_list = []
        for i in range(len(output_scores)):
            input_flag_list.append(
                torch.index_select(
                    valid_flags, 0, input_id_list[i]))
        # Mask padding documents
        output_scores = list(output_scores)
        for i in range(len(output_scores)):
            output_scores[i] = torch.where(
                input_flag_list[i],
                output_scores[i],
                torch.ones_like(output_scores[i]) * self.PADDING_SCORE
            )
        return torch.stack(output_scores, axis=1)

    def ranking_model(self,model, list_size):
        """Construct ranking model with the given list size.

        Args:
            model: (BaseRankingModel) The model that is used to compute the ranking score.
            list_size: (int) The top number of documents to consider in the input docids.
            scope: (string) The name of the variable scope.

        Returns:
            A tensor with the same shape of input_docids.

        """
        output_scores = self.get_ranking_scores(model = model,
             input_id_list= self.docid_inputs, is_training=self.is_training)
        return torch.cat(output_scores, 1)

    def get_ranking_scores(self, model, input_id_list,
                           is_training=False, **kwargs):
        """Compute ranking scores with the given inputs.

        Args:
            model: (BaseRankingModel) The model that is used to compute the ranking score.
            input_id_list: (list<tf.Tensor>) A list of tensors containing document ids.
                            Each tensor must have a shape of [None].
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A tensor with the same shape of input_docids.

        """
            # Build feature padding
        PAD_embed = torch.zeros([1, self.feature_size], dtype = torch.float32)
        letor_features = torch.cat(
            dim=0, tensors=(
                self.letor_features, PAD_embed))
        print(self.letor_features)
        input_feature_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(
                torch.index_select(
                    letor_features,0, input_id_list[i]))
        return model.build(
            input_feature_list, is_training=is_training, **kwargs)

    def create_model(self, feature_size):
        """ Initialize the ranking model.

        ReturnsL
            The ranking model that will be used to computer the ranking score.

        """
        if not hasattr(self, "model"):
            model = utils.find_class(
                self.exp_settings['ranking_model'])(
                self.exp_settings['ranking_model_hparams'], feature_size)
        return model

    def pairwise_cross_entropy_loss(
            self, pos_scores, neg_scores, propensity_weights=None, name=None):
        """Computes pairwise softmax loss without propensity weighting.

        Args:
            pos_scores: (tf.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a positive example.
            neg_scores: (tf.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a negative example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(pos_scores)
        label_dis = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], axis=1)
        # loss = tf.nn.softmax_cross_entropy_with_logits(
        #     logits=torch.cat([pos_scores, neg_scores], axis=1), labels=label_dis
        # ) * propensity_weights
        loss = softmax_cross_entropy_with_logits(
            logits = torch.cat([pos_scores, neg_scores], axis=1), labels = label_dis)* propensity_weights
        return loss

    def sigmoid_loss_on_list(self, output, labels,
                             propensity_weights=None, name=None):
        """Computes pointwise sigmoid loss without propensity weighting.

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
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        label_dis = torch.minimum(labels, 1)
        loss = softmax_cross_entropy_with_logits(
            logits = output, labels = label_dis) * propensity_weights
        return torch.mean(torch.sum(loss, axis=1))

    def pairwise_loss_on_list(self, output, labels,
                              propensity_weights=None, name=None):
        """Computes pairwise entropy loss.

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
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        loss = None
        sliced_output = torch.unbind(output, axis=1)
        sliced_label = torch.unbind(labels, axis=1)
        sliced_propensity = torch.unbind(propensity_weights, axis=1)
        for i in range(len(sliced_output)):
            for j in range(i + 1, len(sliced_output)):
                cur_label_weight = torch.sign(
                    sliced_label[i] - sliced_label[j])
                cur_propensity = sliced_propensity[i] * \
                    sliced_label[i] + \
                    sliced_propensity[j] * sliced_label[j]
                cur_pair_loss = - \
                    torch.exp(
                        sliced_output[i]) / (torch.exp(sliced_output[i]) + torch.exp(sliced_output[j]))
                if loss is None:
                    loss = cur_label_weight * cur_pair_loss
                loss += cur_label_weight * cur_pair_loss * cur_propensity
        batch_size = labels.size()[0]
        # / (tf.reduce_sum(propensity_weights)+1)
        return torch.sum(loss) / batch_size.type(torch.float32)

    def softmax_loss(self, output, labels, propensity_weights=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)
        print(output)
        weighted_labels = (labels + 0.0000001) * propensity_weights
        label_dis = weighted_labels / \
            torch.sum(weighted_labels, 1, keepdim=True)
        label_dis[label_dis!=label_dis] = 0
        loss = softmax_cross_entropy_with_logits(
            logits = output, labels = label_dis)* torch.sum(weighted_labels, 1)
        return torch.sum(loss) / torch.sum(weighted_labels)



