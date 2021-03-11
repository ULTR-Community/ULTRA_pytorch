from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch

from ultra.ranking_model import BaseRankingModel
from ultra.ranking_model import ActivationFunctions
import ultra.utils


class DNN(BaseRankingModel):
    """The deep neural network model for learning to rank.

    This class implements a deep neural network (DNN) based ranking model. It's essientially a multi-layer perceptron network.

    """

    def __init__(self, hparams_str):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """

        self.hparams = ultra.utils.hparams.HParams(
            # Number of neurons in each layer of a ranking_model.
            hidden_layer_sizes=[512, 256, 128],
            # Type for activation function, which could be elu, relu, sigmoid,
            # or tanh
            activation_func='elu',
            initializer='None',                         # Set parameter initializer
            norm="layer"                                # Set the default normalization
        )
        self.hparams.parse(hparams_str)
        self.initializer = None
        self.act_func = None
        self.layer_norm = None

        if self.hparams.activation_func in BaseRankingModel.ACT_FUNC_DIC:
            self.act_func = BaseRankingModel.ACT_FUNC_DIC[self.hparams.activation_func]

        if self.hparams.initializer in BaseRankingModel.INITIALIZER_DIC:
            self.initializer = BaseRankingModel.INITIALIZER_DIC[self.hparams.initializer]

    def build(self, input_list, noisy_params=None,
              noise_rate=0.05, is_training=False, **kwargs):
        """ Create the DNN model

        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features
                        for a list of documents.
            noisy_params: (dict<parameter_name, tf.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """
        input_data = torch.cat(input_list, axis = 0)
        output_data = input_data
        output_sizes = self.hparams.hidden_layer_sizes + [1]
        current_size = output_data.size()[-1]
        if self.layer_norm is None and self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC:
            self.layer_norm = []
            self.layer_norm.append(nn.LayerNorm(current_size))
            for j in range(len(output_sizes)):
                self.layer_norm.append(nn.LayerNorm(output_sizes[j]))


        for j in range(len(output_sizes)):
            if self.layer_norm is not None:
                if self.hparams.norm == "layer":
                    output_data = self.layer_norm[j](
                        output_data.float())
                else:
                    output_data = self.layer_norm[j](
                        output_data.float())
            linear_layer = nn.Linear(current_size, output_sizes[j])
            output_data = linear_layer(output_data.float())
            # Add activation if it is a hidden layer
            if j != len(output_sizes) - 1:
                output_data = self.act_func(output_data)
            current_size = output_sizes[j]

        return torch.split(output_data, len(input_list), dim=0)
