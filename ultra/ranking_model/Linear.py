from __future__ import print_function
from __future__ import absolute_import
from ultra.ranking_model import BaseRankingModel
import ultra

import torch.nn as nn
import torch

device = torch.device('cuda')

class Linear( nn.Module):
    """A linear model for learning to rank.

    This class implements a linear ranking model. It's essientially a logistic regression model.

    """

    def __init__(self, hparams_str, feature_size):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """
        super(Linear, self).__init__()
        self.hparams = ultra.utils.hparams.HParams(
            initializer='None',                         # Set parameter initializer
            norm="layer"                                # Set the default normalization
        )
        self.hparams.parse(hparams_str)
        self.initializer = None
        self.layer_norm = None
        self.output_sizes = [1]

        if self.hparams.initializer in BaseRankingModel.INITIALIZER_DIC:
            self.initializer = BaseRankingModel.INITIALIZER_DIC[self.hparams.initializer]

        modules = []
        for j in range(len(self.output_sizes)):
            if self.layer_norm is None and self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC:
                if self.hparams.norm == "layer":
                    modules.append(nn.LayerNorm(feature_size).to(dtype=torch.float32))
                else:
                    modules.append(nn.BatchNorm2d(feature_size).to(dtype=torch.float32))
            modules.append(nn.Linear(feature_size, self.output_sizes[j]))
            feature_size = self.output_sizes[j]
        self.sequential = nn.Sequential(*modules).to(dtype=torch.float32)

    def build(self, input_list, noisy_params=None,
              noise_rate=0.05, is_training=False, **kwargs):
        """ Create the Linear model

        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features
                        for a list of documents.
            noisy_params: (dict<parameter_name, tf.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """
        input_data = torch.cat(input_list, dim=0)
        input_data = input_data.to(dtype=torch.float32, device=device)
        if (noisy_params==None):
            output_data = self.sequential(input_data)
        else:
            ctr = 0
            for layer in self.sequential:
                if isinstance(layer, nn.Linear):
                    layer.weight += noisy_params[ctr]
                    ctr += 1
                    layer.bias += noisy_params[ctr]
                    ctr += 1
            output_data = self.sequential(input_data)
        output_shape = input_list[0].shape[0]
        return torch.split(output_data, output_shape, dim=0)
