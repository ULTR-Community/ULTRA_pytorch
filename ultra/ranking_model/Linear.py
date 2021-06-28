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
            norm="layer"                                # Set the default normalization
        )
        self.hparams.parse(hparams_str)
        self.initializer = None
        self.layer_norm = None
        self.output_sizes = [1]
        self.sequential = nn.Sequential().to(dtype=torch.float32)

        for j in range(len(self.output_sizes)):
            if self.layer_norm is None and self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC:
                if self.hparams.norm == "layer":
                    self.sequential.add_module('layer_norm{}'.format(j),
                                               nn.LayerNorm(feature_size).to(dtype=torch.float32))
                else:
                    self.sequential.add_module('batch_norm{}'.format(j),
                                               nn.BatchNorm2d(feature_size).to(dtype=torch.float32))

            self.sequential.add_module('linear{}'.format(j), nn.Linear(feature_size, self.output_sizes[j]))
            if j != len(self.output_sizes) - 1:
                self.sequential.add_module('act{}'.format(j), self.act_func)
            feature_size = self.output_sizes[j]

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
        input_data = input_data.to(dtype=torch.float32)
        if torch.cuda.is_available():
            input_data = input_data.to(device=device)
        if (noisy_params==None):
            output_data = self.sequential(input_data)
        else:
            for name, parameter in self.sequential.named_parameters():
                if name in noisy_params:
                    with torch.no_grad():
                        noise = (noisy_params[name]* noise_rate)
                        if torch.cuda.is_available():
                            noise = noise.to(device=device)
                        parameter += noise
            output_data = self.sequential(input_data)
        output_shape = input_list[0].shape[0]
        return torch.split(output_data, output_shape, dim=0)
