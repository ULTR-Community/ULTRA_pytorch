"""Training and testing the SetRank model.

See the following paper for more information.

	* Liang Pang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, Jirong Wen. 2020. SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval. In Proceedings of SIGIR '20
"""

from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import random
from ultra.ranking_model import BaseRankingModel
import ultra.utils
import torch.nn as nn
import torch


# encodr part the transformer is borrowed from
# https://www.tensorflow.org/tutorials/text/transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MultiHeadAttention(nn.Module):
    def __init__(self, input_feature_size, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # self.wq = nn.Linear(input_feature_size, d_model)
        # self.wk = nn.Linear(input_feature_size, d_model)
        # self.wv = nn.Linear(input_feature_size, d_model)

        self.dense = nn.Linear(input_feature_size, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask, training=True):
        batch_size = q.size()[0]
        #         print(q.get_shape(), "q.get_shape()")
        #         print("being called" + "*" * 100)
        if q.size()[-2] > 10:
            collect = ["eval"]
        else:
            collect = ["train"]
        #         q = self.wq(q)  # (batch_size, seq_len, d_model)
        #         k = self.wk(k)  # (batch_size, seq_len, d_model)
        #         v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = q  # (batch_size, seq_len, d_model)
        k = k  # (batch_size, seq_len, d_model)
        v = v  # (batch_size, seq_len, d_model)
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)  # (batch_size, num_heads, seq_len_q, depth)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        concat_attention = torch.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(input_feature_size, d_model, dff):
    return nn.Sequential(
        # (batch_size, seq_len, dff)
        nn.Linear(input_feature_size, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)  # (batch_size, seq_len, d_model)
    )


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, d_model, num_heads)
        self.mha = self.mha.to(device=device)
        self.ffn = point_wise_feed_forward_network(d_model, d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, training, mask):
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask, training)
        attn_output = self.dropout1(attn_output)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, init_feature_size,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.init_feature_size = init_feature_size

        #         self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        #         self.pos_encoding = positional_encoding(maximum_position_encoding,
        #                                                 self.d_model)
        self.input_layer_norm = nn.LayerNorm(self.init_feature_size,
            eps=1e-6)
        self.input_embedding = point_wise_feed_forward_network(self.init_feature_size, d_model, dff)
        self.output_layer = point_wise_feed_forward_network(d_model, 1, dff)
        self.enc_layers = nn.Sequential().to(dtype=torch.float32)
        for i in range(num_layers):
            encoder_layer = EncoderLayer(d_model, num_heads, dff, rate).to(device=device)
            self.enc_layers.add_module('encoder{}'.format(i), encoder_layer)
        #         self.softmax_output=tf.keras.layers.Softmax(1)
        self.dropout = nn.Dropout(rate)

    def forward(self, x, training, mask):
        seq_len = x.size()[1]
        # (batch_size, input_seq_len, d_model)
        x = self.input_embedding(self.input_layer_norm(x))
        x = self.dropout(x)

        for layer in self.enc_layers:
            x = layer(x, training, mask)
        x = self.output_layer(x)
        #         x=self.softmax_output(x)
        return x  # (batch_size, input_seq_len, 1)


def scaled_dot_product_attention(
        q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    #         q_length=tf.norm(q,ord=2, axis=3,keepdims=True)
    #         k_length=tf.norm(k,ord=2, axis=3,keepdims=True)
    # (..., seq_len_q, seq_len_k)
    matmul_qk = torch.matmul(q, k.permute(0,1,3,2))
    #         matmul_qk=q_length*k_length
    print()
    # scale matmul_qk
    dk = torch.tensor(k.size()[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    softmax = nn.Softmax(dim=-1)
    attention_weights = softmax(scaled_attention_logits)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class SetRank(nn.Module):
    """The SetRank model for learning to rank.

    This class implements the SetRank model for ranking.

    See the following paper for more information.

    * Liang Pang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, Jirong Wen. 2020. SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval. In Proceedings of SIGIR '20

    """

    def __init__(self, hparams_str, feature_size=None):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """
        print("build SetRank")
        super(SetRank, self).__init__()
        self.hparams = ultra.utils.hparams.HParams(
            d_model=256,
            num_heads=8,
            num_layers=2,
            diff=64,
            rate=0.0,
            initializer=None
        )
        self.hparams.parse(hparams_str)
        self.initializer = None
        self.feature_size = feature_size
        if self.hparams.initializer == 'constant':
            self.initializer = 0.001
        self.Encoder_layer = Encoder(self.hparams.num_layers, self.hparams.d_model,
                                     self.hparams.num_heads, self.hparams.diff, self.feature_size, self.hparams.rate)

    def build(self, input_list, noisy_params=None,
              noise_rate=0.05, is_training=False, **kwargs):
        """ Create the SetRank model (no supports for noisy parameters)

        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features
                        for a list of documents.
            noisy_params: (dict<parameter_name, tf.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """
        mask = None
        list_size = len(input_list)
        ind = list(range(0, list_size))
        random.shuffle(ind)
        x = [torch.unsqueeze(e, 1) for e in input_list]
        x = torch.cat(x, dim=1)  # [batch,len_seq,feature_size]
        x = x.float()
        x = x.to(device=device)
        x = self.Encoder_layer(x, is_training, mask)  # [batch,len_seq,1]
        output = []
        for i in range(list_size):
            output.append(x[:, i, :])
        return output  # [len_seq,batch,1]
