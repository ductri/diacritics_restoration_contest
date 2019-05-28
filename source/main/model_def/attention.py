import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, enc_output_size, emb_size):
        super(Attention, self).__init__()
        self.attention_size = 256
        self.context_weight = nn.Linear(enc_output_size, emb_size)

    def forward(self, encoder_output, emb_size, *args):
        """

        :param encoder_output: shape == (seq_len, batch, num_directions * hidden_size)
        :param emb_size: shape == (batch, embedding_size)
        :param args:
        :return:
        """
        temp = self.context_weight()
        weights = self.context_weight(encoder_output)
        # 1*3  x,y  3*1
