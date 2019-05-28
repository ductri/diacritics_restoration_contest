import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, encoder_output_size):
        super(Attention, self).__init__()
        self.attention_size = 256
        self.context_weight = nn.Linear(encoder_output_size, 1)

    def forward(self, encoder_output, current_embedding_word, *args):
        """

        :param encoder_output:
        :param current_embedding_word:
        :param args:
        :return:
        """
        weights = self.context_weight(encoder_output)
