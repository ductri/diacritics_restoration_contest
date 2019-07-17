import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class AttentionHead(nn.Module):
    def __init__(self, size):
        super(AttentionHead, self).__init__()
        self.inner_size = 64
        self.Q = nn.Linear(size, self.inner_size)
        self.K = nn.Linear(size, self.inner_size)
        self.V = nn.Linear(size, self.inner_size)
        self.__size = size

    def forward(self, emb_input, *input):
        """

        :param emb_input:  (batch, seq_len, emb_size)
        :param input:
        :return:
        """
        batch, seq_len = emb_input.size(0), emb_input.size(1)

        # (batch, seq_len, size)
        q = self.Q(emb_input)

        # (batch, seq_len, size)
        k = self.K(emb_input)

        # (batch, seq_len, size)
        v = self.V(emb_input)

        # (batch, seq_len, seq_len)
        attn = q.matmul(k.permute(0, 2, 1))
        attn = attn / np.sqrt(self.__size)
        assert attn.size(0) == batch
        assert attn.size(1) == attn.size(2)
        assert attn.size(1) == seq_len

        # (batch, seq_len, seq_len)
        attn = F.softmax(attn, dim=2)

        # (batch, seq_len, seq_len, size)
        attn = torch.mul(attn.view(batch, seq_len, seq_len,1), v.view(batch, 1, seq_len, -1))

        # (batch, seq_len, size)
        attn = attn.sum(dim=2)

        return attn

