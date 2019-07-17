import torch
from torch import nn

from model_def.transformer.attention_head import AttentionHead


class AttentionMultipleHead(nn.Module):
    def __init__(self, num, size):
        super(AttentionMultipleHead, self).__init__()
        self.heads = nn.ModuleList([AttentionHead(size) for _ in range(num)])
        self.projection = nn.Linear(num*self.heads[0].inner_size, out_features=size)

    def forward(self, emb_input, *input):
        """

        :param emb_input: (batch, seq_len, emb_size)
        :param input:
        :return:
        """
        # list of (batch, seq_len, size)
        tmp = [head(emb_input) for head in self.heads]
        tmp = torch.cat(tmp, dim=2)
        tmp = self.projection(tmp)
        return tmp
