import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, enc_output_size, dec_output_size, output_size):
        super(Attention, self).__init__()
        self.attention_size = 256
        self.scoring = nn.Linear(dec_output_size, enc_output_size)
        self.softmax = nn.Softmax(dim=1)
        self.shrink_output = nn.Linear(enc_output_size + dec_output_size, output_size)

    def forward(self, enc_outputs, dec_output, *args):
        """

        :param enc_outputs: shape == (seq_len, batch, enc_output_size)
        :param dec_output: shape == (batch, dec_output_size)
        :param args:
        :return: shape == (batch_size, enc_output_size+dec_output_size)
        """
        # shape == (batch, seq_len, enc_output_size)
        enc_outputs = enc_outputs.permute(1, 0, 2)

        # shape == (batch, enc_output_size)
        temp = self.scoring(dec_output)

        # shape == (batch, enc_output_size, 1)
        temp = temp.view(*temp.size(), 1)

        # shape == (batch, seq_len, 1)
        attn_weight = torch.bmm(enc_outputs, temp)

        attn_weight = self.softmax(attn_weight)

        # shape == (batch, seq_len, enc_output_size)
        context = torch.mul(enc_outputs, attn_weight)

        # shape == (batch, enc_output_size)
        context = torch.mean(context, dim=1)

        # shape == (batch, enc_output_size + dec_output_size)
        context = torch.cat((context, dec_output), dim=1)

        context = self.shrink_output(context)
        context = F.relu(context)

        return context, attn_weight.squeeze(dim=2)

