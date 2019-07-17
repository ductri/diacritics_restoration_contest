import math

import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    """
    https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term.float())
        pe[:, 1::2] = torch.cos(position.float() * div_term.float())
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


if __name__ == '__main__':
    pe = PositionalEncoding(512, 0, 5000)
    x = torch.zeros(10, 20, 512)
    y = pe(x)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(y[0])
    plt.show()
