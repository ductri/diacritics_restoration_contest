import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self):
        super(PositionEmbedding, self).__init__()
        self.const = 100

    def forward(self, batch_size, seq_len, word_emb_size):
        emb = torch.arange(0, seq_len*word_emb_size).view(seq_len, word_emb_size).float()

        indices = torch.arange(0, word_emb_size, 2)
        dim_indices = emb[:, indices].long() % word_emb_size
        word_indices = emb[:, indices].long() / word_emb_size
        pow_factor = dim_indices.float()/word_emb_size
        sin_factor = word_indices.float() / torch.pow(self.const, pow_factor)
        emb[:, indices] = torch.sin(sin_factor)

        indices = torch.arange(1, word_emb_size, 2)
        dim_indices = emb[:, indices].long() % word_emb_size
        word_indices = emb[:, indices].long() / word_emb_size
        pow_factor = dim_indices.float() / word_emb_size
        sin_factor = word_indices.float() / torch.pow(self.const, pow_factor)
        emb[:, indices] = torch.cos(sin_factor)

        return emb.view(1, seq_len, word_emb_size).repeat(batch_size, 1, 1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure()
    x = PositionEmbedding()
    m = x(10, 20, 512)
    plt.imshow(m[0])
    plt.show()
