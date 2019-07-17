import torch
from torch import nn


class MyEmbedding(nn.Module):
    def __init__(self, weight):
        super(MyEmbedding, self).__init__()
        if weight is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(weight).float(), freeze=False)
        else:
            self.embedding = nn.Embedding(num_embeddings=20000, embedding_dim=512)

    def forward(self, *input):
        return self.embedding(*input)
