from torch import nn
import torch
from naruto_skills.new_voc import Voc

from model_def.transformer.layer import Layer
from model_def.transformer.attention_multiple_head import AttentionMultipleHead
from model_def.transformer.embedding import MyEmbedding
from model_def.transformer.position_embedding import PositionalEncoding
from model_def.transformer import constants
from data_for_train import dataset


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        size = 512
        dataset.bootstrap()
        self.embedding = nn.Embedding(num_embeddings=len(dataset.voc_src.index2word), embedding_dim=size)
        self.positional_embedding = PositionalEncoding(self.embedding.embedding_dim, 0.1, 5000)

        self.stacked_layers = nn.Sequential(*[Layer(AttentionMultipleHead(8, size)) for _ in range(6)])
        self.output_projection = nn.Linear(size, len(dataset.voc_tgt.index2word))

    def get_logits(self, word_input):
        """

        :param
        word_input: (batch, seq_len)
        :param
        input:
        :return:
        """
        # (batch, seq_len, size)
        temp = self.embedding(word_input)
        temp = self.positional_embedding(temp)
        temp = self.stacked_layers(temp)
        output = self.output_projection(temp)
        return output

    def forward(self, word_input, *input):
        """

        :param word_input:
        :param input:
        :return:
        """
        logits = self.get_logits(word_input)
        return torch.argmax(logits, dim=2)
