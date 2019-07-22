from torch import nn
from torch.nn import functional as F


class Layer(nn.Module):
    def __init__(self, attention_multi_head):
        super(Layer, self).__init__()
        self.attention_multi_head = attention_multi_head
        __size = attention_multi_head.projection.out_features
        self.norm1 = nn.LayerNorm(__size)
        self.norm2 = nn.LayerNorm(__size)
        self.fc1 = nn.Linear(__size, __size)
        self.fc2 = nn.Linear(__size, __size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, word_input, *input):
        """

        :param word_input: (batch, seq_len, emb_size)
        :param input:
        :return:
        """
        temp = self.attention_multi_head(word_input)
        temp = self.dropout(temp)
        temp = temp + word_input
        temp_store = self.norm1(temp)

        temp = self.fc1(temp_store)
        temp = F.relu(temp)
        temp = self.fc2(temp)
        temp = temp + temp_store
        temp = self.norm2(temp)
        return temp
