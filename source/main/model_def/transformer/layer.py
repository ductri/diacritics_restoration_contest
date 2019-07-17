from torch import nn
from torch.nn import functional as F


class Layer(nn.Module):
    def __init__(self, attention_multi_head):
        super(Layer, self).__init__()
        self.attention_multi_head = attention_multi_head
        __size = attention_multi_head.projection.out_features
        self.fc = nn.Linear(__size, __size)
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
        temp = F.layer_norm(temp, temp.size()[1:])
        return temp
