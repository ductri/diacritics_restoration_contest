import torch
from torch import nn, optim
from torch.nn import functional as F

from utils import pytorch_utils


class Block(nn.Module):
    def __init__(self, channel_size, hidden_size):
        super(Block, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=channel_size, out_channels=hidden_size, kernel_size=3, padding=1)]
                      + [nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1) for _ in range(3)])
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs, *args):
        """

        :param inputs: (batch, _, len)
        :param args:
        :return:
        """
        temp = inputs
        for conv in self.convs:
            temp = conv(temp)
            temp = F.relu(temp)
        temp = self.dropout(temp)
        temp = self.batch_norm(temp)
        temp = torch.cat((temp, inputs), dim=1)
        return temp


class Simple(nn.Module):

    def __init__(self, enc_embedding, dec_embedding):
        super(Simple, self).__init__()
        enc_embedding_size = enc_embedding.weight.size(1)

        self.enc_embedding = enc_embedding

        self.block_1 = Block(enc_embedding_size, 512)
        self.block_2 = Block(enc_embedding_size + 512, 512)
        self.block_3 = Block(enc_embedding_size + 512*2, 512)

        self.fc1 = nn.Linear(enc_embedding_size + 512*3, 2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=dec_embedding.weight.size(0))

        self.xent = None
        self.optimizer = None

    def inner_forward(self, input_word, *params):
        """

        :param input_word shape == (batch_size, max_word_len)
        :return: Tensor shape == (batch, max_len, vocab_size)
        """

        # shape == (batch_size, max_word_len, hidden_size)
        word_embed = self.enc_embedding(input_word)

        # shape == (batch_size, hidden_size, max_word_len)
        word_embed_permuted = word_embed.permute(0, 2, 1)
        temp = self.block_1(word_embed_permuted)
        temp = self.block_2(temp)
        temp = self.block_3(temp)

        temp = temp.permute(0, 2, 1)

        temp = self.fc1(temp)
        temp = F.relu(temp)

        # shape == (batch_size, max_word_len, no_classes)
        output = self.fc2(temp)

        return output

    def get_loss(self, word_input, target, length):
        """

        :param word_input: shape == (batch_size, max_len)
        :param target: shape == (batch_size, max_len)
        :param length: shape == (batch_size)
        :return:
        """
        max_length = word_input.size(1)

        # shape == (batch, max_len, vocab_size)
        predict = self.inner_forward(word_input)
        # shape == (batch, vocab_size, max_len)
        predict = predict.permute(0, 2, 1)

        loss = self.xent(predict, target)
        loss_mask = pytorch_utils.length_to_mask(length, max_len=max_length, dtype=torch.float)
        loss = torch.mul(loss, loss_mask)
        loss = torch.div(loss.sum(dim=1), length.float())
        loss = loss.mean(dim=0)
        return loss

    def forward(self, input_word, *params):
        """

        :param input_word: shape == (batch, max_len)
        :param params:
        :return: Tensor shape == (batch, max_len, vocab_size)
        """
        logits = self.inner_forward(input_word)
        return torch.argmax(logits, dim=2)

    def train_batch(self, word_input, target, length):
        """

        :param word_input: shape == (batch_size, max_len)
        :param target: shape == (batch_size, max_len)
        :return:
        """
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(word_input, target, length)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, mode=True):
        if self.xent is None:
            self.xent = nn.CrossEntropyLoss(reduction='none')
        if self.optimizer is None:
            # self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
            self.optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        super().train(mode)