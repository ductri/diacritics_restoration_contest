import torch
from torch import nn, optim

from utils import pytorch_utils


"""
DEPRECATED
diacritics_19-05-04_03_40_28
------------------ 	Evaluation	------------------
INFO:root:Step: 1320000
INFO:root:Number of batchs: 157
INFO:root:L_mean: 13362815.6725±114847554.0418(8.0722) 	  w_a: 0.9791±0.0053 	 s_a: 0.4656±0.0885 	 Duration: 6.1710 s/step
INFO:root:Current best score: 0.9800801243230254 recorded at step 470000

"""


class SimpleButHuge(nn.Module):

    def __init__(self, src_word_vocab_size, tgt_word_vocab_size):
        super(SimpleButHuge, self).__init__()

        self.input_embedding = nn.Embedding(num_embeddings=src_word_vocab_size, embedding_dim=1024)

        # Conv1d slides on last axis
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm1d(1024)

        self.dropout = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv1d(in_channels=1024 + 1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm1d(1024)

        self.conv5 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm1d(512)

        self.conv6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv6_bn = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(in_features=512, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=tgt_word_vocab_size)

        self.relu = nn.ReLU()
        self.xent = None
        self.optimizer = None

    def train(self, mode=True):
        if self.xent is None:
            self.xent = nn.CrossEntropyLoss(reduction='none')
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        super().train(mode)

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

    def inner_forward(self, input_word, *params):
        """

        :param input_word shape == (batch_size, max_word_len)
        :return: Tensor shape == (batch, max_len, vocab_size)
        """

        # shape == (batch_size, max_word_len, hidden_size)
        word_embed = self.input_embedding(input_word)

        # shape == (batch_size, hidden_size, max_word_len)
        word_embed_permuted = word_embed.permute(0, 2, 1)

        word_pipe = self.conv1(word_embed_permuted)
        word_pipe = self.relu(word_pipe)

        word_pipe = self.conv2(word_pipe)
        word_pipe = self.relu(word_pipe)

        word_pipe = self.conv3(word_pipe)
        word_pipe = self.relu(word_pipe)
        word_pipe = self.conv3_bn(word_pipe)

        # shape == (batch_size, max_word_len, hidden_size)
        word_pipe = word_pipe.permute(0, 2, 1)
        pipe = torch.cat((word_pipe, word_embed), dim=2)

        # shape == (batch_size, 1024+1024, max_word_len)
        pipe = pipe.permute(0, 2, 1)
        pipe = self.dropout(pipe)

        pipe = self.conv4(pipe)
        pipe = self.relu(pipe)
        pipe = self.conv4_bn(pipe)
        pipe = self.dropout(pipe)

        pipe = self.conv5(pipe)
        pipe = self.relu(pipe)
        pipe = self.conv5_bn(pipe)

        pipe = self.conv6(pipe)
        pipe = self.relu(pipe)
        pipe = self.conv6_bn(pipe)

        # shape == (batch_size, max_word_len, 1024)
        pipe = pipe.permute(0, 2, 1)

        # shape == (batch_size, max_word_len, no_classes)
        output = self.fc1(pipe)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output

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


