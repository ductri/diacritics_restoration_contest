from torch import nn, optim
import torch

from utils import pytorch_utils


class TrainingFunction(nn.Module):
    def __init__(self, model):
        super(TrainingFunction, self).__init__()
        self.model = model
        self.xent = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(model.parameters(), lr=3e-4)

    def forward(self, *input):
        return self.model(*input)

    def get_loss(self, inputs):
        src, tgt, seq_len = inputs
        max_length = src.size(1)
        assert max_length >= torch.max(seq_len).int().item()

        # shape == (batch, max_len, vocab_size)
        predict = self.model.get_logits(src)
        # shape == (batch, vocab_size, max_len)
        predict = predict.permute(0, 2, 1)
        loss = self.xent(predict, tgt)
        loss_mask = pytorch_utils.length_to_mask(seq_len, max_len=max_length, dtype=torch.float)
        loss = torch.mul(loss, loss_mask)
        loss = torch.div(loss.sum(dim=1), seq_len.float())
        loss = loss.mean(dim=0)
        return loss

    def train_batch(self, inputs):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(inputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()
