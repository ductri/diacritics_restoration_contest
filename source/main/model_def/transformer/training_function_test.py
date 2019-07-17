import unittest

import torch

from model_def.transformer.training_function import TrainingFunction
from model_def.transformer.model import Model


class TestModel(unittest.TestCase):
    def test_loss(self):
        model = Model()
        training_model = TrainingFunction(model)

        batch = 2
        seq_len = 10
        vocab_size = 100
        word_input = torch.randint(0, vocab_size, size=(batch, seq_len))
        target_input = torch.randint(0, vocab_size, size=(batch, seq_len))
        seq_len_input = torch.ones(size=(batch, )) * 5

        loss = training_model.get_loss([word_input, target_input, seq_len_input])
        self.assertAlmostEqual(loss.item(), 11., places=0)

    def test_train(self):
        model = Model()
        training_model = TrainingFunction(model)

        batch = 2
        seq_len = 10
        vocab_size = 100
        word_input = torch.randint(0, vocab_size, size=(batch, seq_len))
        target_input = torch.randint(0, vocab_size, size=(batch, seq_len))
        seq_len_input = torch.ones(size=(batch,)) * 5

        for i in range(50):
            l = training_model.train_batch([word_input, target_input, seq_len_input])
            if i % 10 == 0:
                print('%s - %s' % (i, l))

        self.assertAlmostEqual(l, 0, places=1)


if __name__ == '__main__':
    unittest.main()
