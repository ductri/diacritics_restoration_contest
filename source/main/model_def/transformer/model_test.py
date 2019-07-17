import unittest

import torch

from model_def.transformer.model import Model


class TestModel(unittest.TestCase):
    def test_forward(self):
        model = Model()

        batch = 2
        seq_len = 10
        word_input = torch.randint(0, 100, size=(batch, seq_len))
        output = model.get_logits(word_input)
        self.assertListEqual(list(output.size()), [2, 10, 54806])


if __name__ == '__main__':
    unittest.main()
