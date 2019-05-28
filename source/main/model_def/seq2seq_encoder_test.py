import unittest
import torch

from model_def.seq2seq_encoder import Encoder


class TestEncoder(unittest.TestCase):

    def test_encoder(self):

        docs = torch.Tensor([[1, 2, 3, 4], [1, 2, 2, 4]]).long()
        batch_size = docs.size(0)

        encoder = Encoder(vocab_size=5)
        h_n, c_n, _ = encoder(docs)

        self.assertEqual(h_n.shape, (6, batch_size, 512))
        self.assertEqual(c_n.shape, (6, batch_size, 512))


if __name__ == '__main__':
    unittest.main()
