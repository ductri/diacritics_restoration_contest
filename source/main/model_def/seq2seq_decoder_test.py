import unittest
import torch

from model_def.seq2seq_decoder import RawDecoder, DecoderGreedyInfer


class TestDecoder(unittest.TestCase):

    def test_raw_decoder(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10

        decoder = RawDecoder(vocab_size=vocab_size)

        inputs_idx = torch.randint(vocab_size, size=(max_length, batch_size))
        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)

        outputs, _ = decoder(inputs_idx, (h_n, c_n))

        self.assertEqual(outputs.size(), (max_length, batch_size, vocab_size))

    def test_infer_decoder(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10

        raw_decoder = RawDecoder(vocab_size=vocab_size)
        decoder = DecoderGreedyInfer(core_decoder=raw_decoder, max_length=max_length, start_idx=0)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)

        outputs = decoder(h_n, c_n)
        self.assertEqual(outputs.size(), (batch_size, max_length))


if __name__ == '__main__':
    unittest.main()


