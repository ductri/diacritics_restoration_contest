import unittest

import numpy as np
import torch

from model_def.seq2seq_decoder import RawDecoder, DecoderGreedyInfer, AttnRawDecoder


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

    def test_raw_decoder_run_by_step(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10

        decoder = RawDecoder(vocab_size=vocab_size)
        decoder.eval()
        with torch.no_grad():
            inputs_idx = torch.randint(vocab_size, size=(max_length, batch_size))
            h_n = torch.rand(3, batch_size, 512)
            c_n = torch.rand(3, batch_size, 512)

            outputs1, (h1, c1) = decoder(inputs_idx, (h_n, c_n))

            h2, c2 = h_n, c_n
            outputs2 = []
            for step in range(inputs_idx.size(0)):
                inputs_idx_step = inputs_idx[step: step+1]
                output_, (h2, c2) = decoder(inputs_idx_step, (h2, c2))
                outputs2.append(output_)
            outputs2 = torch.cat(outputs2, dim=0)

            outputs1 = outputs1.numpy()
            outputs2 = outputs2.numpy()

            self.assertAlmostEqual(np.sum(np.abs((outputs1 - outputs2))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((h1.numpy() - h2.numpy()))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((c1.numpy() - c2.numpy()))), 0, places=5)

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

    def test_attn_decoder(self):
        vocab_size = 5
        batch_size = 2
        seq_len = 10
        enc_output_size = 13

        decoder = AttnRawDecoder(vocab_size=vocab_size, enc_output_size=enc_output_size)

        inputs_idx = torch.randint(vocab_size, size=(seq_len, batch_size))
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)

        outputs, _ = decoder(inputs_idx, (h_n, c_n), enc_outputs)

        self.assertEqual(outputs.size(), (seq_len, batch_size, vocab_size))

    def test_infer_attn_decoder(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10
        enc_output_size = 13
        seq_len = 3

        raw_decoder = AttnRawDecoder(vocab_size=vocab_size, enc_output_size=enc_output_size)
        decoder = DecoderGreedyInfer(core_decoder=raw_decoder, max_length=max_length, start_idx=0)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)

        outputs = decoder(h_n, c_n, enc_outputs)
        self.assertEqual(outputs.size(), (batch_size, max_length))


if __name__ == '__main__':
    unittest.main()


