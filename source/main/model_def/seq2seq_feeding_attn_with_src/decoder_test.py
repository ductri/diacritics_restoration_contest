import unittest

import numpy as np
import torch

from model_def.seq2seq_feeding_attn_with_src.decoder import DecoderGreedyInfer, AttnRawDecoderWithSrc
from model_def.seq2seq_feeding_attn_with_src.encoder import create_my_embedding


class TestDecoder(unittest.TestCase):

    def test_attn_with_src_decoder(self):
        vocab_size = 5
        batch_size = 2
        seq_len = 10
        enc_output_size = 13
        enc_embedding_size = 11

        decoder = AttnRawDecoderWithSrc(embedding=create_my_embedding(np.random.rand(vocab_size, 10)),
                                        enc_output_size=enc_output_size,
                                        enc_embedding_size=enc_embedding_size)

        inputs_idx = torch.randint(vocab_size, size=(seq_len, batch_size))
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
        enc_inputs = torch.randn(seq_len, batch_size, enc_embedding_size)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)

        outputs, _, _ = decoder(inputs_idx, (h_n, c_n), enc_outputs, None, enc_inputs)

        self.assertEqual(outputs.size(), (seq_len, batch_size, vocab_size))

    def test_infer_attn_with_src_decoder(self):
        vocab_size = 5
        batch_size = 2
        enc_output_size = 13
        seq_len = 3
        enc_embedding_size = 11

        raw_decoder = AttnRawDecoderWithSrc(embedding=create_my_embedding(np.random.rand(vocab_size, 10)),
                                            enc_output_size=enc_output_size,
                                            enc_embedding_size=enc_embedding_size)
        decoder = DecoderGreedyInfer(core_decoder=raw_decoder, start_idx=0)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
        enc_inputs = torch.randn(seq_len, batch_size, enc_embedding_size)

        outputs = decoder((h_n, c_n), enc_outputs, enc_inputs)

        self.assertEqual(outputs.size(), (batch_size, seq_len))

    def test_attn_with_src_decoder_run_by_step(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10
        seq_len = 13
        enc_lstm_size = 512
        enc_output_size = enc_lstm_size*2
        enc_embedding_size = 11

        core_decoder = AttnRawDecoderWithSrc(embedding=create_my_embedding(np.random.rand(vocab_size, 10)),
                                             enc_output_size=enc_output_size,
                                             enc_embedding_size=enc_embedding_size)

        core_decoder.eval()
        with torch.no_grad():
            inputs_idx = torch.randint(vocab_size, size=(max_length, batch_size))
            h_n = torch.rand(3, batch_size, enc_lstm_size)
            c_n = torch.rand(3, batch_size, enc_lstm_size)
            enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
            enc_inputs = torch.randn(seq_len, batch_size, enc_embedding_size)

            outputs1, (h1, c1), _ = core_decoder(inputs_idx, (h_n, c_n), enc_outputs, None, enc_inputs)

            h2, c2 = h_n, c_n
            outputs2 = []
            _pred_dec_output = None
            for step in range(inputs_idx.size(0)):
                inputs_idx_step = inputs_idx[step: step+1]
                output_, (h2, c2), _pred_dec_output = core_decoder(inputs_idx_step, (h2, c2), enc_outputs,
                                                                   _pred_dec_output, enc_inputs[step:step+1])
                outputs2.append(output_)
            outputs2 = torch.cat(outputs2, dim=0)

            outputs1 = outputs1.numpy()
            outputs2 = outputs2.numpy()

            self.assertAlmostEqual(np.sum(np.abs((outputs1 - outputs2))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((h1.numpy() - h2.numpy()))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((c1.numpy() - c2.numpy()))), 0, places=5)


if __name__ == '__main__':
    unittest.main()


