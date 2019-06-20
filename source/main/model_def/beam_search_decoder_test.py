import unittest

import torch

from model_def.seq2seq_feeding_attn_with_src.decoder import AttnRawDecoderWithSrc
from model_def.beam_search_decoder import BeamSearchWithSrcInfer


class TestDecoder(unittest.TestCase):

    def test_beam_infer(self):
        vocab_size = 5
        enc_output_size = 7
        enc_embedding_size = 5
        batch_size = 1
        seq_len = 4
        beam_width = 3
        device = torch.device('cpu')

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
        enc_inputs = torch.randn(seq_len, batch_size, enc_embedding_size)

        core_decoder = AttnRawDecoderWithSrc(vocab_size=vocab_size, enc_output_size=enc_output_size,
                                                  enc_embedding_size=enc_embedding_size)
        core_decoder.eval()
        infer_module = BeamSearchWithSrcInfer(core_decoder=core_decoder, start_idx=0, beam_width=beam_width,
                                              device=device)

        output = infer_module(h_n, c_n, enc_outputs, enc_inputs)
        self.assertListEqual(list(output.size()), list([beam_width, batch_size, seq_len]))

    def test_beam_decode(self):
        vocab_size = 5
        enc_output_size = 7
        enc_embedding_size = 5
        batch_size = 1
        seq_len = 4
        beam_width = 3
        device = torch.device('cpu')

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
        enc_inputs = torch.randn(seq_len, batch_size, enc_embedding_size)

        core_decoder = AttnRawDecoderWithSrc(vocab_size=vocab_size, enc_output_size=enc_output_size,
                                                  enc_embedding_size=enc_embedding_size)
        core_decoder.eval()
        infer_module = BeamSearchWithSrcInfer(core_decoder=core_decoder, start_idx=0, beam_width=beam_width,
                                              device=device)

        output = infer_module(h_n, c_n, enc_outputs, enc_inputs)
        output = infer_module.decode_output_matrix(output)
        self.assertListEqual(list(output.size()), list([batch_size, seq_len]))


if __name__ == '__main__':
    unittest.main()


