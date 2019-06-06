from torch import nn
import torch
import numpy as np

from utils import pytorch_utils


class BeamSearchWithSrcInfer(nn.Module):

    def __init__(self, core_decoder, start_idx, beam_width, device):
        """

        :param core_decoder:
        :param start_idx:
        :param beam_width:
        """
        super(BeamSearchWithSrcInfer, self).__init__()
        self.core_decoder = core_decoder
        self.register_buffer('start_idx', torch.Tensor([start_idx]))
        self.beam_width = beam_width
        self.softmax2 = nn.Softmax(dim=2)
        self.softmax1 = nn.Softmax(dim=1)
        self.device = device

    def forward(self, enc_h_n, enc_c_n, enc_outputs, enc_inputs, *args):
        """

        :param enc_h_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_c_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_outputs: shape = (seq_len, batch, _)
        :param enc_inputs: shape = (seq_len, batch, enc_embedding_size)
        :param args:
        :return: shape == (beam_width, batch, seq_len)
        """
        with torch.no_grad():
            batch_size = enc_c_n.size(1)
            seq_len = enc_inputs.size(0)
            vocab_size = self.core_decoder.output_mapping.out_features

            space_search = torch.ones(batch_size, vocab_size * self.beam_width).to(self.device)
            decoder_outputs = torch.zeros(batch_size, seq_len, self.beam_width).to(self.device)

            top_k_v, top_k_i, (enc_h_n, enc_c_n) = self.init_beam_search(enc_h_n, enc_c_n, enc_outputs, enc_inputs)

            current_scores = top_k_v
            decoder_outputs[:, 0, :] = top_k_i

            # shape == (batch_size, beam_width)
            current_word = top_k_i
            # shape == (batch_size, beam_width, 1)
            current_word = current_word.view(*current_word.size(), 1)

            beam_h_n = [enc_h_n.clone() for _ in range(self.beam_width)]
            beam_c_n = [enc_c_n.clone() for _ in range(self.beam_width)]
            for step in range(1, seq_len):
                for beam_idx in range(self.beam_width):
                    # shape == (1, batch_size)
                    beam_current_word = current_word[:, beam_idx, :].permute(1, 0)
                    assert beam_current_word.size(0) == 1
                    assert beam_current_word.size(1) == batch_size
                    # shape == (1, batch_size, vocab_size)
                    __temp = self.core_decoder(beam_current_word, (beam_h_n[beam_idx], beam_c_n[beam_idx]), enc_outputs,
                                               enc_inputs[step: step + 1], step)
                    output, (beam_h_n[beam_idx], beam_c_n[beam_idx]) = __temp
                    output = self.softmax2(output)
                    output = output.squeeze(dim=0)

                    space_search[:, beam_idx*vocab_size:(beam_idx+1)*vocab_size] = output
                # Update space search
                space_search = torch.mul(space_search, pytorch_utils.tile(current_scores, vocab_size, dim=1))

                # Explore and get top #beam_width out of #beam_width*vocab_size candidates
                # shape == (batch_size, beam_width)
                top_k_v, top_k_i = torch.topk(space_search, k=self.beam_width, dim=1)

                # current_scores = self.softmax1(top_k_v)
                # The value of top_k_i contains 2 info: the current indices and the link to previous indices
                decoder_outputs[:, step, :] = top_k_i
                current_word = torch.fmod(top_k_i, vocab_size).view(*top_k_i.size(), 1)

            decoder_outputs = decoder_outputs.permute(2, 0, 1)
            return decoder_outputs.int()

    def init_beam_search(self, enc_h_n, enc_c_n, enc_outputs, enc_inputs):
        """

        :param enc_h_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_c_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_outputs: shape = (seq_len, batch, _)
        :param enc_inputs: shape = (seq_len, batch, enc_embedding_size)
        :param args:
        :return: (top_k_v, top_k_i)
        top_k_v: top probabilities of step 0, shape == (batch, beam_width)
        top_k_i: top probabilities of step 0, shape == (batch, beam_width)
        """
        with torch.no_grad():
            batch_size = enc_c_n.size(1)

            # shape == (1, batch_size)
            current_word = self.start_idx.repeat(1, batch_size).long()

            __temp = self.core_decoder(current_word, (enc_h_n, enc_c_n), enc_outputs, enc_inputs[0: 1], 0)

            # output shape == (1, batch_size, vocab_size)
            output, (h_n, c_n) = __temp

            output_prob = self.softmax2(output)
            output_prob = output_prob.squeeze(dim=0)
            # shape == (batch_size, beam_width)
            top_k_v, top_k_i = torch.topk(output_prob, k=self.beam_width, dim=1)
            # top_k_v = self.softmax1(top_k_v)
            return top_k_v, top_k_i, (h_n, c_n)

    def decode_output_matrix(self, decoder_outputs):
        """

        :param decoder_outputs: shape == (beam_width, batch, seq_len)
        :return: Tensor shape == (beam_width, batch, seq_len)
        """
        batch_size = decoder_outputs.size(1)
        seq_len = decoder_outputs.size(2)
        vocab_size = self.core_decoder.output_mapping.out_features
        batch_indices = torch.from_numpy(np.arange(batch_size)).long()

        output = torch.zeros(batch_size, seq_len).long().to(self.device) - 1
        beam_indices = torch.zeros(batch_size).long()

        for step in range(seq_len-1, -1, -1):
            # shape == (batch, )
            output[:, step] = torch.fmod(decoder_outputs[beam_indices, batch_indices, step], vocab_size)
            beam_indices = decoder_outputs[beam_indices, batch_indices, step].int() / int(vocab_size)
            beam_indices = beam_indices.long()
        return output
