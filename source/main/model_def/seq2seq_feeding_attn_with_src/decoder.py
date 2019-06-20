import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from model_def.seq2seq_feeding_attn_with_src.attention import Attention


class DecoderGreedyInfer(nn.Module):

    def __init__(self, core_decoder, start_idx):
        """
        Output a fixed vector with size of `output_size` for each doc
        :param vocab_size:
        :param max_length: scala int
        :param start_idx: scala int
        """
        super(DecoderGreedyInfer, self).__init__()
        self.core_decoder = core_decoder
        self.register_buffer('start_idx', torch.tensor([[start_idx]]))

    def forward(self, h_c, enc_outputs, enc_inputs, *args):
        """

        :param enc_h_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_c_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_inputs: shape == (seq_len, batch_size, _)
        :param args:
        :return:
        """
        with torch.no_grad():
            batch_size = h_c[0].size(1)
            seq_len = enc_inputs.size(0)
            decoder_output = torch.zeros(batch_size, seq_len)

            current_word = self.start_idx.repeat(1, batch_size).long()
            dec_pre_output = None
            for step in range(seq_len):
                # shape == (1, batch_size, vocab_size)
                output, h_c, dec_pre_output = self.core_decoder(current_word, h_c, enc_outputs, dec_pre_output,
                                                                enc_inputs[step:step+1], *args)

                # shape == (1, batch_size)
                current_word = torch.argmax(output, dim=2)

                decoder_output[:, step] = current_word[0]

            return decoder_output.int()


class AttnRawDecoderWithSrc(nn.Module):
    def __init__(self, embedding, enc_output_size, enc_embedding_size, lstm_size=512, lstm_num_layer=3,
                 use_pred_prob=0.):
        """
        Common use for both Training and Inference
        :param vocab_size:
        """
        super(AttnRawDecoderWithSrc, self).__init__()

        self.lstm_size = lstm_size
        self.lstm_num_layer = lstm_num_layer
        self.dropout_rate = 0.3
        self.half_window_size = 50
        self.use_pred_prob = use_pred_prob

        self.embedding = embedding
        self.embedding_size = embedding.weight.size(1)
        attn_output_size = 512
        self.lstm = nn.LSTM(input_size=self.embedding_size+enc_embedding_size+attn_output_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_num_layer,
                            bidirectional=False, dropout=self.dropout_rate)

        self.attention = Attention(enc_output_size=enc_output_size, dec_output_size=self.lstm_size,
                                   output_size=attn_output_size)

        self.output_mapping = nn.Linear(attn_output_size, self.embedding.weight.size(0))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.register_buffer('dec_pred_output_default', torch.zeros(attn_output_size))

    def forward(self, inputs_idx, h_c, enc_outputs, dec_pre_output, enc_inputs, *args):
        """
        Implemented by running step by step
        :param inputs_idx: shape == (seq_len, batch_size)
        :param h_c: tuple of (h_n, c_n) from LSTM. Each has size of (num_layers * num_directions, batch, hidden_size)
        :param enc_outputs: shape == (seq_len, batch, hidden_size)
        :param dec_pre_output: shape == (batch, _) or None
        :param enc_inputs shape == (seq_len, batch_size, encoder_embedding_size)
        :param args:
        :return: output shape == (seq_len, batch, vocab_size)
        """
        batch_size = inputs_idx.size(1)

        def decode_one_step(inputs, h_c, _dec_pre_output):
            """

            :param inputs: shape == (batch, dec_embedding_size)
            :param h_c:
            :param _dec_pre_output: shape == (batch, _)
            :return: Tensor shape == (batch, _)
            """
            if _dec_pre_output is None:
                _dec_pre_output = self.dec_pred_output_default.repeat(batch_size, 1)

            assert _dec_pre_output.size(0) == batch_size
            inputs = torch.cat((inputs, _dec_pre_output), dim=1)
            inputs = F.relu(inputs)
            inputs = torch.unsqueeze(inputs, dim=0)
            _output, _h_c = self.lstm(inputs, h_c)

            # shape == (batch, _)
            _output, _ = self.attention(enc_outputs, _output[0])

            return _output, _h_c

        __half_window_size = self.half_window_size

        # shape == (seq_len, batch_size, hidden_size)
        embedding_input = self.embedding(inputs_idx)
        embedding_input = self.dropout(embedding_input)
        outputs = []
        for step in range(inputs_idx.size(0)):
            rand = np.random.rand()
            if step != 0:
                if rand < self.use_pred_prob:
                    step_output = self.output_mapping(dec_pre_output)
                    assert step_output.dim() == 2, step_output.dim()
                    assert step_output.size(0) == batch_size, step_output.size(0)
                    next_input_idx = torch.argmax(step_output, dim=1)
                    assert next_input_idx.dim() == 1, next_input_idx.dim()
                    inputs = self.embedding(next_input_idx)
                else:
                    inputs = embedding_input[step]
            else:
                inputs = embedding_input[step]
            inputs = torch.cat((inputs, enc_inputs[step]), dim=1)
            dec_pre_output, h_c = decode_one_step(inputs, h_c, dec_pre_output)
            outputs.append(self.output_mapping(torch.unsqueeze(dec_pre_output, dim=0)))

        # output shape == (seq_len, batch, _)
        outputs = torch.cat(tuple(outputs), dim=0)

        return outputs, h_c, dec_pre_output

