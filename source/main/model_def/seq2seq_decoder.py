import torch
from torch import nn


class DecoderGreedyInfer(nn.Module):

    def __init__(self, core_decoder, max_length, start_idx):
        """
        Output a fixed vector with size of `output_size` for each doc
        :param vocab_size:
        :param max_length: scala int
        :param start_idx: scala int
        """
        super(DecoderGreedyInfer, self).__init__()
        self.core_decoder = core_decoder
        self.register_buffer('start_idx', torch.Tensor([[start_idx]]))

        self.max_length = max_length

    def forward(self, enc_h_n, enc_c_n, *args):
        """

        :param enc_h_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_c_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param args:
        :return:
        """
        with torch.no_grad():
            batch_size = enc_c_n.size(1)
            decoder_output = torch.zeros(batch_size, self.max_length)

            current_word = self.start_idx.repeat(1, batch_size).long()
            h_n, c_n = (enc_h_n, enc_c_n)
            for step in range(self.max_length):
                # shape == (1, batch_size, vocab_size)
                output, (h_n, c_n) = self.core_decoder(current_word, (h_n, c_n))

                # import pdb; pdb.set_trace()

                # shape == (1, batch_size)
                current_word = torch.argmax(output, dim=2)

                decoder_output[:, step] = current_word[0]

            return decoder_output.int()


class RawDecoder(nn.Module):
    def __init__(self, vocab_size):
        """
        Common use for both Training and Inference
        :param vocab_size:
        """
        super(RawDecoder, self).__init__()
        self.embedding_size = 256
        self.lstm_size = 512
        self.lstm_num_layer = 3
        self.dropout_rate = 0.3

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.lstm_size, num_layers=self.lstm_num_layer,
                            bidirectional=False, dropout=self.dropout_rate)
        self.output_mapping = nn.Linear(self.lstm_size, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs_idx, h_n_c_n, *args):
        """

        :param inputs_idx: shape == (max_length, batch_size)
        :param h_n_c_n: tuple of (h_n, c_n) from LSTM. Each has size of (num_layers * num_directions, batch, hidden_size)
        :param args:
        :return: output shape == (max_length, batch, vocab_size)
        """

        # shape == (max_length, batch_size, hidden_size)
        embedding_input = self.embedding(inputs_idx)
        # output shape == (max_length, batch, num_directions * hidden_size)
        output, (h_n, c_n) = self.lstm(embedding_input, h_n_c_n)
        output = self.dropout(output)
        output = self.output_mapping(output)
        return output, (h_n, c_n)