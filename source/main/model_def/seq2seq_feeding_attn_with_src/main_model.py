import torch
from torch import nn, optim

from model_def.seq2seq_feeding_attn_with_src.decoder import DecoderGreedyInfer, AttnRawDecoderWithSrc
from model_def.seq2seq_feeding_attn_with_src.encoder import Encoder, FlattenHiddenLSTM, create_my_embedding
from utils import pytorch_utils


class MainModel(nn.Module):

    def __init__(self, enc_embedding_weight, dec_embedding_weight, start_idx, end_idx):
        super(MainModel, self).__init__()
        self.lr_rate = 1e-3
        self.max_length = 100
        self.__start_idx_int = start_idx

        self.encoder = Encoder(embedding=create_my_embedding(enc_embedding_weight), lstm_num_layer=1, lstm_size=1024)
        _enc_output_size = 2*self.encoder.lstm_size if self.encoder.is_bidirectional else self.encoder.lstm_size
        self.flatten_hidden_lstm = FlattenHiddenLSTM(lstm_num_layer=self.encoder.lstm_num_layer, is_bidirectional=self.encoder.is_bidirectional)

        self.core_decoder = AttnRawDecoderWithSrc(embedding=create_my_embedding(dec_embedding_weight),
                                                  enc_output_size=_enc_output_size, use_pred_prob=0.1,
                                                  lstm_size=self.encoder.lstm_size, lstm_num_layer=1,
                                                  enc_embedding_size=self.encoder.embedding_size
                                                  )
        self.greedy_infer = DecoderGreedyInfer(core_decoder=self.core_decoder,
                                               start_idx=start_idx)

        self.xent = None
        self.optimizer = None

        self.register_buffer('start_idx', torch.Tensor([[start_idx]]).long())
        # self.register_buffer('end_idx', torch.Tensor([[end_idx]]).long())

    def forward(self, word_input, *args):
        """

        :param word_input: shape == (batch_size, max_len)
        :param start_idx: int scala
        :param end_idx: int scala
        :param args:
        :return:
        """
        h_n, c_n, outputs = self.encoder(word_input)
        h_n, c_n = self.flatten_hidden_lstm(h_n, c_n)
        h_c = (h_n, c_n)

        enc_inputs = self.encoder.embedding(word_input)
        enc_inputs = enc_inputs.permute(1, 0, 2)

        output = self.greedy_infer(h_c, outputs, enc_inputs)
        return output

    def train(self, mode=True):
        if self.xent is None:
            self.xent = nn.CrossEntropyLoss(reduction='none')
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr_rate)
        super().train(mode)

    def get_loss(self, word_input, target, length):
        """

        :param word_input: shape == (batch_size, max_len)
        :param target: shape == (batch_size, max_len)
        :param length: shape == (batch_size)
        :return:
        """
        enc_h_n, enc_c_n, enc_outputs = self.encoder(word_input)
        enc_h_n, enc_c_n = self.flatten_hidden_lstm(enc_h_n, enc_c_n)
        batch_size = enc_h_n.size(1)

        init_words = self.start_idx.repeat(batch_size, 1)
        # shape == (batch_size, max_len)
        dec_input = torch.cat((init_words, target), dim=1)

        # shape == (max_len, batch_size)
        dec_input = dec_input.permute(1, 0)[:-1]

        # shape ==  (batch_size, max_len, _)
        enc_inputs = self.encoder.embedding(word_input)
        # shape ==  (max_len, batch_size, _)
        enc_inputs = enc_inputs.permute(1, 0, 2)
        # shape == (max_len+1, batch_size, tgt_vocab_size)
        predict, _, _ = self.core_decoder(dec_input, (enc_h_n, enc_c_n), enc_outputs, None, enc_inputs)

        # shape == (batch_size, tgt_vocab_size, max_len+1)
        predict = predict.permute(1, 2, 0)

        # end_words = self.end_idx.repeat(batch_size, 1)
        # dec_target = torch.cat((target, end_words), dim=1)
        dec_target = target

        loss = self.xent(predict, dec_target)
        loss_mask = pytorch_utils.length_to_mask(length, max_len=self.max_length, dtype=torch.float)
        loss = torch.mul(loss, loss_mask)
        loss = torch.div(loss.sum(dim=1), length.float())
        loss = loss.mean(dim=0)
        return loss

    def train_batch(self, word_input, target, length):
        """

        :param word_input: shape == (batch_size, max_len)
        :param target: shape == (batch_size, max_len)
        :return:
        """
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(word_input, target, length)
        loss.backward()
        self.optimizer.step()

        return loss.item()
