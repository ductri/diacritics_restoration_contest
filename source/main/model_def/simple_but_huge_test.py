import unittest
import torch

from model_def.simple_but_huge import SimpleButHuge


class TestSeq2SeqAttnWithSrc(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward(self):
        src_vocab_size = 5
        batch_size = 2
        seq_len = 10

        inputs_idx = torch.randint(src_vocab_size, size=(batch_size, seq_len))
        model = SimpleButHuge(src_vocab_size=src_vocab_size, tgt_vocab_size=5)
        output = model(inputs_idx)
        self.assertEqual(output.size(), (batch_size, seq_len))

    def test_train(self):
        src_vocab_size = 7
        tgt_vocab_size = 9
        batch_size = 5
        seq_length = 100
        inputs_idx = torch.randint(low=0, high=src_vocab_size, size=(batch_size, seq_length)).to(self.device)
        target_idx = torch.randint(low=0, high=src_vocab_size, size=(batch_size, seq_length)).to(self.device)
        length = torch.mul(torch.ones(batch_size), 50).to(self.device)

        model = SimpleButHuge(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
        model.train()
        model.to(self.device)
        for step in range(50):
            model.train()
            loss = model.train_batch(inputs_idx, target_idx, length)
            print('Step: %s - Loss: %.4f' % (step, loss))

        model.eval()
        pred = model(inputs_idx)
        pred_np = pred.int().cpu().numpy()
        length_np = length.int().cpu().numpy()
        target_idx_np = target_idx.int().cpu().numpy()

        pred_list = []
        for i, l in enumerate(length_np):
            pred_list.extend(pred_np[i, :l])
        target_list = []
        for i, l in enumerate(length_np):
            target_list.extend(target_idx_np[i, :l])

        self.assertListEqual(pred_list, target_list)


if __name__ == '__main__':
    unittest.main()


