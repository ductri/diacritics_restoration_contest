import unittest
import torch

from utils.pytorch_utils import tile


class TestPytorchUtils(unittest.TestCase):

    def test_tile(self):
        x = torch.tensor([[1., 2], [3, 4], [5, 6]])
        y = tile(x, 3, dim=1)
        expect = torch.tensor([[1., 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [5, 5, 5, 6, 6, 6]])

        self.assertEqual(torch.norm(y-expect), 0)


if __name__ == '__main__':
    unittest.main()
