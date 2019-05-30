import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from naruto_skills.voc import Voc

voc_src = None
MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'


class Docs(Dataset):
    def __init__(self, path_to_file, column_name, voc, size=None):
        super(Docs, self).__init__()
        df = pd.read_csv(path_to_file)
        data = df[column_name]
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)

        size = size or data.shape[0]
        data = data.sample(size)

        self.tgt = list(data)
        self.voc = voc

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):
        tgt = self.tgt[idx]

        word_output = self.voc.docs2idx([tgt], equal_length=MAX_LENGTH)[0]
        word_length = len(tgt.split())

        return word_output, word_output, word_length


def bootstrap():
    global voc_src
    path = ROOT + 'main/vocab/output/tgt.pkl'
    voc = Voc.load(path)
    logging.info('Vocab from file %s contains %s tokens', path, len(voc.index2word))


def create_data_loader(path_to_csv, column_name, batch_size, num_workers, size=None, shuffle=True):
    def collate_fn(list_data):
        """
        shape == (batch_size, col1, col2, ...)
        """
        data = zip(*list_data)
        data = [np.stack(col, axis=0) for col in data]
        data = [torch.from_numpy(col) for col in data]
        return data

    my_dataset = Docs(path_to_csv, column_name, voc_src, size=size)
    logging.info('Data at %s contains %s samples', path_to_csv, len(my_dataset))
    dl = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dl

