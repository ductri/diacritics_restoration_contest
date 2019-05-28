import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from naruto_skills.voc import Voc


class MyDataset(Dataset):
    def __init__(self, path_to_file, voc_src, voc_tgt, max_word_len, size=None):
        super(MyDataset, self).__init__()
        df = pd.read_csv(path_to_file)
        df.dropna(inplace=True)
        if size is None:
            start_idx = 0
        else:
            start_idx = df.shape[0] - size
        if start_idx < 0:
            raise Exception('Can not require %s data while there are only %s' % (size, df.shape[0]))

        self.src = list(df['src'].iloc[start_idx:])
        self.tgt = list(df['tgt'].iloc[start_idx:])
        self.voc_src = voc_src
        self.voc_tgt = voc_tgt
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        tgt = self.tgt[idx]

        word_input = self.voc_src.docs2idx([src], equal_length=self.max_word_len)[0]
        word_output = self.voc_tgt.docs2idx([tgt], equal_length=self.max_word_len)[0]
        word_length = len(src.split())

        return word_input, word_length, word_output


def bootstrap():
    global voc_src
    global voc
    voc_src = Voc.load(ROOT + 'main/vocab/output/src.json', name='src')
    voc_src.tokenize_func = str.split
    voc_src.space_char = ' '

    voc_tgt = Voc.load(ROOT + 'main/vocab/output/tgt.json', name='tgt')
    voc_tgt.tokenize_func = str.split
    voc_tgt.space_char = ' '

    logging.info('Src vocab contains %s tokens', len(voc_src.index2word))
    logging.info('Tgt vocab contains %s tokens', len(voc_tgt.index2word))


def create_data_loader(path_to_csv, batch_size, num_workers, size=None, shuffle=True):
    def collate_fn(list_data):
        """
        shape == (batch_size, col1, col2, ...)
        """
        data = zip(*list_data)
        data = [np.stack(col, axis=0) for col in data]
        data = [torch.from_numpy(col) for col in data]
        return data

    my_dataset = MyDataset(path_to_csv, voc_src, voc, MAX_LENGTH, size=size)
    logging.info('Data at %s contains %s samples', path_to_csv, len(my_dataset))
    dl = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dl


def get_dl_train(batch_size, size=None):
    return create_data_loader(ROOT + 'main/data_for_train/output/my_train.csv', batch_size, NUM_WORKERS, size=size)


def get_dl_test(batch_size):
    return create_data_loader(ROOT + 'main/data_for_train/output/my_test.csv', batch_size, NUM_WORKERS, shuffle=False)


def get_dl_eval(batch_size):
    return create_data_loader(ROOT + 'main/data_for_train/output/my_eval.csv', batch_size, NUM_WORKERS, shuffle=False)


voc_src = None
voc = None
MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'
