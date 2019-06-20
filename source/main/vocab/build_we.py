import logging
import pandas as pd

from naruto_skills.word_embedding import WordEmbedding
from naruto_skills.voc import Voc


def build_vocab(df, name, min_freq):
    src = list(df[name])
    we = WordEmbedding(preprocessed_docs=src, min_freq=min_freq, embedding_size=512, worker=8)
    we.add_vocab(['__s__', '__e__', '__p__', '__o__'])
    we_vocabs = we.get_vocab()
    we.save_it(path_for_weight='/source/main/vocab/output/we/%s_weight.npy' % name,
               path_for_vocab='/source/main/vocab/output/we/%s.txt' % name)

    voc = Voc(tokenize_func=Voc.WORD_LV_TOK_FUNC, space_char=Voc.WORD_LV_SPACE_CHR)
    voc.build_from_tokens(we_vocabs, padding_idx=len(we_vocabs) - 2, oov_idx=len(we_vocabs) - 1)
    voc.dump('/source/main/vocab/output/we/%s.pkl' % name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv('/source/main/data_for_train/output/my_train.csv')
    build_vocab(df, 'src', 10)
    build_vocab(df, 'tgt', 5)
