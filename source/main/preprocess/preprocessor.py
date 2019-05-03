import logging
import argparse
import pandas as pd

from nltk.tokenize import word_tokenize
import numpy as np

from preprocess import preprocess_supp


def __split_by_dot_tokenize(doc):
    return doc.replace('.', ' . ')


def __split_by_comma_tokenize(doc):
    return doc.replace(',', ' , ')


def __tokenize_single_doc(doc):
    return ' '.join(word_tokenize(doc))


def __cut_off(doc, length):
    return ' '.join(doc.split()[:length])


def preprocess_text(docs):
    docs = preprocess_supp.remove_html_tag(docs)
    docs = preprocess_supp.replace_url(docs, keep_url_host=False)
    docs = preprocess_supp.replace_phoneNB(docs)
    docs = preprocess_supp.remove_line_break(docs)
    docs = preprocess_supp.replace_all_number(docs)
    docs = preprocess_supp.replace_email(docs)
    docs = preprocess_supp.lowercase(docs)

    docs = [__split_by_dot_tokenize(doc) for doc in docs]
    docs = [__split_by_comma_tokenize(doc) for doc in docs]
    docs = [__tokenize_single_doc(doc) for doc in docs]

    return docs


def infer_preprocess(docs, max_length):
    docs = preprocess_text(docs)
    docs = [__cut_off(doc, max_length) for doc in docs]
    return docs


def train_preprocess(docs, max_length):
    docs = infer_preprocess(docs, max_length)

    logging.info('Pre-processing done')
    logging.info('-- Some samples: ')
    random_index = list(range(len(docs)))
    np.random.shuffle(random_index)
    for i in random_index[:15]:
        logging.info('-- -- %s', docs[i])

    return docs
