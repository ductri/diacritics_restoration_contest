from nltk.tokenize import word_tokenize
import re


p1 = re.compile('[0-9]+')
p2 = re.compile('\n+')


def __replace_digit(doc):
    return p1.sub('__d__', doc)


def __replace_breakline(doc):
    return p2.sub(' ', doc)


def __split_by_dot_tokenize(doc):
    return doc.replace('.', ' . ')


def __split_by_comma_tokenize(doc):
    return doc.replace(',', ' , ')


def __tokenize_single_doc(doc):
    return ' '.join(word_tokenize(doc))


def __cut_off(doc, length):
    return ' '.join(doc.split()[:length])


def infer_preprocess(doc):
    doc = __replace_breakline(doc)
    doc = __split_by_dot_tokenize(doc)
    doc = __split_by_comma_tokenize(doc)
    doc = __tokenize_single_doc(doc)
    doc = doc.lower()
    doc = __replace_digit(doc)
    return doc


def train_preprocess(doc, max_length):
    doc = infer_preprocess(doc)
    doc = __cut_off(doc, max_length)

    return doc
