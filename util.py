import os

import pandas as pd

import logging
import logging.handlers as handlers
from time import strftime


def load_word(path):
    words = list()
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def load_word_re(path):
    words = load_word(path)
    return '(' + ')|('.join(words) + ')'


def load_pair(path):
    vocab = dict()
    for word1, word2 in pd.read_csv(path).values:
        if word1 not in vocab:
            vocab[word1] = word2
        if word2 not in vocab:
            vocab[word2] = word1
    return vocab


def load_triple(path):
    triples = list()
    for field1, field2, field3 in pd.read_csv(path).values:
        triples.append((field1, field2, field3))
    return triples


def get_logger(name, path_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file = strftime('%Y%m%d_%H%M%S') + '.log'
    path = os.path.join(path_dir, file)
    fh = handlers.RotatingFileHandler(path, 'a', 0, 1)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
