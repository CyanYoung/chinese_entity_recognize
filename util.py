import os

import pandas as pd


def load_word(path):
    words = list()
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def load_word_re(path):
    words = load_word(path)
    return '(' + ')|('.join(words) + ')'


def save_entity(path_train_dir, path_entity):
    entity_set = set()
    files = os.listdir(path_train_dir)
    for file in files:
        entity_strs = pd.read_csv(os.path.join(path_train_dir, file), usecols=[1]).values
        for entity_str in entity_strs:
            entitys = entity_str[0].strip().split()
            for entity in entitys:
                entity_set.add(entity)
    with open(path_entity, 'w') as f:
        for entity in entity_set:
            f.write(entity + '\n')


def load_pair(path):
    vocab = dict()
    for word1, word2 in pd.read_csv(path).values:
        if word1 not in vocab:
            vocab[word1] = word2
        if word2 not in vocab:
            vocab[word2] = word1
    return vocab
