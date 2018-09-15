import os

import json
import pandas as pd

import re
import jieba
from jieba.posseg import cut as pos_cut

from util import load_word, load_word_re, load_pair


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


def add_entity(path_word, path_entity):
    words = load_word(path_word)
    with open(path_entity, 'a') as f:
        for word in words:
            f.write(word + '\n')


path_train_dir = 'data/train'
path_entity = 'dict/entity.txt'
path_person = 'dict/person.txt'
path_zh_en = 'dict/zh_en.csv'
path_stop_word = 'dict/stop_word.txt'
save_entity(path_train_dir, path_entity)
add_entity(path_person, path_entity)
jieba.load_userdict(path_entity)
zh_en = load_pair(path_zh_en)
stop_word_re = load_word_re(path_stop_word)


def label_word(text, entitys, label):
    triples = list()
    pairs = list(pos_cut(text))
    for pair in pairs:
        triple = dict()
        triple['word'] = list(pair)[0]
        triple['pos'] = list(pair)[1]
        if triple['word'] in entitys:
            triple['label'] = label
        else:
            triple['label'] = 'O'
        triples.append(triple)
    return triples


def merge(sents, text, triples):
    pre_triples = sents[text]
    for triple, pre_triple in zip(triples, pre_triples):
        if triple['label'] != 'O' and pre_triple['label'] == 'O':
            pre_triple['label'] = triple['label']


def prepare(path, path_dir):
    sents = dict()
    files = os.listdir(path_dir)
    for file in files:
        label = zh_en[os.path.splitext(file)[0]]
        for text, entity_str in pd.read_csv(os.path.join(path_dir, file)).values:
            text = re.sub(stop_word_re, '', text.strip())
            entitys = entity_str.strip().split()
            triples = label_word(text, entitys, label)
            if text in sents:
                merge(sents, text, triples)
            else:
                sents[text] = triples
    with open(path, 'w') as f:
        json.dump(sents, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    path_train = 'data/train.json'
    path_train_dir = 'data/train'
    path_test = 'data/test.json'
    path_test_dir = 'data/test'
    prepare(path_train, path_train_dir)
    prepare(path_test, path_test_dir)
