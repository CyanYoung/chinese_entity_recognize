import os

import json
import pandas as pd

import jieba
from jieba.posseg import cut as pos_cut

from random import shuffle, choice

from util import load_word, load_pair, load_poly, flat_read


path_zh_en = 'dict/zh_en.csv'
path_pre_name = 'dict/pre_name.txt'
path_end_name = 'dict/end_name.txt'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
zh_en = load_pair(path_zh_en)
pre_names = load_word(path_pre_name)
end_names = load_word(path_end_name)
homo_dict = load_poly(path_homo)
syno_dict = load_poly(path_syno)


def merge(path_slot_dir, path_extra, path_cut_word):
    entitys = list()
    files = os.listdir(path_slot_dir)
    for file in files:
        words = load_word(os.path.join(path_slot_dir, file))
        entitys.extend(words)
    entity_strs = flat_read(path_extra, 'entity')
    for entity_str in entity_strs:
        words = entity_str.split()
        entitys.extend(words)
    entity_set = set(entitys)
    with open(path_cut_word, 'w') as f:
        for entity in entity_set:
            f.write(entity + '\n')


def save(path, sents):
    with open(path, 'w') as f:
        json.dump(sents, f, ensure_ascii=False, indent=4)


def make_name(pre_names, end_names, num):
    names = list()
    for i in range(num):
        pre_name = choice(pre_names)
        end_name = choice(end_names)
        names.append(pre_name + end_name)
    return names


def dict2list(sents):
    word_mat, label_mat = list(), list()
    for pairs in sents.values():
        words, labels = list(), list()
        for pair in pairs:
            words.append(pair['word'])
            labels.append(pair['label'])
        word_mat.append(words)
        label_mat.append(labels)
    return word_mat, label_mat


def list2dict(word_mat, label_mat):
    sents = dict()
    for words, labels in zip(word_mat, label_mat):
        text = ''.join(words)
        pairs = list()
        for word, label in zip(words, labels):
            pair = dict()
            pair['word'] = word
            pair['label'] = label
            pairs.append(pair)
        sents[text] = pairs
    return sents


def select(part):
    if part[0] == '[' and part[-1] == ']':
        word = part[1:-1]
        cands = set()
        cands.add(word)
        if word in syno_dict:
            cands.update(syno_dict[word])
        if word in homo_dict:
            cands.update(homo_dict[word])
        return choice(list(cands))
    elif part[0] == '(' and part[-1] == ')':
        word = part[1:-1]
        return choice([word, ''])
    else:
        return part


def generate(temps, slots, num):
    word_mat, label_mat = list(), list()
    for i in range(num):
        parts = choice(temps)
        words, labels = list(), list()
        for part in parts:
            if part in slots:
                entity = choice(slots[part])
                words.append(entity)
                labels.append(part)
            else:
                word = select(part)
                if word:
                    words.append(word)
                    labels.append('O')
        word_mat.append(words)
        label_mat.append(labels)
    return word_mat, label_mat


def sync_shuffle(list1, list2):
    pairs = list(zip(list1, list2))
    shuffle(pairs)
    return zip(*pairs)


def label_sent(path):
    sents = dict()
    for text, entity_str, label_str in pd.read_csv(path).values:
        pairs = list(pos_cut(text))
        entitys, labels = entity_str.split(), label_str.split()
        if len(entitys) != len(labels):
            print('skip: %s', text)
            continue
        triples = list()
        for word, pos in pairs:
            triple = dict()
            triple['word'] = word
            triple['pos'] = pos
            if word in entitys:
                ind = entitys.index(word)
                triple['label'] = labels[ind]
            else:
                triple['label'] = 'O'
            triples.append(triple)
        sents[text] = triples
    return sents


def expand(sents, gen_word_mat, gen_label_mat):
    word_mat, label_mat = dict2list(sents)
    word_mat.extend(gen_word_mat)
    label_mat.extend(gen_label_mat)
    word_mat, label_mat = sync_shuffle(word_mat, label_mat)
    bound = int(len(word_mat) * 0.9)
    train_sents = list2dict(word_mat[:bound], label_mat[:bound])
    test_sents = list2dict(word_mat[bound:], label_mat[bound:])
    return train_sents, test_sents


def prepare(paths):
    merge(paths['slot_dir'], paths['extra'], paths['cut_word'])
    jieba.load_userdict(paths['cut_word'])
    temps = list()
    with open(paths['temp'], 'r') as f:
        for line in f:
            parts = line.strip().split()
            temps.append(parts)
    slots = dict()
    files = os.listdir(paths['slot_dir'])
    for file in files:
        label = zh_en[os.path.splitext(file)[0]]
        slots[label] = list()
        with open(os.path.join(paths['slot_dir'], file), 'r') as f:
            for line in f:
                slots[label].append(line.strip())
    names = make_name(pre_names, end_names, num=1000)
    slots['PER'].extend(names)
    gen_word_mat, gen_label_mat = generate(temps, slots, num=5000)
    sents = label_sent(paths['extra'])
    train_sents, test_sents = expand(sents, gen_word_mat, gen_label_mat)
    save(paths['train'], train_sents)
    save(paths['test'], test_sents)


if __name__ == '__main__':
    paths = dict()
    paths['train'] = 'data/train.json'
    paths['test'] = 'data/test.json'
    paths['temp'] = 'data/template.txt'
    paths['slot_dir'] = 'data/slot'
    paths['extra'] = 'data/extra.csv'
    paths['cut_word'] = 'dict/cut_word.txt'
    prepare(paths)
