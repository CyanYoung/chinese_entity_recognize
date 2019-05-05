import json
import pickle as pk

import re

from util import load_word_re


path_pre_name = 'dict/pre_name.txt'
path_digit = 'dict/digit.txt'
pre_name_re = load_word_re(path_pre_name)
digit_re = load_word_re(path_digit)

path_label_ind = 'feat/label_ind.pkl'


def include_pre_name(word):
    if re.findall(pre_name_re, word):
        return True
    else:
        return False


def include_digit(word):
    if re.findall(digit_re, word):
        return True
    else:
        return False


def sent2feat(triples):
    sent_feat = list()
    for i in range(len(triples)):
        triple = triples[i]
        word_feat = {
            'word': triple['word'],
            'len': len(triple['word']),
            'pos': triple['pos'],
            'pre_name': include_pre_name(triple['word']),
            'digit': include_digit(triple['word'])
        }
        if i > 0:
            last_triple = triples[i - 1]
            word_feat.update({
                'last_word': last_triple['word'],
                'last_len': len(last_triple['word']),
                'last_pos': last_triple['pos'],
                'last_pre_name': include_pre_name(last_triple['word']),
                'last_digit': include_digit(last_triple['word'])
            })
        else:
            word_feat['bos'] = True
        if i < len(triples) - 1:
            next_triple = triples[i + 1]
            word_feat.update({
                'next_word': next_triple['word'],
                'next_len': len(next_triple['word']),
                'next_pos': next_triple['pos'],
                'next_pre_name': include_pre_name(next_triple['word']),
                'next_digit': include_digit(next_triple['word'])
            })
        else:
            word_feat['eos'] = True
        sent_feat.append(word_feat)
    return sent_feat


def sent2label(triples):
    label = list()
    for triple in triples:
        label.append(triple['label'])
    return label


def label2ind(sents, path_label_ind):
    labels = list()
    for pairs in sents.values():
        labels.extend(sent2label(pairs))
    labels = sorted(list(set(labels)))
    label_inds = dict()
    for i in range(len(labels)):
        label_inds[labels[i]] = i
    with open(path_label_ind, 'wb') as f:
        pk.dump(label_inds, f)


def featurize(path_data, path_sent, path_label):
    with open(path_data, 'r') as f:
        sents = json.load(f)
    label2ind(sents, path_label_ind)
    sent_feats, labels = list(), list()
    for triples in sents.values():
        sent_feats.append(sent2feat(triples))
        labels.append(sent2label(triples))
    with open(path_sent, 'w') as f:
        json.dump(sent_feats, f, ensure_ascii=False, indent=4)
    with open(path_label, 'w') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    path_data = 'data/train.json'
    path_sent = 'feat/sent_train.json'
    path_label = 'feat/label_train.json'
    featurize(path_data, path_sent, path_label)
