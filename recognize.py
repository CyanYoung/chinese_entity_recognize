import pickle as pk

import re

from util import load_word_re

from preprocess import label_word

from represent import sent2feat


path_stop_word = 'dict/stop_word.txt'
path_crf = 'model/crf.pkl'
stop_word_re = load_word_re(path_stop_word)
with open(path_crf, 'rb') as f:
    crf = pk.load(f)


def restore_word(triples):
    words = list()
    for triple in triples:
        words.append(triple['word'])
    return words


def predict(text):
    text = re.sub(stop_word_re, '', text.strip())
    triples = label_word(text, [], '')
    words = restore_word(triples)
    sent_feat = sent2feat(triples)
    preds = crf.predict([sent_feat])[0]
    pairs = list()
    for word, pred in zip(words, preds):
        pairs.append((word, pred))
    return pairs


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('pred: %s' % predict(text))
