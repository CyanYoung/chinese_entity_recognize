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


def restore(triples):
    words = list()
    for triple in triples:
        words.append(triple['word'])
    return ' '.join(words)


def predict(text):
    text = re.sub(stop_word_re, '', text.strip())
    triples = label_word(text, [], '')
    cut_text = restore(triples)
    sent_feat = sent2feat(triples)
    if __name__ == '__main__':
        print(cut_text)
    return crf.predict([sent_feat])[0]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('pred: %s' % predict(text))
