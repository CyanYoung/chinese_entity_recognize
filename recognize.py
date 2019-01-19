import pickle as pk

import jieba
from jieba.posseg import cut as pos_cut

from represent import sent2feat

path_cut_word = 'dict/cut_word.txt'
path_crf = 'model/crf.pkl'
jieba.load_userdict(path_cut_word)
with open(path_crf, 'rb') as f:
    crf = pk.load(f)


def restore(triples):
    words = list()
    for triple in triples:
        words.append(triple['word'])
    return words


def predict(text):
    text = text.strip()
    pairs = list(pos_cut(text))
    words = [word for word, tag in pairs]
    tags = [tag for word, tag in pairs]
    triples = list()
    for word, tag in zip(words, tags):
        triple = dict()
        triple['word'] = word
        triple['pos'] = tag
        triples.append(triple)
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
