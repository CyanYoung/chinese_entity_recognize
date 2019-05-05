import pickle as pk

import jieba
from jieba.posseg import cut as pos_cut

from represent import sent2feat


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


path_cut_word = 'dict/cut_word.txt'
jieba.load_userdict(path_cut_word)

path_label_ind = 'feat/label_ind.pkl'
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

path_crf = 'model/crf.pkl'
with open(path_crf, 'rb') as f:
    crf = pk.load(f)


def predict(text):
    pairs = list(pos_cut(text))
    words, tags = zip(*pairs)
    triples = list()
    for word, tag in zip(words, tags):
        triple = dict()
        triple['word'] = word
        triple['pos'] = tag
        triples.append(triple)
    sent = sent2feat(triples)
    preds = crf.predict([sent])[0]
    if __name__ == '__main__':
        pairs = list()
        for word, pred in zip(words, preds):
            pairs.append((word, pred))
        return pairs
    else:
        inds = list()
        for pred in preds:
            inds.append(label_inds[pred])
        return words, inds


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('crf: %s' % predict(text))
