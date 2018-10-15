import json
import pickle as pk

from random import shuffle

from sklearn_crfsuite import CRF


min_freq = 1


def fit(path_sent, path_label, path_crf):
    with open(path_sent, 'r') as f:
        sent_feats = json.load(f)
    with open(path_label, 'r') as f:
        labels = json.load(f)
    sents_labels = list(zip(sent_feats, labels))
    shuffle(sents_labels)
    sent_feats, labels = zip(*sents_labels)
    crf = CRF(algorithm='lbfgs', min_freq=min_freq, c1=0.1, c2=0.1,
              max_iterations=100, all_possible_transitions=True)
    crf.fit(sent_feats, labels)
    with open(path_crf, 'wb') as f:
        pk.dump(crf, f)


if __name__ == '__main__':
    path_sent = 'feat/sent_train.json'
    path_label = 'feat/label_train.json'
    path_crf = 'model/crf.pkl'
    fit(path_sent, path_label, path_crf)
