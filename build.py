import json
import pickle as pk

from random import shuffle

from scipy.stats import expon

from sklearn_crfsuite import CRF
from sklearn.model_selection import RandomizedSearchCV

from sklearn_crfsuite.metrics import flat_accuracy_score
from sklearn.metrics import make_scorer


min_freq = 1


def fit(path_sent, path_label, path_crf):
    with open(path_sent, 'r') as f:
        sent_feats = json.load(f)
    with open(path_label, 'r') as f:
        labels = json.load(f)
    sents_labels = list(zip(sent_feats, labels))
    shuffle(sents_labels)
    sent_feats, labels = zip(*sents_labels)
    crf = CRF(algorithm='lbfgs', min_freq=min_freq, max_iterations=100,
              all_possible_transitions=True)
    params = {'c1': expon(scale=0.1),
              'c2': expon(scale=0.1)}
    flat_acc = make_scorer(flat_accuracy_score)
    crf_rs = RandomizedSearchCV(crf, params, cv=10, n_iter=100, scoring=flat_acc)
    crf_rs.fit(sent_feats, labels)
    with open(path_crf, 'wb') as f:
        pk.dump(crf, f)
    if __name__ == '__main__':
        print(crf_rs.best_params_)


if __name__ == '__main__':
    path_sent = 'feat/sent_train.json'
    path_label = 'feat/label_train.json'
    path_crf = 'model/crf.pkl'
    fit(path_sent, path_label, path_crf)
