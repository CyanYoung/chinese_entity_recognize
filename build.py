import json
import pickle as pk

from scipy.stats import expon

from sklearn_crfsuite import CRF
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer
from sklearn_crfsuite.metrics import flat_accuracy_score


min_freq = 1

path_sent = 'feat/sent_train.json'
path_label = 'feat/label_train.json'
with open(path_sent, 'r') as f:
    sent_feats = json.load(f)
with open(path_label, 'r') as f:
    labels = json.load(f)


def fit(sent_feats, labels, path_crf, tune):
    crf = CRF(algorithm='lbfgs', min_freq=min_freq, c1=0.1, c2=0.1,
              max_iterations=100, all_possible_transitions=True)
    if tune:
        params = {'c1': expon(scale=0.1),
                  'c2': expon(scale=0.1)}
        flat_acc = make_scorer(flat_accuracy_score)
        crf = RandomizedSearchCV(crf, params, cv=5, n_iter=10, scoring=flat_acc)
    crf.fit(sent_feats, labels)
    with open(path_crf, 'wb') as f:
        pk.dump(crf, f)


if __name__ == '__main__':
    path_crf = 'model/crf.pkl'
    fit(sent_feats, labels, path_crf, tune=False)
