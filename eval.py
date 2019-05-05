import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from recognize import label_inds, ind_labels, predict


path_test = 'data/test.json'
with open(path_test, 'r') as f:
    sents = json.load(f)

class_num = len(label_inds)

slots = list(ind_labels.keys())
slots.remove(label_inds['O'])

path_crf = 'metric/crf.csv'


def test(sents):
    flat_labels, flat_preds = list(), list()
    for text, triples in sents.items():
        word1s, labels = list(), list()
        for triple in triples:
            word1s.append(triple['word'])
            labels.append(label_inds[triple['label']])
        word2s, preds = predict(text)
        for i in range(len(word2s)):
            if word2s[i] == word2s[i]:
                flat_labels.append(labels[i])
                flat_preds.append(preds[i])
    precs = precision_score(flat_labels, flat_preds, average=None)
    recs = recall_score(flat_labels, flat_preds, average=None)
    with open(path_crf, 'w') as f:
        f.write('label,prec,rec' + '\n')
        for i in range(class_num):
            f.write('%s,%.2f,%.2f\n' % (ind_labels[i], precs[i], recs[i]))
    f1 = f1_score(flat_labels, flat_preds, average='weighted', labels=slots)
    print('\n%s f1: %.2f - acc: %.2f' % ('crf', f1, accuracy_score(flat_labels, flat_preds)))


if __name__ == '__main__':
    test(sents)
