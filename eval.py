import json

from sklearn_crfsuite.metrics import flat_accuracy_score

from recognize import predict


path_test = 'data/test.json'
with open(path_test, 'r') as f:
    sents = json.load(f)


def test(sents):
    label_mat, pred_mat = list(), list()
    error_mat = dict()
    for text, triples in sents.items():
        labels, errors = list(), list()
        for triple in triples:
            labels.append(triple['label'])
        label_mat.append(labels)
        pairs = predict(text)
        words = [word for word, pred in pairs]
        preds = [pred for word, pred in pairs]
        pred_mat.append(preds)
        for word, pred, label in zip(words, preds, labels):
            if pred != label:
                errors.append((word, label, pred))
        if errors:
            error_mat[text] = errors
    print('\n%s %.2f\n' % ('acc:', flat_accuracy_score(label_mat, pred_mat)))
    for text, errors in error_mat.items():
        error_str = text
        for word, label, pred in errors:
            error_str = error_str + ' | {}: {} -> {}'.format(word, label, pred)
        print(error_str)


if __name__ == '__main__':
    test(sents)
