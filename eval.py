import json

from sklearn_crfsuite.metrics import flat_accuracy_score

from recognize import restore_word, predict


def test(path):
    with open(path, 'r') as f:
        sents = json.load(f)
    label_mat = list()
    pred_mat = list()
    error_mat = dict()
    for triples in sents.values():
        labels = list()
        preds = list()
        errors = list()
        for triple in triples:
            labels.append(triple['label'])
        label_mat.append(labels)
        words = restore_word(triples)
        text = ''.join(words)
        pairs = predict(text)
        for word, pred in pairs:
            preds.append(pred)
        pred_mat.append(preds)
        for word, pred, label in zip(words, preds, labels):
            if pred != label:
                errors.append((word, label, pred))
        if errors:
            error_mat[text] = errors
    print('%s %.2f\n' % ('acc:', flat_accuracy_score(label_mat, pred_mat)))
    for text, errors in error_mat.items():
        error_str = text
        for word, label, pred in errors:
            error_str = error_str + ', {}: {} -> {}'.format(word, label, pred)
        print(error_str)


if __name__ == '__main__':
    path = 'data/test.json'
    test(path)
