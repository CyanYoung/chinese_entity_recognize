import pickle as pk

from preprocess import label_word

from represent import sent2feat


path_crf = 'model/crf.pkl'
with open(path_crf, 'rb') as f:
    crf = pk.load(f)


def restore(triples):
    words = list()
    for triple in triples:
        words.append(triple['word'])
    return words


def predict(text):
    text = text.strip()
    triples = label_word(text, [], '')
    words = restore(triples)
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
