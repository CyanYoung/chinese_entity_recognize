import json

from recognize import predict

from util import load_word, load_pair

from flask import Flask, request

from argparse import ArgumentParser


path_slot = 'dict/slot.txt'
path_chn_eng = 'dict/chn_eng.csv'
path_key_label_slot = 'dict/key_label_slot.csv'
slots = load_word(path_slot)
chn_eng = load_pair(path_chn_eng)
key_label_slot = load_pair(path_key_label_slot)

app = Flask(__name__)

parser = ArgumentParser()
parser.add_argument('-host', type=str, default='127.0.0.1')
parser.add_argument('-port', type=str, default=2020)
args = parser.parse_args()


def init_slot(slots):

    for slot in slots


def map_slot(pred):
    for key, label, slot in key_label_slot:
        if key in pred:
            return slot
        else:
            return pred


@app.route('/extract', methods=['POST'])
def response():
    fields = request.get_json()
    pairs = predict(fields['content'])
    slots =
    for word, pred in pairs:
        if pred != 'N':
            slot = map_slot(chn_eng[pred])
            entitys.append((word, label))
    return json.dumps(fields, ensure_ascii=False)


if __name__ == '__main__':
    app.run(args.host, int(args.port))
