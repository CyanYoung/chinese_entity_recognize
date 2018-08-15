import json

from flask import Flask, request

from argparse import ArgumentParser

from recognize import predict

from util import load_word, load_pair, load_triple, get_logger


path_slot = 'dict/slot.txt'
path_chn_eng = 'dict/chn_eng.csv'
path_label_key_slot = 'dict/label_key_slot.csv'
slots = load_word(path_slot)
chn_eng = load_pair(path_chn_eng)
label_key_slot = load_triple(path_label_key_slot)

app = Flask(__name__)

parser = ArgumentParser()
parser.add_argument('-host', type=str, default='127.0.0.1')
parser.add_argument('-port', type=str, default=2020)
args = parser.parse_args()

path_log_dir = 'log'
logger = get_logger('extract', path_log_dir)


def init_entity(slots):
    entitys = dict()
    for slot in slots:
        entitys[slot] = list()
    return entitys


def map_slot(word, pred):
    for label, key, slot in label_key_slot:
        if pred == label:
            if key in word:
                return slot
    return pred


@app.route('/extract', methods=['POST'])
def response():
    entitys = init_entity(slots)
    data = request.get_json()
    pairs = predict(data['content'])
    fill_slots = list()
    for word, pred in pairs:
        if pred != 'N':
            slot = map_slot(word, chn_eng[pred])
            fill_slots.append(slot)
            entitys[slot].append(word)
    data['intent'] = '_'.join(fill_slots)
    data['entity'] = entitys
    data_str = json.dumps(data, ensure_ascii=False)
    logger.info(data_str)
    return data_str


if __name__ == '__main__':
    app.run(args.host, int(args.port))
