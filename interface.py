import json

from flask import Flask, request

from argparse import ArgumentParser

from recognize import predict

from util import load_pair, load_triple, get_logger


path_zh_en = 'dict/zh_en.csv'
path_label_key_slot = 'dict/label_key_slot.csv'
zh_en = load_pair(path_zh_en)
label_key_slot = load_triple(path_label_key_slot)

app = Flask(__name__)

parser = ArgumentParser()
parser.add_argument('-host', type=str, default='127.0.0.1')
parser.add_argument('-port', type=str, default=2000)
args = parser.parse_args()

path_log_dir = 'log'
logger = get_logger('recognize', path_log_dir)


def map_slot(word, pred):
    for label, key, slot in label_key_slot:
        if pred == label:
            if key in word:
                return slot
    return pred


def make_dict(entitys, slots):
    slot_dict = dict()
    for slot, entity in zip(slots, entitys):
        if slot not in slot_dict:
            slot_dict[slot] = list()
        slot_dict[slot].append(entity)
    return slot_dict


@app.route('/recognize', methods=['POST'])
def response():
    data = request.get_json()
    pairs = predict(data['content'])
    entitys, slots = list(), list()
    for word, pred in pairs:
        if pred != 'O':
            entitys.append(word)
            slot = map_slot(word, zh_en[pred])
            slots.append(slot)
    slot_dict = make_dict(entitys, slots)
    data['slot'] = slot_dict
    data_str = json.dumps(data, ensure_ascii=False)
    logger.info(data_str)
    return data_str


if __name__ == '__main__':
    app.run(args.host, int(args.port))
