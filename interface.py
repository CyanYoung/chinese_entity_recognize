import json

from flask import Flask, request

from argparse import ArgumentParser

from recognize import predict

from util import load_pair, get_logger


path_zh_en = 'dict/zh_en.csv'
zh_en = load_pair(path_zh_en)

app = Flask(__name__)

parser = ArgumentParser()
parser.add_argument('-host', type=str, default='127.0.0.1')
parser.add_argument('-port', type=str, default=2018)
args = parser.parse_args()

path_log_dir = 'log'
logger = get_logger('recognize', path_log_dir)


def make_dict(entitys, labels):
    slot_dict = dict()
    for label, entity in zip(labels, entitys):
        if label not in slot_dict:
            slot_dict[label] = list()
        slot_dict[label].append(entity)
    return slot_dict


@app.route('/recognize', methods=['POST'])
def response():
    data = request.get_json()
    pairs = predict(data['content'])
    entitys, labels = list(), list()
    for word, pred in pairs:
        if pred != 'O':
            entitys.append(word)
            labels.append(zh_en[pred])
    slot_dict = make_dict(entitys, labels)
    data['slot'] = slot_dict
    data_str = json.dumps(data, ensure_ascii=False)
    logger.info(data_str)
    return data_str


if __name__ == '__main__':
    app.run(args.host, int(args.port))
