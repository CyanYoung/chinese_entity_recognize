from argparse import ArgumentParser

import json

from flask import Flask, request

from recognize import predict


parser = ArgumentParser()
parser.add_argument('-host', type=str, default='127.0.0.1')
parser.add_argument('-port', type=str, default=2020)
args = parser.parse_args()

app = Flask(__name__)


@app.route('/extract', methods=['POST'])
def response():
    data = request.get_json()
    data['entity'] = predict(data['content'])
    return json.dumps(data, ensure_ascii=False)


if __name__ == '__main__':
    app.run(args.host, args.port)
