from argparse import ArgumentParser as parser

import json

from flask import Flask, request

from recognize import predict


app = Flask(__name__)


if __name__ == '__main__':
    ip = args.ip
