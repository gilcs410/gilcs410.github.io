# -*- coding: utf-8 -*-
"""

"""

__author__ = "Gil"
__status__ = "test"
__version__ = "0.1"
__date__ = "Nov 2018"

import const
from flask import Flask, request, jsonify, render_template
import json
import traceback
import logging

logger = logging.getLogger('werkzeug')
logger.setLevel(logging.INFO)

app = Flask(__name__)


def get_json(root, key):
    if key not in root:
        raise KeyError("Key: '" + key + "' cannot be found in the json!")

    return root[key]


# @app.route("/")
# def default():
#    return render_template('index.html')


@app.route("/sentiment", methods=["POST"])
def sentiment():

    root = None
    sub_root = None

    try:

        root = request.get_json(force=True)

        if type(root) is str:
            root = json.loads(root)

        sub_root = get_json(root, const.JSON_KEY_ROOT)

        # if top k prediction return config is not set from the request, uses value 1 as default
        top = 1
        if const.JSON_KEY_TOP in sub_root:
            top = get_json(sub_root, const.JSON_KEY_TOP)

    except Exception as e:

        if sub_root is None:
            root = json.loads('{"' + const.JSON_KEY_ROOT + '": {}}')
            sub_root = get_json(root, const.JSON_KEY_ROOT)

        sub_root[const.JSON_KEY_SUCCESS] = False
        sub_root[const.JSON_KEY_ERROR] = str(e)

        traceback.print_tb(e.__traceback__)

    return jsonify(root)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


def main():
    logger.info("Initializing Flask! Loading category prediction models...")

    logger.info("Successfully loaded models! Flask listening to request...")
    logger.setLevel(logging.ERROR)
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
