"""
Developer: Sanjib Sonowal
Email: sanjib.sonowal@gmail.com
Web: www.sanjibsonowal.com
Note: Please email for any collaboration.
"""

import bottle
import json
import os
from bottle import run, request, response, Bottle, HTTPResponse
import train as tr
import test as ts

app = Bottle()
port = int(os.getenv("PORT", 9009))

content_type = "application/json"
headers = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}


class EnableCors(object):
    name = 'enable_cors'
    api = 2

    def apply(self, fn, context):
        def _enable_cors(*args, **kwargs):
            # set CORS headers
            response.headers['Access-Control-Allow-Origin'] = "*"
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, ' \
                                                               'X-CSRF-Token, Authorization'

            if bottle.request.method != 'OPTIONS':
                # actual request; reply with the actual response
                return fn(*args, **kwargs)

        return _enable_cors


@app.route('/')
def hello_world():
    return 'Hello World! I am Credit Score calculator API'


@app.route('/train', method=['POST', 'OPTIONS'])
def train():
    resp = tr.train()
    if resp:
        resp_body = json.dumps({"Status": "Success", "Message": "Trained Successfully!"})
    else:
        resp_body = json.dumps({"Status": "Failure", "Message": "Training Failed!"})
    return HTTPResponse(status=200, body=resp_body, headers=headers)


@app.route('/predict', method=['POST', 'OPTIONS'])
def predict():
    algo, resp = None, None
    if 'algo' in request.json and request.json['algo'] != 'undefined' and request.json['algo'] != '':
        algo = request.json['algo']
    if algo == "logistic-regression":
        resp = ts.logistic_regression_predict()
    elif algo == "decision-tree":
        resp = ts.decision_tree_predict()
    elif algo == "random-forest":
        resp = ts.random_forest_predict()

    if resp is not None:
        resp_body = json.dumps({"Status": "Success", "Accuracy Score": resp})
    else:
        resp_body = json.dumps({"Status": "Failure", "Message": "Something Went Wrong!"})
    return HTTPResponse(status=200, body=resp_body, headers=headers)


app.install(EnableCors())


if __name__ == '__main__':
    run(app, host='0.0.0.0', port=port, debug=True)
