
from werkzeug.utils import secure_filename

import flask
import os
import glob

from flask_cors import CORS
from flask import Flask, flash, request, redirect, url_for, jsonify

from src.predict import predict
from src.elmo import elmoPredict,startElmo

app = flask.Flask(__name__)

app.config["DEBUG"] = False

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

CORS(app)

BASE_PATH = os.path.abspath(os.path.join(__file__, ".."))

TMP_DIR = BASE_PATH+"/out/"

#model = startElmo()


def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate():
    if 'file' not in request.files:
        return None

    file = request.files["file"]

    if file.filename == '':
        return None

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(TMP_DIR+filename)
    return filename


def cleanDir():
    files = glob.glob(TMP_DIR+'*')
    for f in files:
        os.remove(f)


@app.route("/api/v1/extractor/extract", methods=["POST"])
def ner_extractor():
    filename = validate()
    if filename:
        res = predict(TMP_DIR+filename)
        cleanDir()
        return jsonify(res)

    return jsonify("-1")


@app.route("/api/v2/extractor/queryParser",methods=["POST"])
def extractQuery():
    query = request.get_json(force=True)["query"]
    parsed = queryParser(query)
    return jsonify(parsed)

""" @app.route("/api/v2/extractor/extract", methods=["POST"])
def elmo_extractor():
    filename = validate()
    if filename:
        res = elmoPredict(model,TMP_DIR+filename)
        cleanDir()
        return jsonify(res)

    return jsonify("-1") """

app.run(host="0.0.0.0", port=8080,threaded=False)
