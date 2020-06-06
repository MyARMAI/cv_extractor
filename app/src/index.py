


import flask
import tensorflow as tf
import numpy as np
import keras
import textract
import os
import glob

from flask_cors import CORS
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename


from predict import predict
from cv_parser import parser


import pythoncom
pythoncom.CoInitialize()


app = flask.Flask(__name__)

app.config["DEBUG"] = True

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

CORS(app)


def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/v1/extractor/extract", methods=["POST"])
def extractor():
    if 'file' not in request.files:
        return jsonify("no file was provided")

    file = request.files["file"]

    if file.filename == '':
        return jsonify("no file was provided")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save("../temp/"+filename)
        res = parser(os.path.abspath("../temp/"+filename))
        files = glob.glob('../temp/*')
        for f in files:
            os.remove(f)
        
        return jsonify(res)

    return jsonify("-1")


@app.route("/api/v2/extractor/extract", methods=["POST"])
def ner_extractor():
    if 'file' not in request.files:
        return jsonify("no file was provided")

    file = request.files["file"]

    if file.filename == '':
        return jsonify("no file was provided")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save("../temp/"+filename)
        res = predict("../temp/"+filename)
        files = glob.glob('./temp/*')
        for f in files:
            os.remove(f)
        
        return jsonify(res)

    return jsonify("-1")

app.run()
