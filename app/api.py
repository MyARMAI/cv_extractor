


import flask
import tensorflow as tf
import numpy as np
import keras

from flask_cors import CORS
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import textract
from predict import predict

TF_PREDICT_SERVER = " http://localhost:8501/v1/models/extractor:predict"

app = flask.Flask(__name__)

app.config["DEBUG"] = True

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

CORS(app)


def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/extractor/extract", methods=["POST"])
def extractor():
    if 'file' not in request.files:
        return jsonify("no file was provided")

    file = request.files["file"]

    if file.filename == '':
        return jsonify("no file was provided")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save("./temp/"+filename)
        res = predict("./temp/"+filename)
        return jsonify(res)

    return jsonify("-1")


app.run()
