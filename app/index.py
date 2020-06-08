


import flask
import os
import glob

from flask_cors import CORS
from flask import Flask, flash, request, redirect, url_for, jsonify

from werkzeug.utils import secure_filename

from src.predict import predict

app = flask.Flask(__name__)

app.config["DEBUG"] = True

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

CORS(app)

BASE_PATH = os.path.abspath(os.path.join(__file__, ".."))

TMP_DIR = BASE_PATH+"/temp/" 

def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/v2/extractor/extract", methods=["POST"])
def ner_extractor():
    if 'file' not in request.files:
        return jsonify("no file was provided")

    file = request.files["file"]

    if file.filename == '':
        return jsonify("no file was provided")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(TMP_DIR+filename)
        res = predict(TMP_DIR+filename)
        files = glob.glob(TMP_DIR+'*')
        for f in files:
            os.remove(f)
        
        return jsonify(res)

    return jsonify("-1")

app.run(host="0.0.0.0",port=8080)
