from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from collections import Counter

import math
import textract
import os
import re
import json
import numpy as np
import tensorflow as tf
import keras
import gensim
import nltk

import string
printable = set(string.printable)

tf.compat.v1.disable_v2_behavior()

session = keras.backend.get_session()
init = tf.global_variables_initializer()

session.run(init)

graph = tf.get_default_graph()

BASE_PATH = os.path.abspath(os.path.join(__file__, "../../.."))

##################
model_path = BASE_PATH + "/models/saved_model/lstm_ner_model_F1_37_3.h5"

phraser_path = BASE_PATH + "/models/saved_model/phraser"

word2idx_path = BASE_PATH + "/app/assets/word2idx.json"

tag2idx_path = BASE_PATH + "/app/assets/tag2idx.json"
##################

model = tf.keras.models.load_model(model_path)

phraser = gensim.models.Phrases.load(phraser_path)

word2idx = {}
max_len = 50

with open(word2idx_path, encoding="utf-8") as f:
    word2idx = json.load(f)

with open(tag2idx_path, encoding="utf-8") as f:
    tag2idx = json.load(f)

tags = tag2idx.keys()

tags = list(tag2idx.keys())


def loadFile(filepath):

    if not os.path.isfile(filepath):
        raise Exception("OpenFileException", "File doesn't exist")

    raw_file = textract.process(os.path.join(filepath)).decode()
    return raw_file


def dataCleaning(raw_data, stop_lang="english"):
    r = re.sub("[\\r\\t\\n\|\-/]", " ", raw_data)
    #r = ''.join(filter(lambda x: x in printable, r))
    r = r.split()
    return r


def pre_process_doc(data):
    array_nbr = math.ceil(len(data)/max_len)
    sequences = np.array_split(np.array(data), array_nbr)
    return sequences


def post_process(out):
    # if a number follow a skills --> transform to level
    # if a string has a duration has a tag --> O
    ##
    # for key,value in out.items():
    category = []
    for key, value in out.items():
        category.append(value)
        if value == "Skills" and re.match("[0-9]", key.split("_")[1]):
            out[key] = "Level"
    bi_gram = []
    result = {}
    category = list(set(category))
    for _c in category:
        c = [x.split("_")[1].lower() for x, y in out.items() if y == _c]
        # apply bigram model
        result[_c] = phraser[c]
    return result


def predict(filepath):

   # print(filepath)
    raw_test_file = loadFile(filepath)

    cleaned_test_file = dataCleaning(raw_test_file)

    sequences = pre_process_doc(cleaned_test_file)
    pred_dict = {}

    output_file = ""
    pred_sentence = []

    for seq in sequences:
        pred_sentence = pad_sequences(sequences=[[word2idx.get(w, 0) for w in seq]],
                                      padding="post", value=0, maxlen=max_len)
        global session
        global graph
        with graph.as_default():
            set_session(session)
            p = model.predict(np.array([pred_sentence[0]]))
            p = np.argmax(p, axis=-1)
            i = 0
            for w, pred in zip(seq, p[0]):
                pred_dict[str(i)+'_'+w] = tags[pred]
                output_file += w+"  "+tags[pred]+"\n"
                i += 1
                #print("{:15}: {:5}".format(w, tags[pred]))

    # print(pred_dict)
    return post_process(pred_dict)


if __name__ == "__main__":

    print("main")

    test_file = r'C:\Users\Cheikh\Desktop\Projet_memoire\myArmAi\samples\base_cv\cv\CV ATOS Fassou Mathias Niamy - ENGLISH.docx'
    r = predict(test_file)
    print(r)
    #res = predict("./out/CV_ATOS_Amadou_NDIAYE_-_ENGLISH.doc")
   # print(res)
