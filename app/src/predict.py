from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
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
from keras.preprocessing.sequence import pad_sequences

session = keras.backend.get_session()
init = tf.global_variables_initializer()

session.run(init)

graph = tf.get_default_graph()

BASE_PATH = os.path.abspath(".")

##################
model_path = BASE_PATH + "/models/saved_model/lstm_ner_model_F1_37_3.h5"

phraser_path = BASE_PATH + "/models/saved_model/phraser"

word2idx_path = BASE_PATH + "/app/assets/word2idx.json"

tag2idx_path = BASE_PATH + "/app/assets/tag2idx.json"
##################

print(model_path)

model = tf.keras.models.load_model(model_path)

phraser = gensim.models.Phrases.load(phraser_path)

word2idx = {}
max_len = 50
tokenizer = RegexpTokenizer(r'\w+')


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
    cleaned_data = re.sub("[\\r\\n\|]*", "", raw_data)
    stop_words = set(stopwords.words(stop_lang))
    cleaned_data = [x for x in tokenizer.tokenize(cleaned_data)]
  #  print(cleaned_data)
    return cleaned_data


def pre_process_doc(data):
    array_nbr = math.ceil(len(data)/40)
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
        if value == "Skills" and re.match("[0-9]", key):
            out[key] = "Level"
    bi_gram = []
    result = {}
    category = list(set(category))
    for _c in category:
        c = [x.lower() for x, y in out.items() if y == _c]
        # apply bigram model
        result[_c] = phraser[c]
    result
    # print(result)
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
            for w, pred in zip(seq, p[0]):
                pred_dict[w] = tags[pred]
                output_file += w+"  "+tags[pred]+"\n"
                #print("{:15}: {:5}".format(w, tags[pred]))

    return post_process(pred_dict)


if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    print("main")

    test_file = r'..\samples\base_cv\cv\CV ATOS Amadou NDIAYE - ENGLISH.docx'
    # predict(test_file)
    #res = predict("./out/CV_ATOS_Amadou_NDIAYE_-_ENGLISH.doc")
   # print(res)
