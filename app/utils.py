import math
import tensorflow as tf
import numpy as np
import json
import re
import textract

import os
from keras.preprocessing.sequence import pad_sequences


from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
word2idx = {}
max_len = 50


with open("word2idx.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)

with open("tag2idx.json", "r", encoding="utf-8") as f:
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
   # print(cleaned_data)
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

    result = {}
    category = list(set(category))
    for _c in category:
        c = [x for x, y in out.items() if y == _c]
        result[_c] = c
    result.pop("O")
    return result


def prepare_data(filepath):

   # print(filepath)
    raw_test_file = loadFile(filepath)

    cleaned_test_file = dataCleaning(raw_test_file)

    sequences = pre_process_doc(cleaned_test_file)

    pred_dict = {}

    output_file = ""
    pred_sentence = []
    sentences = [[word2idx.get(w, 0) for w in s] for s in sequences]
    x_test_sent = pad_sequences(sequences=sentences,
                                padding="post", value=0, maxlen=50)
    return x_test_sent[0]


def convertToSavedModel():

    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    model = tf.keras.models.load_model(
        "./saved_model/lstm_ner_model_F1_37_3.h5")
    export_path = './SavedFormat/'

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs})
