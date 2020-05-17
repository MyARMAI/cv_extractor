import textract
import os
import re
import json
import numpy as np

import keras
from keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import math

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
    array_nbr = math.ceil(len(data)/max_len)
    sequences = np.array_split(np.array(data), array_nbr)
    return sequences


def predict(sequences):
    model = keras.models.load_model("./saved_model/lstm_ner_model.h5")
    pred_dict = {}
    for seq in sequences:
        pred_sentence = pad_sequences(sequences=[[word2idx.get(w, 0) for w in seq]],
                                      padding="post", value=0, maxlen=max_len)
        p = model.predict(np.array([pred_sentence[0]]))
        p = np.argmax(p, axis=-1)
        for w, pred in zip(seq, p[0]):
            pred_dict[w] = tags[pred]
            print("{:15}: {:5}".format(w, tags[pred]))

    return pred_dict


def main():
    test_file = r"C:\Users\Cheikh\Desktop\Projet_memoire\myArmAi\samples\cv\cv_atos\eng\CV ATOS Amadou NDIAYE - ENGLISH.doc"

    raw_test_file = loadFile(test_file)

    cleaned_test_file = dataCleaning(raw_test_file)

    sequences = pre_process_doc(cleaned_test_file)

    pred_dict = predict(sequences)


if __name__ == "__main__":
    main()
