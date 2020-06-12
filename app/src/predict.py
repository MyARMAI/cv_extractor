import os
import json
import tensorflow as tf
import keras
import gensim
import numpy as np
import sys
import sys

from tensorflow.python.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.backend import set_session

tf.compat.v1.disable_v2_behavior()

session = keras.backend.get_session()
init = tf.global_variables_initializer()

session.run(init)
graph = tf.get_default_graph()

BASE_PATH = os.path.abspath(os.path.join(__file__, "../../.."))


sys.path.append(BASE_PATH+"/app/src/")
from utils import loadFile, pre_process_doc, post_process, dataCleaning

##################
model_path = BASE_PATH + "/models/saved_model/lstm_ner_model_F1_37_3.h5"


word2idx_path = BASE_PATH + "/app/assets/word2idx.json"

tag2idx_path = BASE_PATH + "/app/assets/tag2idx.json"
##################

model = tf.keras.models.load_model(model_path)

word2idx = {}
max_len = 50

with open(word2idx_path, encoding="utf-8") as f:
    word2idx = json.load(f)

with open(tag2idx_path, encoding="utf-8") as f:
    tag2idx = json.load(f)

tags = tag2idx.keys()
tags = list(tag2idx.keys())

def predict(filepath):
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
                #print("{:15}: {:5}".format(w, pred))
                
    res = post_process(pred_dict)
    #print(res)
    return res


if __name__ == "__main__":

    print("main")

    test_file = r'C:\Users\Cheikh\Desktop\Projet_memoire\myArmAi\samples\base_cv\cv\CV ATOS Amadou NDIAYE - ENGLISH.docx'
    r = predict(test_file)
    print(r)
    #res = predict("./out/CV_ATOS_Amadou_NDIAYE_-_ENGLISH.doc")
   # print(res)
