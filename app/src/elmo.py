import os
import json
import keras
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import gensim
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras import backend as K

import sys

sys.path.append(BASE_PATH+"/app/src/")
from utils import loadFile, pre_process_doc, post_process, dataCleaning,fill_out,prepare,divide_chunks

batch_size = 35
max_len = 117

tf.compat.v1.disable_v2_behavior()

session = K.get_session()

init = tf.global_variables_initializer()

session.run(init)

graph = tf.get_default_graph()

BASE_PATH = os.path.abspath(os.path.join(__file__, "../../.."))

sys.path.append(BASE_PATH+"/app/src/")
from utils import loadFile, pre_process_doc, post_process, dataCleaning,fill_out,prepare,divide_chunks


##################
model_path = BASE_PATH + "/models/saved_model/lstm_elmo.h5"

phraser_path = BASE_PATH + "/models/saved_model/phraser"

tag2idx_path = BASE_PATH + "/app/assets/tag.json"

phraser_path = BASE_PATH + "/models/saved_model/phraser"

phraser = gensim.models.Phrases.load(phraser_path)

with open(tag2idx_path, encoding="utf-8") as f:
    tag2idx = json.load(f)

tags = tag2idx.keys()
tags = list(tag2idx.keys())
n_tags = len(tags)

def loadElmo():
    sess = K.get_session()
    K.set_session(sess)
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)
    sess.run(init)
    sess.run(tf.tables_initializer())
    return elmo_model


def ElmoEmbedding(x):
    elmo_model = loadElmo()
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, tf.string)),
        "sequence_len": tf.constant(batch_size*[max_len])
    },
        signature="tokens",
        as_dict=True)["elmo"]


def createModel():
    input_text = Input(shape=(max_len,), dtype="string")
    embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024))(input_text)
    x = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.3))(embedding)
    x_rnn = Bidirectional(
        LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.3))(x)
    x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)
    model = Model(input_text, out)
    return model


def elmoPredict(filepath):
    model = createModel()
    model.load_weights(model_path)
    model.summary()

    raw_file = loadFile(filepath)
    cleaned_file = dataCleaning(raw_file)

    res = list(divide_chunks(cleaned_file, max_len))
    pred_dict = {}
    for r in res:
        sent = prepare(r, batch_size, max_len)
        offset = batch_size - len(sent)
        for i in range(offset):
            sent.append(fill_out([]), max_len)
        p = model.predict(np.array(sent), batch_size=35)[0]

        p = np.argmax(p, axis=-1)
        i = 0
        for w, pred in zip(sent[0], p):
            pred_dict[str(i)+'_'+w] = tags[pred]
            i += 1
            print("{:15}: {:5}".format(w, tags[pred]))
    return pred_dict


def prepareSentence(sentence):
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(sentence[i])
        except:
            new_seq.append("__PAD__")
    return new_seq


if __name__ == "__main__":
    test_file = r"C:\Users\Cheikh\Desktop\Projet_memoire\myArmAi\samples\base_cv\cv\CV ATOS Amadou NDIAYE - ENGLISH.docx"
    elmoPredict(test_file)
