import math
#import tensorflow as tf
import numpy as np
import json
import re
import textract
import os
import nltk
import string
import gensim

BASE_PATH = os.path.abspath(os.path.join(__file__, "../../.."))


phraser_path = BASE_PATH + "/models/saved_model/phraser"
phraser = gensim.models.Phrases.load(phraser_path)

printable = set(string.printable)
max_len = 50

def loadFile(filepath):
    if not os.path.isfile(filepath):
        raise Exception("OpenFileException", "File doesn't exist")

    raw_file = textract.process(os.path.join(filepath)).decode()
    return raw_file


def dataCleaning(raw_data, stop_lang="english"):
    r = re.sub("1-Knowledge|2-Mastery|3-Expertise", "", raw_data)
    r = re.sub("Date de mise à jour : mois année", "", r)
    r = re.sub("[\\r\\t\\n\|\-\_–/\(\)]", " ", r)
    #r = ''.join(filter(lambda x: x in printable, r))
    r = r.split()
    return r


def pre_process_doc(data, max_len=50):
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


def convertToSavedModel(modelpath, output_dir):

    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    model = tf.keras.models.load_model(
        modelpath)
    export_path = output_dir

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs})


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def prepare(data, batch_size, max_len):
    sent = []
    for i in range(batch_size):
        tmp = []
        for j in range(max_len):
            try:
                tmp.append(data[j])
            except:
                tmp.append('__PAD__')
        sent.append(tmp)
    return sent


def fill_out(data, max_len):
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(data[i])
        except:
            new_seq.append("__PAD__")
    return new_seq


if __name__ == "__main__":
    test_file = r"C:\Users\Cheikh\Desktop\Projet_memoire\myArmAi\samples\base_cv\cv\CV ATOS Amadou NDIAYE - ENGLISH.docx"
    res = loadFile(test_file)

    print(dataCleaning(res))
