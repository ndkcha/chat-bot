# natural language processing tool
import nltk
from nltk.stem.lancaster import LancasterStemmer
# statistical analysis and deep learning
import numpy as np
# import tensorflow as tf
import tflearn
# built-in math
import random
# i/o operations
import json
import pickle
# web
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
# internal modules
import db.global_helpers as gbl

# load the stemmer to abbreviate the input words
stemmer = LancasterStemmer()

# load the words
data = pickle.load(open("model/shareen.zdwords", "rb"))
words = data["words"]
classes = data["classes"]
training_features = data["training_features"]
training_classes = data["training_classes"]

# define web server
app = Flask(__name__)
cors = CORS(app)

# convert the sentence to model readable format
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # abbreviate the words
    sentence_words = [stemmer.stem(w.lower()) for w in sentence_words]
    return sentence_words


# convert the words to feature vectors for network input
def math_sentence(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


# classify the sentence
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([math_sentence(sentence)])[0]
    # filter out the predictions based on the threshold value
    results = [[i, r] for i, r in enumerate(results)
               if r > (gbl.THRESHOLD_WOA if gbl.question is None else gbl.THRESHOLD_WA)]

    # sort by the strength of the probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


# perform actions based on the response
@app.route('/', methods=['POST'])
@cross_origin()
def response():
    incoming = request.get_json()
    sentence = incoming["body"]
    results = classify(sentence)
    out = dict()

    if results:
        while results:
            if results[0][0] in tags:
                tag = tags[results[0][0]]
                if 'exit' in tag and tag['exit'] is True:
                    out["message"] = random.choice(tag['response'])
                    return jsonify(out)
                if 'response' in tag:
                    out["message"] = random.choice(tag['response'])
                    return jsonify(out)
            results.pop(0)

    return jsonify(out)


# load tags
with open("patterns/tags.json") as train_data:
    tags = json.load(train_data)
# with open("patterns/questions.json") as questions_data:
#     questions = json.load(questions_data)

# build the neural network that is exactly same as the training replica (3 hidden layers, one regression layer)
net = tflearn.input_data(shape=[None, len(training_features[0])])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(training_classes[0]), activation="softmax")
net = tflearn.regression(net)

# define model
model = tflearn.DNN(net)

# load saved model
model.load("./model/shareen.zdmodel")

# load web server
app.run(port=8200)
