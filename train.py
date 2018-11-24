# natural language processing tool
import nltk
from nltk.stem.lancaster import LancasterStemmer
# statistical analysis and deep learning
import numpy as np
import tensorflow as tf
import tflearn
# math
import random
# i/o operations
import json
import pickle

# load the stemmer to abbreviate the input words
stemmer = LancasterStemmer()

# get the tags for training
with open("patterns/tags.json") as train_data:
    tags = json.load(train_data)

# initialize the memory assignments to hold data dictionary
words = []
classes = []
documents = []
ignore_words = ["?", "!", "."]

# extract features
for tag in tags:
    if "patterns" not in tags[tag]:
        continue
    for pattern in tags[tag]["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, tag))
        if tag not in classes:
            classes.append(tag)

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]

# remove duplicates
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes")
print(len(words), "unique stemmed words", words)

# prepare to train
training_data = []
output_schema = [0] * len(classes)

# prepare features and classes for deep learning classifier
for doc in documents:
    # initialize the training data
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in doc[0]]
    # make numerical pattern in bag
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # assign numerical classes according to the pattern in bag
    output_row = list(output_schema)
    output_row[classes.index(doc[1])] = 1

    training_data.append([bag, output_row])

# shuffle the training data
random.shuffle(training_data)
training_data = np.array(training_data)

training_features = list(training_data[:, 0])
training_classes = list(training_data[:, 1])

# reset the tensorflow graph
tf.reset_default_graph()

# build the neural network (3 hidden layers, one regression layer)
net = tflearn.input_data(shape=[None, len(training_features[0])])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(training_classes[0]), activation="softmax")
net = tflearn.regression(net)

# define model
model = tflearn.DNN(net, tensorboard_dir='model/tflearn_logs')
model.fit(training_features, training_classes, n_epoch=1000, batch_size=16, show_metric=True)
model.save("model/jason.zdmodel")

pickle.dump({'words':words, 'classes':classes, 'training_features':training_features, 'training_classes':training_classes}, open("model/jason.zdwords", "wb"))
