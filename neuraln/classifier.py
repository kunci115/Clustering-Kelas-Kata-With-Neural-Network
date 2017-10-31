# use natural language & sastrawi toolkit
import nltk
import numpy as np
import time
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import json
import datetime
stemmer = StemmerFactory().create_stemmer()
kata_dasar = open('kata_dasar.txt','r').readlines()
kata_imbuh = open('kata_imbuh.txt','r').readlines()
training_data = []
for d in kata_dasar:
    stammed = d.strip()
    training_data.append({"class":"katadasar", "sentence":stammed})
for i in kata_imbuh:
    unstammed = i.strip()
    training_data.append({"class":"kataimbuh", "sentence":unstammed})

    print ("%s sentences in training data" % len(training_data))

kata = []
classes = []
documents = []
ignore_words = ['?']
# Ngulang setiap kata di training data
for pattern in training_data:
    # tokenize setiap kata dengan nltk(kalo di trainingda
    w = nltk.word_tokenize(pattern['sentence'])
    # add ke wordlist
    kata.extend(w)
    # add add dokumen kedalem korpus
    documents.append((w, pattern['class']))
    # add ke dalem list class
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem & lowering setiap kata di sentence
kata = [stemmer.stem(w.lower()) for w in kata if w not in ignore_words]
kata = list(set(kata))

# remove duplikat data
classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(kata), "unique stemmed words", kata)

# Membuat training data
training = []
output = []
# Bikin array kosong untuk output
output_empty = [0] * len(classes)

# training set, bag of words dari setiap sentence
for doc in documents:
    # initialisasi untuk bagofwords
    bag = []
    # list of tokenized words dari setiap pola(pattern)
    pattern_words = doc[0]
    # stem setiap kata
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # membuat bag of word berbentuk array
    for w in kata:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]



# menghitung sigmoid nonlinear
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output dari fungsi sigmoid ke derivativenya
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def clean_up_sentence(sentence):
    # tokenize pola sentence
    sentence_words = nltk.word_tokenize(sentence)
    # stem setiap kata
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return (np.array(bag))


def think(sentence, show_details=False):
    x = bow(sentence.lower(), kata, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    l0 = x
    l1 = sigmoid(np.dot(l0, synapse_0))

    l2 = sigmoid(np.dot(l1, synapse_1))


    return l2


def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (
    hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))
    np.random.seed(1)

    last_mean_error = 1
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs + 1)):

        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        if (dropout):
            layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
            1.0 / (1 - dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j % 10000) == 0 and j > 5000:
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                break
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if (j > 0):
            synapse_0_direction_count += np.abs(
                ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(
                ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'kata': kata,
               'classes': classes
               }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)


    print ("saved synapses to:", synapse_file)

X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
# probability threshold
ERROR_THRESHOLD = 0.2
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results
classify("coba")