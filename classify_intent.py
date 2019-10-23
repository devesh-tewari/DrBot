from __future__ import unicode_literals
import re
import os
import codecs
import json
import csv
import spacy
import numpy as np
from time import time
import numpy as np
import pickle


nlp=spacy.load('en_core_web_lg')

def preprocess(doc):
    clean_tokens = []
    doc = nlp(doc)
    for token in doc:
        if not token.is_stop:
            clean_tokens.append(token.lemma_)
    return " ".join(clean_tokens)

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def semhash_tokenizer(text):
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens

def semhash_corpus(corpus):
    new_corpus = []
    for sentence in corpus:
        sentence = preprocess(sentence)
        tokens = semhash_tokenizer(sentence)
        new_corpus.append(" ".join(map(str,tokens)))
    return new_corpus


def ngram_encode(str_test, HD_aphabet, aphabet, n_size): # method for mapping n-gram statistics of a word to an N-dimensional HD vector
    HD_ngram = np.zeros(HD_aphabet.shape[1]) # will store n-gram statistics mapped to HD vector
    full_str = '#' + str_test + '#' # include extra symbols to the string

    for il, l in enumerate(full_str[:-(n_size-1)]): # loops through all n-grams
        hdgram = HD_aphabet[aphabet.find(full_str[il]), :] # picks HD vector for the first symbol in the current n-gram
        for ng in range(1, n_size): #loops through the rest of symbols in the current n-gram
            hdgram = hdgram * np.roll(HD_aphabet[aphabet.find(full_str[il+ng]), :], ng) # two operations simultaneously; binding via elementvise multiplication; rotation via cyclic shift

        HD_ngram += hdgram # increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram

    HD_ngram_norm = np.sqrt(HD_aphabet.shape[1]) * (HD_ngram/ np.linalg.norm(HD_ngram) )  # normalizes HD-vector so that its norm equals sqrt(N)
    return HD_ngram_norm # output normalized HD mapping



N = 1000 # set the desired dimensionality of HD vectors
n_size=3 # n-gram size
aphabet = 'abcdefghijklmnopqrstuvwxyz#' #fix the alphabet. Note, we assume that capital letters are not in use
np.random.seed(1) # for reproducibility
HD_aphabet = 2 * (np.random.randn(len(aphabet), N) < 0) - 1 # generates bipolar {-1, +1}^N HD vectors; one random HD vector per symbol in the alphabet

filename = 'Models/main_intent_model.sav'
# pickle.dump(clf, open(filename, 'wb'))
clf = pickle.load(open(filename, 'rb'))

def get_intent(text):
    X_test_raw = [text]
    X_test_raw = semhash_corpus(X_test_raw)
    X_test_raw[0] = ngram_encode(X_test_raw[0], HD_aphabet, aphabet, n_size)
    X_test = X_test_raw

    pred = clf.predict(X_test)
    return pred[0]

filename = 'Models/OOD_intent_model.sav'
# pickle.dump(clf, open(filename, 'wb'))
clf_OOD = pickle.load(open(filename, 'rb'))

def get_intent_OOD(text):
    X_test_raw = [text]
    X_test_raw = semhash_corpus(X_test_raw)
    X_test_raw[0] = ngram_encode(X_test_raw[0], HD_aphabet, aphabet, n_size)
    X_test = X_test_raw

    pred = clf_OOD.predict(X_test)
    return pred[0]
