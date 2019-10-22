from __future__ import unicode_literals
import re
import os
import codecs
import json
import csv
import spacy
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
import pickle


nlp=spacy.load('en_core_web_lg')

intent_dict = {'Consult':0, 'Symptom/Disease':1, 'OutOfDomain':2}

def read_CSV_datafile(filename):
    X = []
    y = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            X.append(row[0])
            y.append(row[1])
    return X,y

filename_train = 'Datasets/main_intent_train.tsv'
filename_test = 'Datasets/main_intent_test.tsv'

X_train_raw, y_train_raw = read_CSV_datafile(filename = filename_train)
X_test_raw, y_test_raw = read_CSV_datafile(filename = filename_test)

def tokenize(doc):
    """
    Returns a list of strings containing each token in `sentence`
    """
    #return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
    #                            doc) if i != '' and i != ' ' and i != '\n']
    tokens = []
    doc = nlp.tokenizer(doc)
    for token in doc:
        tokens.append(token.text)
    return tokens

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

X_train_raw = semhash_corpus(X_train_raw)
X_test_raw = semhash_corpus(X_test_raw)

def get_vectorizer(corpus, preprocessor=None, tokenizer=None):
    vectorizer = CountVectorizer(ngram_range=(2,4),analyzer='char')
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.get_feature_names()

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test, target_names,
              print_report=True, feature_names=None, print_top10=False,
              print_cm=True):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    #print("Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(["Make Update", "Setup Printer", "Shutdown Computer","Software Recommendation", "None"]):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join([feature_names[i] for i in top10]))))
        print()

    if print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


def data_for_training():
    vectorizer, feature_names = get_vectorizer(X_train_raw, preprocessor=preprocess, tokenizer=tokenize)

    X_train_no_HD = vectorizer.transform(X_train_raw).toarray()
    X_test_no_HD = vectorizer.transform(X_test_raw).toarray()

    return X_train_no_HD, y_train_raw, X_test_no_HD, y_test_raw, feature_names

X_train, y_train, X_test, y_test, feature_names = data_for_training()

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


for i in range(len(X_train_raw)):
     X_train_raw[i] = ngram_encode(X_train_raw[i], HD_aphabet, aphabet, n_size) # HD_ngram is a projection of n-gram statistics for str to N-dimensional space. It can be used to learn the word embedding

for i in range(len(X_test_raw)):
    X_test_raw[i] = ngram_encode(X_test_raw[i], HD_aphabet, aphabet, n_size)

X_train, y_train, X_test, y_test = X_train_raw, y_train_raw, X_test_raw, y_test_raw

clf = LogisticRegression(C=1.0, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

clf.fit(X_train, y_train)
filename = 'Models/main_intent_model.sav'
pickle.dump(clf, open(filename, 'wb'))
# clf = pickle.load(open(filename, 'rb'))
