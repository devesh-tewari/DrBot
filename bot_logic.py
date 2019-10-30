import difflib
import copy
from fuzzywuzzy import fuzz
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import random

def stopWords(text):
    #text is a sentence
    stopw = set(stopwords.words('english'))
    filtered = []
    words = word_tokenize(text)
    for i in words:
        if i not in stopw:
            filtered.append(i)
    return filtered

def stemming(text):
    #text could be a sent or word
    ps = PorterStemmer()
    empty = []
    for w in text:
        empty.append(w)
    return empty

doctor_types = [line for line in open('Datasets/Doctor Types.csv', 'r').readlines()[1:]]
def consult(text):
    sent = sent_tokenize(text)
    filt_list = stopWords(text)
    filt = ''
    for f in filt_list:
        filt += str(f)+' '

    for i in range(len(filt)):
        if filt[i] == 'hospital' and i > 0:
            hospital_name = filt[i-1]
            return hospital_name + ' hospital'

    user_type = None

    max = 0.0
    max_len = 0

    possible_doctors = []
    for i in range(len(doctor_types)):
        words = doctor_types[i].split()
        word_len = len(words)
        matched_count = 0
        for word in words:
            if word in filt:
                matched_count += 1

        match_ratio = float(matched_count) / float(word_len)
        if match_ratio != 0:
            possible_doctors.append(doctor_types[i])
        else:
            continue

        if match_ratio == 1 and max == 1 and word_len > max_len:
            user_type = doctor_types[i]
            max_len = word_len

        if match_ratio > max:
            max = match_ratio
            max_len = word_len
            user_type = doctor_types[i]

    if max == 1:
        return [user_type]

    elif max >= 0.5:
        return possible_doctors

    else:
        diff = []
        for i in range(len(doctor_types)):
            # sequence = difflib.SequenceMatcher(isjunk=None, a=filt, b=symptoms[i])
            # diff = sequence.ratio()
            diff.append(fuzz.ratio(filt, doctor_types[i]))
            possible_doctors = sorted(range(len(diff)), key=lambda i: diff[i])[-5:]
        user_type = [doctor_types[possible_doctors[i]] for i in range(len(possible_doctors))]
        return user_type



disease_symptom = pd.read_csv('Datasets/disease_symptom.csv', engine='python')
diseases = disease_symptom['Disease'].unique()
symptoms = disease_symptom['Symptom'].unique()

diseases = np.array(diseases, dtype='str')
symptoms = np.array(symptoms, dtype='str')

def getSymptom(text):
    sent = sent_tokenize(text)
    filt_list = stopWords(text)
    filt = ''
    for f in filt_list:
        filt += str(f)+' '

    user_symptom = None

    max = 0.0
    max_len = 0

    possible_symptoms = []
    for i in range(symptoms.size):
        symp_words = symptoms[i].split()
        word_len = len(symp_words)
        matched_count = 0
        for word in symp_words:
            if word in filt:
                matched_count += 1

        match_ratio = float(matched_count) / float(word_len)
        if match_ratio != 0:
            possible_symptoms.append(symptoms[i])
        else:
            continue

        if match_ratio == 1 and max == 1 and word_len > max_len:
            user_symptom = symptoms[i]
            max_len = word_len

        if match_ratio > max:
            max = match_ratio
            max_len = word_len
            user_symptom = symptoms[i]

    if max == 1:
        return [user_symptom]

    elif max >= 0.5:
        return possible_symptoms

    else:
        diff = []
        for i in range(symptoms.size):
            # sequence = difflib.SequenceMatcher(isjunk=None, a=filt, b=symptoms[i])
            # diff = sequence.ratio()
            diff.append(fuzz.ratio(filt, symptoms[i]))
            possible_symptoms = sorted(range(len(diff)), key=lambda i: diff[i])[-5:]
        user_symptom = [symptoms[possible_symptoms[i]] for i in range(len(possible_symptoms))]
        return user_symptom
