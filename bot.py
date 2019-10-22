import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import random
import difflib
import copy
from fuzzywuzzy import fuzz


#greeting file
gr = pd.read_csv('Greeting Dataset.csv', engine='python')
gr = np.array(gr)
gd = gr[:,0]

#thankyou file
tu = pd.read_csv('ThankYou.csv', engine='python')
tu = np.array(tu)
td = gr[:,0]

#welcome file
wc = pd.read_csv('Welcome Dataset.csv', engine='python')
wc = np.array(wc)
wd = wc[:,0]

#age file
ag = pd.read_csv('AGE Dataset.csv', engine='python')
ag = np.array(ag)
ad = ag[:,0]

#bye file
by = pd.read_csv('BYE Dataset.csv', engine='python')
by = np.array(by)
bd = by[:,0]

#name file
nm = pd.read_csv('Name Dataset.csv', engine='python')
nm = np.array(nm)
nd = nm[:,0]

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


def getName(text):
    #text is a/many sentence
    #takes the user response and returns name of the user
    filtered = stopWords(text)
    stemmed = stemming(filtered)
##    print("stemmed",stemmed)
    tag = nltk.pos_tag(stemmed)
    #print(tag)
    noun=[]
    for i in range(len(tag)):
##        print(tag[i][1])
        if ((str(tag[i][1])=='NN' or str(tag[i][1])=='NNP') and str(tag[i][0])!='name'):
            noun.append(tag[i][0])
##    print(noun)
##    chunkGram = r"""Chunk: {<NN+>*}  """
##    chunkParser = nltk.RegexpParser(chunkGram)
##    chunked = chunkParser.parse(tag)
##    print(chunked)
##    for i in chunked:
##        if i != ('name', 'NN'):
##            name = i
##            print('i=',i[0])
##
##    print(name[0])
    return noun

def greet():
    k = random.randint(0,50)
    print(gd[k%11])

def askName():
    k = random.randint(0,50)
    print(nd[k%7])
    inp = input()
    return inp

def askAge():
    k = random.randint(0,50)
    print(ad[k%7])
    inp = input()
    return inp

def getAge(text):
    #text is a sentence(string)
    #expected output: age in number
    filtered = stopWords(text)
    for i in filtered:
        try:
            age = int(i)
        except Exception as e:
            continue
    return age

def askGender():
    print('Are you a Male or a Female?')
    inp = input()
    return inp

def sorry():
    print('I\'m sorry I could not understand that. Let\'s try again.')

def getGender(text):
    #text is a sentence(string)
    #expected output: 'Male' or 'Female'
    filtered = stopWords(text)
    flag=0
    for i in filtered:
        if i.lower()=='male' or i.lower()=='female':
            gender = i
            flag=1
    if flag!=1:
        return 0
    else:
        return gender

def getEmail():
    inp = input()
##    sent = sent_tokenize(input)
##    words = word_tokenize(inp)
##    for i in words:
##        if '@' in i:
##            email = i
    #tokenizing not working :(
    return inp

def smokeAndAlc():
    print('Do you smoke?')
    inp1 = input()
    res1=0
    for i in inp1:
        stem = stemming(i)
        if 'yes' in stem or 'yea' in stem or 'yeah' in stem:
            res1=1
    print('Do you consume Alcohol?')
    inp2 = input()
    res2=0
    for i in inp2:
        stem = stemming(i)
        if 'yes' in stem or 'yea' in stem or 'yeah' in stem:
            res2=1
    return (res1*10)+res2

def getZip():
    inp = input()
    #tok = word_tokenize()
    code=0
    for i in inp:
        try:
            code =code*10+int(i)
        except Exception as e:
            continue
    return code

def extDisease():
    print('Before we ask you your symptoms, we would like to know your health status.')
    print('If yout have any existing Medical Conditions or Problems, please provide them here.')
    print('If you dont, you can reply with a \'no\'')
    inp = input()
    tok = word_tokenize(inp)
    fl=0
    for i in tok:
        stem = stemming(i)
        for i in tok:
            if 'no' in tok:
                fl=1
                break
    if fl==0:
            return inp
    else:
        return 'Nothing Sevre'



doctor_types = [line for line in open('Doctor Types.csv', 'r').readlines()[1:]]
def consult():
    print('Who do you want to consult?')
    inp = input()
    sent = sent_tokenize(inp)
    filt_list = stopWords(inp)
    filt = ''
    for f in filt_list:
        filt += str(f)+' '

    for i in range(len(filt)):
        if filt[i] == 'hospital' and i > 0:
            hospital_name = filt[i-1]
            return hospital_name

    user_type = None

    max = 0.0
    max_len = 0

    possible_doctors = []
    for i in range(doctor_types.size):
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
        for i in range(doctor_types.size):
            # sequence = difflib.SequenceMatcher(isjunk=None, a=filt, b=symptoms[i])
            # diff = sequence.ratio()
            diff.append(fuzz.ratio(filt, doctor_types[i]))
            possible_doctors = sorted(range(len(diff)), key=lambda i: diff[i])[-5:]
        user_type = [doctor_types[possible_doctors[i]] for i in range(len(possible_doctors))]
        return user_type



disease_symptom = pd.read_csv('dataset_clean1.csv', engine='python')
diseases = disease_symptom['Disease'].unique()
symptoms = disease_symptom['Symptom'].unique()

diseases = np.array(diseases, dtype='str')
symptoms = np.array(symptoms, dtype='str')
print(len(diseases))
print(len(symptoms))

def getSymptom():
    print('Please tell me about your symptoms')
    inp = input()
    sent = sent_tokenize(inp)
    filt_list = stopWords(inp)
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


def diagnose():
    symptoms = []
    symp = getSymptom()
    for s in symp:
        symptoms.append(s)
    print('Okay so far you have provided that you have the following symptoms:')
    for i in range(len(symptoms)):
        print(str(i+1)+'. '+symptoms[i])

    print('Are there any other symptoms that you want ot explain? You can reply to this with a "no".')
    inp = input()
    tok = word_tokenize(inp)
    fl=0
    for i in tok:
        stem = stemming(i)
        for i in tok:
            if 'no' in tok:
                fl=1
                break
    if fl==0:
        symp = getSymptom()
        for s in symp:
            symptoms.append(s)

    possible_diseases = copy.deepcopy(disease_symptom)
    for s in symptoms:
        possible_diseases = possible_diseases[possible_diseases['Symptom'] == s]

    possible_diseases = possible_diseases.sort_values(by='Weight', ascending=False)

    print(len(possible_diseases['Disease'].unique()))
    while len(possible_diseases['Disease'].unique()) > 4:
        print(len(possible_diseases['Disease'].unique()))
        symptom_options = []
        for disease in possible_diseases['Disease'].unique():
            symp_in_this_dis = disease_symptom[possible_diseases['Disease'] == disease]['Symptom']
            s = random.choice(symp_in_this_dis)
            if s in symptom:
                s = random.choice(symp_in_this_dis)
                if s in symptom:
                    s = random.choice(symp_in_this_dis)
            symptom_options.append(s)

        # take symptoms from user in the list new_symptoms (ask max 5 options)
        for s in new_symptoms:
            possible_diseases = possible_diseases[possible_diseases['Symptom'] == s]

    return possible_diseases['Disease'].unique()


#Starting the conversation
greet()
print('I\'m DrBot, your personal health assistant.')
print("I can help you find a doctor or I can diagnose with a simple symptom assisment.")
ufName = askName()
name = getName(ufName)
# ufAge = askAge()
# age = getAge(ufAge)
# ufGender = askGender()
# gender = getGender(ufGender)
# while gender==0:
#     sorry()
#     ufGender = askGender()
#     gender = getGender(ufGender)
# print('To help you keep a record of your symptoms and enable us to provide you with better assistance, we would like you to provide us with your email. This is mandatory.')
# email = getEmail()
# print('Your ZipCode would enable us to provide personalised suggestions for hospitals. This is mandatory.')
# zip = getZip()
# sa=smokeAndAlc()
# #sa = (smoke*10)+alc
# existingDiseases = extDisease()

##print('name = {}, age = {}'.format(name[0],age))
#print Everything
##print(name, age, gender, email, zip, sa, existingDiseases)
print('Okay {} '.format(name[0]))
disease = diagnose()
print(disease)
