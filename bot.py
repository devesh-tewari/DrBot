import bot_logic
import classify_intent
import OOD_handler
import scrapper
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import random
from flask import Flask, render_template, request, Response
from flask import jsonify, make_response
import time
import copy


app = Flask(__name__)

push_data = []
message_id = 0
asked_name = False
name = ''
asked_location = False
location = ''
user_symptoms = []
message_id_intent = {}
message_id_options = {}
first_symptom_done = False
asked_symptom = False

def write_chat(text):
    global push_data
    push_data.append(text)

#greeting file
gr = pd.read_csv('Datasets/Greeting Dataset.csv', engine='python')
gr = np.array(gr)
gd = gr[:,0]

#thankyou file
tu = pd.read_csv('Datasets/ThankYou.csv', engine='python')
tu = np.array(tu)
td = gr[:,0]

#welcome file
wc = pd.read_csv('Datasets/Welcome Dataset.csv', engine='python')
wc = np.array(wc)
wd = wc[:,0]

#bye file
by = pd.read_csv('Datasets/BYE Dataset.csv', engine='python')
by = np.array(by)
bd = by[:,0]

#name file
nm = pd.read_csv('Datasets/Name Dataset.csv', engine='python')
nm = np.array(nm)
nd = nm[:,0]

#name file
ln = pd.read_csv('Datasets/Location.csv', engine='python')
ln = np.array(ln)
ld = ln[:,0]

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
    tag = nltk.pos_tag(stemmed)
    noun=[]
    for i in range(len(tag)):
        if ((str(tag[i][1])=='NN' or str(tag[i][1])=='NNP') and str(tag[i][0])!='name'):
            noun.append(tag[i][0])
    return noun

def greet():
    k = random.randint(0,50)
    write_chat(gd[k%11])

def askName():
    global asked_name
    asked_name = True
    k = random.randint(0,50)
    write_chat(nd[k%7])

def askLocation():
    global asked_location
    asked_location = True
    k = random.randint(0,50)
    print(ld[k%5])
    write_chat(ld[k%5])


disease_symptom = pd.read_csv('Datasets/disease_symptom.csv', engine='python')

def generate_response(user_text, options_flag=False):
    global location, user_symptoms, message_id, message_id_intent, message_id_options, first_symptom_done, asked_symptom, disease_symptom

    intent = classify_intent.get_intent(user_text)
    print(intent)
    message_id_intent[message_id] = intent
    data = None

    if options_flag:
        options_indices = user_text.split(',')
        option_msg_id = None
        options = []
        for i in range(len(options_indices)):
            option_msg_id = int(options_indices[i].split(':')[0])
            print(message_id_options)
            options.append( message_id_options[option_msg_id][int(options_indices[i].split(':')[1])] )

        if message_id_intent[option_msg_id] == 'Consult':
            type_of_doctor = options[0]
            type_of_doctor = type_of_doctor.replace('\n','')
            type_of_doctor = type_of_doctor.title()
            if type_of_doctor == 'Doctor':
                type_of_doctor = 'General Physician'
            type_of_doctor = type_of_doctor.replace(' ','%20')
            print('*'+type_of_doctor+'*')
            scrapped_data = scrapper.scrap_data(location, type_of_doctor, 'Doctor', 0)
            data = {'message': scrapped_data, 'msg_type': 'display_cards'}

        elif message_id_intent[option_msg_id] == 'Symptom/Disease':
            for option in options:
                user_symptoms.append(option)

            if not first_symptom_done:
                reply = 'Please tell any other symptoms that you want to explain? You can reply to this with a "no".'
                data = {'message': [reply], 'msg_type': 'chat_msg'}
                asked_symptom = True

            else:
                possible_diseases = copy.deepcopy(disease_symptom)
                for s in user_symptoms:
                    possible_diseases = possible_diseases[possible_diseases['Symptom'] == s]

                possible_diseases = possible_diseases.sort_values(by='Weight', ascending=False)

                if len(possible_diseases['Disease'].unique()) > 4:
                    print(len(possible_diseases['Disease'].unique()))
                    symptom_options = []
                    for disease in possible_diseases['Disease'].unique():
                        symp_in_this_dis = disease_symptom[disease_symptom['Disease'] == disease]['Symptom']
                        s = random.choice(symp_in_this_dis)
                        if s in user_symptoms:
                            s = random.choice(symp_in_this_dis)
                            if s in user_symptoms:
                                s = random.choice(symp_in_this_dis)
                        symptom_options.append(s)

                    write_chat('Please mark the symptoms that you have')
                    time.sleep(2)
                    msg = [{'headline':'', 'body':symptom_options[i]} for i in range(len(symptom_options))]
                    data = {'message': msg, 'msg_type': 'ask_options', 'id': message_id}

                else:
                    if len(possible_diseases['Disease'].unique()) == 0:
                        reply = 'Sorry! There is no disease in my knowledge with the symptoms you provided.'
                    elif len(possible_diseases['Disease'].unique()) == 1:
                        reply = 'Most likely you are suffing from '+possible_diseases[0]
                    else:
                        reply = 'Possible diseases are : '
                        for dis in possible_diseases['Disease'].unique():
                            reply += dis + ' and '
                        reply = reply[:-5]
                    data = {'message': [reply], 'msg_type': 'chat_msg'}

            first_symptom_done = True
        return data

    if intent == 'Consult':
        type_of_doctors = bot_logic.consult(user_text)
        if len(type_of_doctors) == 1:
            type_of_doctors[0] = type_of_doctors[0].replace('\n','')
            type_of_doctors[0] = type_of_doctors[0].title()
            if type_of_doctors[0] == 'Doctor':
                type_of_doctors[0] = 'General Physician'
            type_of_doctors[0] = type_of_doctors[0].replace(' ','%20')
            print('*'+type_of_doctors[0]+'*')
            scrapped_data = scrapper.scrap_data(location, type_of_doctors[0], 'Doctor', 0)
            data = {'message': scrapped_data, 'msg_type': 'display_cards'}
        else:
            write_chat('Which of the these Doctor type did you mean?')
            time.sleep(2)
            message_id_options[message_id] = type_of_doctors
            msg = [{'headline':'', 'body':type_of_doctors[i]} for i in range(len(type_of_doctors))]
            data = {'message': msg, 'msg_type': 'ask_options', 'id': message_id}

    elif intent == 'Symptom/Disease':
        symp = bot_logic.getSymptom(user_text)
        if len(symp) == 1:
            user_symptoms.append(symp[0])
            if not first_symptom_done:
                reply = 'Please tell any other symptoms that you want to explain? You can reply to this with a "no".'
                data = {'message': [reply], 'msg_type': 'chat_msg'}
                asked_symptom = True

            else:
                possible_diseases = copy.deepcopy(disease_symptom)
                for s in user_symptoms:
                    possible_diseases = possible_diseases[possible_diseases['Symptom'] == s]

                possible_diseases = possible_diseases.sort_values(by='Weight', ascending=False)

                if len(possible_diseases['Disease'].unique()) > 4:
                    print(len(possible_diseases['Disease'].unique()))
                    symptom_options = []
                    for disease in possible_diseases['Disease'].unique():
                        symp_in_this_dis = disease_symptom[disease_symptom['Disease'] == disease]['Symptom']
                        s = random.choice(symp_in_this_dis)
                        if s in user_symptoms:
                            s = random.choice(symp_in_this_dis)
                            if s in user_symptoms:
                                s = random.choice(symp_in_this_dis)
                        symptom_options.append(s)

                    write_chat('Please mark the symptoms that you have')
                    time.sleep(2)
                    msg = [{'headline':'', 'body':symptom_options[i]} for i in range(len(symptom_options))]
                    data = {'message': msg, 'msg_type': 'ask_options', 'id': message_id}

                else:
                    if len(possible_diseases['Disease'].unique()) == 0:
                        reply = 'Sorry! There is no disease in my knowledge with the symptoms you provided.'
                    elif len(possible_diseases['Disease'].unique()) == 1:
                        reply = 'Most likely you are suffing from '+possible_diseases[0]
                    else:
                        reply = 'Possible diseases are : '
                        for dis in possible_diseases['Disease'].unique():
                            reply += dis + ' and '
                        reply = reply[:-5]
                    data = {'message': [reply], 'msg_type': 'chat_msg'}

        else:
            msg = [{'headline':'', 'body':symp[i]} for i in range(len(symp))]
            data = {'message': msg, 'msg_type': 'ask_options', 'id': message_id}
            message_id_options[message_id] = symp
            write_chat('Which of the these symptoms did you mean?')
            time.sleep(2)

        first_symptom_done = True


    else: # Out of Domain
        reply = OOD_handler.get_reply(user_text)
        data = {'message': [reply], 'msg_type': 'chat_msg'}

    print(data)
    return data

@app.route('/stream')
def stream():
    global push_data
    def eventStream():
        global push_data
        while True:
            yield 'data: {}\n\n'.format(push_data[0])
            push_data = push_data[1:]
            break
    while(len(push_data) != 0):
        time.sleep(0.5)
        return Response(eventStream(), mimetype="text/event-stream")
    return Response('', mimetype="text/event-stream")


#Starting the conversation
greet()
write_chat('I\'m DrBot, your personal health assistant.')
askName()


@app.route('/get_reply', methods=['GET','POST'])
def get_reply():
    global push_data, message_id, asked_name, name, asked_location, location
    # chat_msg, display_cards
    print('REQUEST:'+str(request.form.to_dict()))

    received_data = request.form.to_dict()['data']
    reply = ''

    if request.form.to_dict()['request_type'] == 'indices':
        reply = generate_response(received_data, options_flag=True)

    elif asked_name:
        name = getName(received_data)[0].title()
        reply = ['Okay ' + name + ', I just have one more question.']
        reply = {'message': reply, 'msg_type': 'chat_msg'}
        asked_name = False
        askLocation()

    elif asked_location:
        location = getName(received_data)[0].title()
        location = location.replace(' ','%20')
        reply = ['Got it. I can help you find a doctor or I can diagnose with a simple symptom assisment.']
        reply = {'message': reply, 'msg_type': 'chat_msg'}
        asked_location = False

    else:
        reply = generate_response(received_data)
        print('Message sent by client: ', received_data)

    message_id += 1
    return make_response(jsonify(reply), 201)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port='8000')
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config["REDIS_URL"] = "redis://localhost"
    app.register_blueprint(sse, url_prefix='/stream')
