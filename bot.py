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

app = Flask(__name__)

push_data = []
message_id = 0
asked_name = False
name = ''

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


#Starting the conversation
greet()
write_chat('I\'m DrBot, your personal health assistant.')
askName()


def generate_response(user_text):

    intent = classify_intent.get_intent(user_text)
    print(intent)
    data = None

    if intent == 'Consult':
        type_of_doctors = bot_logic.consult(user_text)
        if len(type_of_doctors) == 1:
            type_of_doctors[0] = type_of_doctors[0].strip()
            type_of_doctors[0] = type_of_doctors[0].replace('\n','')
            type_of_doctors[0] = type_of_doctors[0].title()
            if type_of_doctors[0] == 'Doctor':
                type_of_doctors[0] = 'General Physician'
            type_of_doctors[0] = type_of_doctors[0].replace(' ','%20')
            print('*'+type_of_doctors[0]+'*')
            scrapped_data = scrapper.scrap_data('Hyderabad', type_of_doctors[0], 'Doctor', 0)
            data = {'message': scrapped_data, 'msg_type': 'display_cards'}
        else:
            write_chat('Which of the following type of Doctor do you mean?')
            msg = [{'headline':'', 'body':type_of_doctors[i]} for i in range(len(type_of_doctors))]
            data = {'message': msg, 'msg_type': 'ask_options', 'id': message_id}
        # scrap data and show doctors
    #
    # elif intent == 'Symptom/Disease':
    #     disease = bot_logic.diagnose(user_text)
    #     return disease
        # tell about the disease

    # else: # Out of Domain
    #     reply = OOD_handler.get_reply(user_text)
    #     write_chat(reply)
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

@app.route('/get_reply', methods=['GET','POST'])
def get_reply():
    global push_data, message_id, asked_name, name
    # chat_msg, display_cards
    print('REQUEST:'+str(request.form.to_dict()))
    received_data = request.form.to_dict()['data']
    message_id += 1

    if asked_name:
        name = getName(received_data)
        reply = ['Okay ' + name[0] + ', I can help you find a doctor or I can diagnose with a simple symptom assisment.']
        reply = {'message': reply, 'msg_type': 'chat_msg'}
        asked_name = False

    else:
        # scrapped_data = [{'headline':'This is first headline', 'body': 'This is first body'}, {'headline':'This is second headline', 'body': 'This is second body'}]
        # data = {'message': scrapped_data, 'msg_type': 'ask_options', 'id': message_id}
        # print('About to push again...')
        # push_data.append('Push data again...')
        reply = generate_response(received_data)
        print('Message sent by client: ', received_data)
    # =============================================================================
    #     scrapped_data = scrap_data('Bangalore', 'asdfasdfsf', 'Doctor Name', 0)
    #     data = {'message': scrapped_data, 'msg_type': 'display_cards'}
    #     print(scrapped_data)
    # =============================================================================
        # reply = [reply]
        # reply = {'message': reply, 'msg_type': 'chat_msg'}

    # =============================================================================
    #     scrapped_data = scrap_data('Bangalore', 'Psychiatrist', 'Doctor', 0)
    #     data = {'message': scrapped_data, 'msg_type': 'display_cards'}
    #
    #     scrapped_data = scrap_data('Bangalore', 'Apollo', 'Hospital', 0)
    #     data = {'message': scrapped_data, 'msg_type': 'display_cards'}
    #
    #     scrapped_data = scrap_data('Bangalore', 'Apollo', 'Clinic', 0)
    #     data = {'message': scrapped_data, 'msg_type': 'display_cards'}
    # =============================================================================

    return make_response(jsonify(reply), 201)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config["REDIS_URL"] = "redis://localhost"
    app.register_blueprint(sse, url_prefix='/stream')
