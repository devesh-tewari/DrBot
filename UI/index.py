from flask import Flask, request, jsonify, render_template
import os
#import dialogflow
import requests
import json
#import pusher

app = Flask(__name__)

# initialize Pusher
# =============================================================================
# pusher_client = pusher.Pusher(
#     app_id=os.getenv('PUSHER_APP_ID'),
#     key=os.getenv('PUSHER_KEY'),
#     secret=os.getenv('PUSHER_SECRET'),
#     cluster=os.getenv('PUSHER_CLUSTER'),
#     ssl=True)
# =============================================================================

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/process',methods=['POST'])
def process():
    buf = request.form['user_input']
    resp='Current response'
    url ='#'
    try:
        if (resp.find('https') != -1):
            resp = resp.replace(" -", "-")
            resp = resp.lower()
            url = resp.split("https",1)[1]
            url="https"+url
            print(url)

    except:
        print("CRASHED")
        pass
    print("Trip Planner: "+resp)
    return render_template('index.html',user_input=buf, bot_response=resp, url_input=url)

#main body
if __name__ == "__main__":
    while(1):
	    app.run(debug=True,port=5003)