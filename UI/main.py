from flask import Flask, render_template, request
from flask import jsonify, make_response
from scrapper import scrap_data

app = Flask(__name__)

@app.route('/get_reply', methods=['POST'])
def get_reply():
    # chat_msg, display_cards
# =============================================================================
#     scrapped_data = scrap_data('Bangalore', 'Meena', 'Doctor Name', 0)
#     data = {'message': scrapped_data, 'msg_type': 'display_cards'}
# =============================================================================
    print(request.data)
    scrapped_data = ['Reply from the bot']
    data = {'message': scrapped_data, 'msg_type': 'chat_msg'}

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

# =============================================================================
#     scrapped_data = scrap_data('Bangalore', 'Apollo', 'Clinic', 0)
#     data = {'message': scrapped_data, 'msg_type': 'display_cards'}
# =============================================================================

    return make_response(jsonify(data), 201)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
