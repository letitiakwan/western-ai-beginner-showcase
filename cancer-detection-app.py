from flask import Flask
from flask import request
from flask_cors import CORS
import cancer_detection as cd

app = Flask (__name__, static_folder='./build', static_url_path='/')
CORS(app)

app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/detect-cancer', methods=['POST'])
def detect_cancer():

    return cd.receive_data(request.data)

