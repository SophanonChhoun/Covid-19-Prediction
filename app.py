import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import json
import jsonpickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# model = pickle.load(open('./models/model.pkl', 'rb'))


@app.route("/", methods=['GET'])
def index():
    return "Hello World."


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    columns = [
        "Breathing Problem", "Fever", "Dry Cough", "Sore throat",
        "Running Nose", "Asthma", "Chronic Lung Disease", "Headache",
        "Heart Disease", "Diabetes", "Hyper Tension", "Fatigue ",
        "Gastrointestinal ", "Abroad travel", "Contact with COVID Patient",
        "Attended Large Gathering", "Visited Public Exposed Places",
        "Family working in Public Exposed Places", "Wearing Masks",
        "Sanitization from Market"
    ]
    df = pd.DataFrame([data])
    model = pickle.load(open('./models/model.pkl', 'rb'))
    output = model.predict(df)
    if output[0] == 1:
        return app.response_class(
            response=json.dumps({
                "message": "You have a Covid 19.",
                "precaution": [],
                "has_diabetes": False,
                'has_heart_disease': False,
                "has_tuberculosis": False
            }),
            status=200,
            mimetype='application/json'
        )
    else:
        return app.response_class(
            response=json.dumps({
                "message": "You don't have a Covid 19.",
                "precaution": [],
                "has_diabetes": False,
                'has_heart_disease': False,
                "has_tuberculosis": False
            }),
            status=200,
            mimetype='application/json'
        )


if __name__ == '__main__':
    app.run(debug=True)
