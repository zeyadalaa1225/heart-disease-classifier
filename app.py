from flask import Flask, request, render_template
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# Load models
ml_model = joblib.load('ml_model.pkl')
knn_model = joblib.load('knn_model.pkl')
dp_model = tf.keras.models.load_model('dp_model.h5')

# Initialize PolynomialFeatures with degree=3
poly = PolynomialFeatures(degree=3, include_bias=True)

# Preprocessing function
def preprocess_input(data):
    data = np.array(data).reshape(1, -1)  # Reshape input
    poly_data = poly.fit_transform(data)   # Apply polynomial transformation
    return poly_data

@app.route('/')
def home():
    # Render the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cholesterol']),
        float(request.form['fastingBS']),
        float(request.form['maxHR']),
        float(request.form['oldpeak']),
        
        1. if request.form['chestPainType'] == 'ASY' else 0,
        1. if request.form['chestPainType'] == 'ATA' else 0,
        float(request.form['exerciseAngina_N']),
        float(request.form['st_slope_up'])
    ]
    print("Input Data:", input_data)
    # Preprocess the input (with polynomial transformation)
    processed_data = preprocess_input(input_data)
    
    # Make predictions with each model
    ml_prediction = ml_model.predict(processed_data)
    knn_prediction = knn_model.predict(processed_data)
    dp_prediction = dp_model.predict(processed_data)

    # Combine results or provide individual ones
    response = {
        'ml_prediction': "3ayan " if bool(ml_prediction[0]) else "mesh 3ayan ",  # True/False
        'knn_prediction':"3ayan " if  bool(knn_prediction[0]) else "mesh 3ayan ",  # True/False
        'dp_prediction': "3ayan " if bool(dp_prediction[0] > 0.5)else "mesh 3ayan "  # Assuming binary classification
    }

    return render_template('result.html', ml_pred=response['ml_prediction'],
                           knn_pred=response['knn_prediction'],
                           dp_pred=response['dp_prediction'])

import webbrowser
import threading

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Open the browser in a separate thread to prevent blocking the Flask server
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
