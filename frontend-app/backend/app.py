import os

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


API_URL = os.getenv("API_URL", "http://localhost:8000")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        response = requests.post(f"{API_URL}/predict", json=data, timeout=30)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        files = {'file': (file.filename, file.stream, 'text/csv')}
        response = requests.post(f"{API_URL}/predict/csv", files=files, timeout=60)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
