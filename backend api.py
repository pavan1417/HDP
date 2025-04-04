from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(HDP)
CORS(app)  # To handle CORS issues

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from the backend!"})

@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.json
    return jsonify({"received_data": data})

if _name_ == '_main_':
    app.run(debug=True, port=5000)
