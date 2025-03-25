from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Load the model and label encoders
model = pickle.load(open('model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Expect JSON input
        features = []

        for key, value in data.items():
            if key in label_encoders:  # Apply label encoding if needed
                value = label_encoders[key].transform([value])[0]
            features.append(float(value))

        # Convert features into a NumPy array
        final_features = np.array(features).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return jsonify({'prediction': output})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
