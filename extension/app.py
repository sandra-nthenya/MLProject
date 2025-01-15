import re
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask_limiter import Limiter
import logging
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Allow requests only from the Chrome extension
#CORS(app, origins=["chrome-extension://chgpfphcckckkpiaaicepnbmkpkobahk"])

limiter = Limiter(app)

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model("model.h5")

# Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Sentiment Prediction Function
def predict_classes(text):
    cleaned_text = clean_text(text)

    # Tokenize and pad the text
    sample_sequence = tokenizer.texts_to_sequences([cleaned_text])
    sample_padded = pad_sequences(sample_sequence, maxlen=100)

    # Predict the class using the model
    sample_prediction = model.predict(sample_padded)
    predicted_class = np.argmax(sample_prediction, axis=1)[0]


    class_names = ['Negative', 'Neutral', 'Positive']
    if predicted_class < 0 or predicted_class >= len(class_names):
        return "Unknown"  # Return a fallback value for unexpected indices

    return class_names[predicted_class]

# Sarcasm Prediction Function
def predict_sarcasm(text):
    if not isinstance(text, str):
        return "Text entered is not a string"

    # Normalize text
    text = text.lower().strip()

    # Sarcasm-related patterns
    sarcasm_patterns = [
        r'\byeah,? right\b',
        r'\btotally\b',
        r'\bsure\b',
        r'\bof course\b',
        r'\bas if\b',
        r'\bgreat,? just what i needed\b',
        r'\blove that for me\b',
        r'\bwhat a surprise\b',
        r'\bthanks a lot\b',
    ]

    # Punctuation or formatting cues
    punctuation_patterns = [
        r'!{2,}',  # Multiple exclamation marks
        r'\.{3,}',  # Ellipses
        r'\b(not|never|no way) (really|totally|at all)\b',  # Contrasting sentiments
    ]

    # Check sarcasm patterns
    is_sarcastic = any(
        re.search(pattern, text) for pattern in sarcasm_patterns + punctuation_patterns
    )

    return "Sarcastic" if is_sarcastic else "Not Sarcastic"

# API Endpoint
@limiter.limit("5 per minute", key_func=get_remote_address)
@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    logging.info(f"Request from {request.remote_addr} at {request.method} {request.url}")
    data = request.get_json()

    # Validate input
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]

    # Get predictions
    sentiment = predict_classes(text)
    sarcasm = predict_sarcasm(text)

    # Return results as JSON
    return jsonify({"sentiment": sentiment, "sarcasm": sarcasm}), 200

@app.errorhandler(429)
def handle_rate_limit_error(error):
    return jsonify({"error": "Too many requests, please try again later."}), 429

if __name__ == "__main__":
    app.run(debug=True)
