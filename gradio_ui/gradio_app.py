import gradio as gr
import numpy as np
import requests
import tempfile
import pickle
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fake_news_detection.predict_model import process_text

# Set max sequence length to match training
MAXLEN = 150

# Download and load model from GCS
model_url = "https://storage.googleapis.com/mlops_fake_news/lstm_model.h5"
tokenizer_url = "https://storage.googleapis.com/mlops_fake_news/tokenizer.pkl"

# Load model
model_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
model_temp.write(requests.get(model_url).content)
model_temp.close()
model = load_model(model_temp.name)

# Load tokenizer
tokenizer_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
tokenizer_temp.write(requests.get(tokenizer_url).content)
tokenizer_temp.close()
with open(tokenizer_temp.name, "rb") as f:
    tokenizer = pickle.load(f)

# Prediction function
def predict(text):
    """
    Predict whether the input text is Fake or Real news.
    Args:
        text (str): The news article body text to classify.
    Returns:
        str: "Fake" or "Real" based on the model's prediction.  
    """
    cleaned = process_text(text)
    sequence = tokenizer.texts_to_sequences([" ".join(cleaned)])
    padded = pad_sequences(sequence, maxlen=MAXLEN)
    prediction = model.predict(padded)[0][0]
    return "Fake" if prediction >= 0.5 else "Real"

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=6, label="Paste Full News Article Body Text"),
    outputs=gr.Label(label="Prediction"),
    title="Fake News Detection App",
    description="Predict whether a news article is Fake or Real using an LSTM-based model."
)

iface.launch()
