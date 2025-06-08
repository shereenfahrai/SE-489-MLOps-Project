import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fake_news_detection.predict_model import process_text
from fake_news_detection.model_loader import model, tokenizer

MAXLEN = 150

def predict(text):
    cleaned = process_text(text)
    sequence = tokenizer.texts_to_sequences([" ".join(cleaned)])
    padded = pad_sequences(sequence, maxlen=MAXLEN)
    prediction = model.predict(padded)[0][0]
    return "Fake" if prediction >= 0.5 else "Real"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=6, label="Paste Full News Article Body Text"),
    outputs=gr.Label(label="Prediction"),
    title="Fake News Detection App",
    description="Predict whether a news article is Fake or Real using an LSTM-based model."
)

iface.launch()
