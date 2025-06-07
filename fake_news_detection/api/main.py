from fastapi import FastAPI
from fake_news_detection.api.schemas import InputText
from fake_news_detection.api.model_loader import model, tokenizer
import tensorflow as tf
from typing import Dict  # For return type annotation

app = FastAPI()


@app.post("/predict")
def predict(input: InputText) -> Dict[str, float]:
    """
    Predict whether the given text is fake news or not.

    Args:
        input (InputText): A Pydantic model containing the input text.

    Returns:
        Dict[str, float]: A dictionary with the prediction score.
                          The value is a float between 0 and 1,
                          where closer to 1 indicates higher likelihood of fake news.
    """
    # Tokenize the input text using the loaded tokenizer
    seq = tokenizer.texts_to_sequences([input.text])

    # Pad the sequence to match the model's expected input length (e.g., 150)
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=150)

    # Make prediction using the preloaded model
    pred = model.predict(padded)[0][0]

    # Return prediction as a float in a dictionary
    return {"prediction": float(pred)}
