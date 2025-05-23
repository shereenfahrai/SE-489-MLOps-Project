# fake_news_detection/predict_model.py
"""
Prediction script for LSTM fake news classifier.

Loads a trained Keras model and tokenizer, applies preprocessing to input text,
generates predictions, evaluates results if ground truth is available,
and saves a confusion matrix plot.
"""

import logging
import os
import pickle
import re
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and tokenizer
MODEL_PATH = os.path.join(BASE_DIR, "models/lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models/tokenizer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/predict.csv")
# RESULT_PATH = "../data/processed/predicted_results.csv"
CONF_MATRIX_PATH = os.path.join(BASE_DIR, "fake_news_detection/reports/figures/predict_confusion_matrix.png")

# nltk.data.path.append("./data/raw/")
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

maxlen = 150  # same as training


# Preprocessing
def safe_word_tokenize(text: str) -> List[str]:
    """
    Tokenizes text into words using NLTK's PunktSentenceTokenizer and TreebankWordTokenizer.

    Args:
        text (str): Input string.

    Returns:
        List[str]: List of word tokens.
    """
    sentences = sent_tokenizer.tokenize(text)
    return [token for sent in sentences for token in word_tokenizer.tokenize(sent)]


def process_text(text: str) -> List[str]:
    """
    Cleans and preprocesses input text:
    - Removes non-alphabetic characters
    - Lowercases text
    - Tokenizes and lemmatizes
    - Removes stopwords and short words
    - Deduplicates tokens while preserving order

    Args:
        text (str): Raw input text.

    Returns:
        List[str]: Cleaned and deduplicated word list.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    words = safe_word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 3]
    _, idx = np.unique(words, return_index=True)
    return [words[i] for i in sorted(idx)]


def plot_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], save_path: str) -> None:
    """
    Plots and saves a confusion matrix as a heatmap.

    Args:
        y_true (Sequence[int]): Ground truth labels.
        y_pred (Sequence[int]): Predicted labels.
        save_path (str): Path to save the plot.
    """
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def predict() -> None:
    """
    Loads LSTM model and tokenizer, preprocesses the input data,
    makes predictions, and evaluates results (if labels are present).

    Saves confusion matrix to file if applicable.
    """
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    df = pd.read_csv(DATA_PATH)
    df = df[df["text"].apply(lambda x: isinstance(x, str))]

    texts = df["text"].tolist()
    cleaned_texts = [process_text(text) for text in texts]

    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=maxlen)
    predictions = model.predict(padded)
    predicted_labels = np.argmax(predictions, axis=1)

    df["predicted_label"] = predicted_labels

    # Evaluate
    if "label" in df.columns:
        y_true = df["label"].tolist()
        acc = accuracy_score(y_true, predicted_labels)
        precision = precision_score(y_true, predicted_labels)
        recall = recall_score(y_true, predicted_labels)
        f1 = f1_score(y_true, predicted_labels)

        logger.info(f"Prediction Accuracy: {acc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        plot_confusion_matrix(y_true, predicted_labels, CONF_MATRIX_PATH)
        logger.info(f"Confusion matrix saved to {CONF_MATRIX_PATH}")


if __name__ == "__main__":
    predict()
