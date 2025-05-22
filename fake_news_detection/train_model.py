"""
Training script for LSTM-based fake news classifier.

This script:
- Loads cleaned text data
- Applies tokenization and preprocessing
- Trains an LSTM model using Keras
- Logs training process with MLflow
- Saves model, tokenizer, and performance metrics
"""

import os
import pickle
import re
from typing import List, Tuple, Union

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd

# Tokenization setup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from fake_news_detection.models.model import build_lstm_model
from fake_news_detection.visualizations.visualize import plot_accuracy_loss, plot_confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MLflow setup
mlflow.set_experiment("fake-news-lstm")
print("MLflow experiment:", mlflow.get_experiment_by_name("fake-news-lstm"))

# nltk.data.path.append("./data/raw/")
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

# Download NLTK resources
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))


def safe_word_tokenize(text: str) -> List[str]:
    """
    Tokenizes input text into word-level tokens.

    Args:
        text (str): Raw input text.

    Returns:
        List[str]: List of word tokens.
    """
    sentences = sent_tokenizer.tokenize(text)
    return [token for sent in sentences for token in word_tokenizer.tokenize(sent)]


def process_text(text: str) -> List[str]:
    """
    Cleans and processes raw text:
    - Removes non-alphabetic characters
    - Lowercases
    - Tokenizes and lemmatizes
    - Removes stopwords and short tokens
    - Deduplicates while preserving order

    Args:
        text (str): Input text.

    Returns:
        List[str]: Preprocessed and deduplicated word tokens.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    words = safe_word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 3]
    _, idx = np.unique(words, return_index=True)
    return [words[i] for i in sorted(idx)]


def load_cleaned_data() -> Tuple[List[str], List[Union[str, int]]]:
    """
    Loads cleaned training data from CSV file.

    Returns:
        Tuple[List[str], List[Union[str, int]]]: List of texts and corresponding labels.
    """
    data_path = os.path.join(BASE_DIR, "data/processed/train.csv")
    df = pd.read_csv(data_path)
    df = df[df["text"].apply(lambda x: isinstance(x, str))]
    return df["text"].tolist(), df["label"].tolist()


def train() -> None:
    """
    Main training routine:
    - Loads data and applies preprocessing
    - Splits into training and test sets
    - Trains LSTM model and logs parameters/metrics with MLflow
    - Saves trained model and tokenizer
    - Generates accuracy/loss and confusion matrix visualizations
    """

    texts, y = load_cleaned_data()
    cleaned_texts = [process_text(text) for text in texts]

    X_train, X_test, y_train, y_test = train_test_split(cleaned_texts, y, test_size=0.2, random_state=42)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    maxlen = 150
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

    label_encoder = LabelEncoder()
    y_train_enc = to_categorical(label_encoder.fit_transform(y_train))
    y_test_enc = to_categorical(label_encoder.transform(y_test))

    with mlflow.start_run():
        model = build_lstm_model(
            vocab_size=vocab_size,
            maxlen=maxlen,
            embed_dim=100,
            lstm_units=150,
            dropout_rate=0.5,
            learning_rate=0.0001,
        )

        mlflow.log_param("epochs", 15)
        mlflow.log_param("maxlen", maxlen)
        mlflow.log_param("embedding_dim", 100)
        mlflow.log_param("lstm_units", 150)
        mlflow.log_param("dropout", 0.5)
        mlflow.log_param("learning_rate", 0.0001)

        history = model.fit(
            X_train_pad,
            y_train_enc,
            epochs=15,
            validation_data=(X_test_pad, y_test_enc),
            verbose=1,
        )

        loss, accuracy = model.evaluate(X_test_pad, y_test_enc)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Predict & compute precision / recall / F1
        y_pred_probs = model.predict(X_test_pad)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test_enc, axis=1)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        model.save(os.path.join(BASE_DIR, "models/lstm_model.h5"))
        mlflow.tensorflow.log_model(model, artifact_path="model")

        with open(os.path.join(BASE_DIR, "models/tokenizer.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)

        plot_accuracy_loss(history)
        plot_confusion_matrix(model, X_test_pad, y_test_enc)


if __name__ == "__main__":
    train()
