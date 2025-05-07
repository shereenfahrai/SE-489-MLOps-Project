# fake_news_detection/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import nltk
import mlflow
import mlflow.tensorflow
import os
import pickle

from fake_news_detection.models.model import build_lstm_model
from fake_news_detection.visualizations.visualize import plot_accuracy_loss, plot_confusion_matrix


# MLflow setup
mlflow.set_experiment("fake-news-lstm")
print("MLflow experiment:", mlflow.get_experiment_by_name("fake-news-lstm"))


# Tokenization setup
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.data.path.append("./data/raw/")
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def safe_word_tokenize(text):
    sentences = sent_tokenizer.tokenize(text)
    return [token for sent in sentences for token in word_tokenizer.tokenize(sent)]

def process_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = safe_word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 3]
    _, idx = np.unique(words, return_index=True)
    return [words[i] for i in sorted(idx)]

def load_cleaned_data():
    df = pd.read_csv("../data/processed/clean_data.csv")
    df = df[df["text"].apply(lambda x: isinstance(x, str))]
    return df["text"].tolist(), df["label"].tolist()

def train():
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
            learning_rate=0.0001
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
            verbose=1
        )

        loss, accuracy = model.evaluate(X_test_pad, y_test_enc)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)

        model.save("../models/lstm_model.h5")
        mlflow.tensorflow.log_model(model, artifact_path="model")

        with open("../models/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

        plot_accuracy_loss(history)
        plot_confusion_matrix(model, X_test_pad, y_test_enc)

if __name__ == "__main__":
    train()
