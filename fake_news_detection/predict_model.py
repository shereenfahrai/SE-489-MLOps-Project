# fake_news_detection/predict_model.py

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load model and tokenizer
MODEL_PATH = "../models/lstm_model.h5"
TOKENIZER_PATH = "../models/tokenizer.pkl"
DATA_PATH = "../data/processed/predict.csv"
# RESULT_PATH = "../data/processed/predicted_results.csv"
CONF_MATRIX_PATH = "/reports/figures/confusion_matrix_predict.png"

nltk.data.path.append("./data/raw/")
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

maxlen = 150  # same as training


# Preprocessing
def safe_word_tokenize(text):
    sentences = sent_tokenizer.tokenize(text)
    return [token for sent in sentences for token in word_tokenizer.tokenize(sent)]


def process_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = safe_word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 3]
    _, idx = np.unique(words, return_index=True)
    return [words[i] for i in sorted(idx)]


def plot_confusion_matrix(y_true, y_pred, save_path):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def predict():
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

        print(f"Prediction Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        plot_confusion_matrix(y_true, predicted_labels, CONF_MATRIX_PATH)
        print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

    # df.to_csv(RESULT_PATH, index=False)
    # print(f"Prediction complete. Results saved to {RESULT_PATH}")


if __name__ == "__main__":
    predict()
