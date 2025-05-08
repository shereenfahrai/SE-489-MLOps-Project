"""
Baseline training script for fake news classification using TF-IDF and Logistic Regression.

Performs 5-fold cross-validation and outputs evaluation metrics and a confusion matrix plot.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(path: str) -> pd.DataFrame:
    """
    Load and clean the training dataset.

    Args:
        path (str): Full path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with only valid text rows.
    """
    df = pd.read_csv(path)
    return df[df["text"].apply(lambda x: isinstance(x, str))]


def vectorize_text(texts: np.ndarray) -> TfidfVectorizer:
    """
    Vectorize the input text using TF-IDF.

    Args:
        texts (np.ndarray): Array of raw text strings.

    Returns:
        scipy.sparse matrix: TF-IDF transformed feature matrix.
    """
    vectorizer = TfidfVectorizer(max_features=10000)
    return vectorizer.fit_transform(texts)


def train_and_evaluate(X_tfidf: scipy.sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    """
    Train logistic regression with 5-fold cross-validation and evaluate metrics.

    Args:
        X_tfidf: TF-IDF feature matrix.
        y: Labels.

    Returns:
        Tuple containing average metrics and confusion matrix from last fold.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    conf_matrix_final = None
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tfidf, y)):
        print(f"\nFold {fold + 1}")

        X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        if fold == skf.get_n_splits() - 1:
            conf_matrix_final = confusion_matrix(y_test, y_pred)

    print("\nAverage across 5 folds:")
    print(f"Accuracy:  {np.mean(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f}")
    print(f"F1 Score:  {np.mean(f1s):.4f}")

    assert conf_matrix_final is not None, "Confusion matrix was not computed."
    return conf_matrix_final


def plot_confusion_matrix(matrix: np.ndarray, save_path: str) -> None:
    """
    Plot and save confusion matrix as a heatmap.

    Args:
        matrix (np.ndarray): Confusion matrix.
        save_path (str): Path to save the image.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.title("Baseline Confusion Matrix (Last Fold)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    df = load_data(os.path.join(BASE_DIR, "data/processed/train.csv"))
    X = df["text"].values
    y = df["label"].values
    X_tfidf = vectorize_text(X)
    conf_matrix = train_and_evaluate(X_tfidf, y)
    plot_confusion_matrix(
        conf_matrix,
        os.path.join(
            BASE_DIR,
            "fake_news_detection/reports/figures/baseline_confusion_matrix.png",
        ),
    )
