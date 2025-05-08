"""
Visualization utilities for model training and evaluation.

This module includes:
- Accuracy and loss curve plotting
- Confusion matrix plotting for classification results
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_accuracy_loss(history: History) -> None:
    """
    Plot and save training/validation accuracy and loss over epochs.

    Args:
        history (History): Keras History object from model.fit().
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "reports/figures/accuracy.png"))
    # plt.show()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("reports/figures/loss.png")


def plot_confusion_matrix(
        model: Model,
        X_test: np.ndarray,
        y_test_one_hot: np.ndarray,
        filename: str = "train_confusion_matrix.png",
) -> None:
    """
        Generate and save confusion matrix heatmap from model predictions.

        Args:
            model (Model): Trained Keras model.
            X_test (np.ndarray): Test feature set.
            y_test_one_hot (np.ndarray): One-hot encoded true labels.
            filename (str): Output file name for the plot image.
    """
    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test_one_hot, axis=1)

    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
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
    plt.savefig(os.path.join(BASE_DIR, f"reports/figures/{filename}"))
