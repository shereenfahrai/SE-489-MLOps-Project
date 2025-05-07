import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# Ensure output directory exists
os.makedirs("reports/figures", exist_ok=True)


def plot_accuracy_loss(history):
    """Plot and save training and validation accuracy/loss curves."""
    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("reports/figures/accuracy.png")
    # plt.show()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("reports/figures/loss.png")
    # plt.show()


def plot_confusion_matrix(model, X_test, y_test_one_hot, filename="confusion_matrix.png"):
    """Plot and save confusion matrix for predictions on the test set."""
    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test_one_hot, axis=1)

    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"reports/figures/{filename}")
    # plt.show()
