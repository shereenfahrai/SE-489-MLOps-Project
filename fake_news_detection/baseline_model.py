import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load cleaned data
df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/train.csv"))
df = df[df["text"].apply(lambda x: isinstance(x, str))]

X = df["text"].values
y = df["label"].values

vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = vectorizer.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []
f1s = []

conf_matrix_final = None
clf_final = None

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
        clf_final = clf  # Save the last fold's model

# Plot confusion matrix of last fold

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix_final,
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

plt.savefig(os.path.join(BASE_DIR, "fake_news_detection/reports/figures/baseline_confusion_matrix.png"))

# Output average metrics
print("\nAverage across 5 folds:")
print(f"Accuracy:  {np.mean(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall:    {np.mean(recalls):.4f}")
print(f"F1 Score:  {np.mean(f1s):.4f}")
