"""
make_dataset.py

This script processes raw news data (True.csv and Fake.csv), labels it,
cleans the text, outputs a full cleaned dataset, and splits it into
train (90%) and predict (10%) CSV files.

References:
- Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Notebook: https://www.kaggle.com/code/yossefmohammed/true-and-fake-news-lstm-accuracy-97-90
"""

import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


def ensure_nltk_data() -> None:
    """Ensure required NLTK resources are available."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            find(resource)
        except LookupError:
            nltk.download(resource)


def clean_text(text: str) -> str:
    """Clean and preprocess a string of text.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and preprocessed text.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\W", " ", str(text))
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    stop_words = set(stopwords.words("english"))
    tokens = [tok for tok in tokens if tok not in stop_words and len(tok) > 3]

    return " ".join(tokens)


def main() -> None:
    """Main function to load, clean, split, and save datasets."""
    ensure_nltk_data()

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    fake_path = os.path.join(raw_dir, "Fake.csv")
    true_path = os.path.join(raw_dir, "True.csv")

    print("Loading data...")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    print("Merging datasets...")
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df.drop(columns=["title", "subject", "date"], inplace=True)

    print("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    os.makedirs(processed_dir, exist_ok=True)
    clean_path = os.path.join(processed_dir, "clean_data.csv")
    print(f"Saving full cleaned data to {clean_path}...")
    df.to_csv(clean_path, index=False)

    # Split off a small portion for prediction/testing
    print("Splitting data into train (90%) and predict (10%) sets...")
    train_df, predict_df = train_test_split(
        df,
        test_size=0.10,
        stratify=df["label"],
        random_state=42,
    )

    train_path = os.path.join(processed_dir, "train.csv")
    pred_path = os.path.join(processed_dir, "predict.csv")
    print(f"Saving train data to {train_path} ({len(train_df)} rows)...")
    train_df.to_csv(train_path, index=False)
    print(f"Saving predict data to {pred_path} ({len(predict_df)} rows)...")
    predict_df.to_csv(pred_path, index=False)

    print("✅ Data processing and splitting complete.")


if __name__ == "__main__":
    main()
