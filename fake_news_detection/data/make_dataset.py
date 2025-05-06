"""
make_dataset.py

This script processes raw news data (True.csv and Fake.csv), labels it,
cleans the text, and outputs a single processed dataset.

"""

import os
import re
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


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
    words = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words and len(word) > 3]

    return " ".join(words)


def main() -> None:
    """Main function to load, clean, and save the dataset."""

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
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    combined_df.drop(columns=["title", "subject", "date"], inplace=True)

    print("Cleaning text...")
    combined_df["text"] = combined_df["text"].apply(clean_text)

    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "clean_data.csv")

    print(f"Saving cleaned data to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    print("Data processing complete.")


if __name__ == "__main__":
    main()
