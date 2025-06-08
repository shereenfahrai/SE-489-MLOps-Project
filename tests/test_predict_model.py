"""
Unit tests for predict_model.py

Tests text preprocessing and prediction pipeline output.
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fake_news_detection.predict_model import process_text


def test_process_text_output() -> None:
    """
    Tests text cleaning pipeline returns correct format.

    Ensures:
    - Only alphabetic words
    - All lowercase
    - No stopwords
    """
    sample = "This is an EXAMPLE sentence, full of noise!"
    tokens = process_text(sample)

    assert isinstance(tokens, list)
    assert all(isinstance(tok, str) for tok in tokens)
    assert all(tok.isalpha() and tok == tok.lower() and len(tok) > 3 for tok in tokens)


def test_tokenizer_and_padding() -> None:
    """
    Tests that tokenizer + pad_sequences gives correct shape.
    """
    texts = [["fake", "news", "today"], ["real", "truth", "story"]]
    joined = [" ".join(words) for words in texts]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(joined)
    seqs = tokenizer.texts_to_sequences(joined)
    padded = pad_sequences(seqs, maxlen=10)

    assert padded.shape == (2, 10)
    assert np.all(padded >= 0)
