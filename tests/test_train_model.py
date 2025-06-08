"""
Unit tests for train_model.py

Tests tokenizer processing, label encoding, and model structure.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from fake_news_detection.models.model import build_lstm_model


def test_tokenizer_sequence_generation() -> None:
    """
    Tests that Tokenizer correctly encodes text to sequence of integers.
    """
    sample_texts = [["fake", "news"], ["real", "story"]]
    texts = [" ".join(words) for words in sample_texts]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    assert isinstance(sequences, list)
    assert all(isinstance(seq, list) for seq in sequences)
    assert all(all(isinstance(num, int) for num in seq) for seq in sequences)


def test_label_encoding() -> None:
    """
    Tests one-hot encoding of labels using LabelEncoder + to_categorical.
    """
    labels = ["real", "fake", "fake", "real"]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)
    y_onehot = to_categorical(y_encoded)

    assert y_onehot.shape == (4, 2)
    assert np.allclose(np.sum(y_onehot, axis=1), 1.0)


def test_model_structure() -> None:
    """
    Tests that LSTM model can be built and compiled.
    """
    model = build_lstm_model(
        vocab_size=1000,
        maxlen=150,
        embed_dim=64,
        lstm_units=64,
        dropout_rate=0.3,
        learning_rate=0.001,
    )

    # Check model input/output shape
    assert model.input_shape == (None, 150)
    assert model.output_shape == (None, 2)

    # Just confirm it's compiled (loss and optimizer present)
    assert model.loss is not None
    assert model.optimizer is not None
