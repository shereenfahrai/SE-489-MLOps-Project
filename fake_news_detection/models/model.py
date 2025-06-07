"""
model.py

Defines the LSTM-based text classification model.
"""

from typing import Optional

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    vocab_size: int,
    maxlen: int,
    embed_dim: int,
    lstm_units: int,
    dropout_rate: float,
    learning_rate: float,
    dense_units: Optional[int] = 64,
) -> Model:
    """
    Builds an LSTM-based text classification model.

    Args:
        vocab_size (int): Size of the vocabulary.
        maxlen (int): Maximum length of input sequences.
        embed_dim (int): Dimension of embedding vectors.
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: A compiled Keras model ready for training.
    """
    inputs = Input(shape=(maxlen,), name="input")
    x = Embedding(input_dim=vocab_size + 1, output_dim=embed_dim, name="embedding")(
        inputs
    )
    x = Dropout(dropout_rate, name="dropout1")(x)
    x = LSTM(lstm_units, return_sequences=True, name="lstm")(x)
    x = Dropout(dropout_rate, name="dropout2")(x)
    x = GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = Dense(dense_units, activation="relu", name="dense1")(x)
    x = Dropout(dropout_rate, name="dropout3")(x)
    outputs = Dense(2, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="LSTM_TextClassifier")
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
