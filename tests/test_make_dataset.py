"""
Unit tests for make_dataset.py

Tests text cleaning and CSV output generation.
"""


from fake_news_detection.data.make_dataset import clean_text


def test_clean_text() -> None:
    """
    Tests that clean_text removes noise and returns processed tokens.

    Verifies:
    - Only alphabetic, lowercase words
    - Stopwords are excluded
    - Tokens have length > 3
    """
    raw = "Breaking News! The quick brown fox jumps over the lazy dog."
    cleaned = clean_text(raw)
    tokens = cleaned.split()

    assert all(token.isalpha() for token in tokens)
    assert all(token == token.lower() for token in tokens)
    assert all(len(token) > 3 for token in tokens)
    assert "the" not in tokens and "over" not in tokens
