"""Pytest suite for naive_bayes.py and evaluate.py."""

import math

import pytest

from naive_bayes import NaiveBayesClassifier, tokenize
from evaluate import (
    accuracy,
    confusion_counts,
    precision_recall_f1,
    train_test_split,
)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, WORLD!! Win $$$ now.") == ["hello", "world", "win", "now"]


def test_tokenize_handles_apostrophes():
    assert tokenize("Don't stop believing") == ["don't", "stop", "believing"]


def test_fit_rejects_mismatched_lengths():
    clf = NaiveBayesClassifier()
    with pytest.raises(ValueError):
        clf.fit(["a", "b"], ["only one label"])


def test_fit_rejects_empty_dataset():
    clf = NaiveBayesClassifier()
    with pytest.raises(ValueError):
        clf.fit([], [])


def test_predict_before_fit_raises():
    clf = NaiveBayesClassifier()
    with pytest.raises(RuntimeError):
        clf.predict("anything")


def test_classifier_separates_obvious_classes():
    texts = [
        "free money now claim your prize",
        "win a free prize click now",
        "urgent claim your free cash reward",
        "let's grab lunch tomorrow at noon",
        "can you send me the meeting notes",
        "thanks for helping me move this weekend",
    ]
    labels = ["spam", "spam", "spam", "ham", "ham", "ham"]

    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(texts, labels)

    assert clf.predict("free cash prize, claim now") == "spam"
    assert clf.predict("let's meet for lunch tomorrow") == "ham"


def test_predict_log_proba_returns_one_score_per_class():
    clf = NaiveBayesClassifier()
    clf.fit(["spam text here", "ham text here"], ["spam", "ham"])
    scores = clf.predict_log_proba("some text")
    assert set(scores.keys()) == {"spam", "ham"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in scores.values())


def test_laplace_smoothing_handles_unseen_words():
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(["known words here"], ["ham"])
    # Should not raise even though "zzbrandnewword" never appeared in training.
    result = clf.predict("zzbrandnewword that was never seen")
    assert result == "ham"


def test_train_test_split_is_deterministic_and_covers_all_rows():
    texts = [f"text {i}" for i in range(10)]
    labels = ["ham"] * 10

    a = train_test_split(texts, labels, test_fraction=0.3, seed=42)
    b = train_test_split(texts, labels, test_fraction=0.3, seed=42)
    assert a == b

    train_texts, _, test_texts, _ = a
    assert len(train_texts) + len(test_texts) == 10
    assert set(train_texts).isdisjoint(set(test_texts))


def test_confusion_counts_and_metrics():
    y_true = ["spam", "spam", "ham", "ham"]
    y_pred = ["spam", "ham", "ham", "spam"]

    counts = confusion_counts(y_true, y_pred)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}

    precision, recall, f1 = precision_recall_f1(counts)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)

    assert accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_accuracy_handles_empty_input():
    assert accuracy([], []) == 0.0
