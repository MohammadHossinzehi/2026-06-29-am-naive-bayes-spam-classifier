"""
A from-scratch multinomial Naive Bayes classifier for text classification
(e.g. spam vs. ham), with Laplace (add-one) smoothing.

No external ML libraries are used -- this is meant to make the underlying
math visible: word counts, log-probabilities, and class priors.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens, stripping punctuation."""
    return TOKEN_RE.findall(text.lower())


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace smoothing.

    Usage:
        clf = NaiveBayesClassifier()
        clf.fit(texts, labels)
        clf.predict("free money now!!!")
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_priors: dict[str, float] = {}
        self.word_counts: dict[str, Counter] = {}
        self.class_totals: dict[str, int] = {}
        self.vocab: set[str] = set()
        self._fitted = False

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "NaiveBayesClassifier":
        texts = list(texts)
        labels = list(labels)
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")
        if not texts:
            raise ValueError("cannot fit on an empty dataset")

        label_counts = Counter(labels)
        n_docs = len(labels)
        self.class_priors = {c: count / n_docs for c, count in label_counts.items()}

        self.word_counts = {c: Counter() for c in label_counts}
        self.class_totals = {c: 0 for c in label_counts}

        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            self.word_counts[label].update(tokens)
            self.class_totals[label] += len(tokens)
            self.vocab.update(tokens)

        self._fitted = True
        return self

    def _log_likelihood(self, tokens: list[str], cls: str) -> float:
        vocab_size = len(self.vocab)
        denom = self.class_totals[cls] + self.alpha * vocab_size
        log_prob = math.log(self.class_priors[cls])
        counts = self.word_counts[cls]
        for token in tokens:
            count = counts.get(token, 0)
            log_prob += math.log((count + self.alpha) / denom)
        return log_prob

    def predict_log_proba(self, text: str) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("classifier has not been fit yet")
        tokens = tokenize(text)
        return {cls: self._log_likelihood(tokens, cls) for cls in self.class_priors}

    def predict(self, text: str) -> str:
        scores = self.predict_log_proba(text)
        return max(scores, key=scores.get)

    def predict_many(self, texts: Iterable[str]) -> list[str]:
        return [self.predict(t) for t in texts]
