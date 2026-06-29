"""
Train/test split + evaluation harness for NaiveBayesClassifier.

Computes accuracy, precision, recall, and F1 (treating "spam" as the
positive class) and prints a small confusion matrix, all without any
external ML libraries.
"""

from __future__ import annotations

import random

from dataset import load
from naive_bayes import NaiveBayesClassifier

POSITIVE_LABEL = "spam"


def train_test_split(
    texts: list[str], labels: list[str], test_fraction: float = 0.3, seed: int = 42
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Shuffle and split (texts, labels) into train/test sets."""
    indices = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split_at = int(len(indices) * (1 - test_fraction))
    train_idx, test_idx = indices[:split_at], indices[split_at:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    return train_texts, train_labels, test_texts, test_labels


def confusion_counts(
    y_true: list[str], y_pred: list[str], positive: str = POSITIVE_LABEL
) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p != positive)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(counts: dict[str, int]) -> tuple[float, float, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def main() -> None:
    texts, labels = load()
    train_texts, train_labels, test_texts, test_labels = train_test_split(
        texts, labels, test_fraction=0.3
    )

    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(train_texts, train_labels)
    predictions = clf.predict_many(test_texts)

    acc = accuracy(test_labels, predictions)
    counts = confusion_counts(test_labels, predictions)
    precision, recall, f1 = precision_recall_f1(counts)

    print(f"Train size: {len(train_texts)}  Test size: {len(test_texts)}")
    print(f"Accuracy:   {acc:.3f}")
    print(f"Precision:  {precision:.3f}")
    print(f"Recall:     {recall:.3f}")
    print(f"F1 score:   {f1:.3f}")
    print()
    print("Confusion matrix (positive class = 'spam'):")
    print(f"  TP={counts['tp']}  FP={counts['fp']}  FN={counts['fn']}  TN={counts['tn']}")
    print()
    print("Sample predictions:")
    for text, true, pred in list(zip(test_texts, test_labels, predictions))[:5]:
        marker = "OK" if true == pred else "XX"
        print(f"  [{marker}] true={true:<5} pred={pred:<5} {text[:60]}")


if __name__ == "__main__":
    main()
