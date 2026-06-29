# Naive Bayes Spam Classifier

A multinomial Naive Bayes text classifier built entirely from scratch in
Python â no scikit-learn, no NLTK, no external ML libraries. It's a spam
vs. ham (legitimate message) classifier, but the implementation is generic
text classification: swap in any labeled text dataset and it works the same
way.

## Why this exists

Naive Bayes is usually a one-liner with `sklearn.naive_bayes.MultinomialNB`.
This repo implements the actual math underneath that call: word-frequency
counting, log-space probability accumulation (to avoid floating-point
underflow from multiplying many small probabilities), Laplace (add-one)
smoothing for words never seen during training, and class priors. The goal
is to make every step inspectable rather than hidden behind a library API.

## How it works

- `naive_bayes.py` â the classifier itself. `NaiveBayesClassifier.fit(texts, labels)`
  builds per-class word counts and vocabulary; `predict(text)` tokenizes the
  input and picks the class with the highest log-posterior probability.
  Laplace smoothing (`alpha=1.0` by default) means a word that never
  appeared in a class during training doesn't zero out that class's
  probability outright.
- `dataset.py` â a small, hand-labeled set of 40 spam/ham SMS-style
  messages, embedded directly in code so the project has zero external
  data dependencies.
- `evaluate.py` â splits the dataset into train/test sets (deterministic
  shuffle via a fixed seed), trains the classifier, and reports accuracy,
  precision, recall, and F1 score (with "spam" as the positive class),
  plus a small confusion matrix and a handful of sample predictions.
- `test_naive_bayes.py` â a pytest suite covering tokenization, input
  validation (mismatched lengths, empty datasets, predicting before
  fitting), correct classification on clearly-separable examples, Laplace
  smoothing on unseen words, and the evaluation helpers (split determinism,
  confusion matrix math).

## Running it

Requires Python 3.9+ (uses `list[str]`/`dict[str, ...]` style type hints)
and `pytest` for the tests.

```bash
# Run the evaluation: trains on 70% of the dataset, evaluates on the rest
python evaluate.py

# Run the test suite
pip install pytest
pytest -v
```

Example use of the classifier directly:

```python
from naive_bayes import NaiveBayesClassifier

clf = NaiveBayesClassifier()
clf.fit(["free money now, click here!", "let's get lunch tomorrow"],
        ["spam", "ham"])
clf.predict("claim your free prize now")  # -> "spam"
```

## Design notes

- **Log-space math**: probabilities for a 20-word message would otherwise
  multiply together into numbers far smaller than floating-point can
  represent. Working in log-space (summing log-probabilities instead of
  multiplying probabilities) avoids that entirely.
- **Laplace smoothing**: without it, a single never-before-seen word would
  give that class a likelihood of exactly zero, regardless of how strongly
  every other word pointed to it. Adding 1 to every word count (and the
  vocabulary size to the denominator) keeps every class's probability
  nonzero.
- **Deterministic train/test split**: `evaluate.py` shuffles with a fixed
  random seed so results are reproducible across runs, while still giving
  a randomized (not just first-N) split.
- **Why a tiny embedded dataset instead of a downloaded corpus**: keeping
  the dataset in `dataset.py` means the repo has no network dependency and
  runs identically anywhere Python is installed â at the cost of a smaller
  sample size than a real-world spam corpus would have.
