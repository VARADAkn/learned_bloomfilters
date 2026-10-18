import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

N_FEATURES = 2 ** 10


def make_vectorizer(n_features=N_FEATURES):
    return HashingVectorizer(analyzer="char", ngram_range=(2, 4),
                             n_features=n_features, alternate_sign=False,
                             norm="l2")


def make_classifier(name, seed=0):
    if name == "LR":
        return LogisticRegression(max_iter=300, C=5.0)
    if name == "NB":
        return MultinomialNB(alpha=0.1)
    if name == "RF":
        return RandomForestClassifier(n_estimators=8, max_depth=8,
                                      n_jobs=-1, random_state=seed)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(8,), max_iter=60,
                             random_state=seed)
    raise ValueError(name)


def train_classifier(name, vec, keys, neg, seed=0):
    """Train keys (label 1) vs safe URLs (label 0). Returns (clf, train_seconds)."""
    X = vec.transform(list(keys) + list(neg))
    y = np.r_[np.ones(len(keys)), np.zeros(len(neg))]
    clf = make_classifier(name, seed)
    t = time.perf_counter()
    clf.fit(X, y)
    return clf, time.perf_counter() - t


def model_size_bits(clf):
    """Model size = learned parameters only, counted as float32 (32 bits each).
    Random forest: 128 bits per tree node (feature, threshold, children, value)."""
    if isinstance(clf, LogisticRegression):
        n = clf.coef_.size + clf.intercept_.size
    elif isinstance(clf, MultinomialNB):
        n = clf.feature_log_prob_.size + clf.class_log_prior_.size
    elif isinstance(clf, MLPClassifier):
        n = sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)
    elif isinstance(clf, RandomForestClassifier):
        return sum(e.tree_.node_count for e in clf.estimators_) * 128
    else:
        return len(pickle.dumps(clf)) * 8
    return int(n) * 32


def score(clf, vec, urls):
    """Score in [0,1]: how malicious the URL looks."""
    return clf.predict_proba(vec.transform(list(urls)))[:, 1]