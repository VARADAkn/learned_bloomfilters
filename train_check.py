import numpy as np
from sklearn.metrics import roc_auc_score

from data import load
from features import make_vectorizer, train_classifier, score, model_size_bits

d = load(0, "kaggle", n_keys=30000, n_train=30000, n_val=30000, n_test=50000)
print({k: len(v) for k, v in d.items()})

vec = make_vectorizer()
for name in ("LR", "NB"):
    clf, secs = train_classifier(name, vec, d["keys"], d["train_neg"])
    s_neg = score(clf, vec, d["test_neg"])
    # held-out check on keys: split the keys in half, train on the first half only
    half = len(d["keys"]) // 2
    clf2, _ = train_classifier(name, vec, d["keys"][:half], d["train_neg"])
    s_unseen_mal = score(clf2, vec, d["keys"][half:])      # malicious URLs it never saw
    s_unseen_neg = score(clf2, vec, d["test_neg"])
    auc = roc_auc_score(np.r_[np.ones(len(s_unseen_mal)), np.zeros(len(s_unseen_neg))],
                        np.r_[s_unseen_mal, s_unseen_neg])
    print(f"{name}: train {secs:.1f}s | model {model_size_bits(clf)} bits | "
          f"AUC on UNSEEN malicious URLs: {auc:.3f}")