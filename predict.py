import os
import pickle
import sys

import numpy as np

from data import load
from features import make_vectorizer, train_classifier, score
from lbf import LearnedBF

MODEL_FILE = "results/lbf_model.pkl"
BITS_PER_KEY = 12      # memory budget of the filter
ALERT_FPR = 0.05       # about 5% of safe URLs may be flagged UNSAFE (raise it to catch more)


def classify(lbf, url):
    """Returns ('SAFE' or 'UNSAFE', score, reason)."""
    s = float(score(lbf.clf, lbf.vec, [url])[0])
    if s >= lbf.tau:
        return "UNSAFE", s, "classifier is very confident"
    if lbf.backup.query_many([url])[0]:
        return "UNSAFE", s, "found in the known malicious list"
    if s >= lbf.alert_tau:
        return "UNSAFE", s, "classifier score above the alert level"
    return "SAFE", s, "not detected as malicious"


def build():
    """Train on the real dataset and build the filter (done once)."""
    print("Loading dataset and training... (a few minutes the first time)")
    d = load(0, "kaggle", n_keys=100000, n_train=100000, n_val=50000, n_test=50000)
    vec = make_vectorizer()
    clf, secs = train_classifier("LR", vec, d["keys"], d["train_neg"])
    total_bits = BITS_PER_KEY * len(d["keys"])
    lbf = LearnedBF(clf, vec, total_bits).fit(d["keys"], d["val_neg"])

    # alert threshold: flag the top ALERT_FPR share of safe validation URLs
    vs = score(lbf.clf, lbf.vec, d["val_neg"])
    lbf.alert_tau = float(np.quantile(vs, 1 - ALERT_FPR))

    # quality report
    missed = sum(classify(lbf, u)[0] == "SAFE" for u in d["keys"][:2000])
    false_alarms = np.mean([classify(lbf, u)[0] == "UNSAFE" for u in d["test_neg"][:5000]])
    print(f"Trained in {secs:.1f}s")
    print(f"Main threshold {lbf.tau:.4f} | alert threshold {lbf.alert_tau:.4f}")
    print(f"Keys stored in the backup filter: {lbf.n_backup_keys} of {len(d['keys'])}")
    print(f"Malicious URLs wrongly called SAFE (out of 2000 checked): {missed} (must be 0)")
    print(f"Safe URLs wrongly called UNSAFE (false alarm rate): {false_alarms:.2%}")

    os.makedirs("results", exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(lbf, f)
    return lbf


def load_or_build():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            lbf = pickle.load(f)
        if hasattr(lbf, "alert_tau"):       # old saved model without alert level -> rebuild
            return lbf
    return build()


def show(lbf, url):
    label, s, why = classify(lbf, url)
    print(f"  -> {label}  (score {s:.3f}; alert level {lbf.alert_tau:.3f}; {why})")


if __name__ == "__main__":
    lbf = load_or_build()
    urls = sys.argv[1:]
    if urls:
        for u in urls:
            print(u)
            show(lbf, u)
    else:
        print("\nType a URL and press Enter (empty line to quit).")
        while True:
            u = input("URL> ").strip()
            if not u:
                break
            show(lbf, u)
            print()