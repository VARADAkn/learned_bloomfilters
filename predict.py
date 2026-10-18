import os
import pickle
import sys

from data import load
from features import make_vectorizer, train_classifier, score
from lbf import LearnedBF

MODEL_FILE = "results/lbf_model.pkl"
BITS_PER_KEY = 12          # memory budget of the filter


def build():
    """Train on the real dataset and build the Learned Bloom Filter (done once)."""
    print("Loading dataset and training... (a few minutes the first time)")
    d = load(0, "kaggle", n_keys=100000, n_train=100000, n_val=50000, n_test=50000)
    vec = make_vectorizer()
    clf, secs = train_classifier("LR", vec, d["keys"], d["train_neg"])
    total_bits = BITS_PER_KEY * len(d["keys"])
    lbf = LearnedBF(clf, vec, total_bits).fit(d["keys"], d["val_neg"])

    # quality report on data the filter has not been tuned on
    missed = int((~lbf.query_many(d["keys"])).sum())
    fpr = float(lbf.query_many(d["test_neg"]).mean())
    print(f"Trained in {secs:.1f}s | malicious URLs missed: {missed} (must be 0)")
    print(f"Safe URLs wrongly flagged (FPR): {fpr:.4%}")

    os.makedirs("results", exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(lbf, f)
    return lbf


def load_or_build():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return build()


def check(lbf, url):
    s = float(score(lbf.clf, lbf.vec, [url])[0])
    if s >= lbf.tau:
        return "MALICIOUS  (the classifier flagged it)", s
    if lbf.backup.query_many([url])[0]:
        return "MALICIOUS  (found in the backup filter)", s
    return "SAFE       (not flagged, not in the malicious list)", s


if __name__ == "__main__":
    lbf = load_or_build()
    urls = sys.argv[1:]
    if urls:
        for u in urls:
            verdict, s = check(lbf, u)
            print(f"{u}\n  -> {verdict} | score {s:.3f}")
    else:
        print("\nType a URL and press Enter (empty line to quit).")
        while True:
            u = input("URL> ").strip()
            if not u:
                break
            verdict, s = check(lbf, u)
            print(f"  -> {verdict} | score {s:.3f}\n")