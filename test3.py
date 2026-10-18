from data import synthetic_dataset
from features import make_vectorizer, train_classifier
from lbf import LearnedBF
from bloom import BloomFilter

d = synthetic_dataset(10000, 10000, 10000, 20000)
vec = make_vectorizer()
clf, _ = train_classifier("LR", vec, d["keys"], d["train_neg"])
total = 10 * len(d["keys"])                       # 10 bits per key

lbf = LearnedBF(clf, vec, total).fit(d["keys"], d["val_neg"])
print("false negatives:", (~lbf.query_many(d["keys"])).sum())    # MUST be 0
print("LBF FPR:", lbf.query_many(d["test_neg"]).mean())

std = BloomFilter(total, n_expected=len(d["keys"])); std.add_many(d["keys"])
print("Standard FPR:", std.query_many(d["test_neg"]).mean())     # LBF should be much lower