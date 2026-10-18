from data import synthetic_dataset
from features import *
d = synthetic_dataset(5000, 5000, 100, 100)
vec = make_vectorizer()
clf, t = train_classifier("LR", vec, d["keys"], d["train_neg"])
print(score(clf, vec, d["keys"][:3]))       # high (close to 1)
print(score(clf, vec, d["test_neg"][:3]))   # low (close to 0)
print(model_size_bits(clf))                 # 32800