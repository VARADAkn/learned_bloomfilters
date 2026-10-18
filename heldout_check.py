import pandas as pd

from data import load
from predict import load_or_build, classify

N_KEYS = 100000          # must equal n_keys in predict.build()
N_TEST = 3000

lbf = load_or_build()

# same filtering/shuffling as data.kaggle_dataset, so held-out = not used as keys
df = pd.read_csv("data/malicious_phish.csv").dropna(subset=["url"]).drop_duplicates("url")
df["url"] = df["url"].astype(str)
mal = df[df["type"] != "benign"]["url"].sample(frac=1, random_state=0).tolist()
held = mal[N_KEYS:N_KEYS + N_TEST]
print("held-out malicious URLs available:", len(held))

d = load(0, "kaggle", n_keys=N_KEYS, n_train=100000, n_val=50000, n_test=50000)
safe = d["test_neg"][:N_TEST]

detected = sum(classify(lbf, u)[0] == "UNSAFE" for u in held) / len(held)
false_alarm = sum(classify(lbf, u)[0] == "UNSAFE" for u in safe) / len(safe)
print(f"Unseen malicious URLs called UNSAFE: {detected:.1%}")
print(f"Safe URLs wrongly called UNSAFE:     {false_alarm:.1%}")