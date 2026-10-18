import random
import string

SUS = ["login", "verify", "secure", "account", "update", "free", "bonus", "confirm",
       "bank", "paypal", "signin", "wallet", "password", "claim", "prize"]
GOOD = ["news", "shop", "blog", "wiki", "tech", "travel", "food", "sports", "music",
        "school", "health", "garden", "books", "movies", "science", "weather",
        "recipes", "forum", "photos", "store"]
BAD_TLD = ["xyz", "top", "tk", "ru", "info", "click", "cc", "biz"]
GOOD_TLD = ["com", "org", "edu", "net", "in", "co.uk", "gov"]


def _rs(rng, n, chars=string.ascii_lowercase):
    return "".join(rng.choice(chars) for _ in range(n))


def _digits(rng, lo, hi):
    return "".join(rng.choice(string.digits) for _ in range(rng.randint(lo, hi)))


def _malicious(rng):
    if rng.random() < 0.60:  # "obvious" malicious pattern (rest look like normal sites)
        host = rng.choice([_rs(rng, rng.randint(5, 11)),
                           rng.choice(SUS) + "-" + _rs(rng, 4)])
        if rng.random() < 0.6:
            host += "-" + rng.choice(SUS)
        host += _digits(rng, 0, 4)
        tld = rng.choice(BAD_TLD) if rng.random() < 0.8 else rng.choice(GOOD_TLD)
        scheme = "http" if rng.random() < 0.75 else "https"
        segs = [rng.choice(SUS) if rng.random() < 0.5 else _rs(rng, 6) + _digits(rng, 1, 4)
                for _ in range(rng.randint(1, 4))]
        q = "?id=" + _digits(rng, 3, 8) if rng.random() < 0.5 else ""
    else:  # "stealthy" malicious: looks like a normal site
        host = rng.choice(GOOD) + _rs(rng, 5)
        tld = rng.choice(GOOD_TLD)
        scheme = "https" if rng.random() < 0.6 else "http"
        segs = [rng.choice(GOOD) for _ in range(rng.randint(1, 3))] + [_rs(rng, 5)]
        q = ""
    return f"{scheme}://{host}.{tld}/" + "/".join(segs) + q


def _benign(rng, shifted=False):
    host = rng.choice(GOOD) + (rng.choice(GOOD) if rng.random() < 0.4 else "") + _rs(rng, 4)
    p_bad_tld = 0.30 if shifted else 0.02
    tld = rng.choice(BAD_TLD) if rng.random() < p_bad_tld else rng.choice(GOOD_TLD)
    scheme = "https" if rng.random() < (0.55 if shifted else 0.85) else "http"
    segs = [rng.choice(GOOD) for _ in range(rng.randint(1, 3))]
    if rng.random() < (0.35 if shifted else 0.12):
        segs.insert(rng.randint(0, len(segs)), rng.choice(SUS))
    if rng.random() < (0.6 if shifted else 0.2):
        segs.append(_digits(rng, 2, 6))
    q = "?page=" + _digits(rng, 1, 3) if rng.random() < 0.3 else ""
    return f"{scheme}://{host}.{tld}/" + "/".join(segs) + q


def _unique(gen, n, forbidden):
    out, seen = [], set(forbidden)
    while len(out) < n:
        u = gen()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def synthetic_dataset(n_keys=30000, n_train=30000, n_val=30000, n_test=50000,
                      n_shift=50000, seed=0):
    rng = random.Random(seed)
    keys = _unique(lambda: _malicious(rng), n_keys, [])
    used = set(keys)
    train = _unique(lambda: _benign(rng), n_train, used); used |= set(train)
    val = _unique(lambda: _benign(rng), n_val, used); used |= set(val)
    test = _unique(lambda: _benign(rng), n_test, used); used |= set(test)
    shift = _unique(lambda: _benign(rng, shifted=True), n_shift, used)
    return dict(keys=keys, train_neg=train, val_neg=val, test_neg=test, shift_neg=shift)


def kaggle_dataset(path="data/malicious_phish.csv", n_keys=30000, n_train=30000,
                   n_val=30000, n_test=50000, seed=0):
    """Kaggle 'Malicious URLs dataset': CSV with columns url,type (type=='benign' is safe)."""
    import pandas as pd
    df = pd.read_csv(path).drop_duplicates("url")
    mal = df[df["type"] != "benign"]["url"].sample(frac=1, random_state=seed).tolist()
    ben = df[df["type"] == "benign"]["url"].sample(frac=1, random_state=seed).tolist()
    keys = mal[:n_keys]
    keyset = set(keys)
    need = n_train + n_val + n_test
    ben = [u for u in ben if u not in keyset]
    train, val = ben[:n_train], ben[n_train:n_train + n_val]
    test = ben[n_train + n_val:need]
    rest = sorted(ben[need:], key=len)              # drift proxy: unusually long benign URLs
    shift = rest[int(0.75 * len(rest)):][:50000]
    return dict(keys=keys, train_neg=train, val_neg=val, test_neg=test, shift_neg=shift)


def load(seed=0, source="synthetic", **kw):
    if source == "synthetic":
        return synthetic_dataset(seed=seed, **kw)
    return kaggle_dataset(seed=seed, **kw)


if __name__ == "__main__":
    d = synthetic_dataset(n_keys=5, n_train=3, n_val=3, n_test=3, n_shift=3)
    for name, rows in d.items():
        print(name)
        for r in rows:
            print("   ", r)