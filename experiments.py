import argparse
import os
import time

import numpy as np
import pandas as pd

from bloom import BloomFilter, optimal_k, theory_fpr
from data import load
from features import make_vectorizer, train_classifier, model_size_bits
from lbf import LearnedBF, PartitionedLBF

os.makedirs("results", exist_ok=True)


def evaluate(filt, keys, neg):
    """FPR on safe URLs, false negatives on keys, latency (us/query) on safe URLs."""
    t = time.perf_counter()
    r = filt.query_many(neg)
    us = (time.perf_counter() - t) / len(neg) * 1e6
    fn = int((~filt.query_many(keys)).sum())
    return dict(fpr=float(r.mean()), false_neg=fn, query_us=us,
                memory_bits=int(filt.size_bits))


def build_standard(keys, total_bits):
    bf = BloomFilter(total_bits, n_expected=len(keys))
    bf.add_many(keys)
    return bf


# ------------------------------------------------ Exp 0: validate Bloom filter
def validate_bloom(d):
    keys, neg = d["keys"], d["test_neg"]
    rows = []
    for bpk in (4, 6, 8, 10, 12):
        m = bpk * len(keys)
        k = optimal_k(m, len(keys))
        bf = build_standard(keys, m)
        rows.append(dict(bits_per_key=bpk, k=k,
                         measured_fpr=float(bf.query_many(neg).mean()),
                         theory_fpr=theory_fpr(m, len(keys), k),
                         false_neg=int((~bf.query_many(keys)).sum())))
    df = pd.DataFrame(rows)
    df.to_csv("results/exp0_bloom_validation.csv", index=False)
    print("\n[Exp 0] Standard Bloom filter: measured vs theory (Eq. 4)")
    print(df.to_string(index=False))


# ------------------------------------------------ Exp 1: FPR vs memory
def exp1(seeds, args):
    rows = []
    for seed in seeds:
        d = load(seed, args.source, **args.sizes)
        vec = make_vectorizer()
        clf, tt = train_classifier("LR", vec, d["keys"], d["train_neg"], seed)
        for bpk in args.budgets:
            total = bpk * len(d["keys"])
            std = build_standard(d["keys"], total)
            rows.append(dict(seed=seed, bits_per_key=bpk, filter="Standard",
                             **evaluate(std, d["keys"], d["test_neg"])))
            try:
                t = time.perf_counter()
                lbf = LearnedBF(clf, vec, total).fit(d["keys"], d["val_neg"])
                bt = time.perf_counter() - t
                rows.append(dict(seed=seed, bits_per_key=bpk, filter="LBF", build_s=bt + tt,
                                 **evaluate(lbf, d["keys"], d["test_neg"])))
                t = time.perf_counter()
                pl = PartitionedLBF(clf, vec, total, 5).fit(d["keys"], d["val_neg"])
                bt = time.perf_counter() - t
                rows.append(dict(seed=seed, bits_per_key=bpk, filter="PLBF", build_s=bt + tt,
                                 **evaluate(pl, d["keys"], d["test_neg"])))
            except ValueError as e:
                print(f"  budget {bpk} bits/key skipped for learned filters: {e}")
    df = pd.DataFrame(rows)
    df.to_csv("results/exp1_fpr_vs_memory.csv", index=False)
    print("\n[Exp 1] FPR vs memory budget (mean over seeds)")
    print(df.groupby(["bits_per_key", "filter"])[["fpr", "false_neg", "query_us"]]
            .mean().round(5).to_string())


# ------------------------------------------------ Exp 2: classifier comparison
def exp2(seeds, args):
    rows = []
    for seed in seeds:
        d = load(seed, args.source, **args.sizes)
        vec = make_vectorizer()
        for name in ("LR", "NB", "RF", "MLP"):
            clf, tt = train_classifier(name, vec, d["keys"], d["train_neg"], seed)
            for bpk in args.exp2_budgets:
                total = bpk * len(d["keys"])
                try:
                    lbf = LearnedBF(clf, vec, total).fit(d["keys"], d["val_neg"])
                except ValueError:   # model alone is bigger than the memory budget
                    rows.append(dict(seed=seed, classifier=name, bits_per_key=bpk,
                                     model_bits=model_size_bits(clf), train_s=tt,
                                     fpr=np.nan))
                    continue
                ev = evaluate(lbf, d["keys"], d["test_neg"])
                rows.append(dict(seed=seed, classifier=name, bits_per_key=bpk,
                                 model_bits=lbf.model_bits, train_s=tt,
                                 backup_keys=lbf.n_backup_keys, **ev))
    df = pd.DataFrame(rows)
    df.to_csv("results/exp2_classifiers.csv", index=False)
    print("\n[Exp 2] Classifier comparison (NaN fpr = model does not fit the budget)")
    print(df.groupby(["bits_per_key", "classifier"]).mean(numeric_only=True)
            .drop(columns="seed").round(5).to_string())


# ------------------------------------------------ Exp 3: number of partitions
def exp3(seeds, args):
    rows = []
    bpk = args.fixed_bpk
    for seed in seeds:
        d = load(seed, args.source, **args.sizes)
        vec = make_vectorizer()
        clf, _ = train_classifier("LR", vec, d["keys"], d["train_neg"], seed)
        total = bpk * len(d["keys"])
        for p in (2, 3, 4, 6, 8, 12, 16):
            t = time.perf_counter()
            pl = PartitionedLBF(clf, vec, total, p).fit(d["keys"], d["val_neg"])
            bt = time.perf_counter() - t
            rows.append(dict(seed=seed, partitions=p, build_s=bt,
                             **evaluate(pl, d["keys"], d["test_neg"])))
    df = pd.DataFrame(rows)
    df.to_csv("results/exp3_partitions.csv", index=False)
    print(f"\n[Exp 3] Partitions vs FPR / build time at {bpk} bits/key")
    print(df.groupby("partitions")[["fpr", "build_s", "false_neg"]].mean().round(5).to_string())


# ------------------------------------------------ Exp 4: distribution shift
def exp4(seeds, args):
    rows = []
    bpk = args.fixed_bpk
    for seed in seeds:
        d = load(seed, args.source, **args.sizes)
        vec = make_vectorizer()
        clf, _ = train_classifier("LR", vec, d["keys"], d["train_neg"], seed)
        total = bpk * len(d["keys"])
        filters = {"Standard": build_standard(d["keys"], total),
                   "LBF": LearnedBF(clf, vec, total).fit(d["keys"], d["val_neg"]),
                   "PLBF": PartitionedLBF(clf, vec, total, 5).fit(d["keys"], d["val_neg"])}
        for name, f in filters.items():
            for setting, neg in (("same distribution", d["test_neg"]),
                                 ("shifted distribution", d["shift_neg"])):
                rows.append(dict(seed=seed, filter=name, setting=setting,
                                 **evaluate(f, d["keys"], neg)))
    df = pd.DataFrame(rows)
    df.to_csv("results/exp4_drift.csv", index=False)
    print(f"\n[Exp 4] Distribution shift at {bpk} bits/key")
    print(df.groupby(["filter", "setting"])[["fpr", "false_neg"]].mean().round(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "kaggle"])
    args = ap.parse_args()

    if args.quick:
        args.sizes = dict(n_keys=10000, n_train=10000, n_val=10000, n_test=20000)
        seeds = [0]
        args.budgets = [8, 10, 14]
        args.fixed_bpk = 14
        args.exp2_budgets = [14, 24]
    else:
        args.sizes = dict(n_keys=30000, n_train=30000, n_val=30000, n_test=50000)
        seeds = [0, 1, 2]
        args.budgets = [8, 10, 12, 15, 20]
        args.fixed_bpk = 12
        args.exp2_budgets = [12, 20]

    validate_bloom(load(0, args.source, **args.sizes))
    exp1(seeds, args)
    exp2(seeds, args)
    exp3(seeds, args)
    exp4(seeds, args)
    print("\nDone. CSVs are in results/. Next: python plots.py")