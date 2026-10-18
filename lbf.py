import math
import time

import numpy as np

from bloom import BloomFilter, optimal_k, theory_fpr
from features import model_size_bits, score

LN2 = math.log(2)


class LearnedBF:
    """Classifier first, backup Bloom filter for the keys the classifier misses."""
    name = "LBF"

    def __init__(self, clf, vec, total_bits):
        self.clf, self.vec, self.total_bits = clf, vec, total_bits
        self.model_bits = model_size_bits(clf)

    def fit(self, keys, val_neg):
        keys = list(keys)
        budget = self.total_bits - self.model_bits
        if budget < 64:
            raise ValueError(f"model ({self.model_bits} bits) does not fit in budget "
                             f"({self.total_bits} bits)")
        ks, vs = score(self.clf, self.vec, keys), score(self.clf, self.vec, val_neg)

        # choose tau on VALIDATION non-keys: minimise estimated total FPR
        best = None
        for q in np.geomspace(1e-4, 0.5, 60):          # model-only FPR targets
            tau = float(np.quantile(vs, 1 - q))
            f_model = float((vs >= tau).mean())
            n_fn = int((ks < tau).sum())               # keys the model misses
            if n_fn == 0:
                est = f_model
            else:
                est = f_model + (1 - f_model) * theory_fpr(budget, n_fn,
                                                           optimal_k(budget, n_fn))
            if best is None or est < best[0]:
                best = (est, tau)
        self.tau = best[1]

        missed = [u for u, s in zip(keys, ks) if s < self.tau]
        self.backup = BloomFilter(budget, n_expected=max(1, len(missed)))
        self.backup.add_many(missed)
        self.n_backup_keys = len(missed)
        return self

    def query_many(self, urls):
        urls = list(urls)
        out = score(self.clf, self.vec, urls) >= self.tau
        idx = np.where(~out)[0]
        if len(idx):
            out[idx] = self.backup.query_many([urls[i] for i in idx])
        return out

    @property
    def size_bits(self):
        return self.model_bits + self.backup.size_bits


class PartitionedLBF:
    """PLBF-lite: p score regions, each with its own backup filter."""
    name = "PLBF"

    def __init__(self, clf, vec, total_bits, n_regions=5):
        self.clf, self.vec, self.total_bits, self.p = clf, vec, total_bits, n_regions
        self.model_bits = model_size_bits(clf)

    def _bits_for(self, lam, n, g):
        f = np.minimum(1.0, lam * n / g)
        bits = np.where(f < 1.0,
                        n * (1 / LN2) * np.log2(1.0 / np.maximum(f, 1e-300)), 0.0)
        return f, bits

    def fit(self, keys, val_neg):
        t0 = time.perf_counter()
        keys = list(keys)
        budget = self.total_bits - self.model_bits
        if budget < 64 * self.p:
            raise ValueError("model does not fit in budget")
        ks, vs = score(self.clf, self.vec, keys), score(self.clf, self.vec, val_neg)

        qs = np.linspace(0, 1, self.p + 1)[1:-1]
        self.edges = np.unique(np.quantile(ks, qs))
        P = len(self.edges) + 1
        kr = np.searchsorted(self.edges, ks, side="right")
        vr = np.searchsorted(self.edges, vs, side="right")
        n = np.bincount(kr, minlength=P).astype(float)               # keys per region
        g = (np.bincount(vr, minlength=P) + 1.0) / (len(vs) + P)     # safe-URL share

        has = n > 0
        lo, hi = -80.0, 20.0                       # bisection on log(lambda)
        for _ in range(200):
            mid = (lo + hi) / 2
            _, bits = self._bits_for(math.exp(mid), n[has], g[has])
            if bits.sum() > budget:
                lo = mid
            else:
                hi = mid
        f_has, bits_has = self._bits_for(math.exp(hi), n[has], g[has])
        f = np.zeros(P); f[has] = f_has
        bits = np.zeros(P); bits[has] = bits_has

        self.mode, self.filters = [], []
        for i in range(P):
            if n[i] == 0:
                self.mode.append("reject"); self.filters.append(None)
            elif f[i] >= 1.0:
                self.mode.append("accept"); self.filters.append(None)
            else:
                m_i = max(8, int(bits[i]))
                bf = BloomFilter(m_i, n_expected=int(n[i]))
                bf.add_many([u for u, r in zip(keys, kr) if r == i])
                self.mode.append("bloom"); self.filters.append(bf)
        self.n_regions, self.region_f = P, f
        self.build_seconds = time.perf_counter() - t0
        return self

    def query_many(self, urls):
        urls = list(urls)
        s = score(self.clf, self.vec, urls)
        reg = np.searchsorted(self.edges, s, side="right")
        out = np.zeros(len(urls), dtype=bool)
        for i, mode in enumerate(self.mode):
            idx = np.where(reg == i)[0]
            if not len(idx):
                continue
            if mode == "accept":
                out[idx] = True
            elif mode == "bloom":
                out[idx] = self.filters[i].query_many([urls[j] for j in idx])
        return out

    @property
    def size_bits(self):
        return self.model_bits + sum(f.size_bits for f in self.filters if f is not None)