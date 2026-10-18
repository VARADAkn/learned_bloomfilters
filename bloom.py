import hashlib
import math
import numpy as np


def optimal_k(m, n):
    """Eq. (5): k_opt = (m/n) ln 2."""
    if n <= 0:
        return 1
    return max(1, int(round((m / n) * math.log(2))))


def theory_fpr(m, n, k):
    """Eq. (4): (1 - e^{-kn/m})^k."""
    if n <= 0:
        return 0.0
    return (1.0 - math.exp(-k * n / m)) ** k


def _hash_pairs(items):
    h1 = np.empty(len(items), dtype=np.uint64)
    h2 = np.empty(len(items), dtype=np.uint64)
    for i, s in enumerate(items):
        d = hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=16).digest()
        h1[i] = int.from_bytes(d[:8], "little")
        h2[i] = int.from_bytes(d[8:], "little") | 1   # odd -> better spread
    return h1, h2


class BloomFilter:
    def __init__(self, m, k=None, n_expected=None):
        self.m = int(max(1, m))
        if k is None:
            k = optimal_k(self.m, n_expected) if n_expected else 1
        self.k = int(k)
        self.bits = np.zeros(self.m, dtype=bool)   # logically m bits
        self.n = 0

    def _positions(self, items):
        h1, h2 = _hash_pairs(items)
        i = np.arange(self.k, dtype=np.uint64)[None, :]
        with np.errstate(over="ignore"):
            pos = (h1[:, None] + i * h2[:, None]) % np.uint64(self.m)
        return pos.astype(np.int64)

    def add_many(self, items):
        items = list(items)
        if not items:
            return
        self.bits[self._positions(items).ravel()] = True
        self.n += len(items)

    def query_many(self, items):
        items = list(items)
        if not items:
            return np.zeros(0, dtype=bool)
        if self.n == 0:
            return np.zeros(len(items), dtype=bool)
        return self.bits[self._positions(items)].all(axis=1)

    def add(self, x):
        self.add_many([x])

    def query(self, x):
        return bool(self.query_many([x])[0])

    @property
    def size_bits(self):
        return self.m