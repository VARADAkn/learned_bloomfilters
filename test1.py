from bloom import BloomFilter
bf = BloomFilter(m=1000, n_expected=100)
bf.add("evil.com")
print(bf.query("evil.com"))    # True
print(bf.query("google.com"))  # almost always False