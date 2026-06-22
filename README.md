# Bloom filter + dataset deduplication

A from-scratch [Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter) and a
small program that uses it to deduplicate a text dataset, plus a sweep that maps
out the space vs. false-positive-rate trade-off.

## What's here

| file           | what it does |
|----------------|--------------|
| `bloom.py`     | The Bloom filter. Stdlib only (`hashlib`, `math`). ~60 lines. |
| `dedup.py`     | Streams the 20 Newsgroups corpus through the filter to drop duplicates, and grades the result against an exact set. |
| `tradeoff.py`  | Sweeps bits-per-element and plots empirical vs. theoretical FP rate → `tradeoff.png`. |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python dedup.py        # dedup demo + stats
python tradeoff.py     # trade-off table + tradeoff.png
```

`dedup.py` takes `--unique`, `--dup-fraction`, `--bits-per-element`, and `--hashes` flags.

## How it works

**Parameterization.** The filter is constructed from its two physical dials and
nothing else: `BloomFilter(num_bits, num_hashes=7)` = `m` and `k`. Size `m` to
your data (~10 bits per item is the rule of thumb); `k` defaults to 7, the
optimal hash count at ~10 bits/item (~1% false positives). `tradeoff.py` sweeps
both to show how they trade space against the false-positive rate.

**Hashing.** Instead of `k` separate hash functions, we take one 128-bit
`blake2b` digest, split it into two 64-bit halves, and combine them as
`(h1 + i*h2) mod m` for `i = 0..k-1` (the Kirsch–Mitzenmacher double-hashing
trick). One hash computation, `k` well-distributed positions.

**Why a Bloom filter for dedup?** Membership test and insert are O(1) in a fixed
bit array, and the filter has *no false negatives* — anything truly seen always
tests positive, so every real duplicate is caught. The price is *false
positives*: it occasionally claims a brand-new item was seen, so dedup can drop a
genuinely-unique record. `dedup.py` reports exactly how many. In the demo the
filter is ~7 KB while an exact set of the document text would be ~4 MB — that
600× memory win, at the cost of a tunable trickle of wrongly-dropped items, is
the entire reason you'd reach for one (e.g. deduping a corpus too large to hold
an exact set of).

## The trade-off curve

`tradeoff.py` holds the data fixed and sweeps memory (bits per element):

```
bits/elem  memory(KB)   k  theoretical FP  empirical FP
      2.0         1.0   1         0.32684       0.34325
      4.0         1.9   3         0.13448       0.05376
      6.0         2.9   4         0.05398       0.01661
      8.0         3.8   6         0.02131       0.00334
     10.0         4.8   7         0.00816       0.00077
     12.0         5.7   8         0.00313       0.00077
     15.0         7.2  10         0.00074       0.00000
     20.0         9.5  14         0.00007       0.00000
```

Every few extra bits per element drops the FP rate by roughly 10×. Empirical
generally sits a bit *below* theory because the textbook rate assumes a full
filter, whereas in a streaming dedup most lookups happen while the filter is
still filling up.

## Things I'd revisit with more time

- **Near-duplicates, not just exact.** Real corpus dedup wants near-duplicate
  detection (MinHash / SimHash over shingles), feeding fingerprints into the
  Bloom filter. Here a single edited character counts as a new document.
- **Hash speed.** `blake2b` is cryptographic and overkill; a non-crypto hash like
  `mmh3` would be faster. I stuck to stdlib to keep it dependency-free.
- **Bit storage.** `bytearray` is clear; for large `m`, NumPy or `int.bit_*`
  tricks would cut memory/time. Also no serialization (save/load) yet.
- **Plot artifact.** Zero empirical FPs are clamped to 1e-5 so they show on the
  log axis, which reads as a floor — it isn't one.
- **Scale.** I used ~4k docs so it runs in seconds. The interesting regime is
  billions of items, where the exact-set baseline stops fitting in memory at all.
```
