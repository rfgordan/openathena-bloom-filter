"""Explore the Bloom filter space vs. false-positive-rate trade-off.

We hold the dataset fixed and sweep the only real dial: bits per element (m/n).
For each setting we (a) compute the theoretical FP rate and (b) measure the
empirical FP rate by actually deduplicating the stream. The two should track
closely, which validates that the double-hashing scheme behaves like ideal
independent hashes.

Output: a printed table plus tradeoff.png.
"""

import math

import matplotlib
matplotlib.use("Agg")  # headless: write a file, don't open a window
import matplotlib.pyplot as plt

from dedup import dedup, load_stream


def run():
    stream = load_stream(num_unique=4000, dup_fraction=0.30)
    n = len(set(stream))  # distinct items -> the "elements" in bits-per-element
    bits_per_element = [2, 4, 6, 8, 10, 12, 15, 20]

    rows = []
    for bpe in bits_per_element:
        # Sweep the dials directly: m = bits/element * n, and k at its optimum
        # for that ratio, k = (m/n)*ln2 = bpe*ln2.
        num_bits = round(bpe * n)
        num_hashes = max(1, round(bpe * math.log(2)))
        _, bloom, stats = dedup(stream, num_bits, num_hashes)
        rows.append({
            "bpe": bpe,  # the dial we set, m / n
            "kb": bloom.num_bytes / 1024,
            "k": bloom.num_hashes,
            "theoretical": bloom.expected_fp_rate(),  # (1 - e^-kn/m)^k at actual load
            "empirical": stats["empirical_fp_rate"],
        })

    print(f"{'bits/elem':>9} {'memory(KB)':>11} {'k':>3} {'theoretical FP':>15} {'empirical FP':>13}")
    for r in rows:
        print(f"{r['bpe']:>9.1f} {r['kb']:>11.1f} {r['k']:>3} {r['theoretical']:>15.5f} {r['empirical']:>13.5f}")

    plot(rows)


def plot(rows):
    kb = [r["kb"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(kb, [r["theoretical"] for r in rows], "o-", label="theoretical")
    ax.plot(kb, [max(r["empirical"], 1e-5) for r in rows], "s--", label="empirical (measured)")
    ax.set_yscale("log")
    ax.set_xlabel("filter memory (KB)")
    ax.set_ylabel("false-positive rate (log scale)")
    ax.set_title("Bloom filter: space vs. false-positive rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("tradeoff.png", dpi=120)
    print("\nwrote tradeoff.png")


if __name__ == "__main__":
    run()
