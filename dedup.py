import argparse
import random

from sklearn.datasets import fetch_20newsgroups

from bloom import BloomFilter


def load_stream(num_unique, dup_fraction, seed=0):
    docs = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).data
    rng = random.Random(seed)

    unique = rng.sample(docs, num_unique)
    num_dupes = round(num_unique * dup_fraction / (1 - dup_fraction))
    stream = unique + [rng.choice(unique) for _ in range(num_dupes)]
    rng.shuffle(stream)
    return stream


def dedup(stream, num_bits, num_hashes):
    bloom = BloomFilter(num_bits, num_hashes)
    seen_exact = set()

    kept = []
    true_dupes_removed = 0
    false_positives = 0  # unique docs wrongly dropped because the Bloom filter lied

    for item in stream:
        print(f"item:{item}")
        if item in bloom:
            print("COPY")
            # Bloom says "seen before" -> we drop it as a duplicate.
            if item not in seen_exact:
                print("FALSE POSITIVE")
                false_positives += 1  # ...but it was actually new. Data loss.
            else:
                print("TRUE POSITIVE")
                true_dupes_removed += 1
            continue
        bloom.add(item)
        seen_exact.add(item)
        kept.append(item)

    return kept, bloom, {
        "stream_size": len(stream),
        "actual_unique": len(seen_exact),
        "kept": len(kept),
        "true_dupes_removed": true_dupes_removed,
        "false_positives": false_positives,
        "empirical_fp_rate": false_positives / max(1, len(seen_exact)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unique", type=int, default=4000, help="number of distinct documents")
    ap.add_argument("--dup-fraction", type=float, default=0.30, help="fraction of stream that is duplicates")
    ap.add_argument("--bits-per-element", type=float, default=10.0, help="filter size in bits per stream item (~10 ≈ 1%% FP)")
    ap.add_argument("--hashes", type=int, default=7, help="number of hash functions k")
    args = ap.parse_args()

    stream = load_stream(args.unique, args.dup_fraction)
    num_bits = round(args.bits_per_element * len(stream))
    kept, bloom, stats = dedup(stream, num_bits, args.hashes)

    print(bloom)
    print()
    print(f"  stream size              {stats['stream_size']:>8,}")
    print(f"  actually-unique docs     {stats['actual_unique']:>8,}")
    print(f"  kept after dedup         {stats['kept']:>8,}")
    print(f"  true duplicates removed  {stats['true_dupes_removed']:>8,}")
    print(f"  false positives (lost)   {stats['false_positives']:>8,}  <- unique docs wrongly dropped")
    print()
    print(f"  formula FP rate          {bloom.expected_fp_rate():>8.4f}  (predicted at this load)")
    print(f"  empirical FP rate        {stats['empirical_fp_rate']:>8.4f}")
    print(f"  filter memory            {bloom.num_bytes / 1024:>8.1f} KB")
    exact_bytes = sum(len(d.encode()) for d in set(stream))
    print(f"  exact set would need     {exact_bytes / 1024:>8.1f} KB of raw document text")


if __name__ == "__main__":
    main()
