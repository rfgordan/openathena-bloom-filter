"""A small, dependency-free Bloom filter.

A Bloom filter is a probabilistic set. `add` and "is x in the set?" run in
constant time using a fixed bit array, no matter how many items you store.
The trade-off: membership tests can return false positives (it may say "yes"
for something never added), but *never* false negatives (a real member always
tests positive).
"""

import math
from hashlib import blake2b


class BloomFilter:
    def __init__(self, num_bits, num_hashes=7):
        """Construct directly from the two physical dials:

        num_bits (m):   size of the underlying bit array. Pick this to match how
                        many items you'll add; ~10 bits per item is a good rule
                        of thumb. (No universal default -- it scales with data.)
        num_hashes (k): number of hash functions / bits set per item. Defaults to
                        7, the optimal k at ~10 bits per item, which lands around
                        a 1% false-positive rate -- a common sweet spot.

        """
        self.num_bits = max(1, int(num_bits))
        self.num_hashes = max(1, int(num_hashes))

        self.bits = bytearray((self.num_bits + 7) // 8)  # ceil(num_bits / 8) bytes
        self.count = 0

    def _indices(self, item):
        """Yield the k bit positions for `item`.

        Computing k independent hashes is wasteful. Instead we take one 128-bit
        blake2b digest, split it into two 64-bit halves h1 and h2, and build k
        positions as (h1 + i*h2) mod m. This is the Kirsch-Mitzenmacher double-
        hashing trick: it gives k well-distributed positions from a single hash
        with no measurable hit to the false-positive rate.
        """

        # TODO: faster hash function, potentially hash k inputs instead to avoid degenerate case
        # TODO: locally sensitive hashing for near-dup
        data = item if isinstance(item, bytes) else str(item).encode("utf-8")
        digest = blake2b(data, digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:], "big")
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    def add(self, item):
        for idx in self._indices(item):
            self.bits[idx >> 3] |= 1 << (idx & 7)
        self.count += 1

    def __contains__(self, item):
        # Positive only if *every* one of the k bits is set. A single unset bit
        # proves the item was never added -> no false negatives, ever.
        return all(self.bits[idx >> 3] & (1 << (idx & 7)) for idx in self._indices(item))

    @property
    def num_bytes(self):
        return len(self.bits)

    def expected_fp_rate(self):
        """Theoretical FP rate given how many items have actually been added:
        (1 - e^(-k*count/m))^k."""
        if self.count == 0:
            return 0.0
        return (1 - math.exp(-self.num_hashes * self.count / self.num_bits)) ** self.num_hashes

    def __repr__(self):
        return (f"BloomFilter(m={self.num_bits} bits / {self.num_bytes} bytes, "
                f"k={self.num_hashes}, count={self.count})")
