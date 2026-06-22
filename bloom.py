import math
from hashlib import blake2b


class BloomFilter:
    def __init__(self, num_bits, num_hashes=7):
        self.num_bits = max(1, int(num_bits))
        self.num_hashes = max(1, int(num_hashes))

        self.bits = bytearray((self.num_bits + 7) // 8)  # ceil(num_bits / 8) bytes
        self.count = 0

    def _indices(self, item):
        # TODO: faster hash function, potentially hash k inputs instead to avoid degenerate case
        # TODO: think more about possible degenerate case for large gcd(h2, m)
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
        if self.count == 0:
            return 0.0
        return (1 - math.exp(-self.num_hashes * self.count / self.num_bits)) ** self.num_hashes

    def __repr__(self):
        return (f"BloomFilter(m={self.num_bits} bits / {self.num_bytes} bytes, "
                f"k={self.num_hashes}, count={self.count})")
