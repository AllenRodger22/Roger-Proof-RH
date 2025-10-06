"""Prime number utilities."""

from __future__ import annotations

from typing import List


def primes_up_to(limit: int) -> List[int]:
    """Return all primes ``<= limit`` using a simple sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start: limit + 1: step] = [False] * ((limit - start) // step + 1)
    return [i for i, flag in enumerate(sieve) if flag]
