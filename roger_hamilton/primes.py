"""Prime number utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _sieve_limit(n: int) -> int:
    """Return an upper bound for the nth prime using the Rosser-Schoenfeld estimate."""
    if n < 6:
        return 15
    return int(n * (math.log(n) + math.log(math.log(n)))) + 10


def first_primes(n: int) -> np.ndarray:
    """Return the first *n* primes as a NumPy array."""
    if n <= 0:
        return np.array([], dtype=int)

    limit = _sieve_limit(n)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = False
    primes = np.nonzero(sieve)[0]
    if primes.size < n:
        # Recursively increase the bound until enough primes are found.
        return first_primes(n + 10)[:n]
    return primes[:n]


__all__ = ["first_primes"]
