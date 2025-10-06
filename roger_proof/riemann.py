"""Helpers for retrieving Riemann zeta zeros."""

from __future__ import annotations

from functools import lru_cache
from typing import List

import mpmath as mp


@lru_cache(maxsize=None)
def riemann_zeros(count: int) -> List[float]:
    """Return the first ``count`` ordinates of Riemann zeta zeros.

    The values are computed using :func:`mpmath.zetazero` with 80 digits of
    precision and returned as Python floats.
    """
    mp.mp.dps = 80
    zeros = [float(mp.zetazero(n + 1)) for n in range(count)]
    return zeros
