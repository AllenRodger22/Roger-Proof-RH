"""Statistical helpers for spectrum analysis."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def wigner_surmise_cdf(s: np.ndarray) -> np.ndarray:
    """Return the cumulative distribution function of the GUE Wigner surmise."""
    s = np.asarray(s)
    return 1.0 - np.exp(-4.0 * s ** 2 / math.pi) * (1.0 + 2.0 * s / math.pi)


def kolmogorov_smirnov(empirical: Sequence[float], theoretical_cdf) -> float:
    """Compute the Kolmogorov–Smirnov statistic for 1D data."""
    data = np.sort(np.asarray(empirical))
    n = data.size
    if n == 0:
        return float("nan")
    cdf_values = theoretical_cdf(data)
    empirical_cdf = (np.arange(1, n + 1)) / n
    empirical_cdf_prev = (np.arange(n)) / n
    diffs = np.maximum(np.abs(empirical_cdf - cdf_values), np.abs(cdf_values - empirical_cdf_prev))
    return float(np.max(diffs))


def gap_ratio(spacings: Sequence[float]) -> float:
    """Compute the average consecutive gap ratio."""
    s = np.asarray(spacings)
    if s.size < 2:
        return float("nan")
    ratios = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.mean(ratios))


def operator_norm(matrix: np.ndarray) -> float:
    """Compute the spectral norm of a matrix."""
    eigvals = np.linalg.eigvalsh(matrix.conj().T @ matrix)
    return float(np.sqrt(np.max(eigvals.real)))
