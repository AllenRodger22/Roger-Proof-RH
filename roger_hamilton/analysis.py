"""Analysis utilities for comparing the Roger-Hamilton spectrum with zeta zeros."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .zeros import load_riemann_zeros


@dataclass
class ComparisonResult:
    m: int
    zeros: np.ndarray
    spectrum: np.ndarray
    delta: np.ndarray
    rmse: float
    max_abs_error: float


@dataclass
class SpacingStatistics:
    ks_statistic: float
    mean_gap_ratio: float
    gaps: np.ndarray
    normalised_gaps: np.ndarray


_DEF_PI = np.pi


def compare_spectrum(eigenvalues: np.ndarray, m: int) -> ComparisonResult:
    zeros = load_riemann_zeros(m)
    m = min(m, len(eigenvalues), len(zeros))
    spectrum = eigenvalues[:m]
    delta = spectrum - zeros[:m]
    rmse = float(np.sqrt(np.mean(delta**2)))
    max_abs = float(np.max(np.abs(delta)))
    return ComparisonResult(m=m, zeros=zeros[:m], spectrum=spectrum, delta=delta, rmse=rmse, max_abs_error=max_abs)


def _gue_cdf(s: np.ndarray) -> np.ndarray:
    a = 4.0 / _DEF_PI
    exp_term = np.exp(-a * s**2)
    return 1.0 - exp_term * (1.0 + 2.0 * s / np.sqrt(_DEF_PI))


def _empirical_ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    sorted_vals = np.sort(values)
    return np.searchsorted(sorted_vals, grid, side="right") / values.size


def spacing_statistics(eigenvalues: np.ndarray, count: int) -> SpacingStatistics:
    count = min(count, len(eigenvalues))
    if count < 3:
        raise ValueError("Need at least three eigenvalues for spacing statistics")
    values = eigenvalues[:count]
    gaps = np.diff(values)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        raise ValueError("No positive gaps available for statistics")
    normalised = positive / np.mean(positive)
    grid = np.linspace(0.0, normalised.max() * 1.2, 2048)
    empirical = _empirical_ecdf(normalised, grid)
    theoretical = _gue_cdf(grid)
    ks = float(np.max(np.abs(empirical - theoretical)))

    next_gaps = positive[1:]
    prev_gaps = positive[:-1]
    ratios = np.minimum(next_gaps, prev_gaps) / np.maximum(next_gaps, prev_gaps)
    mean_ratio = float(np.mean(ratios))
    return SpacingStatistics(ks_statistic=ks, mean_gap_ratio=mean_ratio, gaps=gaps, normalised_gaps=normalised)


def save_level2_table(result: ComparisonResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n", "gamma_n", "E_n", "delta_n"])
        for idx, (g, e, d) in enumerate(zip(result.zeros, result.spectrum, result.delta, strict=False), start=1):
            writer.writerow([idx, f"{g:.12f}", f"{e:.12f}", f"{d:.12e}"])


def save_unitarity_report(path: Path, norm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"unitarity_norm={norm:.3e}\n")


def save_params(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(params, f, indent=2, sort_keys=True)


__all__ = [
    "ComparisonResult",
    "SpacingStatistics",
    "compare_spectrum",
    "spacing_statistics",
    "save_level2_table",
    "save_unitarity_report",
    "save_params",
]
