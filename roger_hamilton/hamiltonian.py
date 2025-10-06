"""Diagonalisation utilities for the Roger-Hamilton operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class Spectrum:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def eigenvalues_from_unitary(
    U: np.ndarray,
    mode: Literal["log", "cayley"] = "log",
) -> Spectrum:
    """Return the eigenvalues of ``H`` obtained from the unitary ``U``."""
    w, V = np.linalg.eig(U)
    theta = np.angle(w)
    if mode == "log":
        E = theta
    elif mode == "cayley":
        with np.errstate(divide="ignore", invalid="ignore"):
            E = -np.cos(theta / 2.0) / np.sin(theta / 2.0)
        # Replace infinities (where theta == 0 modulo pi) with large values
        mask = ~np.isfinite(E)
        E[mask] = np.sign(theta[mask]) * np.inf
    else:
        raise ValueError("mode must be 'log' or 'cayley'")

    order = np.argsort(E)
    return Spectrum(eigenvalues=E[order], eigenvectors=V[:, order])


__all__ = ["Spectrum", "eigenvalues_from_unitary"]
