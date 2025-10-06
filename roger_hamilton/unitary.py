"""Construction of the Roger-Hamilton unitary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .primes import first_primes


@dataclass
class UnitaryComponents:
    """Container for intermediate data used to build the unitary operator."""

    t: np.ndarray
    phi: np.ndarray
    shift_matrix: np.ndarray
    modulation_matrix: np.ndarray


@dataclass
class UnitaryResult:
    """Return type for :func:`construct_unitary`."""

    matrix: np.ndarray
    components: UnitaryComponents


def build_azimuthal_phase(
    t: np.ndarray,
    P: int,
    beta: float,
    omega: float,
    seed: int | None = None,
    phase_mode: str = "deterministic",
) -> np.ndarray:
    """Construct the Azimuthal phase ``Phi`` evaluated on the grid ``t``."""
    primes = first_primes(P)
    logp = np.log(primes)
    rng = np.random.default_rng(seed)
    if phase_mode == "random":
        theta_p = rng.uniform(0.0, 2 * np.pi, size=logp.shape)
    elif phase_mode == "deterministic":
        theta_p = np.zeros_like(logp)
    else:
        raise ValueError("phase_mode must be 'deterministic' or 'random'")

    phi = np.zeros_like(t, dtype=np.float64)
    weights = np.exp(-beta * logp)
    for lp, theta, weight in zip(logp, theta_p, weights, strict=False):
        phi += np.sin(omega * lp * t + theta) * weight

    std = phi.std()
    if std == 0.0:
        return phi
    return phi / std


def _unitary_fft_matrix(N: int) -> Tuple[np.ndarray, np.ndarray]:
    j = np.arange(N)[:, None]
    k = np.arange(N)[None, :]
    F = np.exp(-2j * np.pi * j * k / N) / np.sqrt(N)
    return F, F.conjugate().T


def _fractional_shift_matrix(N: int, tau: float) -> np.ndarray:
    kvec = np.arange(-N // 2, N // 2)
    phase = np.exp(1j * kvec * tau)
    return np.fft.ifftshift(np.diag(phase))


def construct_unitary(
    N: int,
    T: float,
    tau: float,
    P: int,
    beta: float,
    omega: float,
    seed: int | None = None,
    phase_mode: str = "deterministic",
) -> UnitaryResult:
    """Build the unitary operator ``U = M_phi S_tau`` on the chosen grid."""
    t = np.linspace(-T, T, N, endpoint=False)
    phi = build_azimuthal_phase(t, P=P, beta=beta, omega=omega, seed=seed, phase_mode=phase_mode)
    modulation = np.diag(np.exp(1j * phi))

    F, Fh = _unitary_fft_matrix(N)
    Dk = _fractional_shift_matrix(N, tau)
    shift = Fh @ Dk @ F
    unitary = modulation @ shift
    return UnitaryResult(matrix=unitary, components=UnitaryComponents(t=t, phi=phi, shift_matrix=shift, modulation_matrix=modulation))


def serialise_params(params: Dict[str, float | int | str]) -> str:
    return json.dumps(params, indent=2, sort_keys=True)


__all__ = ["UnitaryComponents", "UnitaryResult", "build_azimuthal_phase", "construct_unitary", "serialise_params"]
