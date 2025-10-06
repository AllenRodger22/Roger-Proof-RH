"""Core utilities for constructing and analysing the Roger-Hamilton operator."""

from .primes import first_primes
from .unitary import build_azimuthal_phase, construct_unitary
from .hamiltonian import eigenvalues_from_unitary
from .analysis import (
    load_riemann_zeros,
    compare_spectrum,
    spacing_statistics,
    save_level2_table,
)
from .refinement import refine_pipeline, save_refinement_summary

__all__ = [
    "first_primes",
    "build_azimuthal_phase",
    "construct_unitary",
    "eigenvalues_from_unitary",
    "load_riemann_zeros",
    "compare_spectrum",
    "spacing_statistics",
    "save_level2_table",
    "refine_pipeline",
    "save_refinement_summary",
]
