from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .analysis import (
    ComparisonResult,
    SpacingStatistics,
    save_level2_table,
    save_params,
    save_unitarity_report,
    spacing_statistics,
)
from .hamiltonian import eigenvalues_from_unitary
from .unitary import construct_unitary


@dataclass
class ParameterSet:
    omega: float
    tau: float
    P: int
    beta: float

    def as_dict(self) -> dict[str, float | int]:
        return {"omega": self.omega, "tau": self.tau, "P": self.P, "beta": self.beta}


@dataclass
class StepSizes:
    omega: float
    tau: float
    P: int
    beta: float

    def shrink(self, factor: float, min_P_step: int = 50) -> None:
        self.omega /= factor
        self.tau /= factor
        self.beta /= factor
        next_P = max(min_P_step, int(round(self.P / factor)))
        self.P = max(min_P_step, next_P)


@dataclass
class CandidateResult:
    params: ParameterSet
    comparison: ComparisonResult
    spacing: SpacingStatistics
    unitarity_norm: float
    min_distance_to_minus_one: float
    levels_used: int
    N: int
    T: float
    seed: int | None
    phase_mode: str

    @property
    def rmse(self) -> float:
        return self.comparison.rmse

    @property
    def max_delta(self) -> float:
        return self.comparison.max_abs_error

    @property
    def ks(self) -> float:
        return self.spacing.ks_statistic

    @property
    def gap_ratio(self) -> float:
        return self.spacing.mean_gap_ratio


@dataclass
class RefinementResult:
    best: CandidateResult
    replicate: CandidateResult
    loop_count: int
    convergence_reached: bool
    history: list[CandidateResult]


def _unitarity_norm(matrix: np.ndarray) -> float:
    identity = np.eye(matrix.shape[0], dtype=complex)
    residue = matrix.conjugate().T @ matrix - identity
    return float(np.linalg.norm(residue, ord=2))


def _distance_to_minus_one(eigenvalues: np.ndarray) -> float:
    angles = np.angle(eigenvalues)
    distance = np.minimum(np.abs(angles - math.pi), np.abs(angles + math.pi))
    return float(np.min(distance))


def _plot_delta(delta: np.ndarray, path: Path, loop: int, rmse: float, ks: float) -> None:
    count = min(30, delta.size)
    indices = np.arange(1, count + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(indices, delta[:count], marker="o", linestyle="-", label="Δₙ")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("n")
    ax.set_ylabel("Δₙ")
    ax.set_title(f"Loop {loop} | RMSE={rmse:.2e}, KS={ks:.3f}")
    ax.grid(True)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_spacing_histogram(normalised_gaps: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(normalised_gaps, bins=40, density=True, alpha=0.7, label="Empirical")
    s = np.linspace(0, normalised_gaps.max() * 1.1, 400)
    pdf = (32.0 / (math.pi**2)) * (s**2) * np.exp(-4.0 * s**2 / math.pi)
    ax.plot(s, pdf, color="black", label="GUE surmise")
    ax.set_xlabel("Normalised spacing")
    ax.set_ylabel("Density")
    ax.set_title("Spacing histogram")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_spacing_ecdf(normalised_gaps: np.ndarray, path: Path) -> None:
    sorted_gaps = np.sort(normalised_gaps)
    ecdf = np.arange(1, sorted_gaps.size + 1) / sorted_gaps.size
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(sorted_gaps, ecdf, where="post", label="Empirical")
    s = np.linspace(0, sorted_gaps.max() * 1.1, 400)
    cdf = 1.0 - np.exp(-4.0 * s**2 / math.pi) * (1.0 + 2.0 * s / math.sqrt(math.pi))
    ax.plot(s, cdf, color="black", label="GUE surmise")
    ax.set_xlabel("Normalised spacing")
    ax.set_ylabel("ECDF")
    ax.set_title("Spacing ECDF")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_grid(center: ParameterSet, steps: StepSizes, bounds: dict[str, tuple[float, float] | tuple[int, int]], grid_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omega_bounds = bounds["omega"]
    tau_bounds = bounds["tau"]
    beta_bounds = bounds["beta"]
    P_bounds = bounds["P"]

    omegas = np.linspace(center.omega - steps.omega, center.omega + steps.omega, grid_points)
    omegas = np.clip(omegas, omega_bounds[0], omega_bounds[1])
    omegas = np.unique(np.round(omegas, decimals=9))

    taus = np.linspace(center.tau - steps.tau, center.tau + steps.tau, grid_points)
    taus = np.clip(taus, tau_bounds[0], tau_bounds[1])
    taus = np.unique(np.round(taus, decimals=9))

    betas = np.linspace(center.beta - steps.beta, center.beta + steps.beta, grid_points)
    betas = np.clip(betas, beta_bounds[0], beta_bounds[1])
    betas = np.unique(np.round(betas, decimals=9))

    if steps.P == 0:
        candidate_P = [center.P]
    else:
        candidate_P = np.linspace(center.P - steps.P, center.P + steps.P, grid_points)
    candidate_P = np.unique(np.round(candidate_P).astype(int))
    candidate_P = candidate_P[(candidate_P >= P_bounds[0]) & (candidate_P <= P_bounds[1])]
    if candidate_P.size == 0:
        candidate_P = np.array([int(np.clip(center.P, P_bounds[0], P_bounds[1]))])

    return omegas, taus, candidate_P, betas


def _evaluate_candidate(
    N: int,
    T: float,
    params: ParameterSet,
    m: int,
    spacing_count: int,
    zeros: np.ndarray,
    seed: int | None,
    phase_mode: str,
) -> CandidateResult:
    unitary_result = construct_unitary(
        N=N,
        T=T,
        tau=params.tau,
        P=params.P,
        beta=params.beta,
        omega=params.omega,
        seed=seed,
        phase_mode=phase_mode,
    )
    matrix = unitary_result.matrix
    norm = _unitarity_norm(matrix)
    if norm > 1e-10:
        raise ValueError(f"Unitarity violation detected: norm={norm:.3e}")

    raw_eigs = np.linalg.eigvals(matrix)
    distance = _distance_to_minus_one(raw_eigs)
    if distance < 1e-6:
        raise ValueError("Eigenvalues too close to -1, aborting to maintain principal branch")

    spectrum = eigenvalues_from_unitary(matrix, mode="log")
    eigenvalues = spectrum.eigenvalues
    m_eval = min(m, zeros.size, eigenvalues.size)
    zeros_slice = zeros[:m_eval]
    spectrum_slice = eigenvalues[:m_eval]
    if m_eval == 0:
        delta = np.empty(0, dtype=float)
        rmse = float("inf")
        max_abs = float("inf")
    else:
        delta = spectrum_slice - zeros_slice
        rmse = float(np.sqrt(np.mean(delta**2)))
        max_abs = float(np.max(np.abs(delta)))
    comparison = ComparisonResult(
        m=m_eval,
        zeros=zeros_slice,
        spectrum=spectrum_slice,
        delta=delta,
        rmse=rmse,
        max_abs_error=max_abs,
    )
    spacing = spacing_statistics(eigenvalues, spacing_count)
    levels_used = min(spacing_count, eigenvalues.size)
    return CandidateResult(
        params=params,
        comparison=comparison,
        spacing=spacing,
        unitarity_norm=norm,
        min_distance_to_minus_one=distance,
        levels_used=levels_used,
        N=N,
        T=T,
        seed=seed,
        phase_mode=phase_mode,
    )


def refine_pipeline(
    zeros: np.ndarray,
    N: int = 512,
    T: float = 24.0,
    tau: float = 0.37,
    P: int = 700,
    beta: float = 1.0,
    omega: float = 1.05,
    seed: int | None = 42,
    phase_mode: str = "deterministic",
    m: int = 30,
    spacing_count: int = 200,
    loops: int = 8,
    grid_points: int = 5,
    shrink_factor: float = 1.5,
    rmse_tolerance: float = 1e-9,
    secondary_grid: tuple[int, float] = (768, 28.0),
    output_dir: Path | str = Path("artifacts/refine"),
    d_omega: float = 0.02,
    d_tau: float = 0.04,
    d_P: int = 100,
    d_beta: float = 0.1,
) -> RefinementResult:
    output_path = Path(output_dir)
    snapshot_dir = output_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    center = ParameterSet(omega=omega, tau=tau, P=P, beta=beta)
    steps = StepSizes(omega=d_omega, tau=d_tau, P=d_P, beta=d_beta)
    bounds = {
        "omega": (1.0, 1.10),
        "tau": (0.05, 0.9),
        "P": (400, 1200),
        "beta": (0.8, 1.2),
    }

    best_global: CandidateResult | None = None
    history: list[CandidateResult] = []
    previous_rmse: float | None = None
    convergence_reached = False

    for loop in range(1, loops + 1):
        omegas, taus, Ps, betas = _build_grid(center, steps, bounds, grid_points)
        candidates: list[CandidateResult] = []
        for omega_val, tau_val, P_val, beta_val in itertools.product(omegas, taus, Ps, betas):
            params = ParameterSet(omega=float(omega_val), tau=float(tau_val), P=int(P_val), beta=float(beta_val))
            try:
                candidate = _evaluate_candidate(
                    N=N,
                    T=T,
                    params=params,
                    m=m,
                    spacing_count=spacing_count,
                    zeros=zeros,
                    seed=seed,
                    phase_mode=phase_mode,
                )
            except ValueError:
                continue
            candidates.append(candidate)

        if not candidates:
            raise RuntimeError("All candidates rejected due to unitarity or branch issues")

        best_candidate = min(candidates, key=lambda c: (c.rmse, c.ks))
        history.append(best_candidate)

        print(
            f"[loop {loop:02d}] ω={best_candidate.params.omega:.5f} "
            f"τ={best_candidate.params.tau:.5f} P={best_candidate.params.P} "
            f"β={best_candidate.params.beta:.5f} "
            f"RMSE={best_candidate.rmse:.3e} KS={best_candidate.ks:.3f}"
        )

        _plot_delta(
            best_candidate.comparison.delta,
            snapshot_dir / f"loop_{loop:02d}_delta.png",
            loop,
            best_candidate.rmse,
            best_candidate.ks,
        )
        with (snapshot_dir / f"loop_{loop:02d}_best.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "loop": loop,
                    "params": best_candidate.params.as_dict(),
                    "rmse": best_candidate.rmse,
                    "max_delta": best_candidate.max_delta,
                    "KS": best_candidate.ks,
                    "gap_ratio": best_candidate.gap_ratio,
                    "unitarity_norm": best_candidate.unitarity_norm,
                    "distance_to_minus_one": best_candidate.min_distance_to_minus_one,
                },
                handle,
                indent=2,
            )

        if best_global is None or best_candidate.rmse < best_global.rmse:
            best_global = best_candidate
        center = best_candidate.params

        if previous_rmse is not None:
            improvement = previous_rmse - best_candidate.rmse
            if improvement < rmse_tolerance:
                convergence_reached = True
                print(
                    f"Convergence reached at loop {loop:02d}; "
                    f"ΔRMSE={improvement:.3e}."
                )
                loop_count = loop
                break
        previous_rmse = best_candidate.rmse
        steps.shrink(shrink_factor)
    else:
        loop_count = loops

    if best_global is None:
        raise RuntimeError("No successful candidate found during refinement")

    if not convergence_reached:
        loop_count = len(history)

    secondary_N, secondary_T = secondary_grid
    try:
        replicate = _evaluate_candidate(
            N=secondary_N,
            T=secondary_T,
            params=best_global.params,
            m=m,
            spacing_count=spacing_count,
            zeros=zeros,
            seed=seed,
            phase_mode=phase_mode,
        )
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Failed to evaluate replicate grid") from exc

    print(f"Refinement finished after {loop_count} loops.")

    return RefinementResult(
        best=best_global,
        replicate=replicate,
        loop_count=loop_count,
        convergence_reached=convergence_reached,
        history=history,
    )


def save_refinement_summary(
    result: RefinementResult,
    m: int,
    spacing_count: int,
    output_dir: Path | str = Path("artifacts/refine"),
) -> dict:
    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    tables_dir = output_path / "tables"
    logs_dir = output_path / "logs"

    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    best = result.best
    spacing = best.spacing
    comparison = best.comparison

    _plot_spacing_histogram(spacing.normalised_gaps, plots_dir / "hist_spacing.png")
    _plot_spacing_ecdf(spacing.normalised_gaps, plots_dir / "ecdf_spacing.png")
    _plot_delta(comparison.delta, plots_dir / "delta_vs_n.png", result.loop_count, best.rmse, best.ks)

    table_path = tables_dir / f"Tabela_refine_m{comparison.m}.csv"
    save_level2_table(comparison, table_path)
    save_params(
        logs_dir / "params.json",
        {
            "mode": "schrodinger",
            "N": best.N,
            "T": best.T,
            "tau": best.params.tau,
            "P": best.params.P,
            "beta": best.params.beta,
            "omega": best.params.omega,
            "seed": best.seed,
            "phase_mode": best.phase_mode,
            "m": comparison.m,
            "spacing_count": spacing_count,
        },
    )
    save_unitarity_report(logs_dir / "unitarity_check.txt", best.unitarity_norm)

    replicate = result.replicate
    shared_count = min(comparison.m, replicate.comparison.m, m)
    shift = 0.0
    if shared_count > 0:
        shift = float(
            np.max(
                np.abs(
                    comparison.spectrum[:shared_count]
                    - replicate.comparison.spectrum[:shared_count]
                )
            )
        )

    passed_digits = best.rmse <= 1e-8 and best.max_delta <= 1e-8
    ks_ok = best.ks <= 0.05
    gap_ratio_target = 0.60266
    verdict = "fail"
    if passed_digits and ks_ok and shift <= 1e-9:
        verdict = "pass-level2-c1c2"

    summary = {
        "status": "ok",
        "mode": "schrodinger",
        "params": {
            "N": best.N,
            "T": best.T,
            "tau": best.params.tau,
            "P": best.params.P,
            "beta": best.params.beta,
            "omega": best.params.omega,
            "seed": best.seed,
            "m": comparison.m,
        },
        "unitarity_norm": best.unitarity_norm,
        "branch": "principal",
        "matching": {
            "rmse": best.rmse,
            "max_delta": best.max_delta,
            "passed_digits": passed_digits,
            "table_path": str(table_path),
        },
        "spacing_stats": {
            "KS": best.ks,
            "gap_ratio": best.gap_ratio,
            "target_gap_ratio": gap_ratio_target,
            "levels_used": best.levels_used,
        },
        "stability": {
            "replicated_on": {"N": result.replicate.N, "T": result.replicate.T},
            "max_abs_shift_first_m": shift,
        },
        "euler_brief": "",
        "artifacts": {
            "hist_path": str(plots_dir / "hist_spacing.png"),
            "ecdf_path": str(plots_dir / "ecdf_spacing.png"),
            "delta_path": str(plots_dir / "delta_vs_n.png"),
            "params_log": str(logs_dir / "params.json"),
            "unitarity_log": str(logs_dir / "unitarity_check.txt"),
        },
        "verdict": verdict,
        "notes": (
            f"Refinement completed in {result.loop_count} loops; "
            f"RMSE={best.rmse:.3e}, KS={best.ks:.3f}; "
            f"convergence={'yes' if result.convergence_reached else 'no'}."
        ),
        "next_actions": [
            "Increase m to 100 for additional comparison",
            "Extend spacing analysis beyond first 200 levels",
        ],
    }

    with (output_path / "refine_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


__all__ = [
    "ParameterSet",
    "StepSizes",
    "CandidateResult",
    "RefinementResult",
    "refine_pipeline",
    "save_refinement_summary",
]
