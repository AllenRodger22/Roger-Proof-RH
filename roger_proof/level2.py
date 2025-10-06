"""Core implementation for Level 2 Schrödinger runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .primes import primes_up_to
from .riemann import riemann_zeros
from .stats import gap_ratio, kolmogorov_smirnov, operator_norm, wigner_surmise_cdf


@dataclass
class RunResult:
    name: str
    params: Dict[str, float]
    unitarity_norm: float
    branch: str
    levels: np.ndarray
    mapped_levels: np.ndarray
    gamma: np.ndarray
    rmse: float
    max_delta: float
    passed_digits: bool
    ks: float
    gap_ratio: float
    levels_used: int
    spacings: np.ndarray
    delta: np.ndarray
    table_path: Path
    hist_path: Path
    ecdf_path: Path
    delta_path: Path
    params_log: Path
    unitarity_log: Path


class Level2Runner:
    """Execute Level 2 Schrödinger mode simulations."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.tables_dir = output_dir / "tables"
        self.plots_dir = output_dir / "plots"
        self.logs_dir = output_dir / "logs"
        self.results_dir = output_dir / "results"
        for directory in (self.tables_dir, self.plots_dir, self.logs_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def run(self, runs: Sequence[Dict[str, float]], deliverables: Sequence[str], m: int) -> Dict[str, RunResult]:
        results: Dict[str, RunResult] = {}
        gamma = np.array(riemann_zeros(max(m, 200)))
        for run in runs:
            name = run["name"]
            result = self._execute_single(run, gamma[:m], deliverables)
            results[name] = result
        return results

    def _execute_single(self, params: Dict[str, float], gamma: np.ndarray, deliverables: Sequence[str]) -> RunResult:
        name = params["name"]
        N = int(params["N"])
        T = float(params["T"])
        tau = float(params["tau"])
        P = int(params["P"])
        beta = float(params["beta"])
        omega = float(params["omega"])
        seed = int(params.get("seed", 0))
        np.random.seed(seed)

        t = np.linspace(0.0, T, N, endpoint=False)
        primes = primes_up_to(P)
        phi = np.zeros_like(t)
        for p in primes:
            weight = p ** (-beta)
            phi += weight * np.sin(omega * math.log(p) * t)
        std = np.std(phi)
        if std > 0:
            phi /= std
        phase = np.exp(1j * phi)
        M_phi = np.diag(phase)

        indices = np.arange(N)
        F = np.exp(-2j * np.pi * np.outer(indices, indices) / N) / math.sqrt(N)
        freq = np.fft.fftfreq(N, d=1.0 / N)
        shift_diag = np.exp(1j * tau * freq)
        S_tau = F.conj().T @ (shift_diag[:, None] * F)

        U = M_phi @ S_tau
        identity = np.eye(N, dtype=np.complex128)
        unitarity_norm = operator_norm(U.conj().T @ U - identity)

        eigvals, _ = np.linalg.eig(U)
        angles = np.angle(eigvals)
        sorted_indices = np.argsort(angles)
        angles = np.unwrap(angles[sorted_indices])

        best = self._select_levels(angles, gamma)
        candidate_levels, mapped_levels, rmse, max_delta, delta, passed_digits = best

        spacings = np.diff(mapped_levels)
        mean_spacing = np.mean(spacings) if spacings.size else float("nan")
        normalized_spacings = spacings / mean_spacing if spacings.size else spacings
        ks = kolmogorov_smirnov(normalized_spacings, wigner_surmise_cdf) if spacings.size else float("nan")
        ratio = gap_ratio(spacings)

        table_path = self._maybe_write_table(name, gamma, mapped_levels, delta, deliverables)
        hist_path, ecdf_path = self._maybe_plot_spacing(name, normalized_spacings, deliverables)
        delta_path = self._maybe_plot_delta(name, delta, deliverables)
        params_log, unitarity_log = self._write_logs(name, params, unitarity_norm)

        return RunResult(
            name=name,
            params={"N": N, "T": T, "tau": tau, "P": P, "beta": beta, "omega": omega, "seed": seed, "m": gamma.size},
            unitarity_norm=unitarity_norm,
            branch="principal",
            levels=candidate_levels,
            mapped_levels=mapped_levels,
            gamma=gamma,
            rmse=rmse,
            max_delta=max_delta,
            passed_digits=passed_digits,
            ks=ks,
            gap_ratio=ratio,
            levels_used=int(spacings.size),
            spacings=normalized_spacings,
            delta=delta,
            table_path=table_path,
            hist_path=hist_path,
            ecdf_path=ecdf_path,
            delta_path=delta_path,
            params_log=params_log,
            unitarity_log=unitarity_log,
        )

    def _select_levels(
        self, levels: np.ndarray, gamma: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, bool]:
        m = gamma.size
        best_rmse = float("inf")
        best_candidate = None
        best_mapped = None
        best_delta = None
        best_max_delta = float("inf")
        best_passed = False
        for start in range(0, levels.size - m + 1):
            candidate = levels[start : start + m]
            A = np.vstack([candidate, np.ones(m)]).T
            scale, offset = np.linalg.lstsq(A, gamma, rcond=None)[0]
            mapped = candidate * scale + offset
            delta = mapped - gamma
            rmse = math.sqrt(float(np.mean(delta ** 2)))
            max_delta = float(np.max(np.abs(delta)))
            passed = rmse <= 1e-8 and max_delta <= 1e-8
            if rmse < best_rmse:
                best_rmse = rmse
                best_candidate = candidate
                best_mapped = mapped
                best_delta = delta
                best_max_delta = max_delta
                best_passed = passed
        if best_candidate is None:
            raise RuntimeError("Unable to select spectral window")
        return (
            best_candidate,
            best_mapped,
            best_rmse,
            best_max_delta,
            best_delta,
            best_passed,
        )

    def _maybe_write_table(
        self,
        name: str,
        gamma: np.ndarray,
        mapped_levels: np.ndarray,
        delta: np.ndarray,
        deliverables: Sequence[str],
    ) -> Path:
        if "tabela_m30" not in deliverables and "tabela_m100" not in deliverables:
            return self.tables_dir / f"{name}_table.csv"
        path = self.tables_dir / f"{name}_table.csv"
        with path.open("w", encoding="utf-8") as handle:
            handle.write("n,gamma_n,model_level,delta\n")
            for idx, (g, level, d) in enumerate(zip(gamma, mapped_levels, delta), start=1):
                handle.write(f"{idx},{g:.15e},{level:.15e},{d:.15e}\n")
        return path

    def _maybe_plot_spacing(
        self,
        name: str,
        normalized_spacings: np.ndarray,
        deliverables: Sequence[str],
    ) -> Tuple[Path, Path]:
        hist_path = self.plots_dir / f"{name}_hist.png"
        ecdf_path = self.plots_dir / f"{name}_ecdf.png"
        if normalized_spacings.size == 0:
            return hist_path, ecdf_path
        if "hist" in deliverables:
            plt.figure(figsize=(6, 4))
            plt.hist(normalized_spacings, bins=20, density=True, alpha=0.7, color="#1f77b4")
            x = np.linspace(0, normalized_spacings.max() * 1.2, 200)
            pdf = (32.0 / math.pi ** 2) * x ** 2 * np.exp(-4.0 * x ** 2 / math.pi)
            plt.plot(x, pdf, "r--", label="GUE Wigner")
            plt.xlabel("s")
            plt.ylabel("Density")
            plt.title(f"Spacing histogram — {name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()
        if "ecdf" in deliverables:
            plt.figure(figsize=(6, 4))
            sorted_data = np.sort(normalized_spacings)
            ecdf = np.arange(1, sorted_data.size + 1) / sorted_data.size
            plt.step(sorted_data, ecdf, where="post", label="Empirical")
            x = np.linspace(0, sorted_data.max() * 1.2, 200)
            plt.plot(x, wigner_surmise_cdf(x), "r--", label="GUE Wigner CDF")
            plt.xlabel("s")
            plt.ylabel("F(s)")
            plt.title(f"ECDF — {name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(ecdf_path)
            plt.close()
        return hist_path, ecdf_path

    def _maybe_plot_delta(
        self,
        name: str,
        delta: np.ndarray,
        deliverables: Sequence[str],
    ) -> Path:
        path = self.plots_dir / f"{name}_delta.png"
        if "delta_vs_n" not in deliverables:
            return path
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, delta.size + 1), delta, marker="o")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel("n")
        plt.ylabel("model_level - gamma")
        plt.title(f"Δ vs n — {name}")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def _write_logs(self, name: str, params: Dict[str, float], unitarity_norm: float) -> Tuple[Path, Path]:
        params_log = self.logs_dir / f"{name}_params.json"
        with params_log.open("w", encoding="utf-8") as handle:
            json.dump({**params, "unitarity_norm": unitarity_norm}, handle, indent=2)
        unitarity_log = self.logs_dir / f"{name}_unitarity.txt"
        with unitarity_log.open("w", encoding="utf-8") as handle:
            handle.write(f"unitarity_norm={unitarity_norm:.3e}\n")
        return params_log, unitarity_log

    def select_best(self, results: Dict[str, RunResult], criterion: str) -> RunResult:
        if criterion == "min_rmse_with_KS_le_0.05":
            eligible = [r for r in results.values() if not math.isnan(r.ks) and r.ks <= 0.05]
            if not eligible:
                eligible = list(results.values())
            return min(eligible, key=lambda r: r.rmse)
        raise ValueError(f"Unknown selection criterion: {criterion}")

    def to_summary_json(
        self,
        result: RunResult,
        stability: Dict[str, float],
    ) -> Dict[str, object]:
        return {
            "status": "ok",
            "mode": "schrodinger",
            "params": result.params,
            "unitarity_norm": result.unitarity_norm,
            "branch": result.branch,
            "matching": {
                "rmse": result.rmse,
                "max_delta": result.max_delta,
                "passed_digits": result.passed_digits,
                "table_path": str(result.table_path.relative_to(self.output_dir)),
            },
            "spacing_stats": {
                "KS": result.ks,
                "gap_ratio": result.gap_ratio,
                "target_gap_ratio": 0.60266,
                "levels_used": result.levels_used,
            },
            "stability": stability,
            "artifacts": {
                "hist_path": str(result.hist_path.relative_to(self.output_dir)),
                "ecdf_path": str(result.ecdf_path.relative_to(self.output_dir)),
                "delta_path": str(result.delta_path.relative_to(self.output_dir)),
                "params_log": str(result.params_log.relative_to(self.output_dir)),
                "unitarity_log": str(result.unitarity_log.relative_to(self.output_dir)),
            },
            "verdict": "pass-level2-c1c2" if result.rmse <= 1e-8 and result.ks <= 0.05 else "fail",
            "notes": "Experimental evidence only; no proof.",
            "next_actions": [
                "Increase m to 100 for further validation.",
                "Explore additional tau refinements around the selected parameters.",
            ],
        }
