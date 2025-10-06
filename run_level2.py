"""Command-line utility to execute the Level-2 Roger-Hamilton pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from roger_hamilton.analysis import (
    compare_spectrum,
    save_level2_table,
    save_params,
    save_unitarity_report,
    spacing_statistics,
)
from roger_hamilton.hamiltonian import eigenvalues_from_unitary
from roger_hamilton.unitary import construct_unitary

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_svg(path: Path, svg: str) -> None:
    _ensure_parent(path)
    path.write_text(svg, encoding="utf-8")


def _svg_header(width: int = 600, height: int = 400) -> str:
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"


def _svg_axes(width: int = 600, height: int = 400, margin: int = 50) -> tuple[str, Callable[[float, float], tuple[float, float]]]:
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    def transform(x: float, y: float) -> tuple[float, float]:
        return margin + x * plot_width, height - margin - y * plot_height

    axes = (
        f"<rect x='{margin}' y='{margin}' width='{plot_width}' height='{plot_height}' "
        "fill='none' stroke='black' stroke-width='1' />"
    )
    return axes, transform


def _svg_polyline(points: Sequence[tuple[float, float]], transform: Callable[[float, float], tuple[float, float]], colour: str = "#1f77b4", stroke_width: float = 2.0) -> str:
    transformed = [transform(x, y) for x, y in points]
    path_data = " ".join(f"{x:.2f},{y:.2f}" for x, y in transformed)
    return f"<polyline fill='none' stroke='{colour}' stroke-width='{stroke_width}' points='{path_data}' />"


def _svg_circles(points: Sequence[tuple[float, float]], transform: Callable[[float, float], tuple[float, float]], colour: str = "#ff7f0e", radius: float = 3.0) -> str:
    circles = []
    for x, y in points:
        tx, ty = transform(x, y)
        circles.append(f"<circle cx='{tx:.2f}' cy='{ty:.2f}' r='{radius}' fill='{colour}' />")
    return "".join(circles)


def _svg_text(x: float, y: float, text: str, size: int = 12) -> str:
    return f"<text x='{x}' y='{y}' font-size='{size}' fill='black'>{text}</text>"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["schrodinger"], default="schrodinger")
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--T", type=float, default=24.0)
    parser.add_argument("--tau", type=float, default=0.37)
    parser.add_argument("--P", type=int, default=700, help="Number of primes in the Azimuthal phase")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--phase-mode", choices=["deterministic", "random"], default="deterministic")
    parser.add_argument("--m", type=int, default=30, help="Number of levels to compare against zeros")
    parser.add_argument("--spacing-count", type=int, default=200, help="Number of eigenvalues used for spacing stats")
    parser.add_argument("--eigen-mode", choices=["log", "cayley"], default="log")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def _ensure_unitarity(U: np.ndarray) -> float:
    identity = np.eye(U.shape[0], dtype=complex)
    residue = U.conjugate().T @ U - identity
    norm = float(np.linalg.norm(residue, ord=2))
    if norm > 1e-10:
        raise RuntimeError(f"Unitarity violation detected: norm={norm:.3e}")
    return norm


def _plot_spacing_histogram(normalised_gaps: np.ndarray, output: Path) -> None:
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.hist(normalised_gaps, bins=40, density=True, alpha=0.7, label="Empirical")
        s = np.linspace(0, normalised_gaps.max() * 1.1, 400)
        pdf = (32 / (np.pi**2)) * (s**2) * np.exp(-4 * s**2 / np.pi)
        plt.plot(s, pdf, label="GUE surmise", color="black")
        plt.xlabel("Normalised spacing")
        plt.ylabel("Density")
        plt.title("Spacing histogram")
        plt.legend()
        plt.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output)
        plt.close()
        return

    counts, edges = np.histogram(normalised_gaps, bins=40, density=True)
    width = 600
    height = 400
    margin = 50
    axes, transform = _svg_axes(width, height, margin)
    max_count = counts.max() if counts.size else 1.0
    rects = []
    for count, left, right in zip(counts, edges[:-1], edges[1:], strict=False):
        x0, y0 = transform((left - edges[0]) / (edges[-1] - edges[0] + 1e-12), count / max_count)
        x1, y1 = transform((right - edges[0]) / (edges[-1] - edges[0] + 1e-12), 0.0)
        rects.append(
            f"<rect x='{x0:.2f}' y='{y0:.2f}' width='{max(x1 - x0, 0.1):.2f}' height='{max(y1 - y0, 0.1):.2f}' fill='#1f77b4' fill-opacity='0.7' stroke='none' />"
        )
    s = np.linspace(0, normalised_gaps.max() * 1.1, 400)
    pdf = (32 / (np.pi**2)) * (s**2) * np.exp(-4 * s**2 / np.pi)
    norm_pdf = pdf / (pdf.max() if pdf.size else 1.0)
    points = [((val - edges[0]) / (edges[-1] - edges[0] + 1e-12), p) for val, p in zip(s, norm_pdf, strict=False)]
    polyline = _svg_polyline(points, transform, colour="#000000", stroke_width=1.5)
    svg = [
        _svg_header(width, height),
        axes,
        *rects,
        polyline,
        _svg_text(60, 30, "Spacing histogram"),
    ]
    svg.append("</svg>")
    _write_svg(output, "".join(svg))


def _plot_spacing_ecdf(normalised_gaps: np.ndarray, output: Path) -> None:
    sorted_gaps = np.sort(normalised_gaps)
    ecdf = np.arange(1, sorted_gaps.size + 1) / sorted_gaps.size
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.step(sorted_gaps, ecdf, where="post", label="Empirical")
        s = np.linspace(0, sorted_gaps.max() * 1.1, 400)
        cdf = 1.0 - np.exp(-4 * s**2 / np.pi) * (1.0 + 2.0 * s / np.sqrt(np.pi))
        plt.plot(s, cdf, label="GUE surmise", color="black")
        plt.xlabel("Normalised spacing")
        plt.ylabel("ECDF")
        plt.title("Spacing ECDF")
        plt.legend()
        plt.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output)
        plt.close()
        return

    width = 600
    height = 400
    margin = 50
    axes, transform = _svg_axes(width, height, margin)
    x_vals = (sorted_gaps - sorted_gaps.min()) / (sorted_gaps.max() - sorted_gaps.min() + 1e-12)
    emp_points = list(zip(x_vals, ecdf))
    theo_s = np.linspace(sorted_gaps.min(), sorted_gaps.max(), 400)
    theo_vals = 1.0 - np.exp(-4 * theo_s**2 / np.pi) * (1.0 + 2.0 * theo_s / np.sqrt(np.pi))
    theo_x = (theo_s - theo_s.min()) / (theo_s.max() - theo_s.min() + 1e-12)
    theo_points = list(zip(theo_x, theo_vals))
    svg = [
        _svg_header(width, height),
        axes,
        _svg_polyline(emp_points, transform, colour="#1f77b4"),
        _svg_polyline(theo_points, transform, colour="#000000", stroke_width=1.5),
        _svg_text(60, 30, "Spacing ECDF"),
    ]
    svg.append("</svg>")
    _write_svg(output, "".join(svg))


def _plot_delta_vs_n(delta: np.ndarray, output: Path) -> None:
    indices = np.arange(1, delta.size + 1)
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(indices, delta, marker="o", linestyle="-", label="E_n - gamma_n")
        plt.axhline(0.0, color="black", linewidth=1.0)
        plt.xlabel("n")
        plt.ylabel("Delta")
        plt.title("Spectral deviation")
        plt.legend()
        plt.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output)
        plt.close()
        return

    width = 600
    height = 400
    margin = 50
    axes, transform = _svg_axes(width, height, margin)
    x_vals = (indices - indices.min()) / (indices.max() - indices.min() + 1e-12)
    delta_range = delta.max() - delta.min()
    if delta_range == 0:
        scaled_delta = np.zeros_like(delta)
    else:
        scaled_delta = (delta - delta.min()) / delta_range
    points = list(zip(x_vals, scaled_delta))
    zero_level = 0.0
    if delta_range != 0:
        zero_level = (-delta.min()) / delta_range
        zero_level = max(0.0, min(1.0, zero_level))
    zero_line = _svg_polyline([(0.0, zero_level), (1.0, zero_level)], transform, colour="#000000", stroke_width=1.0)
    svg = [
        _svg_header(width, height),
        axes,
        zero_line,
        _svg_polyline(points, transform, colour="#1f77b4"),
        _svg_circles(points, transform),
        _svg_text(60, 30, "Spectral deviation"),
    ]
    svg.append("</svg>")
    _write_svg(output, "".join(svg))


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir

    params = {
        "mode": args.mode,
        "N": args.N,
        "T": args.T,
        "tau": args.tau,
        "P": args.P,
        "beta": args.beta,
        "omega": args.omega,
        "seed": args.seed,
        "phase_mode": args.phase_mode,
        "m": args.m,
        "spacing_count": args.spacing_count,
        "eigen_mode": args.eigen_mode,
    }

    unitary = construct_unitary(
        N=args.N,
        T=args.T,
        tau=args.tau,
        P=args.P,
        beta=args.beta,
        omega=args.omega,
        seed=args.seed,
        phase_mode=args.phase_mode,
    )

    norm = _ensure_unitarity(unitary.matrix)
    save_unitarity_report(output_dir / "logs" / "unitarity_check.txt", norm)
    save_params(output_dir / "logs" / "params.json", params)

    spectrum = eigenvalues_from_unitary(unitary.matrix, mode=args.eigen_mode)
    comparison = compare_spectrum(spectrum.eigenvalues, args.m)
    save_level2_table(comparison, output_dir / "tables" / f"level2_m{comparison.m}.csv")

    spacing_stats = spacing_statistics(spectrum.eigenvalues, args.spacing_count)

    _plot_spacing_histogram(spacing_stats.normalised_gaps, output_dir / "plots" / "hist_spacing.svg")
    _plot_spacing_ecdf(spacing_stats.normalised_gaps, output_dir / "plots" / "ecdf_spacing.svg")
    _plot_delta_vs_n(comparison.delta, output_dir / "plots" / "delta_vs_n.svg")

    summary = {
        "rmse": comparison.rmse,
        "max_abs_error": comparison.max_abs_error,
        "ks_statistic": spacing_stats.ks_statistic,
        "mean_gap_ratio": spacing_stats.mean_gap_ratio,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
