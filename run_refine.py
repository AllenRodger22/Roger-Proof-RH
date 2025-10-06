"""Adaptive refinement driver for the Roger-Hamilton Level-2 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roger_hamilton.refinement import refine_pipeline, save_refinement_summary
from roger_hamilton.zeros import load_riemann_zeros


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["schrodinger"], default="schrodinger")
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--T", type=float, default=24.0)
    parser.add_argument("--tau", type=float, default=0.37)
    parser.add_argument("--P", type=int, default=700)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-mode", choices=["deterministic", "random"], default="deterministic")
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--zeros-count", type=int, default=None)
    parser.add_argument("--spacing-count", type=int, default=200)
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--grid-points", type=int, default=5)
    parser.add_argument("--shrink-factor", type=float, default=1.5)
    parser.add_argument("--rmse-tolerance", type=float, default=1e-9)
    parser.add_argument("--secondary-N", type=int, default=768)
    parser.add_argument("--secondary-T", type=float, default=28.0)
    parser.add_argument("--d-omega", type=float, default=0.02)
    parser.add_argument("--d-tau", type=float, default=0.04)
    parser.add_argument("--d-P", type=int, default=100)
    parser.add_argument("--d-beta", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/refine"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    zeros_count = args.zeros_count if args.zeros_count is not None else max(args.m, 100)
    zeros = load_riemann_zeros(zeros_count)

    result = refine_pipeline(
        zeros=zeros,
        N=args.N,
        T=args.T,
        tau=args.tau,
        P=args.P,
        beta=args.beta,
        omega=args.omega,
        seed=args.seed,
        phase_mode=args.phase_mode,
        m=args.m,
        spacing_count=args.spacing_count,
        loops=args.loops,
        grid_points=args.grid_points,
        shrink_factor=args.shrink_factor,
        rmse_tolerance=args.rmse_tolerance,
        secondary_grid=(args.secondary_N, args.secondary_T),
        output_dir=args.output_dir,
        d_omega=args.d_omega,
        d_tau=args.d_tau,
        d_P=args.d_P,
        d_beta=args.d_beta,
    )

    summary = save_refinement_summary(
        result,
        m=args.m,
        spacing_count=args.spacing_count,
        output_dir=args.output_dir,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
