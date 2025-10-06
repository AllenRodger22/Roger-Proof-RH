"""Command-line interface to execute Level 2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from roger_proof import Level2Runner, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Level 2 (Schrödinger mode) runs")
    parser.add_argument("config", type=Path, help="Path to the configuration file (JSON or YAML)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where artifacts should be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mode = config.get("mode", "schrodinger")
    if mode != "schrodinger":
        raise ValueError(f"Unsupported mode: {mode}")
    runs = config.get("runs", [])
    deliverables = config.get("deliverables", [])
    m = int(config.get("m", 30))

    runner = Level2Runner(args.output_dir)
    results = runner.run(runs, deliverables, m)
    best = runner.select_best(results, config.get("selection", "min_rmse_with_KS_le_0.05"))

    stability_info = {
        "replicated_on": {},
        "max_abs_shift_first_m": float("nan"),
    }

    for candidate in results.values():
        if candidate is best:
            continue
        if (
            candidate.params["tau"] == best.params["tau"]
            and candidate.params["P"] == best.params["P"]
            and candidate.params["beta"] == best.params["beta"]
            and candidate.params["omega"] == best.params["omega"]
            and candidate.params["m"] == best.params["m"]
            and (candidate.params["N"], candidate.params["T"]) != (best.params["N"], best.params["T"])
        ):
            shift = float(
                np.max(np.abs(candidate.mapped_levels - best.mapped_levels))
            )
            stability_info = {
                "replicated_on": {"N": candidate.params["N"], "T": candidate.params["T"]},
                "max_abs_shift_first_m": shift,
            }
            break

    summary = runner.to_summary_json(best, stability_info)
    summary_path = runner.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
