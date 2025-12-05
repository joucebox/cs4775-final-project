#!/usr/bin/env python3
"""Run the complete analysis workflow."""

import argparse
import subprocess
import sys

STEPS = [
    ("Estimating HMM parameters", "scripts.estimate_parameters"),
    ("Evaluating alignments", "scripts.evaluate_alignments"),
    ("Plotting metrics", "scripts.plot_metrics"),
    (
        "Plotting posteriors (zoom diff)",
        "scripts.plot_posteriors",
        ["--compare", "--overlay", "--per-pair", "--zoom-diff"],
    ),
    (
        "Plotting posteriors",
        "scripts.plot_posteriors",
        ["--compare", "--overlay", "--per-pair"],
    ),
    ("Plotting posterior gain", "scripts.plot_posterior_gain"),
]


def run(module: str, args: list[str] | None = None) -> None:
    """Run a script."""
    cmd = [sys.executable, "-m", module] + (args or [])
    subprocess.run(cmd, check=True)


def main() -> None:
    """Run the complete analysis workflow."""
    parser = argparse.ArgumentParser(description="Run the complete analysis workflow.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Rfam pairs before running (default: skip)",
    )
    opts = parser.parse_args()

    if opts.download:
        print("\n=== Downloading Rfam pairs ===")
        run("scripts.download_rfam_pairs")

    for step in STEPS:
        name, module = step[0], step[1]
        args = step[2] if len(step) > 2 else None
        print(f"\n=== {name} ===")
        run(module, args)

    print("\n=== Workflow complete ===")


if __name__ == "__main__":
    main()
