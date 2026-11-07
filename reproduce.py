"""Regenerate the tables and figures of the study.

Fast path (default): build every figure and table from the result files shipped in
``results``. Full path (``--full``): rerun the experiments first, which takes about an hour
on a laptop, then build the figures and tables. Run from the folder that holds this file
after installing the package with ``pip install -e ofdmest``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = ["validate_engine.py", "run_regime.py", "run_mismatch.py", "run_pilot_maps.py", "run_ber.py", "run_per_tone.py", "run_cost.py"]


def run(script: str, *args: str) -> None:
    """Run one experiment script with a bounded number of linear-algebra threads.

    Several concurrent NumPy processes each spawning one thread per logical processor slow
    each other down by an order of magnitude; four threads per process is a safe setting.
    """
    env = dict(os.environ)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, "4")
    print(f"== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(HERE / "experiments" / script), *args], check=True, cwd=HERE / "experiments", env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="rerun all experiments before building figures and tables")
    args = parser.parse_args()
    if args.full:
        for script in EXPERIMENTS:
            if script == "validate_engine.py":
                run(script, str(HERE / "results" / "validation.csv"))
            else:
                run(script)
    run("make_figures.py")
    run("make_tables.py")


if __name__ == "__main__":
    main()
