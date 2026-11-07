"""Cost study: operation counts and measured NumPy run times of each estimator per grid.

Writes results/cost.csv. Times are wall-clock on the machine that runs the script and are
indicative only; the operation counts are hardware independent.
"""
from __future__ import annotations

import platform
import time

import pandas as pd

from ofdmest import OperatingPoint, noise_variance_from_snr
from ofdmest.cost import benchmark
from common import RESULTS, estimator_set, exponential_profile, grids


def main():
    rows = []
    for label, (grid, pilots) in grids().items():
        profile = exponential_profile(grid, 100e-9)
        point = OperatingPoint(profile, noise_variance_from_snr(15.0, grid.n_subcarriers))
        ests = estimator_set(grid, pilots, profile)
        for res in benchmark(ests, grid, pilots, point, n_symbols=2000, repeats=5):
            rows.append({"grid": label, "n_tones": grid.n_active, "n_pilots": pilots.count, "estimator": res.estimator, "real_multiplications": res.real_multiplications, "design_ms": res.design_ms, "dense_apply_us_per_symbol": res.dense_apply_us, "structured_apply_us_per_symbol": res.structured_apply_us})
        print(f"{label:32s} done", flush=True)
    df = pd.DataFrame(rows)
    df["machine"] = f"{platform.processor()} / Python {platform.python_version()}"
    df.to_csv(RESULTS / "cost.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
