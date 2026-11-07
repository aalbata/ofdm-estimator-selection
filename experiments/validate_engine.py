"""Validation of the exact error against matched-draw Monte Carlo over a broad grid.

For every estimator, pilot map, delay profile and SNR in the grid, the script compares the
exact normalised error with the Monte Carlo mean over 10,000 OFDM symbols and records
whether the exact value lies inside the 95 percent interval of the Monte Carlo estimate.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ofdmest import (
    CubicSplineInterpolation,
    DelayProfile,
    DftInterpolation,
    DftLeastSquares,
    LinearInterpolation,
    NearestPilot,
    OfdmGrid,
    WienerFilter,
    dvbt2_scattered,
    exact_error,
    lte_crs,
    nr_dmrs_type1,
    nr_dmrs_type2,
    robust_wiener,
    simulate,
    uniform_anchored,
    uniform_centred,
)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation.csv")
N_SYMBOLS = 10000


def cases():
    nr = OfdmGrid(12 * 20, 30e3, fft_size=512, cp_duration_s=2.34e-6)
    lte = OfdmGrid(12 * 25, 15e3, fft_size=512, cp_duration_s=4.69e-6)
    t2 = OfdmGrid(768, 1 / 896e-6, fft_size=8192, cp_duration_s=896e-6 / 32)
    prop = OfdmGrid(64, 312.5e3, fft_size=64, cp_duration_s=16 / (64 * 312.5e3))
    yield "NR type 1", nr, nr_dmrs_type1(20, 0), [30e-9, 150e-9, 500e-9]
    yield "NR type 2", nr, nr_dmrs_type2(20, 0), [30e-9, 150e-9, 500e-9]
    yield "LTE CRS", lte, lte_crs(25, 0, 0), [50e-9, 200e-9, 1e-6]
    yield "DVB-T2 PP2", t2, dvbt2_scattered(768, "PP2", 0), [1e-6, 5e-6]
    yield "DVB-T2 PP7", t2, dvbt2_scattered(768, "PP7", 0), [1e-6, 5e-6]
    yield "proprietary anchored", prop, uniform_anchored(64, 8), [50e-9, 200e-9]
    yield "proprietary centred", prop, uniform_centred(64, 8), [50e-9, 200e-9]


def estimators_for(grid, profile, pilots):
    mismatched = DelayProfile.exponential(2.0 * profile.rms_delay_spread_s, grid.sample_period_s)
    out = [NearestPilot(), LinearInterpolation("constant"), LinearInterpolation("linear"), CubicSplineInterpolation("constant"), WienerFilter(), WienerFilter(mismatched, None, name="Wiener (2x delay spread)"), robust_wiener(grid)]
    steps = np.diff(pilots.indices)
    if np.all(steps == steps[0]):
        out.append(DftInterpolation())
    else:
        out.append(DftLeastSquares(ridge=1e-3, name="DFT least squares (ridge)"))
    return out


rows = []
t_start = time.perf_counter()
for label, grid, pilots, spreads in cases():
    for ds in spreads:
        profile = DelayProfile.exponential(ds, grid.sample_period_s, max_delay_s=min(15 * ds, grid.cp_duration_s))
        ests = estimators_for(grid, profile, pilots)
        for snr in (0.0, 15.0, 30.0):
            exact = [exact_error(e, grid, pilots, profile, snr) for e in ests]
            sims = simulate(ests, grid, pilots, profile, snr, n_symbols=N_SYMBOLS, seed=int(1000 * ds * 1e9) % 100000 + int(snr))
            for e, x, s in zip(ests, exact, sims):
                rows.append({
                    "map": label, "n_tones": grid.n_active, "n_pilots": pilots.count, "rms_delay_spread_ns": ds * 1e9,
                    "snr_db": snr, "estimator": e.name, "exact_mse": x.total, "mc_mse": s.mse_mean,
                    "mc_half_width_95": s.mse_half_width_95, "relative_error_percent": 100 * (s.mse_mean - x.total) / x.total,
                    "inside_95": abs(s.mse_mean - x.total) <= s.mse_half_width_95, "mc_ber": s.ber,
                })
        print(f"{label:22s} ds={ds*1e9:7.1f} ns done, elapsed {time.perf_counter()-t_start:6.1f} s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\n{len(df)} cells; inside the 95% interval: {df.inside_95.mean()*100:.1f}%; "
      f"max |relative error| {df.relative_error_percent.abs().max():.2f}%; "
      f"median |relative error| {df.relative_error_percent.abs().median():.2f}%")
print(df.groupby("estimator").agg(cells=("exact_mse", "size"), inside=("inside_95", "mean"), max_abs_rel=("relative_error_percent", lambda s: s.abs().max())).round(3))
