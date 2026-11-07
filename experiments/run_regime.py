"""Regime study: exact error of every estimator over delay spread and SNR for each grid.

Writes results/regime.csv with one row per grid, profile, SNR and estimator, holding the
total error, its channel-mismatch and noise parts, the error on the two band-edge tones,
and the operation count. Profiles are the exponential sweep plus the factory-site profiles.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ofdmest import OperatingPoint, WienerFilter, error_components, error_from_matrix, noise_variance_from_snr, robust_wiener
from common import RESULTS, SNR_DB, SNR_INDEPENDENT, delay_spreads_for, estimator_set, exponential_profile, grids, site_profiles


def rows_for(label, grid, pilots, profile_name, profile, kind):
    out = []
    ests = estimator_set(grid, pilots, profile)
    fixed = {}
    for est in ests:
        if isinstance(est, SNR_INDEPENDENT):
            A = est.matrix(grid, pilots)
            fixed[est.name] = (est, error_components(A, grid, pilots, profile))
    for snr in SNR_DB:
        sigma2 = 10.0 ** (-snr / 10.0)
        point = OperatingPoint(profile, noise_variance_from_snr(snr, grid.n_subcarriers))
        for est in ests:
            if est.name in fixed:
                bias, gain = fixed[est.name][1]
                noise = sigma2 * gain
            else:
                A = est.matrix(grid, pilots, point)
                bias, noise = error_from_matrix(A, grid, pilots, point)
            per_tone = bias + noise
            out.append({
                "grid": label, "map": pilots.name, "n_tones": grid.n_active, "n_pilots": pilots.count,
                "profile_kind": kind, "profile": profile_name, "rms_delay_spread_ns": profile.rms_delay_spread_s * 1e9,
                "snr_db": snr, "estimator": est.name, "error": float(per_tone.mean()), "error_db": float(10 * np.log10(per_tone.mean())),
                "bias": float(bias.mean()), "noise": float(noise.mean()), "edge_error": float(per_tone[[0, -1]].mean()),
                "interior_error": float(per_tone[len(per_tone) // 4: 3 * len(per_tone) // 4].mean()),
                "real_multiplications": est.real_multiplications(grid, pilots),
            })
    return out


def main():
    rows = []
    t0 = time.perf_counter()
    for label, (grid, pilots) in grids().items():
        for ds in delay_spreads_for(grid):
            profile = exponential_profile(grid, ds)
            rows += rows_for(label, grid, pilots, f"exponential {ds*1e9:.1f} ns", profile, "sweep")
        for name, profile in site_profiles(grid).items():
            rows += rows_for(label, grid, pilots, name, profile, "site")
        print(f"{label:32s} done, elapsed {time.perf_counter()-t0:6.1f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "regime.csv", index=False)
    print(f"wrote {RESULTS / 'regime.csv'} with {len(df)} rows")


if __name__ == "__main__":
    main()
