"""Mismatch study: cost of designing the Wiener filter for the wrong delay profile.

For each grid and actual delay spread, Wiener filters are designed for an exponential
profile whose delay spread is the actual one times a factor, and for uniform profiles
spanning multiples of the actual delay spread or the cyclic prefix. The design noise level
equals the actual one. Writes results/mismatch.csv with the loss relative to the matched
filter in dB.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ofdmest import DelayProfile, WienerFilter, exact_error, robust_wiener
from common import RESULTS, delay_spreads_for, exponential_profile, grids, site_profiles

FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
UNIFORM_SPANS = [2.0, 4.0, 8.0, 16.0]
SNR_DB = [5.0, 15.0, 25.0]


def main():
    rows = []
    t0 = time.perf_counter()
    for label, (grid, pilots) in grids().items():
        if label.startswith("DVB-T2") and "spacing 6" not in label:
            continue
        profiles = {f"exponential {ds*1e9:.0f} ns": exponential_profile(grid, ds) for ds in delay_spreads_for(grid, n=6, lowest_s=20e-9)}
        profiles.update(site_profiles(grid))
        for pname, profile in profiles.items():
            ds = profile.rms_delay_spread_s
            for snr in SNR_DB:
                matched = exact_error(WienerFilter(), grid, pilots, profile, snr).total_db
                designs = []
                for f in FACTORS:
                    try:
                        design_profile = DelayProfile.exponential(f * ds, grid.sample_period_s)
                    except ValueError:
                        continue
                    designs.append((f"exponential x{f:g}", WienerFilter(design_profile), f * ds))
                for s in UNIFORM_SPANS:
                    designs.append((f"uniform x{s:g}", robust_wiener(grid, s * ds), s * ds))
                designs.append(("uniform to CP", robust_wiener(grid), grid.cp_duration_s))
                for dname, est, dspan in designs:
                    res = exact_error(est, grid, pilots, profile, snr)
                    rows.append({"grid": label, "profile": pname, "rms_delay_spread_ns": ds * 1e9, "snr_db": snr, "design": dname, "design_scale_ns": dspan * 1e9, "error_db": res.total_db, "matched_db": matched, "loss_db": res.total_db - matched, "bias": res.bias_total, "noise": res.noise_total})
        print(f"{label:32s} done, elapsed {time.perf_counter()-t0:6.1f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "mismatch.csv", index=False)
    print(f"wrote {RESULTS / 'mismatch.csv'} with {len(df)} rows")


if __name__ == "__main__":
    main()
