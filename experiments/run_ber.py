"""Bit error rate study: matched-draw Monte Carlo of QPSK with each estimator.

For the LTE and NR type 1 grids and the NIST factory-site profiles (the exponential profiles
at the reported delay spreads and the measured profiles) plus one long exponential profile,
every estimator is applied to the same 5,000 channel, noise and data draws per SNR. Writes results/ber.csv with the simulated error, its 95 percent half width,
the exact error, and the bit error rate with zero-forcing equalisation (QPSK by default,
16-QAM with ``--16qam``, written to results/ber_16qam.csv).
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from ofdmest import exact_error, simulate
from common import RESULTS, SNR_DB, estimator_set, exponential_profile, grids, site_profiles

N_SYMBOLS = 5000


def main():
    modulation = "16qam" if "--16qam" in sys.argv else "qpsk"
    rows = []
    t0 = time.perf_counter()
    all_grids = grids()
    for label in ("LTE 20 MHz, CRS", "NR 20 MHz, DM-RS type 1"):
        grid, pilots = all_grids[label]
        profiles = {k: v for k, v in site_profiles(grid).items() if k.startswith("NIST")}
        long_ds = 0.2 * grid.cp_duration_s
        profiles[f"exponential {long_ds*1e9:.0f} ns"] = exponential_profile(grid, long_ds)
        for pname, profile in profiles.items():
            ests = estimator_set(grid, pilots, profile)
            for i, snr in enumerate(SNR_DB):
                sims = simulate(ests, grid, pilots, profile, snr, n_symbols=N_SYMBOLS, seed=100 * i + 7, modulation=modulation)
                for est, sim in zip(ests, sims):
                    exact = exact_error(est, grid, pilots, profile, snr)
                    rows.append({"grid": label, "profile": pname, "rms_delay_spread_ns": profile.rms_delay_spread_s * 1e9, "snr_db": snr, "estimator": est.name, "exact_error_db": exact.total_db, "mc_error_db": sim.mse_db, "mc_half_width_db": 10 * __import__("numpy").log10(1 + sim.mse_half_width_95 / sim.mse_mean), "modulation": modulation, "ber": sim.ber, "qpsk_ber": sim.ber if modulation == "qpsk" else float("nan"), "bit_errors": sim.bit_errors, "bits": sim.bits, "n_symbols": sim.n_symbols, "seed": 100 * i + 7})
            print(f"{label:28s} {pname:36s} done, elapsed {time.perf_counter()-t0:7.1f} s", flush=True)
    df = pd.DataFrame(rows)
    name = "ber.csv" if modulation == "qpsk" else "ber_16qam.csv"
    df.to_csv(RESULTS / name, index=False)
    print(f"wrote {RESULTS / name} with {len(df)} rows")


if __name__ == "__main__":
    main()
