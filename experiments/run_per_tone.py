"""Per-subcarrier error of each estimator on the LTE grid, automotive-plant profile, 15 dB.

Writes results/per_tone.csv with one row per estimator and subcarrier, holding the total
normalised error and its channel-mismatch and noise parts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ofdmest import exact_error
from common import RESULTS, estimator_set, grids, site_profiles


def main():
    grid, pilots = grids()["LTE 20 MHz, CRS"]
    profile = site_profiles(grid)["NIST automotive plant (132 ns)"]
    rows = []
    for est in estimator_set(grid, pilots, profile):
        res = exact_error(est, grid, pilots, profile, 15.0)
        for m, (e, b, n) in enumerate(zip(res.per_tone, res.bias_per_tone, res.noise_per_tone)):
            rows.append({"estimator": est.name, "subcarrier": int(grid.active_indices[m]), "error": float(e), "bias": float(b), "noise": float(n)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "per_tone.csv", index=False)
    print(f"wrote {RESULTS / 'per_tone.csv'} with {len(df)} rows")


if __name__ == "__main__":
    main()
