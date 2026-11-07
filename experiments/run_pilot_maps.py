"""Free pilot maps on a proprietary link: rules, exhaustive optima and genetic search.

Part A compares the anchored and centred uniform maps with the exhaustive optimum on grids
small enough to enumerate, for the linear interpolator with constant edges.
Part B, on the 64-tone link with 8 pilots and delay spreads of 25, 50 and 100 ns, compares
anchored, centred and genetic-algorithm maps for linear interpolation, with a clean band and
with a narrowband interferer raising the noise on a block of tones by 20 dB; the genetic algorithm uses the exact objective
averaged over the operating points. Writes results/pilot_maps_small.csv and
results/pilot_maps.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ofdmest import DelayProfile, LinearInterpolation, OfdmGrid, PilotMap, exact_error, uniform_anchored, uniform_centred
from ofdmest.search import MapObjective, exhaustive, genetic, rule_maps
from common import RESULTS, proprietary_grid

SNR_DB = [10.0, 20.0]
DELAY_SPREADS_SAMPLES = [1.0, 2.0, 4.0, 6.0]


def small_grids():
    for n, k in ((24, 4), (32, 4), (32, 5)):
        yield OfdmGrid(n, 312.5e3, fft_size=n, cp_duration_s=(n // 4) / (n * 312.5e3)), k


def part_a():
    rows = []
    est = LinearInterpolation("constant")
    for grid, k in small_grids():
        ts = grid.sample_period_s
        for ds_samples in DELAY_SPREADS_SAMPLES:
            ds = ds_samples * ts
            profile = DelayProfile.exponential(ds, ts)
            for snr in (15.0,):
                obj = MapObjective(est, grid, [(profile, snr)])
                best = exhaustive(obj, k)
                rules = rule_maps(grid, k)
                for name, pm in rules.items():
                    rows.append({"n_tones": grid.n_subcarriers, "n_pilots": k, "rms_delay_spread_samples": ds_samples, "snr_db": snr, "map": name, "indices": " ".join(map(str, pm.indices)), "error_db": 10 * np.log10(obj(pm.indices)), "optimum_db": 10 * np.log10(best.objective), "optimum_indices": " ".join(map(str, best.pilots.indices)), "maps_enumerated": best.evaluations})
        print(f"small grid {grid.n_subcarriers}/{k} done", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "pilot_maps_small.csv", index=False)


def part_b():
    grid = proprietary_grid()
    ts = grid.sample_period_s
    est = LinearInterpolation("constant")
    rows = []
    scenarios = {"clean band": None}
    for start in (8, 20, 44):
        extra = np.zeros(64)
        extra[start:start + 8] = 20.0
        scenarios[f"interferer on tones {start} to {start + 7} (+20 dB)"] = extra
    for ds_samples in (0.5, 1.0, 2.0):
        ds = ds_samples * ts
        profile = DelayProfile.exponential(ds, ts)
        for sname, extra in scenarios.items():
            obj = MapObjective(est, grid, [(profile, snr) for snr in SNR_DB], extra)
            rules = rule_maps(grid, 8)
            moved = {}
            if extra is not None:
                bad = set(np.flatnonzero(extra > 0).tolist())
                active = np.ones(64, dtype=bool)
                active[list(bad)] = False
                blocked = OfdmGrid(64, grid.spacing_hz, fft_size=64, cp_duration_s=grid.cp_duration_s, active=active)
                for name, pm in rule_maps(blocked, 8).items():
                    moved[f"{name}, moved off interferer"] = pm
            candidates = dict(rules)
            candidates.update(moved)
            t0 = time.perf_counter()
            ga = genetic(obj, 8, population=40, generations=60, seed=0, seeds_in=list(candidates.values()))
            ga_time = time.perf_counter() - t0
            candidates["genetic algorithm"] = ga.pilots
            for name, pm in candidates.items():
                rows.append({"rms_delay_spread_ns": ds * 1e9, "scenario": sname, "map": name, "indices": " ".join(map(str, pm.indices)), "objective_db": 10 * np.log10(obj(pm.indices)), "ga_seconds": ga_time if name == "genetic algorithm" else np.nan, "ga_maps_evaluated": ga.evaluations if name == "genetic algorithm" else np.nan})
        print(f"64-tone link, delay spread {ds*1e9:.0f} ns done", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "pilot_maps.csv", index=False)


if __name__ == "__main__":
    import sys

    if "--part-b-only" not in sys.argv:
        part_a()
    part_b()
    print("done")
