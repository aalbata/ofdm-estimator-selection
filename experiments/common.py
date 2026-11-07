"""Shared definitions of the grids, pilot maps, channel profiles and estimators of the study."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ofdmest import (
    CubicSplineInterpolation,
    DelayProfile,
    DftInterpolation,
    DftLeastSquares,
    LinearInterpolation,
    NearestPilot,
    OfdmGrid,
    WienerFilter,
    comb,
    indoor_factory_delay_spread_s,
    lte_crs,
    nr_dmrs_type1,
    nr_dmrs_type2,
    robust_wiener,
    uniform_anchored,
    uniform_centred,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

NR_TC = 1.0 / (480000.0 * 4096.0)
NR_CP_30KHZ = 144 * 64 * 0.5 * NR_TC
LTE_CP = 144.0 / (15000.0 * 2048.0)
T2_TU_8K = 896e-6

SNR_DB = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]


def delay_spreads_for(grid: OfdmGrid, n: int = 12, lowest_s: float = 10e-9) -> list[float]:
    """Geometric sweep of RMS delay spread from ``lowest_s`` to a quarter of the cyclic prefix.

    An exponential profile confined to the prefix cannot exceed about 0.29 times the prefix
    in RMS delay spread, and channels beyond that leave the block-fading model.
    """
    return [float(v) for v in np.geomspace(lowest_s, 0.25 * grid.cp_duration_s, n)]


def grids() -> dict[str, tuple[OfdmGrid, object]]:
    """Standard grids with their pilot maps."""
    nr = OfdmGrid(12 * 51, 30e3, fft_size=1024, cp_duration_s=NR_CP_30KHZ)
    lte = OfdmGrid(12 * 100, 15e3, fft_size=2048, cp_duration_s=LTE_CP)
    # Guard intervals allowed with each 8K scattered pilot pattern in SISO mode
    # (ETSI EN 302 755 V1.4.1, Table 58): PP1 with 1/4, PP3 with 1/8, PP5 with 1/16, PP7 with 1/128.
    def t2(gi: float) -> OfdmGrid:
        return OfdmGrid(1024, 1.0 / T2_TU_8K, fft_size=8192, cp_duration_s=gi * T2_TU_8K)

    return {
        "NR 20 MHz, DM-RS type 1": (nr, nr_dmrs_type1(51)),
        "NR 20 MHz, DM-RS type 2": (nr, nr_dmrs_type2(51)),
        "LTE 20 MHz, CRS": (lte, lte_crs(100)),
        "DVB-T2 8K, PP1 (spacing 3)": (t2(1 / 4), comb(1024, 3, 0, "DVB-T2 spacing 3")),
        "DVB-T2 8K, PP3 (spacing 6)": (t2(1 / 8), comb(1024, 6, 0, "DVB-T2 spacing 6")),
        "DVB-T2 8K, PP5 (spacing 12)": (t2(1 / 16), comb(1024, 12, 0, "DVB-T2 spacing 12")),
        "DVB-T2 8K, PP7 (spacing 24)": (t2(1 / 128), comb(1024, 24, 0, "DVB-T2 spacing 24")),
    }


def proprietary_grid() -> OfdmGrid:
    """A 64-tone link at 312.5 kHz spacing with a 0.8 us cyclic prefix (16 samples)."""
    return OfdmGrid(64, 312.5e3, fft_size=64, cp_duration_s=16 / (64 * 312.5e3))


def site_profiles(grid: OfdmGrid) -> dict[str, DelayProfile]:
    """Exponential profiles with the delay spreads of measured and modelled factory halls.

    The NIST values are the mean expected RMS delay spreads of Table 4.2 of NIST TN 1951
    at 2.2 GHz (automotive plant internal route, machine shop, steam plant). The 3GPP values
    are the medians of the TR 38.901 indoor factory model for a small hall (60 m x 40 m x
    6 m) and a large hall (300 m x 150 m x 12 m), line of sight and non line of sight.
    """
    halls = {"small": (60.0, 40.0, 6.0), "large": (300.0, 150.0, 12.0)}
    out = {}
    out["NIST steam plant (36 ns)"] = 36.1e-9
    out["NIST machine shop (74 ns)"] = 74.3e-9
    out["NIST automotive plant (132 ns)"] = 131.8e-9
    for name, (l, w, h) in halls.items():
        volume = l * w * h
        surface = 2 * (l * w + l * h + w * h)
        for los in (True, False):
            ds = indoor_factory_delay_spread_s(volume, surface, los)
            out[f"3GPP InF {name} hall {'LOS' if los else 'NLOS'} ({ds*1e9:.0f} ns)"] = ds
    profiles = {}
    for name, ds in out.items():
        limit = grid.cp_duration_s
        profiles[name] = DelayProfile.exponential(ds, grid.sample_period_s, max_delay_s=min(15 * ds, limit), name=name)
    return profiles


def exponential_profile(grid: OfdmGrid, ds: float) -> DelayProfile:
    return DelayProfile.exponential(ds, grid.sample_period_s, max_delay_s=min(15 * ds, grid.cp_duration_s))


def estimator_set(grid: OfdmGrid, pilots, profile: DelayProfile | None = None) -> list:
    """The estimators of the study, each with a stated role.

    Interpolators (nearest, linear, cubic spline) need no statistics. The transform-domain
    estimator uses the cyclic prefix as its tap window. The robust Wiener filter assumes a
    uniform profile over the cyclic prefix. The matched Wiener filter is the reference (best
    linear estimator). Mismatched designs are added by the mismatch study.
    """
    out = [NearestPilot(), LinearInterpolation("constant"), CubicSplineInterpolation("constant")]
    steps = np.diff(pilots.indices)
    if np.all(steps == steps[0]):
        out.append(DftInterpolation())
    else:
        out.append(DftLeastSquares(ridge=1e-3, name="DFT least squares (ridge)"))
    out.append(robust_wiener(grid))
    out.append(WienerFilter())
    return out


SNR_INDEPENDENT = (NearestPilot, LinearInterpolation, CubicSplineInterpolation, DftInterpolation, DftLeastSquares)
