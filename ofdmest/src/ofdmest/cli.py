"""Command-line interface.

Examples::

    ofdmest error --grid lte:100 --profile exp:200ns --snr 0,15,30
    ofdmest recommend --grid nr:51:30 --profile inf:48000:14560:los --snr 10 --tolerance 1
    ofdmest sweep --grid lte:100 --ds 20ns:2us:12 --snr 0,10,20,30 --out sweep.csv
    ofdmest validate --grid custom:64:312500:64:0.8us --map centred:8 --profile exp:100ns --snr 15
    ofdmest search --grid custom:64:312500:64:0.8us --pilots 8 --profile exp:150ns --snr 10,20 --interference 20:28:20
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .analysis import exact_error
from .channel import DelayProfile, OfdmGrid, indoor_factory_delay_spread_s
from .estimators import (
    CubicSplineInterpolation,
    DftInterpolation,
    DftLeastSquares,
    LinearInterpolation,
    NearestPilot,
    OperatingPoint,
    WienerFilter,
    noise_variance_from_snr,
    robust_wiener,
)
from .pilots import DVBT2_SCATTERED_PATTERNS, PilotMap, comb, dvbt2_scattered, lte_crs, nr_dmrs_type1, nr_dmrs_type2, uniform_anchored, uniform_centred
from .recommend import recommend
from .search import MapObjective, exhaustive, genetic, rule_maps
from .simulate import simulate

NR_TC = 1.0 / (480000.0 * 4096.0)
LTE_FFT = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 2048, 100: 2048}
DVBT2_TU_8MHZ_US = {1024: 112.0, 2048: 224.0, 4096: 448.0, 8192: 896.0, 16384: 1792.0, 32768: 3584.0}
DVBT2_CARRIERS = {1024: 853, 2048: 1705, 4096: 3409, 8192: 6817, 16384: 13633, 32768: 27265}


def _seconds(text: str) -> float:
    """Parse a duration such as '200ns', '2.5us', '1ms' or a plain number of seconds."""
    t = text.strip().lower()
    for suffix, scale in (("ns", 1e-9), ("us", 1e-6), ("ms", 1e-3), ("s", 1.0)):
        if t.endswith(suffix):
            return float(t[: -len(suffix)]) * scale
    return float(t)


def _pow2_at_least(n: float) -> int:
    return int(2 ** np.ceil(np.log2(n)))


def parse_grid(text: str, window: int | None = None) -> tuple[OfdmGrid, PilotMap | None, str]:
    """Build the grid and, for standard presets, the default pilot map."""
    parts = text.split(":")
    kind = parts[0].lower()
    if kind == "nr":
        prb = int(parts[1])
        scs_khz = float(parts[2]) if len(parts) > 2 else 30.0
        mu = int(round(np.log2(scs_khz / 15.0)))
        n = 12 * prb
        fft = _pow2_at_least(1.2 * n)
        cp = 144 * 64 * 2.0**-mu * NR_TC
        grid = OfdmGrid(n, scs_khz * 1e3, fft_size=fft, cp_duration_s=cp)
        dmrs_type = int(parts[3]) if len(parts) > 3 else 1
        pilots = nr_dmrs_type1(prb) if dmrs_type == 1 else nr_dmrs_type2(prb)
        return grid, pilots, f"NR {prb} PRB at {scs_khz:g} kHz"
    if kind == "lte":
        prb = int(parts[1])
        n = 12 * prb
        fft = LTE_FFT.get(prb, _pow2_at_least(1.2 * n))
        grid = OfdmGrid(n, 15e3, fft_size=fft, cp_duration_s=144 * (1.0 / (15000.0 * 2048.0)))
        return grid, lte_crs(prb), f"LTE {prb} PRB"
    if kind == "dvbt2":
        fft = int(parts[1].lower().replace("k", "")) * 1024
        pattern = parts[2].upper()
        gi = eval(parts[3]) if len(parts) > 3 else 1.0 / 32.0
        tu = DVBT2_TU_8MHZ_US[fft] * 1e-6
        carriers = DVBT2_CARRIERS[fft]
        n = carriers if window is None else min(window, carriers)
        grid = OfdmGrid(n, 1.0 / tu, fft_size=fft, cp_duration_s=gi * tu)
        return grid, dvbt2_scattered(n, pattern), f"DVB-T2 {fft//1024}K {pattern} GI {gi:g} ({n} carriers analysed)"
    if kind == "custom":
        n, spacing, fft = int(parts[1]), float(parts[2]), int(parts[3])
        cp = _seconds(parts[4]) if len(parts) > 4 else None
        return OfdmGrid(n, spacing, fft_size=fft, cp_duration_s=cp), None, f"custom {n} tones at {spacing:g} Hz"
    raise ValueError(f"unknown grid preset {text}")


def parse_map(text: str | None, grid: OfdmGrid, default: PilotMap | None) -> PilotMap:
    if text is None:
        if default is None:
            raise ValueError("this grid has no default pilot map; give --map")
        return default
    parts = text.split(":")
    kind = parts[0].lower()
    if kind == "comb":
        return comb(grid.n_subcarriers, int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    if kind == "anchored":
        return uniform_anchored(grid.n_subcarriers, int(parts[1]))
    if kind == "centred":
        return uniform_centred(grid.n_subcarriers, int(parts[1]))
    if kind == "list":
        return PilotMap(np.array([int(x) for x in parts[1].split(",")]), "list")
    raise ValueError(f"unknown map {text}")


def parse_profile(text: str, grid: OfdmGrid) -> DelayProfile:
    parts = text.split(":")
    kind = parts[0].lower()
    limit = grid.cp_duration_s
    if kind == "exp":
        ds = _seconds(parts[1])
        max_delay = _seconds(parts[2]) if len(parts) > 2 else (min(15 * ds, limit) if limit else 15 * ds)
        return DelayProfile.exponential(ds, grid.sample_period_s, max_delay_s=max_delay)
    if kind == "uniform":
        return DelayProfile.uniform(_seconds(parts[1]), grid.sample_period_s)
    if kind == "inf":
        volume, surface = float(parts[1]), float(parts[2])
        los = parts[3].lower() == "los" if len(parts) > 3 else True
        ds = indoor_factory_delay_spread_s(volume, surface, los)
        max_delay = min(15 * ds, limit) if limit else 15 * ds
        return DelayProfile.exponential(ds, grid.sample_period_s, max_delay_s=max_delay, name=f"InF {'LOS' if los else 'NLOS'} {ds*1e9:.0f} ns")
    if kind == "csv":
        table = pd.read_csv(parts[1])
        return DelayProfile(table["delay_s"].to_numpy(), table["power"].to_numpy(), name=parts[1])
    if kind == "flat":
        return DelayProfile.single_tap()
    raise ValueError(f"unknown profile {text}")


def parse_estimators(text: str | None, grid: OfdmGrid, pilots: PilotMap) -> list:
    names = (text or "nearest,linear,spline,dft,robust,wiener").split(",")
    out = []
    for raw in names:
        parts = raw.strip().split(":")
        key = parts[0].lower()
        if key == "nearest":
            out.append(NearestPilot())
        elif key == "linear":
            out.append(LinearInterpolation("linear" if len(parts) > 1 and parts[1] == "linear" else "constant"))
        elif key == "spline":
            out.append(CubicSplineInterpolation(parts[1] if len(parts) > 1 else "constant"))
        elif key == "dft":
            out.append(DftInterpolation(_seconds(parts[1]) if len(parts) > 1 else None))
        elif key == "ridge":
            out.append(DftLeastSquares(ridge=float(parts[1]) if len(parts) > 1 else 1e-3, name="DFT least squares (ridge)"))
        elif key == "robust":
            out.append(robust_wiener(grid, _seconds(parts[1]) if len(parts) > 1 else None))
        elif key == "wiener":
            if len(parts) == 1:
                out.append(WienerFilter())
            else:
                ds = _seconds(parts[1])
                max_delay = min(15 * ds, grid.cp_duration_s) if grid.cp_duration_s else 15 * ds
                design = DelayProfile.exponential(ds, grid.sample_period_s, max_delay_s=max_delay)
                var = 10 ** (-float(parts[2]) / 10) if len(parts) > 2 else None
                out.append(WienerFilter(design, var, name=f"Wiener (design {ds*1e9:.0f} ns" + (f", {parts[2]} dB)" if len(parts) > 2 else ")")))
        else:
            raise ValueError(f"unknown estimator {raw}")
    return out


def parse_interference(text: str | None, grid: OfdmGrid) -> np.ndarray | None:
    if not text:
        return None
    extra = np.zeros(grid.n_subcarriers)
    for block in text.split(";"):
        start, stop, level = block.split(":")
        extra[int(start):int(stop)] += float(level)
    return extra


def _snr_list(text: str) -> list[float]:
    if ":" in text:
        lo, hi, n = text.split(":")
        return [float(v) for v in np.linspace(float(lo), float(hi), int(n))]
    return [float(v) for v in text.split(",")]


def _ds_list(text: str) -> list[float]:
    lo, hi, n = text.split(":")
    return [float(v) for v in np.geomspace(_seconds(lo), _seconds(hi), int(n))]


def _emit(df: pd.DataFrame, out: str | None) -> None:
    if out:
        df.to_csv(out, index=False)
        print(f"wrote {out} ({len(df)} rows)")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def cmd_error(args):
    grid, default, label = parse_grid(args.grid, args.window)
    pilots = parse_map(args.map, grid, default)
    profile = parse_profile(args.profile, grid)
    extra = parse_interference(args.interference, grid)
    ests = parse_estimators(args.estimators, grid, pilots)
    print(f"{label}; {pilots.name}: {pilots.count} pilots on {grid.n_active} tones (overhead {100*pilots.overhead(grid):.1f}%); profile {profile.name}, RMS delay spread {profile.rms_delay_spread_s*1e9:.1f} ns")
    rows = []
    for snr in _snr_list(args.snr):
        for est in ests:
            try:
                res = exact_error(est, grid, pilots, profile, snr, extra)
                rows.append({"snr_db": snr, "estimator": est.name, "error_db": res.total_db, "bias_db": 10 * np.log10(max(res.bias_total, 1e-30)), "noise_db": 10 * np.log10(max(res.noise_total, 1e-30)), "edge_error_db": 10 * np.log10(max(res.per_tone[[0, -1]].mean(), 1e-30)), "real_multiplications": res.real_multiplications})
            except ValueError as err:
                rows.append({"snr_db": snr, "estimator": est.name, "error_db": np.nan, "note": str(err)})
    _emit(pd.DataFrame(rows), args.out)


def cmd_recommend(args):
    grid, default, label = parse_grid(args.grid, args.window)
    pilots = parse_map(args.map, grid, default)
    profile = parse_profile(args.profile, grid)
    extra = parse_interference(args.interference, grid)
    ests = parse_estimators(args.estimators, grid, pilots)
    print(f"{label}; {pilots.name}; profile {profile.name}, RMS delay spread {profile.rms_delay_spread_s*1e9:.1f} ns")
    for snr in _snr_list(args.snr):
        rec = recommend(ests, grid, pilots, profile, snr, tolerance_db=args.tolerance, target_db=args.target, extra_noise_db=extra)
        print(f"\nSNR {snr:g} dB")
        print(rec)


def cmd_sweep(args):
    grid, default, label = parse_grid(args.grid, args.window)
    pilots = parse_map(args.map, grid, default)
    extra = parse_interference(args.interference, grid)
    ests = parse_estimators(args.estimators, grid, pilots)
    rows = []
    for ds in _ds_list(args.ds):
        limit = grid.cp_duration_s
        profile = DelayProfile.exponential(ds, grid.sample_period_s, max_delay_s=min(15 * ds, limit) if limit else 15 * ds)
        for snr in _snr_list(args.snr):
            for est in ests:
                try:
                    res = exact_error(est, grid, pilots, profile, snr, extra)
                    rows.append({"grid": label, "map": pilots.name, "rms_delay_spread_ns": ds * 1e9, "snr_db": snr, "estimator": est.name, "error_db": res.total_db, "bias_db": 10 * np.log10(max(res.bias_total, 1e-30)), "noise_db": 10 * np.log10(max(res.noise_total, 1e-30)), "real_multiplications": res.real_multiplications})
                except ValueError:
                    continue
    _emit(pd.DataFrame(rows), args.out)


def cmd_validate(args):
    grid, default, label = parse_grid(args.grid, args.window)
    pilots = parse_map(args.map, grid, default)
    profile = parse_profile(args.profile, grid)
    extra = parse_interference(args.interference, grid)
    ests = parse_estimators(args.estimators, grid, pilots)
    rows = []
    for snr in _snr_list(args.snr):
        sims = simulate(ests, grid, pilots, profile, snr, n_symbols=args.symbols, seed=args.seed, extra_noise_db=extra)
        for est, sim in zip(ests, sims):
            exact = exact_error(est, grid, pilots, profile, snr, extra)
            rows.append({"snr_db": snr, "estimator": est.name, "exact_db": exact.total_db, "monte_carlo_db": sim.mse_db, "mc_half_width_db": 10 * np.log10(1 + sim.mse_half_width_95 / sim.mse_mean), "inside_95": abs(sim.mse_mean - exact.total) <= sim.mse_half_width_95, "qpsk_ber": sim.ber})
    _emit(pd.DataFrame(rows), args.out)


def cmd_search(args):
    grid, default, label = parse_grid(args.grid, args.window)
    extra = parse_interference(args.interference, grid)
    profiles = [parse_profile(p, grid) for p in args.profile.split(";")]
    points = [(prof, snr) for prof in profiles for snr in _snr_list(args.snr)]
    est = parse_estimators(args.estimator, grid, None)[0] if args.estimator else LinearInterpolation()
    objective = MapObjective(est, grid, points, extra)
    rules = rule_maps(grid, args.pilots)
    print(f"{label}; objective: mean exact error of {est.name} over {len(points)} operating points")
    for name, pm in rules.items():
        print(f"{name:12s} {pm.indices.tolist()}  error {10*np.log10(objective(pm.indices)):.2f} dB")
    if args.method == "exhaustive":
        res = exhaustive(objective, args.pilots)
    else:
        res = genetic(objective, args.pilots, population=args.population, generations=args.generations, seed=args.seed, seeds_in=list(rules.values()))
    print(f"{res.pilots.name:12s} {res.pilots.indices.tolist()}  error {10*np.log10(res.objective):.2f} dB  ({res.evaluations} distinct maps evaluated)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ofdmest", description="Exact error analysis of linear pilot-aided OFDM channel estimators")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, profile=True):
        p.add_argument("--grid", required=True, help="nr:<prb>[:<scs_khz>[:<dmrs_type>]] | lte:<prb> | dvbt2:<fft_k>:<pattern>[:<gi>] | custom:<n>:<spacing_hz>:<fft>[:<cp>]")
        p.add_argument("--map", help="comb:<spacing>[:<offset>] | anchored:<k> | centred:<k> | list:<i,j,...> (default: the standard map of the grid)")
        p.add_argument("--window", type=int, help="number of carriers analysed for large DVB-T2 symbols")
        if profile:
            p.add_argument("--profile", required=True, help="exp:<rms>[:<max>] | uniform:<max> | inf:<volume_m3>:<surface_m2>[:los|nlos] | csv:<file> | flat")
        p.add_argument("--snr", required=True, help="list such as 0,10,20 or range lo:hi:n in dB")
        p.add_argument("--interference", help="tone blocks with extra noise, start:stop:dB separated by ';'")
        p.add_argument("--out", help="write the table to this CSV file")

    p = sub.add_parser("error", help="exact error of several estimators at one operating point")
    common(p)
    p.add_argument("--estimators", help="comma list: nearest, linear[:linear], spline[:native], dft[:window], ridge[:lambda], robust[:span], wiener[:ds[:snr]]")
    p.set_defaults(func=cmd_error)

    p = sub.add_parser("recommend", help="cheapest estimator within a tolerance of the matched Wiener error")
    common(p)
    p.add_argument("--estimators")
    p.add_argument("--tolerance", type=float, default=1.0, help="dB above the matched Wiener error")
    p.add_argument("--target", type=float, help="absolute error target in dB instead of the tolerance")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("sweep", help="exact error over a delay-spread and SNR grid")
    common(p, profile=False)
    p.add_argument("--ds", required=True, help="geometric range lo:hi:n such as 20ns:2us:12")
    p.add_argument("--estimators")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("validate", help="compare the exact error with Monte Carlo and report the QPSK bit error rate")
    common(p)
    p.add_argument("--estimators")
    p.add_argument("--symbols", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("search", help="optimise a free pilot map for an estimator")
    common(p)
    p.add_argument("--pilots", type=int, required=True)
    p.add_argument("--estimator", help="single estimator name (default linear)")
    p.add_argument("--method", choices=["ga", "exhaustive"], default="ga")
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_search)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
