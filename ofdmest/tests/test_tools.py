import numpy as np

from ofdmest import DelayProfile, DftInterpolation, LinearInterpolation, OfdmGrid, OperatingPoint, WienerFilter, comb, exact_error, noise_variance_from_snr, robust_wiener, uniform_centred
from ofdmest.cli import main, parse_grid
from ofdmest.cost import benchmark
from ofdmest.recommend import recommend
from ofdmest.search import MapObjective, exhaustive, genetic, rule_maps


def grid64():
    return OfdmGrid(64, 312.5e3, fft_size=64, cp_duration_s=16 / (64 * 312.5e3))


def test_dft_interpolation_matches_least_squares_on_full_band_comb():
    grid = OfdmGrid(64, 15e3, fft_size=64, cp_duration_s=8 / (64 * 15e3))
    pilots = comb(64, 4, 0)
    profile = DelayProfile.uniform(7 * grid.sample_period_s, grid.sample_period_s)
    res = exact_error(DftInterpolation(), grid, pilots, profile, 200.0)
    assert res.total < 1e-10


def test_dft_interpolation_exact_against_monte_carlo():
    from ofdmest import simulate

    grid = OfdmGrid(256, 15e3, fft_size=256, cp_duration_s=18 / (256 * 15e3))
    pilots = comb(256, 6, 1)
    profile = DelayProfile.exponential(3 * grid.sample_period_s, grid.sample_period_s, max_delay_s=grid.cp_duration_s)
    est = DftInterpolation()
    sim = simulate([est], grid, pilots, profile, 20.0, n_symbols=3000, seed=5)[0]
    exact = exact_error(est, grid, pilots, profile, 20.0).total
    assert abs(sim.mse_mean - exact) < 3 * sim.mse_half_width_95 + 1e-9


def test_benchmark_reports_all_estimators():
    grid = grid64()
    pilots = comb(64, 8, 0)
    profile = DelayProfile.exponential(2 * grid.sample_period_s, grid.sample_period_s, max_delay_s=grid.cp_duration_s)
    point = OperatingPoint(profile, noise_variance_from_snr(15.0, 64))
    ests = [LinearInterpolation(), DftInterpolation(), WienerFilter()]
    out = benchmark(ests, grid, pilots, point, n_symbols=200, repeats=2)
    assert [r.estimator for r in out] == [e.name for e in ests]
    assert all(r.dense_apply_us > 0 for r in out)
    assert out[0].structured_apply_us is not None and out[2].structured_apply_us is None


def test_recommend_prefers_cheapest_within_tolerance():
    grid = grid64()
    pilots = comb(64, 2, 0)
    profile = DelayProfile.exponential(1.0 * grid.sample_period_s, grid.sample_period_s, max_delay_s=grid.cp_duration_s)
    rec = recommend([LinearInterpolation(), robust_wiener(grid), WienerFilter()], grid, pilots, profile, 30.0, tolerance_db=20.0)
    assert rec.chosen == LinearInterpolation().name
    rec = recommend([LinearInterpolation(), WienerFilter()], grid, pilots, profile, 5.0, tolerance_db=0.1)
    assert rec.chosen == WienerFilter().name


def test_exhaustive_and_genetic_search_agree_on_a_small_grid():
    grid = OfdmGrid(16, 312.5e3, fft_size=16, cp_duration_s=4 / (16 * 312.5e3))
    profile = DelayProfile.exponential(1.0 * grid.sample_period_s, grid.sample_period_s, max_delay_s=grid.cp_duration_s)
    obj = MapObjective(LinearInterpolation(), grid, [(profile, 15.0)])
    best = exhaustive(obj, 3)
    ga = genetic(obj, 3, population=20, generations=30, seed=1)
    assert ga.objective >= best.objective - 1e-12
    assert abs(ga.objective - best.objective) < 1e-9
    rules = rule_maps(grid, 3)
    assert set(rules) == {"anchored", "centred"}


def test_rule_maps_avoid_inactive_tones():
    active = np.ones(64, dtype=bool)
    active[20:28] = False
    grid = OfdmGrid(64, 312.5e3, fft_size=64, active=active)
    for pm in rule_maps(grid, 8).values():
        assert grid.active[pm.indices].all()
        assert pm.count == 8


def test_cli_presets_and_commands(capsys, tmp_path):
    grid, pilots, label = parse_grid("nr:6:30")
    assert grid.n_subcarriers == 72 and pilots.count == 36 and abs(grid.cp_duration_s - 2.34375e-6) < 1e-12
    grid, pilots, label = parse_grid("lte:6")
    assert grid.fft_size == 128 and pilots.count == 12
    grid, pilots, label = parse_grid("dvbt2:8k:PP2", window=240)
    assert grid.n_subcarriers == 240 and pilots.count == 20
    out = tmp_path / "t.csv"
    assert main(["error", "--grid", "lte:6", "--profile", "exp:100ns", "--snr", "10", "--estimators", "linear,wiener", "--out", str(out)]) == 0
    assert out.exists()
    assert main(["recommend", "--grid", "custom:64:312500:64:0.8us", "--map", "centred:8", "--profile", "flat", "--snr", "10", "--estimators", "nearest,linear,wiener"]) == 0
    assert "chosen" in capsys.readouterr().out
    assert main(["error", "--grid", "nosuch:1", "--profile", "flat", "--snr", "10"]) == 2
