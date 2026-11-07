import numpy as np
import pytest

from ofdmest import (
    CubicSplineInterpolation,
    DelayProfile,
    DftLeastSquares,
    LinearInterpolation,
    NearestPilot,
    OfdmGrid,
    OperatingPoint,
    PilotMap,
    WienerFilter,
    comb,
    dvbt2_scattered,
    error_from_matrix,
    exact_error,
    indoor_factory_delay_spread_s,
    lte_crs,
    noise_variance_from_snr,
    nr_dmrs_type1,
    nr_dmrs_type2,
    robust_wiener,
    simulate,
    uniform_anchored,
    uniform_centred,
)


def small_grid(n=64, spacing=15e3, fft=64, cp_samples=16):
    return OfdmGrid(n, spacing, fft_size=fft, cp_duration_s=cp_samples / (fft * spacing))


def test_covariance_matches_dft_model_for_sampled_taps():
    grid = small_grid()
    powers = np.array([0.5, 0.3, 0.2])
    profile = DelayProfile(np.arange(3) * grid.sample_period_s, powers)
    R = profile.covariance(grid.n_subcarriers, grid.spacing_hz)
    F = np.exp(-2j * np.pi * np.outer(np.arange(64), np.arange(3)) / 64)
    expected = F @ np.diag(powers) @ F.conj().T
    assert np.allclose(R, expected)
    assert np.allclose(R, R.conj().T)
    assert np.allclose(np.diag(R).real, 1.0)


def test_exponential_profile_reaches_target_delay_spread():
    grid = small_grid(fft=1024, spacing=30e3)
    target = 120e-9
    profile = DelayProfile.exponential(target, grid.sample_period_s)
    assert abs(profile.rms_delay_spread_s - target) / target < 1e-4
    assert abs(profile.powers.sum() - 1.0) < 1e-12


def test_indoor_factory_delay_spread_formula():
    assert abs(indoor_factory_delay_spread_s(1.0, 1.0, True) - 40.0 * 10**-9.35) < 1e-15
    assert abs(indoor_factory_delay_spread_s(1.0, 1.0, False) - 62.0 * 10**-9.44) < 1e-15


def test_standard_pilot_maps():
    t1 = nr_dmrs_type1(2, port=0)
    assert t1.indices.tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    assert nr_dmrs_type1(1, port=2).indices.tolist() == [1, 3, 5, 7, 9, 11]
    assert nr_dmrs_type2(1, port=0).indices.tolist() == [0, 1, 6, 7]
    assert nr_dmrs_type2(1, port=2).indices.tolist() == [2, 3, 8, 9]
    assert nr_dmrs_type2(1, port=4).indices.tolist() == [4, 5, 10, 11]
    assert lte_crs(2, v_shift=1, v=3).indices.tolist() == [4, 10, 16, 22]
    pp2 = dvbt2_scattered(48, "PP2", symbol_index=1)
    assert pp2.indices.tolist() == [6, 18, 30, 42]
    assert uniform_anchored(64, 8).indices.tolist() == [0, 9, 18, 27, 36, 45, 54, 63]
    assert uniform_centred(64, 8).indices.tolist() == [4, 12, 20, 28, 36, 44, 52, 60]


def test_pilots_must_be_active():
    active = np.ones(64, dtype=bool)
    active[32] = False
    grid = OfdmGrid(64, 15e3, fft_size=64, active=active)
    with pytest.raises(ValueError):
        comb(64, 8, 0).validate(grid)
    comb(64, 8, 1).validate(grid)


def test_interpolators_reproduce_pilot_values_and_partition_unity():
    grid = small_grid()
    pilots = uniform_anchored(64, 8)
    pos = pilots.positions_in(grid)
    for est in (NearestPilot(), LinearInterpolation("constant"), LinearInterpolation("linear"), CubicSplineInterpolation("constant"), CubicSplineInterpolation("native")):
        A = est.matrix(grid, pilots)
        assert A.shape == (64, 8)
        assert np.allclose(A[pos], np.eye(8))
        assert np.allclose(A.sum(axis=1), 1.0)


def test_flat_channel_gives_only_noise_error():
    grid = small_grid()
    pilots = uniform_centred(64, 8)
    profile = DelayProfile.single_tap()
    for est in (NearestPilot(), LinearInterpolation(), CubicSplineInterpolation()):
        res = exact_error(est, grid, pilots, profile, 10.0)
        assert res.bias_total < 1e-12
        assert res.noise_total > 0


def test_dft_least_squares_is_exact_on_grid_without_noise():
    grid = small_grid(cp_samples=8)
    profile = DelayProfile.uniform(7 * grid.sample_period_s, grid.sample_period_s)
    pilots = comb(64, 8, 0)
    res = exact_error(DftLeastSquares(), grid, pilots, profile, 200.0)
    assert res.total < 1e-10


def test_matched_wiener_is_the_best_linear_estimator():
    grid = small_grid(fft=64, cp_samples=16)
    profile = DelayProfile.exponential(4 * grid.sample_period_s, grid.sample_period_s, max_delay_s=15 * grid.sample_period_s)
    pilots = comb(64, 4, 1)
    best = exact_error(WienerFilter(), grid, pilots, profile, 15.0).total
    for est in (NearestPilot(), LinearInterpolation(), CubicSplineInterpolation(), DftLeastSquares(), robust_wiener(grid), WienerFilter(DelayProfile.exponential(2 * grid.sample_period_s, grid.sample_period_s, max_delay_s=15 * grid.sample_period_s), 0.1)):
        assert exact_error(est, grid, pilots, profile, 15.0).total >= best - 1e-12


def test_identity_estimator_with_all_pilots():
    grid = small_grid()
    pilots = PilotMap(np.arange(64))
    profile = DelayProfile.exponential(3 * grid.sample_period_s, grid.sample_period_s, max_delay_s=15 * grid.sample_period_s)
    point = OperatingPoint(profile, noise_variance_from_snr(10.0, 64))
    bias, noise = error_from_matrix(np.eye(64), grid, pilots, point)
    assert np.allclose(bias, 0.0)
    assert np.allclose(noise, 0.1)


def test_exact_error_agrees_with_monte_carlo():
    grid = small_grid(fft=64, cp_samples=16)
    profile = DelayProfile.exponential(3 * grid.sample_period_s, grid.sample_period_s, max_delay_s=15 * grid.sample_period_s)
    pilots = uniform_centred(64, 8)
    estimators = [LinearInterpolation(), CubicSplineInterpolation(), WienerFilter(), DftLeastSquares(n_taps=8)]
    sims = simulate(estimators, grid, pilots, profile, 15.0, n_symbols=4000, seed=3)
    for est, sim in zip(estimators, sims):
        exact = exact_error(est, grid, pilots, profile, 15.0).total
        assert abs(sim.mse_mean - exact) < 3 * sim.mse_half_width_95 + 1e-9


def test_simulation_is_deterministic_for_a_seed():
    grid = small_grid()
    profile = DelayProfile.exponential(3 * grid.sample_period_s, grid.sample_period_s, max_delay_s=15 * grid.sample_period_s)
    pilots = uniform_centred(64, 8)
    a = simulate([LinearInterpolation()], grid, pilots, profile, 10.0, n_symbols=500, seed=7)[0]
    b = simulate([LinearInterpolation()], grid, pilots, profile, 10.0, n_symbols=500, seed=7)[0]
    assert a == b


def test_16qam_mapping_is_gray_and_unit_energy():
    from ofdmest.simulate import _GRAY_LEVELS, _qam16, _qam16_bits

    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=(10, 4000), dtype=np.int8)
    x = _qam16(bits)
    assert abs(np.mean(np.abs(x) ** 2) - 1.0) < 0.02
    assert (_qam16_bits(x) == bits).all()
    codes = [format(int(i), "02b") for i in np.argsort(_GRAY_LEVELS)]
    assert all(sum(a != b for a, b in zip(codes[k], codes[k + 1])) == 1 for k in range(3))


def test_simulate_16qam_runs_and_matches_exact_error():
    grid = small_grid(fft=64, cp_samples=16)
    profile = DelayProfile.exponential(3 * grid.sample_period_s, grid.sample_period_s, max_delay_s=15 * grid.sample_period_s)
    pilots = uniform_centred(64, 8)
    est = LinearInterpolation()
    sim = simulate([est], grid, pilots, profile, 20.0, n_symbols=2000, seed=9, modulation="16qam")[0]
    exact = exact_error(est, grid, pilots, profile, 20.0).total
    assert abs(sim.mse_mean - exact) < 3 * sim.mse_half_width_95 + 1e-9
    assert 0.0 < sim.ber < 0.5
