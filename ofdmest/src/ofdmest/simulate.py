"""Monte Carlo simulation used to validate the exact error and to measure bit error rates.

Channel realisations are synthesised in the frequency domain from the delay profile,
``H_m = sum_l h_l exp(-j 2 pi m spacing_hz tau_l)`` with independent complex Gaussian taps
of the profile powers, so their covariance is exactly the one used by the analysis. Pilots
carry unit symbols, data tones carry Gray-coded QPSK or 16-QAM, and data are equalised by
dividing the received sample by the channel estimate (zero forcing). All estimators passed to one
call see identical channel, noise and data draws.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import DelayProfile, OfdmGrid
from .estimators import Estimator, OperatingPoint, noise_variance_from_snr
from .pilots import PilotMap


@dataclass(frozen=True)
class SimulationResult:
    estimator: str
    n_symbols: int
    mse_mean: float
    mse_half_width_95: float
    ber: float
    bit_errors: int
    bits: int

    @property
    def mse_db(self) -> float:
        return float(10.0 * np.log10(self.mse_mean))


def _qpsk(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(bits.shape[0], -1, 2)
    return ((1 - 2 * b[..., 0]) + 1j * (1 - 2 * b[..., 1])) / np.sqrt(2.0)


def _qpsk_bits(symbols: np.ndarray) -> np.ndarray:
    out = np.empty(symbols.shape + (2,), dtype=np.int8)
    out[..., 0] = symbols.real < 0
    out[..., 1] = symbols.imag < 0
    return out.reshape(symbols.shape[0], -1)


_GRAY_LEVELS = np.array([3.0, 1.0, -3.0, -1.0])  # levels for bit pairs 00, 01, 10, 11: first bit is the sign, second the inner level


def _qam16(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(bits.shape[0], -1, 4)
    i = _GRAY_LEVELS[2 * b[..., 0] + b[..., 1]]
    q = _GRAY_LEVELS[2 * b[..., 2] + b[..., 3]]
    return (i + 1j * q) / np.sqrt(10.0)


def _axis_bits(values: np.ndarray) -> np.ndarray:
    """Gray-decode one axis of a 16-QAM symbol: first bit is the sign, second the magnitude."""
    out = np.empty(values.shape + (2,), dtype=np.int8)
    out[..., 0] = values < 0
    out[..., 1] = np.abs(values) < 2.0
    return out


def _qam16_bits(symbols: np.ndarray) -> np.ndarray:
    scaled = symbols * np.sqrt(10.0)
    out = np.concatenate([_axis_bits(scaled.real), _axis_bits(scaled.imag)], axis=-1)
    return out.reshape(symbols.shape[0], -1)


_MODULATIONS = {"qpsk": (_qpsk, _qpsk_bits, 2), "16qam": (_qam16, _qam16_bits, 4)}


def simulate(
    estimators: list[Estimator],
    grid: OfdmGrid,
    pilots: PilotMap,
    profile: DelayProfile,
    snr_db: float,
    n_symbols: int = 10000,
    seed: int = 0,
    extra_noise_db: np.ndarray | None = None,
    batch: int = 500,
    modulation: str = "qpsk",
) -> list[SimulationResult]:
    """Matched-draw Monte Carlo of several estimators at one operating point.

    ``modulation`` is ``"qpsk"`` or ``"16qam"``; the SNR is per subcarrier for unit symbol
    energy in both cases.
    """
    if modulation not in _MODULATIONS:
        raise ValueError("modulation must be 'qpsk' or '16qam'")
    modulate, demodulate, bits_per_symbol = _MODULATIONS[modulation]
    point = OperatingPoint(profile, noise_variance_from_snr(snr_db, grid.n_subcarriers, extra_noise_db))
    pos = pilots.positions_in(grid)
    act = grid.active_indices
    data_rows = np.setdiff1d(np.arange(act.size), pos)
    matrices = [est.matrix(grid, pilots, point) for est in estimators]
    basis = profile.frequency_response_basis(act, grid.spacing_hz)
    amp = np.sqrt(profile.powers / 2.0)
    std = np.sqrt(point.noise_variance[act] / 2.0)
    rng = np.random.default_rng(seed)
    sums = np.zeros(len(estimators))
    sums_sq = np.zeros(len(estimators))
    errors = np.zeros(len(estimators), dtype=np.int64)
    done = 0
    while done < n_symbols:
        b = min(batch, n_symbols - done)
        taps = amp * (rng.standard_normal((b, profile.n_taps)) + 1j * rng.standard_normal((b, profile.n_taps)))
        H = taps @ basis.T
        noise = std * (rng.standard_normal((b, act.size)) + 1j * rng.standard_normal((b, act.size)))
        bits = rng.integers(0, 2, size=(b, bits_per_symbol * data_rows.size), dtype=np.int8)
        X = modulate(bits)
        Y = H + noise
        Y[:, data_rows] = H[:, data_rows] * X + noise[:, data_rows]
        Yp = Y[:, pos]
        for i, A in enumerate(matrices):
            Hhat = Yp @ A.T
            err = np.mean(np.abs(Hhat - H) ** 2, axis=1)
            sums[i] += err.sum()
            sums_sq[i] += (err**2).sum()
            Xhat = Y[:, data_rows] / Hhat[:, data_rows]
            errors[i] += int(np.count_nonzero(demodulate(Xhat) != bits))
        done += b
    results = []
    bits_total = n_symbols * bits_per_symbol * data_rows.size
    for i, est in enumerate(estimators):
        mean = sums[i] / n_symbols
        var = max(sums_sq[i] / n_symbols - mean**2, 0.0)
        half = 1.96 * np.sqrt(var / n_symbols)
        results.append(SimulationResult(est.name, n_symbols, float(mean), float(half), errors[i] / bits_total, int(errors[i]), int(bits_total)))
    return results
