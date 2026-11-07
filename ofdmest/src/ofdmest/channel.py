"""OFDM frequency grid and second-order channel models.

The analysis in this package needs only the second-order statistics of the channel
frequency response. A channel is described by a discrete power-delay profile: tap delays
in seconds and tap powers that sum to one. The frequency correlation between two tones
separated by ``k`` subcarrier spacings is

    r(k) = sum_l p_l exp(-j 2 pi k spacing_hz tau_l),

which gives a Hermitian Toeplitz covariance matrix over the analysed band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import toeplitz


@dataclass(frozen=True)
class OfdmGrid:
    """Frequency grid of the analysed OFDM band.

    The band holds ``n_subcarriers`` tones with uniform ``spacing_hz``. Tone index 0 is the
    lowest tone of the band. ``fft_size`` sets the transmitter sample period
    ``1 / (fft_size * spacing_hz)``, which is the tap grid of DFT-domain estimators and of
    sampled delay profiles. ``cp_duration_s`` is the cyclic prefix length, used as the
    default tap window of DFT-domain estimators. ``active`` marks the tones that carry
    data or pilots; tones outside the mask (for example a null DC tone) are neither
    estimated nor counted in the error.
    """

    n_subcarriers: int
    spacing_hz: float
    fft_size: int | None = None
    cp_duration_s: float | None = None
    active: np.ndarray | None = None

    def __post_init__(self) -> None:
        if int(self.n_subcarriers) < 2:
            raise ValueError("n_subcarriers must be at least 2")
        if not self.spacing_hz > 0:
            raise ValueError("spacing_hz must be positive")
        object.__setattr__(self, "n_subcarriers", int(self.n_subcarriers))
        if self.fft_size is None:
            size = int(2 ** np.ceil(np.log2(self.n_subcarriers)))
            object.__setattr__(self, "fft_size", size)
        if int(self.fft_size) < self.n_subcarriers:
            raise ValueError("fft_size must be at least n_subcarriers")
        object.__setattr__(self, "fft_size", int(self.fft_size))
        if self.cp_duration_s is not None and not self.cp_duration_s > 0:
            raise ValueError("cp_duration_s must be positive when given")
        if self.active is None:
            mask = np.ones(self.n_subcarriers, dtype=bool)
        else:
            mask = np.asarray(self.active, dtype=bool)
            if mask.shape != (self.n_subcarriers,):
                raise ValueError("active must have one entry per subcarrier")
            if mask.sum() < 2:
                raise ValueError("at least two active subcarriers are required")
        mask.setflags(write=False)
        object.__setattr__(self, "active", mask)

    @property
    def sample_period_s(self) -> float:
        """Transmitter sample period in seconds."""
        return 1.0 / (self.fft_size * self.spacing_hz)

    @property
    def active_indices(self) -> np.ndarray:
        """Indices of the active tones in increasing order."""
        return np.flatnonzero(self.active)

    @property
    def n_active(self) -> int:
        """Number of active tones."""
        return int(self.active.sum())

    @property
    def cp_samples(self) -> int | None:
        """Cyclic prefix length in samples, rounded down, or None when not set."""
        if self.cp_duration_s is None:
            return None
        return int(np.floor(self.cp_duration_s / self.sample_period_s + 1e-9))

    @property
    def bandwidth_hz(self) -> float:
        """Width of the analysed band in hertz."""
        return self.n_subcarriers * self.spacing_hz


@dataclass(frozen=True)
class DelayProfile:
    """Discrete power-delay profile with delays in seconds and powers summing to one."""

    delays_s: np.ndarray
    powers: np.ndarray
    name: str = "custom"

    def __post_init__(self) -> None:
        delays = np.asarray(self.delays_s, dtype=float).reshape(-1)
        powers = np.asarray(self.powers, dtype=float).reshape(-1)
        if delays.size == 0 or delays.shape != powers.shape:
            raise ValueError("delays_s and powers must be non-empty arrays of equal length")
        if np.any(delays < 0):
            raise ValueError("delays must be non-negative")
        if np.any(powers < 0) or not np.isfinite(powers).all() or powers.sum() <= 0:
            raise ValueError("powers must be non-negative with a positive sum")
        order = np.argsort(delays, kind="stable")
        delays = delays[order]
        powers = powers[order] / powers.sum()
        delays.setflags(write=False)
        powers.setflags(write=False)
        object.__setattr__(self, "delays_s", delays)
        object.__setattr__(self, "powers", powers)

    @property
    def n_taps(self) -> int:
        return int(self.delays_s.size)

    @property
    def mean_delay_s(self) -> float:
        return float(np.sum(self.powers * self.delays_s))

    @property
    def rms_delay_spread_s(self) -> float:
        """Root-mean-square delay spread of the profile."""
        mean = self.mean_delay_s
        second = float(np.sum(self.powers * self.delays_s**2))
        return float(np.sqrt(max(second - mean**2, 0.0)))

    @property
    def max_delay_s(self) -> float:
        return float(self.delays_s[-1])

    def frequency_correlation(self, spacing_hz: float, lags: np.ndarray) -> np.ndarray:
        """Correlation r(k) between tones separated by ``lags`` subcarrier spacings."""
        lags = np.asarray(lags, dtype=float)
        phase = -2j * np.pi * spacing_hz * np.outer(lags, self.delays_s)
        return np.exp(phase) @ self.powers.astype(complex)

    def covariance(self, n_subcarriers: int, spacing_hz: float) -> np.ndarray:
        """Hermitian Toeplitz covariance of the frequency response over a band.

        The matrix is cached per band size and spacing, since a profile is typically
        evaluated many times on the same grid; the cached array is read-only.
        """
        key = (int(n_subcarriers), float(spacing_hz))
        cache = self.__dict__.get("_covariance_cache")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_covariance_cache", cache)
        if key not in cache:
            column = self.frequency_correlation(spacing_hz, np.arange(n_subcarriers))
            R = toeplitz(column, np.conj(column))
            R.setflags(write=False)
            cache[key] = R
        return cache[key]

    def frequency_response_basis(self, tones: np.ndarray, spacing_hz: float) -> np.ndarray:
        """Matrix G with G[m, l] = exp(-j 2 pi tone_m spacing_hz tau_l) for synthesis."""
        tones = np.asarray(tones, dtype=float)
        return np.exp(-2j * np.pi * spacing_hz * np.outer(tones, self.delays_s))

    def resampled(self, sample_period_s: float, n_taps: int | None = None) -> "DelayProfile":
        """Profile with each tap moved to the nearest sample-grid delay, powers merged."""
        bins = np.rint(self.delays_s / sample_period_s).astype(int)
        length = int(bins.max()) + 1 if n_taps is None else int(n_taps)
        powers = np.zeros(length)
        for b, p in zip(bins, self.powers):
            if b < length:
                powers[b] += p
        keep = powers > 0
        delays = np.arange(length)[keep] * sample_period_s
        return DelayProfile(delays, powers[keep], name=f"{self.name} (sampled)")

    @staticmethod
    def single_tap() -> "DelayProfile":
        """Frequency-flat channel."""
        return DelayProfile(np.zeros(1), np.ones(1), name="flat")

    @staticmethod
    def uniform(max_delay_s: float, sample_period_s: float, name: str | None = None) -> "DelayProfile":
        """Equal-power taps on the sample grid from 0 up to and including ``max_delay_s``."""
        n_taps = int(np.floor(max_delay_s / sample_period_s + 1e-9)) + 1
        delays = np.arange(n_taps) * sample_period_s
        return DelayProfile(delays, np.ones(n_taps), name=name or f"uniform {max_delay_s*1e9:.0f} ns")

    @staticmethod
    def exponential(
        rms_delay_spread_s: float,
        sample_period_s: float,
        max_delay_s: float | None = None,
        name: str | None = None,
    ) -> "DelayProfile":
        """Sampled exponential profile whose discrete RMS delay spread equals the target.

        Tap powers are proportional to exp(-tau_l / d) on the sample grid, truncated at
        ``max_delay_s`` (default: fifteen times the target delay spread). The decay constant
        ``d`` is found by bisection so that the RMS delay spread of the truncated discrete
        profile equals ``rms_delay_spread_s`` to within 0.01 percent.
        """
        if rms_delay_spread_s <= 0:
            raise ValueError("rms_delay_spread_s must be positive")
        if max_delay_s is None:
            max_delay_s = 15.0 * rms_delay_spread_s
        n_taps = int(np.floor(max_delay_s / sample_period_s + 1e-9)) + 1
        delays = np.arange(n_taps) * sample_period_s
        if n_taps < 2:
            raise ValueError("the sample period is too coarse for the requested delay spread")

        def rms_for(decay: float) -> float:
            powers = np.exp(-delays / decay)
            powers /= powers.sum()
            mean = np.sum(powers * delays)
            return float(np.sqrt(max(np.sum(powers * delays**2) - mean**2, 0.0)))

        lo, hi = 1e-3 * sample_period_s, 1e3 * max_delay_s
        if rms_for(hi) < rms_delay_spread_s:
            raise ValueError("max_delay_s is too short for the requested delay spread")
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            if rms_for(mid) < rms_delay_spread_s:
                lo = mid
            else:
                hi = mid
            if hi / lo < 1 + 1e-6:
                break
        decay = np.sqrt(lo * hi)
        label = name or f"exponential {rms_delay_spread_s*1e9:.0f} ns"
        return DelayProfile(delays, np.exp(-delays / decay), name=label)


def indoor_factory_delay_spread_s(volume_m3: float, surface_m2: float, line_of_sight: bool) -> float:
    """Median RMS delay spread of the 3GPP TR 38.901 indoor factory model.

    Uses the mean of lgDS from Table 7.5-6 Part-3 of TR 38.901 (V19.5.0):
    LOS  lgDS = log10(26 (V/S) + 14) - 9.35,
    NLOS lgDS = log10(30 (V/S) + 32) - 9.44,
    with V the hall volume in cubic metres and S its total surface area in square metres.
    """
    ratio = volume_m3 / surface_m2
    if line_of_sight:
        lg = np.log10(26.0 * ratio + 14.0) - 9.35
    else:
        lg = np.log10(30.0 * ratio + 32.0) - 9.44
    return float(10.0**lg)
