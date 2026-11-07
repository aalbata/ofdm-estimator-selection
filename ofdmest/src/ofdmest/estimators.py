"""Linear pilot-aided channel estimators as matrices.

Every estimator here forms the estimate over the active tones as ``H_hat = A y_p``, where
``y_p`` holds the least-squares pilot observations (received pilot samples divided by the
known pilot symbols). The matrix ``A`` has one row per active tone and one column per
pilot. Because each estimator is linear, its expected error follows exactly from the
second-order statistics of the channel and the noise (see :mod:`ofdmest.analysis`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .channel import DelayProfile, OfdmGrid
from .pilots import PilotMap


@dataclass(frozen=True)
class OperatingPoint:
    """Actual channel statistics and per-pilot noise variances at which an estimator runs."""

    profile: DelayProfile
    noise_variance: np.ndarray

    def __post_init__(self) -> None:
        var = np.asarray(self.noise_variance, dtype=float).reshape(-1)
        if np.any(var < 0):
            raise ValueError("noise variances must be non-negative")
        var.setflags(write=False)
        object.__setattr__(self, "noise_variance", var)


def noise_variance_from_snr(snr_db: float, n_subcarriers: int, extra_db: np.ndarray | None = None) -> np.ndarray:
    """Per-tone noise variance for unit channel power and unit-modulus pilots.

    ``snr_db`` is the average signal-to-noise ratio per tone. ``extra_db`` optionally raises
    the noise level of individual tones, for example to model narrowband interference.
    """
    var = np.full(n_subcarriers, 10.0 ** (-snr_db / 10.0))
    if extra_db is not None:
        extra = np.asarray(extra_db, dtype=float)
        if extra.shape != (n_subcarriers,):
            raise ValueError("extra_db must have one entry per subcarrier")
        var = var * 10.0 ** (extra / 10.0)
    return var


def _edge_rows(A: np.ndarray, tones: np.ndarray, pilots: np.ndarray, mode: str) -> np.ndarray:
    """Overwrite the rows of ``A`` for tones outside the pilot span according to ``mode``."""
    if mode == "native":
        return A
    A = A.copy()
    lo, hi = pilots[0], pilots[-1]
    below = tones < lo
    above = tones > hi
    if mode == "constant":
        A[below, :] = 0.0
        A[below, 0] = 1.0
        A[above, :] = 0.0
        A[above, -1] = 1.0
    elif mode == "linear":
        t = (tones[below] - pilots[0]) / (pilots[1] - pilots[0])
        A[below, :] = 0.0
        A[below, 0] = 1.0 - t
        A[below, 1] = t
        t = (tones[above] - pilots[-2]) / (pilots[-1] - pilots[-2])
        A[above, :] = 0.0
        A[above, -2] = 1.0 - t
        A[above, -1] = t
    else:
        raise ValueError("extrapolation must be 'constant', 'linear' or 'native'")
    return A


class Estimator:
    """Base class: subclasses return the estimation matrix for a grid, pilot map and operating point."""

    name = "estimator"
    needs_statistics = False

    def matrix(self, grid: OfdmGrid, pilots: PilotMap, point: OperatingPoint | None = None) -> np.ndarray:
        raise NotImplementedError

    def real_multiplications(self, grid: OfdmGrid, pilots: PilotMap) -> int:
        """Real multiplications per OFDM symbol for a straightforward implementation."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class NearestPilot(Estimator):
    """Each tone takes the estimate of its nearest pilot (ties go to the lower pilot)."""

    name = "nearest pilot"

    def matrix(self, grid, pilots, point=None):
        pilots.validate(grid)
        tones = grid.active_indices
        p = pilots.indices
        right = np.searchsorted(p, tones, side="left")
        right = np.clip(right, 0, p.size - 1)
        left = np.clip(right - 1, 0, p.size - 1)
        choose_left = (np.abs(tones - p[left]) <= np.abs(tones - p[right]))
        col = np.where(choose_left, left, right)
        A = np.zeros((tones.size, p.size))
        A[np.arange(tones.size), col] = 1.0
        return A

    def real_multiplications(self, grid, pilots):
        return 0


class LinearInterpolation(Estimator):
    """Straight-line interpolation between neighbouring pilots.

    Beyond the outermost pilots the estimate is held constant (``extrapolation='constant'``)
    or continued along the line through the two outermost pilots (``'linear'``).
    """

    def __init__(self, extrapolation: str = "constant") -> None:
        if extrapolation not in ("constant", "linear"):
            raise ValueError("extrapolation must be 'constant' or 'linear'")
        self.extrapolation = extrapolation
        self.name = f"linear ({extrapolation} edges)"

    def matrix(self, grid, pilots, point=None):
        pilots.validate(grid)
        tones = grid.active_indices
        p = pilots.indices.astype(float)
        A = np.empty((tones.size, p.size))
        for k in range(p.size):
            unit = np.zeros(p.size)
            unit[k] = 1.0
            A[:, k] = np.interp(tones, p, unit)
        return _edge_rows(A, tones, pilots.indices, self.extrapolation)

    def real_multiplications(self, grid, pilots):
        return 4 * grid.n_active


class CubicSplineInterpolation(Estimator):
    """Cubic spline through the pilot estimates with not-a-knot end conditions.

    Outside the pilot span the estimate is held constant, continued linearly from the two
    outermost pilots, or taken from the spline's own end polynomials (``'native'``).
    """

    def __init__(self, extrapolation: str = "constant") -> None:
        if extrapolation not in ("constant", "linear", "native"):
            raise ValueError("extrapolation must be 'constant', 'linear' or 'native'")
        self.extrapolation = extrapolation
        self.name = f"cubic spline ({extrapolation} edges)"

    def matrix(self, grid, pilots, point=None):
        pilots.validate(grid)
        if pilots.count < 4:
            raise ValueError("a cubic spline needs at least four pilots")
        tones = grid.active_indices
        spline = CubicSpline(pilots.indices.astype(float), np.eye(pilots.count), axis=0, bc_type="not-a-knot", extrapolate=True)
        A = spline(tones.astype(float))
        return _edge_rows(A, tones, pilots.indices, self.extrapolation)

    def real_multiplications(self, grid, pilots):
        return 6 * grid.n_active + 10 * pilots.count


class WienerFilter(Estimator):
    """Linear minimum-mean-square-error (Wiener) interpolation in frequency.

    The filter is designed from a delay profile and per-pilot noise variances. With
    ``design_profile=None`` and ``design_noise=None`` it uses the actual operating point
    (matched design). A different profile or noise level gives a mismatched design, which is
    how a receiver with an assumed delay profile is modelled.
    """

    needs_statistics = True

    def __init__(self, design_profile: DelayProfile | None = None, design_noise_variance: float | np.ndarray | None = None, name: str | None = None) -> None:
        self.design_profile = design_profile
        self.design_noise_variance = design_noise_variance
        if name is not None:
            self.name = name
        elif design_profile is None and design_noise_variance is None:
            self.name = "Wiener (matched)"
        else:
            label = design_profile.name if design_profile is not None else "actual profile"
            self.name = f"Wiener ({label})"

    def matrix(self, grid, pilots, point=None):
        pilots.validate(grid)
        if point is None and (self.design_profile is None or self.design_noise_variance is None):
            raise ValueError("a matched Wiener filter needs the operating point")
        profile = self.design_profile if self.design_profile is not None else point.profile
        if self.design_noise_variance is None:
            var = point.noise_variance
        else:
            var = np.asarray(self.design_noise_variance, dtype=float)
        var = np.broadcast_to(var, (grid.n_subcarriers,)) if var.ndim == 0 or var.size == 1 else var
        if var.shape != (grid.n_subcarriers,):
            raise ValueError("noise variance must be a scalar or have one entry per subcarrier")
        R = profile.covariance(grid.n_subcarriers, grid.spacing_hz)
        p = pilots.indices
        act = grid.active_indices
        Rpp = R[np.ix_(p, p)] + np.diag(var[p])
        Rap = R[np.ix_(act, p)]
        return np.linalg.solve(Rpp.conj().T, Rap.conj().T).conj().T

    def real_multiplications(self, grid, pilots):
        return 4 * grid.n_active * pilots.count


class DftLeastSquares(Estimator):
    """Least-squares fit of a tap model on the sample grid, evaluated on all tones.

    The channel is modelled as ``n_taps`` taps at integer multiples of the sample period
    (default: the cyclic prefix length of the grid). The tap vector is the least-squares
    solution from the pilot observations, optionally with ridge regularisation, and the
    estimate is its frequency response. With comb pilots and no regularisation this is the
    classical DFT-based estimator (inverse DFT of the pilot estimates, truncation to the
    tap window, DFT back to all tones).
    """

    def __init__(self, n_taps: int | None = None, ridge: float = 0.0, name: str | None = None) -> None:
        if n_taps is not None and n_taps < 1:
            raise ValueError("n_taps must be positive")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.n_taps = n_taps
        self.ridge = ridge
        self.name = name or ("DFT least squares" if n_taps is None else f"DFT least squares ({n_taps} taps)")

    def taps_for(self, grid: OfdmGrid) -> int:
        if self.n_taps is not None:
            return int(self.n_taps)
        if grid.cp_samples is None:
            raise ValueError("give n_taps or a grid with a cyclic prefix")
        return int(grid.cp_samples)

    def matrix(self, grid, pilots, point=None):
        pilots.validate(grid)
        L = self.taps_for(grid)
        if pilots.count < L and self.ridge == 0.0:
            raise ValueError(f"{pilots.count} pilots cannot resolve {L} taps without regularisation")
        taps = np.arange(L)
        Fp = np.exp(-2j * np.pi * np.outer(pilots.indices, taps) / grid.fft_size)
        Fa = np.exp(-2j * np.pi * np.outer(grid.active_indices, taps) / grid.fft_size)
        gram = Fp.conj().T @ Fp + self.ridge * np.eye(L)
        return Fa @ np.linalg.solve(gram, Fp.conj().T)

    def real_multiplications(self, grid, pilots):
        return 4 * grid.n_active * pilots.count


class DftInterpolation(Estimator):
    """Classical transform-domain estimator for a uniform pilot comb.

    The pilot estimates are transformed with a ``K``-point inverse DFT, taps beyond a window
    of ``window_s`` seconds are discarded (default: the cyclic prefix), and the kept taps are
    transformed back onto every active tone. The Fourier period is the span of the comb,
    ``K`` pilots times their spacing, so the tap grid has period ``1 / (K D spacing_hz)``.
    With this choice the transform pair is orthogonal and the estimator is the
    least-squares fit of the windowed taps. The map must be a uniform comb.
    """

    def __init__(self, window_s: float | None = None, name: str | None = None) -> None:
        if window_s is not None and not window_s > 0:
            raise ValueError("window_s must be positive")
        self.window_s = window_s
        self.name = name or ("DFT interpolation" if window_s is None else f"DFT interpolation ({window_s*1e9:.0f} ns window)")

    def window_taps(self, grid: OfdmGrid, pilots: PilotMap) -> int:
        span = grid.cp_duration_s if self.window_s is None else self.window_s
        if span is None:
            raise ValueError("give window_s or a grid with a cyclic prefix")
        spacing = int(pilots.indices[1] - pilots.indices[0])
        tap_period = 1.0 / (pilots.count * spacing * grid.spacing_hz)
        return int(min(pilots.count, np.floor(span / tap_period + 1e-9) + 1))

    def matrix(self, grid, pilots, point=None):
        pilots.validate(grid)
        steps = np.diff(pilots.indices)
        if np.any(steps != steps[0]):
            raise ValueError("DFT interpolation needs a uniform pilot comb")
        K = pilots.count
        D = int(steps[0])
        L = self.window_taps(grid, pilots)
        taps = np.arange(L)
        Fk = np.exp(-2j * np.pi * np.outer(np.arange(K), taps) / K)
        tones = (grid.active_indices - pilots.indices[0]) / D
        Fa = np.exp(-2j * np.pi * np.outer(tones, taps) / K)
        return Fa @ (Fk.conj().T / K)

    def real_multiplications(self, grid, pilots):
        K = pilots.count
        D = int(pilots.indices[1] - pilots.indices[0])
        n_fft = K * D
        return int(2 * K * np.log2(K) + 2 * n_fft * np.log2(n_fft))


def robust_wiener(grid: OfdmGrid, max_delay_s: float | None = None, snr_db: float | None = None, name: str | None = None) -> WienerFilter:
    """Wiener filter designed for a uniform delay profile spanning ``max_delay_s``.

    This is the classical robust design that assumes no knowledge of the profile shape; the
    default span is the cyclic prefix of the grid. ``snr_db`` fixes the design noise level;
    when omitted the actual noise level is used.
    """
    span = grid.cp_duration_s if max_delay_s is None else max_delay_s
    if span is None:
        raise ValueError("give max_delay_s or a grid with a cyclic prefix")
    profile = DelayProfile.uniform(span, grid.sample_period_s)
    var = None if snr_db is None else 10.0 ** (-snr_db / 10.0)
    return WienerFilter(profile, var, name=name or f"Wiener (uniform to {span*1e9:.0f} ns)")
