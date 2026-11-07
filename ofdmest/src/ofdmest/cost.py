"""Computational cost of the estimators: operation counts and measured run times.

Two measures are reported. ``real_multiplications`` is the number of real multiplications
per OFDM symbol of a straightforward implementation, a hardware-independent proxy. The
measured times are wall-clock figures for NumPy implementations on the machine that runs
the benchmark; they depend on the BLAS library and the processor and are indicative only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .channel import OfdmGrid
from .estimators import (
    CubicSplineInterpolation,
    DftInterpolation,
    Estimator,
    LinearInterpolation,
    NearestPilot,
    OperatingPoint,
)
from .pilots import PilotMap


@dataclass(frozen=True)
class CostResult:
    estimator: str
    real_multiplications: int
    design_ms: float
    dense_apply_us: float
    structured_apply_us: float | None


def _structured_apply(estimator: Estimator, grid: OfdmGrid, pilots: PilotMap):
    """Return a function applying the estimator without a dense matrix, or None."""
    tones = grid.active_indices.astype(float)
    p = pilots.indices.astype(float)
    if isinstance(estimator, NearestPilot):
        A = estimator.matrix(grid, pilots)
        col = np.argmax(A, axis=1)
        return lambda y: y[:, col]
    if isinstance(estimator, LinearInterpolation) and estimator.extrapolation == "constant":
        return lambda y: _interp_batch(tones, p, y)
    if isinstance(estimator, CubicSplineInterpolation) and estimator.extrapolation == "native":
        return lambda y: CubicSpline(p, y.T, axis=0, bc_type="not-a-knot")(tones).T
    if isinstance(estimator, DftInterpolation):
        K = pilots.count
        D = int(pilots.indices[1] - pilots.indices[0])
        L = estimator.window_taps(grid, pilots)
        n_fft = K * D
        start = int(pilots.indices[0])
        offsets = grid.active_indices - start

        def apply(y):
            taps = np.fft.ifft(y, axis=1)
            taps[:, L:] = 0.0
            full = np.fft.fft(taps, n=n_fft, axis=1)
            return full[:, offsets % n_fft]

        return apply
    return None


def _interp_batch(tones, p, y):
    out = np.empty((y.shape[0], tones.size), dtype=complex)
    for i in range(y.shape[0]):
        out[i] = np.interp(tones, p, y[i].real) + 1j * np.interp(tones, p, y[i].imag)
    return out


def benchmark(estimators: list[Estimator], grid: OfdmGrid, pilots: PilotMap, point: OperatingPoint, n_symbols: int = 2000, repeats: int = 5, seed: int = 0) -> list[CostResult]:
    """Measure design time and per-symbol application time of each estimator."""
    rng = np.random.default_rng(seed)
    y = (rng.standard_normal((n_symbols, pilots.count)) + 1j * rng.standard_normal((n_symbols, pilots.count))) / np.sqrt(2)
    results = []
    for est in estimators:
        t0 = time.perf_counter()
        A = est.matrix(grid, pilots, point)
        design_ms = 1e3 * (time.perf_counter() - t0)
        At = np.ascontiguousarray(A.T)
        dense = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = y @ At
            dense.append((time.perf_counter() - t0) / n_symbols)
        structured = _structured_apply(est, grid, pilots)
        structured_us = None
        if structured is not None:
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = structured(y)
                times.append((time.perf_counter() - t0) / n_symbols)
            structured_us = 1e6 * float(np.min(times))
        results.append(CostResult(est.name, est.real_multiplications(grid, pilots), design_ms, 1e6 * float(np.min(dense)), structured_us))
    return results
