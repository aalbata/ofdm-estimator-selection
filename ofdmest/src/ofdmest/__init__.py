"""Exact error analysis of linear pilot-aided OFDM channel estimators."""
from .analysis import ErrorResult, error_components, error_from_matrix, exact_error, ideal_error_db
from .channel import DelayProfile, OfdmGrid, indoor_factory_delay_spread_s
from .estimators import (
    CubicSplineInterpolation,
    DftInterpolation,
    DftLeastSquares,
    Estimator,
    LinearInterpolation,
    NearestPilot,
    OperatingPoint,
    WienerFilter,
    noise_variance_from_snr,
    robust_wiener,
)
from .pilots import (
    DVBT2_SCATTERED_PATTERNS,
    PilotMap,
    comb,
    dvbt2_scattered,
    lte_crs,
    nr_dmrs_type1,
    nr_dmrs_type2,
    uniform_anchored,
    uniform_centred,
)
from .simulate import SimulationResult, simulate

__version__ = "0.1.0"

__all__ = [
    "CubicSplineInterpolation",
    "DVBT2_SCATTERED_PATTERNS",
    "DelayProfile",
    "DftInterpolation",
    "DftLeastSquares",
    "ErrorResult",
    "Estimator",
    "LinearInterpolation",
    "NearestPilot",
    "OfdmGrid",
    "OperatingPoint",
    "PilotMap",
    "SimulationResult",
    "WienerFilter",
    "comb",
    "dvbt2_scattered",
    "error_components",
    "error_from_matrix",
    "exact_error",
    "ideal_error_db",
    "indoor_factory_delay_spread_s",
    "lte_crs",
    "noise_variance_from_snr",
    "nr_dmrs_type1",
    "nr_dmrs_type2",
    "robust_wiener",
    "simulate",
    "uniform_anchored",
    "uniform_centred",
]
