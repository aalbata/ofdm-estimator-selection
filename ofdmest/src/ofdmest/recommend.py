"""Choice of the cheapest estimator that meets an error target."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import exact_error
from .channel import DelayProfile, OfdmGrid
from .estimators import Estimator, WienerFilter
from .pilots import PilotMap


@dataclass(frozen=True)
class Recommendation:
    """Outcome of :func:`recommend`: a table of candidates and the chosen one."""

    table: pd.DataFrame
    chosen: str | None
    reference_db: float
    tolerance_db: float

    def __str__(self) -> str:
        head = f"reference (matched Wiener) {self.reference_db:.2f} dB, tolerance {self.tolerance_db:.2f} dB"
        body = self.table.to_string(index=False, float_format=lambda v: f"{v:.2f}")
        tail = f"chosen: {self.chosen}" if self.chosen else "chosen: none meets the target"
        return "\n".join([head, body, tail])


def recommend(
    candidates: list[Estimator],
    grid: OfdmGrid,
    pilots: PilotMap,
    profile: DelayProfile,
    snr_db: float,
    tolerance_db: float = 1.0,
    target_db: float | None = None,
    extra_noise_db: np.ndarray | None = None,
) -> Recommendation:
    """Pick the candidate with the fewest real multiplications whose error meets the target.

    The target is the matched Wiener error plus ``tolerance_db`` unless an absolute
    ``target_db`` is given. Candidates that do not apply to the pilot map (for example a
    transform-domain estimator on a non-uniform map) are listed with an empty error.
    """
    reference = exact_error(WienerFilter(), grid, pilots, profile, snr_db, extra_noise_db).total_db
    limit = reference + tolerance_db if target_db is None else target_db
    rows = []
    for est in candidates:
        try:
            res = exact_error(est, grid, pilots, profile, snr_db, extra_noise_db)
            rows.append({"estimator": est.name, "error_db": res.total_db, "bias_db": 10 * np.log10(max(res.bias_total, 1e-30)), "noise_db": 10 * np.log10(max(res.noise_total, 1e-30)), "real_multiplications": res.real_multiplications, "meets_target": res.total_db <= limit})
        except ValueError as err:
            rows.append({"estimator": est.name, "error_db": np.nan, "bias_db": np.nan, "noise_db": np.nan, "real_multiplications": np.nan, "meets_target": False, "note": str(err)})
    table = pd.DataFrame(rows)
    ok = table[table.meets_target]
    chosen = None
    if not ok.empty:
        chosen = str(ok.sort_values(["real_multiplications", "error_db"]).iloc[0].estimator)
    return Recommendation(table, chosen, reference, limit - reference if target_db is None else tolerance_db)
