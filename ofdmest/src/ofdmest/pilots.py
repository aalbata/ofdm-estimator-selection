"""Pilot maps: which tones of the analysed band carry known pilot symbols."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import OfdmGrid


@dataclass(frozen=True)
class PilotMap:
    """Sorted tone indices of the pilots within an OFDM band."""

    indices: np.ndarray
    name: str = "custom"

    def __post_init__(self) -> None:
        idx = np.unique(np.asarray(self.indices, dtype=int).reshape(-1))
        if idx.size < 2:
            raise ValueError("a pilot map needs at least two pilots")
        if idx.min() < 0:
            raise ValueError("pilot indices must be non-negative")
        idx.setflags(write=False)
        object.__setattr__(self, "indices", idx)

    @property
    def count(self) -> int:
        return int(self.indices.size)

    def validate(self, grid: OfdmGrid) -> None:
        """Raise ValueError unless every pilot lies on an active tone of ``grid``."""
        if self.indices.max() >= grid.n_subcarriers:
            raise ValueError("pilot index beyond the band")
        if not grid.active[self.indices].all():
            raise ValueError("pilots must lie on active tones")

    def positions_in(self, grid: OfdmGrid) -> np.ndarray:
        """Row positions of the pilots within the ordered list of active tones."""
        self.validate(grid)
        return np.searchsorted(grid.active_indices, self.indices)

    def overhead(self, grid: OfdmGrid) -> float:
        """Fraction of active tones used as pilots."""
        return self.count / grid.n_active

    def mean_spacing(self) -> float:
        return float(np.mean(np.diff(self.indices)))


def comb(n_subcarriers: int, spacing: int, offset: int = 0, name: str | None = None) -> PilotMap:
    """Pilots every ``spacing`` tones starting at ``offset``."""
    if spacing < 1 or offset < 0 or offset >= n_subcarriers:
        raise ValueError("invalid comb parameters")
    idx = np.arange(offset, n_subcarriers, spacing)
    return PilotMap(idx, name or f"comb {spacing} offset {offset}")


def uniform_anchored(n_subcarriers: int, n_pilots: int, name: str = "uniform anchored") -> PilotMap:
    """``n_pilots`` equally spaced pilots with the first on tone 0 and the last on the top tone."""
    idx = np.unique(np.round(np.linspace(0, n_subcarriers - 1, n_pilots)).astype(int))
    return PilotMap(idx, name)


def uniform_centred(n_subcarriers: int, n_pilots: int, name: str = "uniform centred") -> PilotMap:
    """``n_pilots`` pilots at the centres of equal-width bins spanning the band."""
    idx = np.round((np.arange(n_pilots) + 0.5) * n_subcarriers / n_pilots - 0.5).astype(int)
    return PilotMap(np.unique(idx), name)


def nr_dmrs_type1(n_resource_blocks: int, port: int = 0, name: str | None = None) -> PilotMap:
    """5G NR DM-RS configuration type 1, single symbol, one of the ports 0 to 3.

    Within each resource block of 12 tones the port occupies tones 4n + 2k' + delta with
    k' in {0, 1}, n in {0, 1, 2}, and delta = 0 for ports 0 and 1, delta = 1 for ports 2 and 3
    (3GPP TS 38.211, clause 7.4.1.1.2). This is a comb with spacing 2.
    """
    if port not in (0, 1, 2, 3):
        raise ValueError("port must be 0, 1, 2 or 3 for a single-symbol type 1 map")
    delta = 0 if port < 2 else 1
    n = 12 * n_resource_blocks
    return comb(n, 2, delta, name or f"NR DM-RS type 1 (port {port})")


def nr_dmrs_type2(n_resource_blocks: int, port: int = 0, name: str | None = None) -> PilotMap:
    """5G NR DM-RS configuration type 2, single symbol, one of the ports 0 to 5.

    Within each resource block the port occupies tones 6n + k' + delta with k' in {0, 1},
    n in {0, 1}, and delta = 0, 2 or 4 for port pairs (0, 1), (2, 3), (4, 5)
    (3GPP TS 38.211, clause 7.4.1.1.2). Pilots come in adjacent pairs with a gap of four.
    """
    if port not in range(6):
        raise ValueError("port must be between 0 and 5 for a single-symbol type 2 map")
    delta = 2 * (port // 2)
    tones = []
    for rb in range(n_resource_blocks):
        for n in (0, 1):
            for kp in (0, 1):
                tones.append(12 * rb + 6 * n + kp + delta)
    return PilotMap(np.array(tones), name or f"NR DM-RS type 2 (port {port})")


def lte_crs(n_resource_blocks: int, v_shift: int = 0, v: int = 0, name: str | None = None) -> PilotMap:
    """LTE cell-specific reference signal of one antenna port in one OFDM symbol.

    Tones k = 6m + (v + v_shift) mod 6, with v = 0 or 3 depending on the symbol and port and
    v_shift the physical cell identity modulo 6 (3GPP TS 36.211, clause 6.10.1.2).
    """
    n = 12 * n_resource_blocks
    return comb(n, 6, (v + v_shift) % 6, name or f"LTE CRS (shift {(v + v_shift) % 6})")


DVBT2_SCATTERED_PATTERNS = {
    "PP1": (3, 4),
    "PP2": (6, 2),
    "PP3": (6, 4),
    "PP4": (12, 2),
    "PP5": (12, 4),
    "PP6": (24, 2),
    "PP7": (24, 4),
    "PP8": (6, 16),
}
"""Separation of pilot-bearing carriers D_x and number of symbols forming one scattered
pilot sequence D_y for each DVB-T2 scattered pilot pattern (ETSI EN 302 755, clause 9.2.3.1).
Within one OFDM symbol the scattered pilots are spaced D_x D_y carriers apart."""


def dvbt2_scattered(n_carriers: int, pattern: str, symbol_index: int = 0, name: str | None = None) -> PilotMap:
    """DVB-T2 scattered pilots of one OFDM symbol: carriers k with k mod (D_x D_y) = D_x (l mod D_y)."""
    if pattern not in DVBT2_SCATTERED_PATTERNS:
        raise ValueError(f"unknown DVB-T2 pattern {pattern}")
    dx, dy = DVBT2_SCATTERED_PATTERNS[pattern]
    offset = dx * (symbol_index % dy)
    return comb(n_carriers, dx * dy, offset, name or f"DVB-T2 {pattern} (symbol {symbol_index % dy})")
