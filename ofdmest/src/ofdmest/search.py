"""Pilot-map optimisation for links whose pilot positions are free.

The objective is the exact error of a chosen estimator averaged over a set of operating
points (delay profiles and SNRs), optionally with per-tone interference. Three searches
are provided: exhaustive enumeration for small maps, closed-form rules (anchored and
centred uniform maps), and a genetic algorithm with a discrete representation.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .analysis import exact_error
from .channel import DelayProfile, OfdmGrid
from .estimators import Estimator
from .pilots import PilotMap, uniform_anchored, uniform_centred


@dataclass(frozen=True)
class SearchResult:
    pilots: PilotMap
    objective: float
    evaluations: int
    history: np.ndarray


class MapObjective:
    """Mean exact error of ``estimator`` over the operating points, with caching."""

    def __init__(self, estimator: Estimator, grid: OfdmGrid, points: list[tuple[DelayProfile, float]], extra_noise_db: np.ndarray | None = None) -> None:
        self.estimator = estimator
        self.grid = grid
        self.points = points
        self.extra_noise_db = extra_noise_db
        self.cache: dict[tuple[int, ...], float] = {}
        self.evaluations = 0

    def __call__(self, indices) -> float:
        key = tuple(int(i) for i in np.sort(np.asarray(indices)))
        if key not in self.cache:
            pilots = PilotMap(np.array(key))
            vals = [exact_error(self.estimator, self.grid, pilots, prof, snr, self.extra_noise_db).total for prof, snr in self.points]
            self.cache[key] = float(np.mean(vals))
            self.evaluations += 1
        return self.cache[key]


def rule_maps(grid: OfdmGrid, n_pilots: int) -> dict[str, PilotMap]:
    """Anchored and centred uniform maps restricted to active tones."""
    maps = {"anchored": uniform_anchored(grid.n_subcarriers, n_pilots), "centred": uniform_centred(grid.n_subcarriers, n_pilots)}
    out = {}
    for name, pm in maps.items():
        idx = _move_to_active(pm.indices, grid)
        out[name] = PilotMap(idx, name)
    return out


def _move_to_active(indices: np.ndarray, grid: OfdmGrid) -> np.ndarray:
    """Move each pilot on an inactive tone to the nearest free active tone."""
    active = grid.active.copy()
    out = []
    for x in indices:
        x = int(x)
        if active[x] and x not in out:
            out.append(x)
            continue
        for d in range(1, grid.n_subcarriers):
            for c in (x - d, x + d):
                if 0 <= c < grid.n_subcarriers and active[c] and c not in out:
                    out.append(c)
                    break
            else:
                continue
            break
    return np.array(sorted(out))


def exhaustive(objective: MapObjective, n_pilots: int, limit: int = 2_000_000) -> SearchResult:
    """Enumerate every map of ``n_pilots`` active tones; refuses more than ``limit`` maps."""
    tones = objective.grid.active_indices
    from math import comb as n_choose_k

    total = n_choose_k(tones.size, n_pilots)
    if total > limit:
        raise ValueError(f"{total} maps exceed the enumeration limit of {limit}")
    best, best_val = None, np.inf
    for cmb in itertools.combinations(tones.tolist(), n_pilots):
        v = objective(cmb)
        if v < best_val:
            best, best_val = cmb, v
    return SearchResult(PilotMap(np.array(best), "exhaustive optimum"), best_val, objective.evaluations, np.array([best_val]))


def genetic(objective: MapObjective, n_pilots: int, population: int = 40, generations: int = 60, mutation: float = 0.35, seed: int = 0, seeds_in: list[PilotMap] | None = None) -> SearchResult:
    """Genetic algorithm over subsets of active tones with elitism, crossover and mutation."""
    rng = np.random.default_rng(seed)
    tones = objective.grid.active_indices
    n = tones.size

    def random_map():
        return np.sort(rng.choice(tones, size=n_pilots, replace=False))

    pop = [random_map() for _ in range(population)]
    for i, pm in enumerate(seeds_in or []):
        if pm.count == n_pilots:
            pop[population - 1 - (i % population)] = np.sort(pm.indices)
    scores = [objective(p) for p in pop]
    history = [min(scores)]
    for _ in range(generations):
        order = np.argsort(scores)
        new = [pop[int(order[0])].copy(), pop[int(order[1])].copy()]
        parents = order[: max(4, population // 2)]
        while len(new) < population:
            a, b = rng.choice(parents, 2, replace=False)
            child = np.unique(np.r_[pop[int(a)][: n_pilots // 2], pop[int(b)][n_pilots // 2:]])
            while child.size < n_pilots:
                cand = int(rng.choice(tones))
                if cand not in child:
                    child = np.append(child, cand)
            child = np.sort(child[:n_pilots])
            if rng.random() < mutation:
                pos = int(rng.integers(0, n_pilots))
                cand = int(rng.choice(tones))
                while cand in child:
                    cand = int(rng.choice(tones))
                child[pos] = cand
                child = np.sort(child)
            new.append(child)
        pop = new
        scores = [objective(p) for p in pop]
        history.append(min(scores))
    best = int(np.argmin(scores))
    return SearchResult(PilotMap(pop[best], "genetic algorithm"), float(scores[best]), objective.evaluations, np.minimum.accumulate(np.array(history)))
