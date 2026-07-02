"""Parallel ring tournaments for training PIMC agents (Phase 5).

PIMC is far heavier than a one-ply lookup, so the N independent ring matches are
farmed out to a process pool. Each match is seeded deterministically from a base
seed (`seed + i`), so results are reproducible and identical regardless of the
worker count — only the wall-clock changes.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from engine.heuristic import Weights, simulate_match
from search.agent import SearchAgent

RingJob = tuple[Weights, Weights, int, int, int]


@dataclass(slots=True)
class TrainingConfig:
    """PIMC-light defaults that keep the evolution loop practical."""

    n_worlds: int = 6
    max_depth: int = 4
    workers: int = 4


def _ring_match(job: RingJob) -> tuple[float, float]:
    """Play one seeded SearchAgent-vs-SearchAgent deal (process-pool worker)."""
    weights_a, weights_b, seed, n_worlds, max_depth = job
    rng = np.random.default_rng(seed)
    a = SearchAgent(weights_a, rng, n_worlds, max_depth)
    b = SearchAgent(weights_b, rng, n_worlds, max_depth)
    return simulate_match(a, b, rng)


def parallel_search_fitness(genomes: list[Weights], seed: int, cfg: TrainingConfig) -> list[float]:
    """Ring fitness for `genomes` using PIMC agents, fanned out over workers.

    Identical results for any `cfg.workers` (each match is independently
    seeded); set `workers <= 1` to run sequentially.
    """
    n = len(genomes)
    jobs: list[RingJob] = [
        (genomes[i], genomes[(i + 1) % n], seed + i, cfg.n_worlds, cfg.max_depth) for i in range(n)
    ]
    if cfg.workers <= 1:
        results = [_ring_match(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            results = list(pool.map(_ring_match, jobs))

    fitness = [0.0] * n
    for i, (sa, sb) in enumerate(results):
        fitness[i] += sa - sb
        fitness[(i + 1) % n] += sb - sa
    return fitness
