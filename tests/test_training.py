"""Phase 5 tests: parallel tournament determinism and search-agent evolution."""

from __future__ import annotations

import numpy as np

from engine.genetic import BotPopulation, GeneticConfig, round_robin_fitness
from engine.heuristic import Weights
from search.agent import SearchAgent
from search.tournament import TrainingConfig, parallel_search_fitness

_TRAIN = TrainingConfig(n_worlds=4, max_depth=3, workers=2)


def _genomes(rng: np.random.Generator, n: int) -> list[Weights]:
    return [Weights.random(rng) for _ in range(n)]


def test_parallel_fitness_independent_of_worker_count() -> None:
    genomes = _genomes(np.random.default_rng(7), 4)
    serial = parallel_search_fitness(genomes, 100, TrainingConfig(4, 3, workers=1))
    parallel = parallel_search_fitness(genomes, 100, TrainingConfig(4, 3, workers=2))
    assert np.allclose(serial, parallel)


def test_parallel_fitness_is_zero_sum() -> None:
    genomes = _genomes(np.random.default_rng(8), 4)
    fitness = parallel_search_fitness(genomes, 50, _TRAIN)
    assert len(fitness) == 4
    assert abs(sum(fitness)) < 1e-9


def test_round_robin_accepts_search_agent_factory() -> None:
    rng = np.random.default_rng(9)
    pop = BotPopulation(GeneticConfig(population_size=3), rng)

    def make_agent(w: Weights, r: np.random.Generator) -> SearchAgent:
        return SearchAgent(w, r, n_worlds=3, max_depth=3)

    fitness = round_robin_fitness(pop, rng, make_agent)
    assert len(fitness) == 3
    assert all(np.isfinite(f) for f in fitness)
