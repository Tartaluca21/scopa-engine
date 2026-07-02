"""Genetic Algorithm for evolving heuristic weights (Phase 3).

A population of weight genomes evolved by elitism + uniform crossover + gaussian
mutation. Fitness comes from self-play matches (engine.heuristic.simulate_match).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from engine.heuristic import HeuristicBot, Player, Weights, simulate_match

AgentFactory = Callable[[Weights, np.random.Generator], Player]


def _heuristic_factory(weights: Weights, rng: np.random.Generator) -> Player:
    """Default factory: a one-ply HeuristicBot (ignores `rng`)."""
    return HeuristicBot(weights)


@dataclass(slots=True)
class GeneticConfig:
    """Hyper-parameters for the evolutionary loop."""

    population_size: int = 1000
    elite_frac: float = 0.1
    mutation_sigma: float = 0.1


class BotPopulation:
    """A population of weight genomes for tournament-based evolution."""

    genomes: list[Weights]
    config: GeneticConfig

    def __init__(self, config: GeneticConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.genomes = [Weights.random(rng) for _ in range(config.population_size)]

    def __len__(self) -> int:
        return len(self.genomes)

    def evolve(self, fitness: list[float], rng: np.random.Generator) -> None:
        """Produce the next generation: elitism + crossover + gaussian mutation."""
        if len(fitness) != len(self.genomes):
            raise ValueError("fitness length must match population size")
        order = np.argsort(fitness)[::-1]
        n_elite = max(1, int(len(self.genomes) * self.config.elite_frac))
        elite = [self.genomes[int(i)] for i in order[:n_elite]]
        children: list[Weights] = list(elite)
        while len(children) < len(self.genomes):
            a, b = (elite[int(i)] for i in rng.integers(0, n_elite, size=2))
            children.append(self._reproduce(a, b, rng))
        self.genomes = children

    def _reproduce(self, a: Weights, b: Weights, rng: np.random.Generator) -> Weights:
        """Uniform crossover followed by gaussian mutation."""
        mask = rng.random(len(a.to_vector())) < 0.5
        child = np.where(mask, a.to_vector(), b.to_vector())
        child = child + rng.normal(0.0, self.config.mutation_sigma, size=child.shape)
        # clamp to non-negative, like Weights.random: negative weights invert
        # the heuristic (e.g. a negative settebello weight) and confuse the agent.
        return Weights.from_vector(np.clip(child, 0.0, None))


def round_robin_fitness(
    population: BotPopulation,
    rng: np.random.Generator,
    make_agent: AgentFactory = _heuristic_factory,
) -> list[float]:
    """Score each genome by a self-play match against the next genome (ring).

    `make_agent` turns a genome into a player, so the same loop trains either
    one-ply heuristic bots (default) or PIMC `SearchAgent`s.
    """
    genomes = population.genomes
    n = len(genomes)
    fitness = [0.0] * n
    for i in range(n):
        a = make_agent(genomes[i], rng)
        b = make_agent(genomes[(i + 1) % n], rng)
        sa, sb = simulate_match(a, b, rng)
        fitness[i] += sa - sb
        fitness[(i + 1) % n] += sb - sa
    return fitness
