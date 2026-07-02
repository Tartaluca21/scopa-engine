"""Evolve PIMC SearchAgent weights with a small, parallel genetic loop.

Uses PIMC-light tournament settings so the loop stays practical, then prints the
best evolved genome. Run with:

    python train.py
"""

from __future__ import annotations

import argparse

import numpy as np

from engine.genetic import BotPopulation, GeneticConfig
from engine.heuristic import Weights
from search.tournament import TrainingConfig, parallel_search_fitness

POPULATION = 24
GENERATIONS = 8
BASE_SEED = 20260630


def train(
    generations: int = GENERATIONS,
    ga_cfg: GeneticConfig | None = None,
    train_cfg: TrainingConfig | None = None,
    verbose: bool = False,
) -> Weights:
    """Run the evolution loop and return the fittest genome found.

    Text recaps are off by default; pass `verbose=True` for per-generation logs.
    """
    ga_cfg = ga_cfg or GeneticConfig(
        population_size=POPULATION, elite_frac=0.25, mutation_sigma=0.1
    )
    train_cfg = train_cfg or TrainingConfig()
    rng = np.random.default_rng(BASE_SEED)
    pop = BotPopulation(ga_cfg, rng)
    best, best_fit = pop.genomes[0], -np.inf
    for gen in range(generations):
        fitness = parallel_search_fitness(pop.genomes, BASE_SEED + gen, train_cfg)
        top = int(np.argmax(fitness))
        if fitness[top] > best_fit:
            best, best_fit = pop.genomes[top], fitness[top]
        if verbose:
            print(f"gen {gen:2d}  best_fit={fitness[top]:+.2f}  {pop.genomes[top]}")
        pop.evolve(fitness, rng)
    if verbose:
        print(f"\nbest genome (fit={best_fit:+.2f}): {best}")
    return best


def main() -> None:
    """Console entry point: run evolution and print the best genome."""
    parser = argparse.ArgumentParser(
        description="Evolve PIMC weights with a genetic loop and print the best genome."
    )
    parser.add_argument(
        "-g", "--generations", type=int, default=GENERATIONS, help="generations to run"
    )
    args = parser.parse_args()
    train(generations=args.generations, verbose=True)


if __name__ == "__main__":
    main()
