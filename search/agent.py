"""Playable agents for tournaments and self-play (Phase 5).

`SearchAgent` wraps the PIMC coordinator behind the `Player` protocol so it can
drop straight into `simulate_match` and the genetic loop; its `Weights` drive
the alpha-beta leaf evaluation. `RandomBot` is a uniform-random baseline used to
validate that search actually buys playing strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.core import ScopaEngine
from engine.heuristic import Weights
from search.alphabeta import Move, SearchConfig, legal_moves
from search.pimc import PimcConfig, pimc_decide


@dataclass(slots=True)
class SearchAgent:
    """PIMC-backed player; `weights` tune the alpha-beta leaf evaluation."""

    weights: Weights
    rng: np.random.Generator
    n_worlds: int = 6
    max_depth: int = 4

    def _config(self) -> PimcConfig:
        return PimcConfig(
            n_worlds=self.n_worlds,
            search=SearchConfig(max_depth=self.max_depth, weights=self.weights),
        )

    def select(self, engine: ScopaEngine, player: int) -> Move:
        """Return the PIMC-chosen (card, capture_indices) for `player`."""
        return pimc_decide(engine, player, self._config(), self.rng)


@dataclass(slots=True)
class RandomBot:
    """Uniform-random baseline over the current legal moves."""

    rng: np.random.Generator

    def select(self, engine: ScopaEngine, player: int) -> Move:
        """Return a uniformly random legal (card, capture_indices)."""
        moves = legal_moves(engine, player)
        if not moves:
            raise ValueError("no legal move available")
        return moves[int(self.rng.integers(len(moves)))]
