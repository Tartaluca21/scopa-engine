"""Perfect-Information Monte Carlo coordinator.

Enumerate the deciding player's legal root moves (drawn from known information
only), then for each of N determinized worlds run alpha-beta on every candidate
move and average the resulting margins. The move with the highest mean score
across worlds is returned. A single TranspositionTable is shared across worlds:
keys are full-state Zobrist hashes, so identical states across worlds collapse
to the same sound entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cognitive.belief import BeliefSystem
from engine.core import ScopaEngine
from engine.transposition import TranspositionTable
from search.alphabeta import Move, SearchConfig, alphabeta, legal_moves
from search.determinize import determinize


@dataclass(slots=True)
class PimcConfig:
    """Coordinator parameters: number of sampled worlds and the search config."""

    n_worlds: int = 30
    search: SearchConfig = field(default_factory=SearchConfig)


def pimc_decide(
    engine: ScopaEngine,
    player: int,
    cfg: PimcConfig,
    rng: np.random.Generator,
    belief: BeliefSystem | None = None,
) -> Move:
    """Return the (card, capture_set) with the best mean score over N worlds.

    When `belief` is supplied its posterior (carrying opponent-play inference)
    biases the determinization; otherwise a fresh uniform belief is built from
    the current state, so worlds are still sampled through the belief layer.
    """
    if engine.current_player != player:
        raise ValueError(f"not player {player}'s turn to move")
    moves = legal_moves(engine, player)
    if not moves:
        raise ValueError("no legal move available")

    if belief is None:
        belief = BeliefSystem(bot_player=player)
        belief.update_on_deal(engine)
    weights = belief.get_opponent_hand_probabilities(as_array=True)

    tt = TranspositionTable()
    totals = [0.0] * len(moves)
    for _ in range(cfg.n_worlds):
        world = determinize(engine, player, rng, weights)
        for i, (card, cap) in enumerate(moves):
            child = world.clone()
            child.apply_legal_move(card, cap)
            # child is the opponent's turn: negate to score from `player`.
            value = -alphabeta(child, cfg.search.max_depth, -math.inf, math.inf, tt, cfg.search)
            totals[i] += value

    best = max(range(len(moves)), key=lambda i: totals[i])
    return moves[best]
