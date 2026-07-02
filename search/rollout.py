"""Heuristic-guided rollouts for ISMCTS (Phase 6+).

Cures the tactical myopia of a uniform-random playout: the rollout *policy* is
the exposure-aware `HeuristicBot`, which trades capture value (denari,
settebello, primiera, cards) against the high-value cards its move would leave
exposed on the table and against handing the opponent a Scopa. A small epsilon
falls back to the cheap greedy move for exploration. Both sides therefore play
sensibly, which sharply lowers the variance of the deal outcome and lets the
Cards / Denari / Primiera / Settebello majorities reach UCB selection instead of
being averaged into noise.

The policy is deliberately costlier per ply than a random playout (~half the
iterations), but the far lower variance wins comfortably at an equal wall-clock
budget. The rollout runs to the end of the deal (Scopa deals terminate in at
most a few dozen plies) and is scored by the real `score_deal` margin, so every
category keeps its true one-point weight -- the sign that a per-category rescale
would have quietly destroyed.
"""

from __future__ import annotations

import numpy as np

from engine.cards import HAND_ZONES, Zone, card_value
from engine.core import ScopaEngine
from engine.heuristic import (
    PRIMIERA_POINTS,
    HeuristicBot,
    Weights,
    _scopa_threat,
    _weighted,
    capture_features,
    score_deal,
)
from search.alphabeta import Move

# Exploration rate of the otherwise-greedy rollout policy.
EPSILON = 0.1


def maybe_deal(world: ScopaEngine, rng: np.random.Generator) -> None:
    """Deal the next round if both hands are empty but the talon still has cards."""
    if (
        world.count(Zone.MANO_P1) == 0
        and world.count(Zone.MANO_P2) == 0
        and world.count(Zone.MAZZO) > 0
    ):
        world.deal_round(rng)


def _safe_laydown(table: list[int], cards: list[int]) -> int:
    """Lay the least damaging card: avoid gifting a Scopa, else dump low primiera."""
    safe = [c for c in cards if not _scopa_threat([*table, c])]
    pool = safe if safe else cards
    return min(pool, key=lambda c: PRIMIERA_POINTS[card_value(c)])


def rollout_move(
    world: ScopaEngine, player: int, weights: Weights, rng: np.random.Generator
) -> Move:
    """Greedy capture-maximizing move (epsilon-random), else a safe lay-down."""
    table = [int(c) for c in world.cards_in(Zone.TAVOLO)]
    captures: list[tuple[int, np.ndarray]] = []
    laydowns: list[int] = []
    for c in world.cards_in(HAND_ZONES[player]):
        card = int(c)
        options = world.captures_for(card)
        if options:
            captures.extend((card, opt) for opt in options)
        else:
            laydowns.append(card)

    if captures:
        if rng.random() < EPSILON:
            card, opt = captures[int(rng.integers(len(captures)))]
            return card, [int(x) for x in opt]
        best_card, best_opt = captures[0]
        best_val = -np.inf
        for card, opt in captures:
            val = _weighted(capture_features(opt), weights)
            if len(opt) == len(table):  # clears the table -> a Scopa
                val += weights.scope
            if val > best_val:
                best_val, best_card, best_opt = val, card, opt
        return best_card, [int(x) for x in best_opt]

    if rng.random() < EPSILON:
        return laydowns[int(rng.integers(len(laydowns)))], []
    return _safe_laydown(table, laydowns), []


def simulate(world: ScopaEngine, weights: Weights, rng: np.random.Generator) -> float:
    """Play `world` to the end of the deal and return its player-0 points margin.

    Each ply is chosen by the exposure-aware `HeuristicBot`, with an epsilon
    fallback to the cheap greedy move for exploration. The near-deterministic,
    exposure-aware policy makes the `score_deal` margin a low-variance estimate
    of the eventual majority split rather than the coin-flip a random playout
    would produce.
    """
    bot = HeuristicBot(weights)
    while True:
        maybe_deal(world, rng)
        if world.is_game_over():
            break
        player = world.current_player
        if rng.random() < EPSILON:
            card, cap = rollout_move(world, player, weights, rng)
        else:
            card, cap = bot.select(world, player)
        world.execute_move(card, cap)
    world.end_of_deal_sweep()
    p0, p1 = score_deal(world)
    return p0 - p1
