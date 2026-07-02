"""Heuristic bot, exposure-aware move selection, and self-play (Phase 3).

A greedy one-ply selector that trades capture value against table exposure. The
evaluation primitives it stands on -- the `Weights` genome, `capture_features`,
the exposure estimate, and deal scoring -- live in `engine.features`; this module
composes them into a playing agent and a self-play driver. The two exposure
components fix the naive "give-away" behaviour of a flat subset count:

  * lost-card risk: a value-weighted estimate of the best single capture the
    opponent could make next turn (so leaving the settebello or extra denari
    on the table is penalized by *what* is exposed, not just how many subsets);
  * anti-Scopa risk: a full-point penalty, scaled by the same w_scope used to
    reward our own Scope, whenever our move hands the opponent a table they can
    clear in a single play.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from engine.cards import HAND_ZONES, Zone
from engine.core import ScopaEngine
from engine.features import (
    PRIMIERA_POINTS,
    RISK_COEF,
    SCOPA_RISK_COEF,
    CaptureFeatures,
    Weights,
    _scopa_threat,
    _weighted,
    _worst_exposure,
    capture_features,
    evaluate,
    score_deal,
)

# Re-exported so existing call sites keep importing the evaluation layer from
# `engine.heuristic`; the definitions now live in `engine.features`.
__all__ = [
    "PRIMIERA_POINTS",
    "CaptureFeatures",
    "HeuristicBot",
    "Player",
    "Weights",
    "_scopa_threat",
    "_weighted",
    "capture_features",
    "evaluate",
    "score_deal",
    "simulate_match",
]


@dataclass(slots=True)
class HeuristicBot:
    """Greedy one-ply bot trading capture value against table exposure."""

    weights: Weights

    def select(self, engine: ScopaEngine, player: int) -> tuple[int, list[int]]:
        """Return (card, capture_indices) maximizing value minus exposure risk."""
        hand = engine.cards_in(HAND_ZONES[player])
        if hand.size == 0:
            raise ValueError("empty hand: no move available")
        table = [int(c) for c in engine.cards_in(Zone.TAVOLO)]
        best_card = int(hand[0])
        best_cap: list[int] = []
        best_score = -np.inf
        for c in hand:
            card = int(c)
            for cap, score in self._candidate_scores(engine, card, table):
                if score > best_score:
                    best_score, best_card, best_cap = score, card, cap
        return best_card, best_cap

    def _exposure(self, rest: list[int]) -> float:
        """Penalty for the table handed to the opponent: lost cards + Scopa risk."""
        penalty = RISK_COEF * _worst_exposure(rest, self.weights)
        if _scopa_threat(rest):
            penalty += SCOPA_RISK_COEF * self.weights.scope
        return penalty

    def _candidate_scores(
        self, engine: ScopaEngine, card: int, table: list[int]
    ) -> list[tuple[list[int], float]]:
        options = engine.captures_for(card)
        out: list[tuple[list[int], float]] = []
        if options:
            for opt in options:
                taken = [int(x) for x in opt]
                rest = [t for t in table if t not in taken]
                val = _weighted(capture_features(opt), self.weights)
                if not rest:
                    val += self.weights.scope  # we clear the table: a scopa
                out.append((taken, val - self._exposure(rest)))
        else:
            rest = [*table, card]
            out.append(([], -self._exposure(rest)))
        return out

    def choose_move(self, engine: ScopaEngine, player: int) -> int:
        return self.select(engine, player)[0]

    def move_scores(self, engine: ScopaEngine, player: int) -> list[tuple[int, list[int], float]]:
        """Exposure-aware value of every legal move: (card, capture_set, score).

        The same quantity `select` maximizes, exposed per move so a search can
        use it as a prior over `player`'s options without re-deriving it.
        """
        table = [int(c) for c in engine.cards_in(Zone.TAVOLO)]
        out: list[tuple[int, list[int], float]] = []
        for c in engine.cards_in(HAND_ZONES[player]):
            card = int(c)
            for cap, score in self._candidate_scores(engine, card, table):
                out.append((card, cap, score))
        return out


class Player(Protocol):
    """Anything that can pick a move: HeuristicBot, SearchAgent, RandomBot."""

    def select(self, engine: ScopaEngine, player: int) -> tuple[int, list[int]]:
        """Return (card, capture_indices) for `player` in the given state."""
        ...


def simulate_match(bot_a: Player, bot_b: Player, rng: np.random.Generator) -> tuple[float, float]:
    """Self-play one full deal between two bots; return (score_a, score_b)."""
    engine = ScopaEngine()
    bots: tuple[Player, Player] = (bot_a, bot_b)
    engine.deal_round(rng)
    while not engine.is_game_over():
        if engine.count(Zone.MANO_P1) == 0 and engine.count(Zone.MANO_P2) == 0:
            engine.deal_round(rng)
            continue
        player = engine.current_player
        card, capture = bots[player].select(engine, player)
        engine.execute_move(card, capture)
    engine.end_of_deal_sweep()
    return score_deal(engine)
