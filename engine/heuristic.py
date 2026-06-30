"""Heuristic bot, exposure-aware move selection, and self-play (Phase 3).

Linear move-evaluation driven by a tunable weight vector, a one-ply selector
that penalizes leaving capturable combinations (<=10) on the table, and a full
self-play match used as the Genetic Algorithm's fitness function.

    V(s) = w_captures*captures + w_denari*denari + w_sette*settebello
         + w_primiera*primiera + w_scope*scope
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
import numpy.typing as npt

from engine.cards import (
    HAND_ZONES,
    PRESE_ZONES,
    Suit,
    Zone,
    card_suit,
    card_value,
)
from engine.core import ScopaEngine

PRIMIERA_POINTS: dict[int, int] = {
    7: 21, 6: 18, 1: 16, 5: 15, 4: 14, 3: 13, 2: 12, 8: 10, 9: 10, 10: 10
}
N_WEIGHTS = 5
RISK_COEF = 0.25  # penalty per opponent-capturable table combination


@dataclass(slots=True)
class Weights:
    """Tunable parameters of the evaluation function (the GA genome)."""

    captures: float = 1.0
    denari: float = 1.0
    settebello: float = 1.0
    primiera: float = 1.0
    scope: float = 1.0

    def to_vector(self) -> npt.NDArray[np.float64]:
        return np.array(astuple(self), dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: npt.NDArray[np.float64]) -> Weights:
        if vec.shape != (N_WEIGHTS,):
            raise ValueError(f"expected {N_WEIGHTS} weights, got {vec.shape}")
        return cls(*(float(x) for x in vec))

    @classmethod
    def random(cls, rng: np.random.Generator) -> Weights:
        # clamp to non-negative: negative weights would invert the heuristic.
        return cls.from_vector(np.clip(rng.uniform(-1.0, 1.0, N_WEIGHTS), 0.0, None))


@dataclass(slots=True)
class CaptureFeatures:
    captures: int
    denari: int
    settebello: int
    primiera: int


def capture_features(captured: npt.NDArray[np.intp]) -> CaptureFeatures:
    """Extract evaluation features from captured card indices."""
    denari = 0
    settebello = 0
    best_per_suit = [0] * len(Suit)
    for c in captured:
        idx = int(c)
        suit = card_suit(idx)
        value = card_value(idx)
        if suit == Suit.DENARI:
            denari += 1
            if value == 7:
                settebello = 1
        pts = PRIMIERA_POINTS[value]
        if pts > best_per_suit[int(suit)]:
            best_per_suit[int(suit)] = pts
    return CaptureFeatures(int(captured.size), denari, settebello, sum(best_per_suit))


def _weighted(f: CaptureFeatures, w: Weights) -> float:
    return (
        w.captures * f.captures
        + w.denari * f.denari
        + w.settebello * f.settebello
        + w.primiera * f.primiera
    )


def evaluate(
    engine: ScopaEngine, player: int, weights: Weights, scope_count: int = 0
) -> float:
    """Linear value of `player`'s captured pile under `weights`."""
    f = capture_features(engine.cards_in(PRESE_ZONES[player]))
    return _weighted(f, weights) + weights.scope * scope_count


def _count_exposed(table: list[int]) -> int:
    """Number of non-empty table subsets summing to <=10 (opponent-capturable)."""
    count = 0

    def rec(start: int, total: int, size: int) -> None:
        nonlocal count
        if size > 0:
            count += 1
        for i in range(start, len(table)):
            v = card_value(table[i])
            if total + v <= 10:
                rec(i + 1, total + v, size + 1)

    rec(0, 0, 0)
    return count


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
                out.append((taken, val - RISK_COEF * _count_exposed(rest)))
        else:
            rest = [*table, card]
            out.append(([], -RISK_COEF * _count_exposed(rest)))
        return out

    def choose_move(self, engine: ScopaEngine, player: int) -> int:
        return self.select(engine, player)[0]


def _award(points: list[float], a: int, b: int) -> None:
    if a > b:
        points[0] += 1.0
    elif b > a:
        points[1] += 1.0


def score_deal(engine: ScopaEngine) -> tuple[float, float]:
    """Score a finished deal: carte, denari, settebello, primiera, scope."""
    f0 = capture_features(engine.cards_in(PRESE_ZONES[0]))
    f1 = capture_features(engine.cards_in(PRESE_ZONES[1]))
    points = [0.0, 0.0]
    _award(points, f0.captures, f1.captures)
    _award(points, f0.denari, f1.denari)
    if f0.settebello:
        points[0] += 1.0
    elif f1.settebello:
        points[1] += 1.0
    _award(points, f0.primiera, f1.primiera)
    points[0] += float(engine.scopa_counts[0])
    points[1] += float(engine.scopa_counts[1])
    return points[0], points[1]


def simulate_match(
    bot_a: HeuristicBot, bot_b: HeuristicBot, rng: np.random.Generator
) -> tuple[float, float]:
    """Self-play one full deal between two bots; return (score_a, score_b)."""
    engine = ScopaEngine()
    bots = (bot_a, bot_b)
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
