"""Evaluation primitives: capture features, the weight genome, exposure, scoring.

The low-level layer beneath `engine.heuristic`: it turns captured cards into the
four majority features (carte, denari, settebello, primiera), applies the tunable
`Weights`, estimates the exposure a table hands the opponent, and scores a
finished deal. Kept dependency-free of the bot itself so both the heuristic bot
and the search leaf-evaluators can share it without a cycle.

    V(s) = w_captures*captures + w_denari*denari + w_settebello*settebello
         + w_primiera*(primiera / PRIMIERA_SCALE) + w_scope*scope
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple, dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from engine.cards import (
    CARD_SUITS,
    CARD_VALUES,
    N_CARDS,
    N_SUITS,
    N_VALUES,
    PRESE_ZONES,
    Suit,
    card_index,
    subsets_summing,
)
from engine.core import ScopaEngine

PRIMIERA_POINTS: dict[int, int] = {
    7: 21,
    6: 18,
    1: 16,
    5: 15,
    4: 14,
    3: 13,
    2: 12,
    8: 10,
    9: 10,
    10: 10,
}
N_WEIGHTS = 5
# Normalizes the prime-point sum (10..84) down to the unit scale of the other
# features, so primiera no longer silently dominates the linear evaluation.
PRIMIERA_SCALE = 21.0
# Weight on the opponent's best one-move capture left on the table.
RISK_COEF = 0.5
# Discounted likelihood that the opponent actually holds the Scopa card; the
# penalty itself is scaled by w_scope so giving a Scopa mirrors scoring one.
SCOPA_RISK_COEF = 0.9

# Per-index feature lookups: a card's primiera points and its settebello/denari
# identity are fixed, so the hot feature loop indexes these instead of a dict
# keyed on a freshly derived value and an IntEnum-constructing `card_suit`.
_DENARI_SUIT = int(Suit.DENARI)
_SETTEBELLO_IDX = card_index(Suit.DENARI, 7)
PRIMIERA_BY_CARD: tuple[int, ...] = tuple(PRIMIERA_POINTS[CARD_VALUES[i]] for i in range(N_CARDS))


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


def _features_from_ids(ids: Sequence[int]) -> CaptureFeatures:
    """Extract evaluation features from a sequence of Python card indices."""
    denari = 0
    settebello = 0
    best_per_suit = [0] * N_SUITS
    for idx in ids:
        s = CARD_SUITS[idx]
        if s == _DENARI_SUIT:
            denari += 1
            if idx == _SETTEBELLO_IDX:
                settebello = 1
        pts = PRIMIERA_BY_CARD[idx]
        if pts > best_per_suit[s]:
            best_per_suit[s] = pts
    return CaptureFeatures(len(ids), denari, settebello, sum(best_per_suit))


def capture_features(captured: npt.NDArray[np.intp]) -> CaptureFeatures:
    """Extract evaluation features from captured card indices."""
    return _features_from_ids(captured.tolist())


def _weighted(f: CaptureFeatures, w: Weights) -> float:
    return (
        w.captures * f.captures
        + w.denari * f.denari
        + w.settebello * f.settebello
        + w.primiera * (f.primiera / PRIMIERA_SCALE)
    )


def evaluate(engine: ScopaEngine, player: int, weights: Weights, scope_count: int = 0) -> float:
    """Linear value of `player`'s captured pile under `weights`."""
    f = capture_features(engine.cards_in(PRESE_ZONES[player]))
    return _weighted(f, weights) + weights.scope * scope_count


def _opponent_options(table: tuple[int, ...], v: int) -> tuple[tuple[int, ...], ...]:
    """Subsets of `table` a single opponent card of value `v` could capture.

    Mirrors the official rule: a single equal-value card forbids sum-captures.
    """
    singles = tuple((c,) for c in table if CARD_VALUES[c] == v)
    if singles:
        return singles
    return subsets_summing(table, v)


@lru_cache(maxsize=1 << 16)
def _exposure_features(table: tuple[int, ...]) -> tuple[CaptureFeatures, ...]:
    """Capture features the opponent could achieve from `table` in one play.

    Depends only on the (canonicalized) table, never on the weights, so it is
    cached across every candidate move and every genome sharing the same table.
    `_worst_exposure` then only runs the cheap weighted max over these.
    """
    feats: list[CaptureFeatures] = []
    for v in range(1, N_VALUES + 1):
        for subset in _opponent_options(table, v):
            feats.append(_features_from_ids(subset))
    return tuple(feats)


def _worst_exposure(table: list[int], weights: Weights) -> float:
    """Weighted value of the most damaging single capture the opponent can make."""
    worst = 0.0
    for f in _exposure_features(tuple(sorted(table))):
        val = _weighted(f, weights)
        if val > worst:
            worst = val
    return worst


def _scopa_threat(table: list[int]) -> bool:
    """True if a single opponent card could clear the whole table (a Scopa).

    Clearing all cards at once requires either a lone table card (equal-value
    capture) or a table whose total value is reachable by one card (<= 10);
    both collapse to `1 <= sum(values) <= N_VALUES`.
    """
    return 1 <= sum(CARD_VALUES[c] for c in table) <= N_VALUES


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


# Winner labels are from player 0's perspective: player 0 -> "p0", 1 -> "p1",
# a split -> "none". Callers naming players (e.g. human vs bot) remap as needed.
Winner = str


@dataclass(slots=True, frozen=True)
class DealBreakdown:
    """Per-component outcome of a finished deal, player 0 vs player 1."""

    p0_score: float
    p1_score: float
    p0_scope: int
    p1_scope: int
    settebello_winner: Winner
    denari_winner: Winner
    primiera_winner: Winner
    cards_winner: Winner


def _winner(a: int, b: int) -> Winner:
    if a > b:
        return "p0"
    if b > a:
        return "p1"
    return "none"


def deal_breakdown(engine: ScopaEngine) -> DealBreakdown:
    """Component-by-component breakdown of a finished deal (no state mutation)."""
    f0 = capture_features(engine.cards_in(PRESE_ZONES[0]))
    f1 = capture_features(engine.cards_in(PRESE_ZONES[1]))
    p0_score, p1_score = score_deal(engine)
    settebello = "p0" if f0.settebello else ("p1" if f1.settebello else "none")
    return DealBreakdown(
        p0_score=p0_score,
        p1_score=p1_score,
        p0_scope=int(engine.scopa_counts[0]),
        p1_scope=int(engine.scopa_counts[1]),
        settebello_winner=settebello,
        denari_winner=_winner(f0.denari, f1.denari),
        primiera_winner=_winner(f0.primiera, f1.primiera),
        cards_winner=_winner(f0.captures, f1.captures),
    )
