"""Card primitives: constants, suits, zones, index helpers, subset combinatorics.

Card index: idx = suit * 10 + (value - 1), suit in [0,3], value in [1,10].
Kept dependency-free so engine.zobrist and engine.core can both import it
without a circular reference.
"""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache

N_CARDS = 40
N_SUITS = 4
N_VALUES = 10


class Suit(IntEnum):
    DENARI = 0
    COPPE = 1
    BASTONI = 2
    SPADE = 3


class Zone(IntEnum):
    """Disjoint zones a card can occupy."""

    MAZZO = 0  # deck
    TAVOLO = 1  # table
    MANO_P1 = 2  # player 1 hand
    MANO_P2 = 3  # player 2 hand
    PRESE_P1 = 4  # player 1 captures
    PRESE_P2 = 5  # player 2 captures


N_ZONES = len(Zone)

HAND_ZONES = (Zone.MANO_P1, Zone.MANO_P2)
PRESE_ZONES = (Zone.PRESE_P1, Zone.PRESE_P2)

# Precomputed per-index lookup tables. The value/suit of a card index never
# changes, so hot inner loops (millions of calls per ISMCTS search) index these
# tuples instead of paying modulo arithmetic and, crucially, IntEnum
# construction — profiling showed `Suit(...)` dominating `card_suit`.
CARD_VALUES: tuple[int, ...] = tuple(i % N_VALUES + 1 for i in range(N_CARDS))
CARD_SUITS: tuple[int, ...] = tuple(i // N_VALUES for i in range(N_CARDS))


def card_index(suit: Suit, value: int) -> int:
    """Index 0..39 of card (suit, value)."""
    if not 1 <= value <= N_VALUES:
        raise ValueError(f"value out of range: {value}")
    return int(suit) * N_VALUES + (value - 1)


def card_value(idx: int) -> int:
    """Card value 1..10 from its index."""
    return idx % N_VALUES + 1


def card_suit(idx: int) -> Suit:
    """Card suit from its index."""
    return Suit(idx // N_VALUES)


@lru_cache(maxsize=1 << 17)
def subsets_summing(cards: tuple[int, ...], target: int) -> tuple[tuple[int, ...], ...]:
    """All subsets (>=2 cards) of `cards` whose value sum equals `target`.

    Memoized on the immutable `(cards, target)` key: the table configuration
    repeats heavily across an ISMCTS search (the same table is re-evaluated for
    every candidate and every value 1..10 in exposure analysis), so caching
    collapses the dominant recursion cost. The recursion reads a precomputed
    `vals` list instead of calling `card_value` per node. The returned tuples
    are shared cache entries and MUST be treated as immutable by callers.
    """
    results: list[tuple[int, ...]] = []
    vals = [CARD_VALUES[c] for c in cards]
    n = len(cards)

    def rec(start: int, acc: list[int], total: int) -> None:
        if total == target and len(acc) >= 2:
            results.append(tuple(acc))
        if total >= target:
            return
        for i in range(start, n):
            v = vals[i]
            if total + v > target:
                continue
            acc.append(cards[i])
            rec(i + 1, acc, total + v)
            acc.pop()

    rec(0, [], 0)
    return tuple(results)
