"""Card primitives: constants, suits, zones, index helpers, subset combinatorics.

Card index: idx = suit * 10 + (value - 1), suit in [0,3], value in [1,10].
Kept dependency-free so engine.zobrist and engine.core can both import it
without a circular reference.
"""

from __future__ import annotations

from enum import IntEnum

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


def subsets_summing(cards: list[int], target: int) -> list[list[int]]:
    """All subsets (>=2 cards) of `cards` whose value sum equals `target`."""
    results: list[list[int]] = []

    def rec(start: int, acc: list[int], total: int) -> None:
        if total == target and len(acc) >= 2:
            results.append(acc.copy())
        if total >= target:
            return
        for i in range(start, len(cards)):
            v = card_value(cards[i])
            if total + v > target:
                continue
            acc.append(cards[i])
            rec(i + 1, acc, total + v)
            acc.pop()

    rec(0, [], 0)
    return results
