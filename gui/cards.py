"""Short, GUI-friendly card labels: index -> code like "7D" / "4C" / "10B".

Value (1..10) followed by the suit initial: Denari=D, Coppe=C, Bastoni=B,
Spade=S. Kept tiny and pygame-free so labels are reusable and testable.
"""

from __future__ import annotations

from engine.cards import Suit, card_suit, card_value

_SUIT_INITIAL: dict[Suit, str] = {
    Suit.DENARI: "D",
    Suit.COPPE: "C",
    Suit.BASTONI: "B",
    Suit.SPADE: "S",
}


def card_code(idx: int) -> str:
    """Compact label for a card index, e.g. 7 of Denari -> "7D"."""
    return f"{card_value(idx)}{_SUIT_INITIAL[card_suit(idx)]}"
