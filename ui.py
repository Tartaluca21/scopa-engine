"""Presentation layer for the human-vs-bot CLI: card names, menu, input.

Split out of `play.py` to keep that file focused on the game loop. Cards are
shown with their Italian names (the game's domain language); all identifiers
and messages stay English.
"""

from __future__ import annotations

from engine.cards import HAND_ZONES, Zone, card_suit, card_value
from engine.core import ScopaEngine
from search.alphabeta import capture_options

Move = tuple[int, list[int]]

_VALUE_NAMES: dict[int, str] = {1: "Asso", 8: "Fante", 9: "Cavallo", 10: "Re"}
_SUIT_NAMES = ("Denari", "Coppe", "Bastoni", "Spade")


def card_name(idx: int) -> str:
    """Readable Italian name of a card index, e.g. 'Asso di Bastoni'."""
    value = card_value(idx)
    label = _VALUE_NAMES.get(value, str(value))
    return f"{label} di {_SUIT_NAMES[int(card_suit(idx))]}"


def _cards(engine: ScopaEngine, zone: Zone) -> list[int]:
    return [int(c) for c in engine.cards_in(zone)]


def describe_move(card: int, capture: list[int]) -> str:
    """One-line description of a (card, capture) move for the menu."""
    if not capture:
        return f"Play {card_name(card)} onto the table"
    taken = ", ".join(card_name(c) for c in capture)
    return f"Play {card_name(card)} to capture {taken}"


def enumerate_moves(engine: ScopaEngine, player: int) -> list[Move]:
    """Flat list of every legal (card, capture-option) the player may pick."""
    moves: list[Move] = []
    for card in _cards(engine, HAND_ZONES[player]):
        for cap in capture_options(engine, card):
            moves.append((card, cap))
    return moves


def show_state(engine: ScopaEngine, player: int) -> None:
    """Print the table and `player`'s hand."""
    table = _cards(engine, Zone.TAVOLO)
    print("\nTable: " + (", ".join(card_name(c) for c in table) or "(empty)"))
    hand = _cards(engine, HAND_ZONES[player])
    print("Your hand: " + ", ".join(card_name(c) for c in hand))


def read_choice(n_options: int) -> int:
    """Prompt until the human enters a valid 1-based menu index, return 0-based."""
    while True:
        raw = input(f"Choose a move [1-{n_options}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= n_options:
            return int(raw) - 1
        print("Invalid choice, try again.")
