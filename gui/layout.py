"""Geometry: turn the live engine state into placeable, hit-testable card slots.

Pure math plus read-only engine queries, with no pygame dependency, so zone
layout stays unit-testable without a display. Each zone is a horizontal band;
its cards are centered as a row.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.cards import HAND_ZONES, PRESE_ZONES
from engine.cards import Zone as CardZone
from engine.core import ScopaEngine
from gui import config

HUMAN = 0
OPPONENT = 1


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle in window pixels."""

    x: int
    y: int
    width: int
    height: int

    def contains(self, px: int, py: int) -> bool:
        """True if (px, py) falls inside this rectangle."""
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


@dataclass(frozen=True, slots=True)
class CardSlot:
    """One drawn card: its rectangle, the card index, and its face direction."""

    rect: Rect
    card: int
    face_up: bool


@dataclass(frozen=True, slots=True)
class Zone:
    """A labeled band and the card slots laid out within it."""

    name: str
    label: str
    band_y: int
    slots: list[CardSlot]


@dataclass(frozen=True, slots=True)
class CapturePile:
    """A player's captured cards: a stacked deck plus its scopa markers."""

    name: str
    center: tuple[int, int]
    cards: list[int]
    scope: int


def card_row(cards: list[int], center_y: int, face_up: bool) -> list[CardSlot]:
    """Center a row of slots for `cards` around the window's horizontal center."""
    count = len(cards)
    if count == 0:
        return []
    step = config.CARD_WIDTH + config.CARD_GAP
    total = count * config.CARD_WIDTH + (count - 1) * config.CARD_GAP
    start_x = config.WINDOW_WIDTH // 2 - total // 2
    top = center_y - config.CARD_HEIGHT // 2
    return [
        CardSlot(
            Rect(start_x + i * step, top, config.CARD_WIDTH, config.CARD_HEIGHT),
            card,
            face_up,
        )
        for i, card in enumerate(cards)
    ]


def _band(name: str, label: str, frac: float, cards: list[int], face_up: bool) -> Zone:
    band_y = int(config.WINDOW_HEIGHT * frac)
    return Zone(name, label, band_y, card_row(cards, band_y, face_up))


def _cards(engine: ScopaEngine, zone: CardZone) -> list[int]:
    return [int(c) for c in engine.cards_in(zone)]


def build_zones(engine: ScopaEngine) -> list[Zone]:
    """Build the three zones (opponent, table, player) from the live engine."""
    return [
        _band(
            "opponent",
            "Opponent",
            config.OPPONENT_BAND_Y,
            _cards(engine, HAND_ZONES[OPPONENT]),
            face_up=False,
        ),
        _band("table", "Table", config.TABLE_BAND_Y, _cards(engine, CardZone.TAVOLO), face_up=True),
        _band(
            "player",
            "Your hand",
            config.PLAYER_BAND_Y,
            _cards(engine, HAND_ZONES[HUMAN]),
            face_up=True,
        ),
    ]


def build_piles(engine: ScopaEngine) -> list[CapturePile]:
    """Build the human and bot capture piles from the live engine state."""
    return [
        CapturePile(
            "human",
            config.HUMAN_PILE_CENTER,
            _cards(engine, PRESE_ZONES[HUMAN]),
            int(engine.scopa_counts[HUMAN]),
        ),
        CapturePile(
            "bot",
            config.BOT_PILE_CENTER,
            _cards(engine, PRESE_ZONES[OPPONENT]),
            int(engine.scopa_counts[OPPONENT]),
        ),
    ]


def card_positions(zones: list[Zone], piles: list[CapturePile]) -> dict[int, tuple[float, float]]:
    """Map every visible card to its center: slot centers and pile anchors."""
    positions: dict[int, tuple[float, float]] = {}
    for zone in zones:
        for slot in zone.slots:
            positions[slot.card] = (
                slot.rect.x + slot.rect.width / 2,
                slot.rect.y + slot.rect.height / 2,
            )
    for pile in piles:
        for card in pile.cards:
            positions[card] = (float(pile.center[0]), float(pile.center[1]))
    return positions


def _hit_zone(zones: list[Zone], name: str, pos: tuple[int, int]) -> int | None:
    px, py = pos
    for zone in zones:
        if zone.name != name:
            continue
        for slot in zone.slots:
            if slot.rect.contains(px, py):
                return slot.card
    return None


def hit_test(zones: list[Zone], pos: tuple[int, int]) -> int | None:
    """Return the human hand card index under `pos`, or None if none hit."""
    return _hit_zone(zones, "player", pos)


def hit_test_table(zones: list[Zone], pos: tuple[int, int]) -> int | None:
    """Return the table card index under `pos`, or None if none hit."""
    return _hit_zone(zones, "table", pos)
