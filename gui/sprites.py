"""Lazy loader and cache for the traditional 40-card regional deck images.

Each card face is loaded on first request from `config.ASSET_DIR` as
"<value>_<suit>.png" (e.g. "7_denari.png"); the shared card back comes from
"back.png". A missing or unreadable file caches `None`, signalling the renderer
to fall back to its drawn rectangle so the game keeps running without assets.
"""

from __future__ import annotations

import pygame

from engine.cards import Suit, card_suit, card_value
from gui import config

_SUIT_FILE: dict[Suit, str] = {
    Suit.DENARI: "denari",
    Suit.COPPE: "coppe",
    Suit.BASTONI: "bastoni",
    Suit.SPADE: "spade",
}


def face_filename(card: int) -> str:
    """Asset filename for a card index, e.g. 7 of Denari -> "7_denari.png"."""
    return f"{card_value(card)}_{_SUIT_FILE[card_suit(card)]}.png"


class SpriteCache:
    """Loads and memoizes card surfaces, yielding `None` when an asset is absent."""

    _faces: dict[int, pygame.Surface | None]
    _back_loaded: bool
    _back: pygame.Surface | None

    def __init__(self) -> None:
        self._faces = {}
        self._back_loaded = False
        self._back = None

    def _load(self, filename: str) -> pygame.Surface | None:
        path = config.ASSET_DIR / filename
        if not path.is_file():
            return None
        try:
            image = pygame.image.load(str(path))
        except pygame.error:
            return None
        return image.convert_alpha()

    def face(self, card: int) -> pygame.Surface | None:
        """The face surface for `card`, or `None` if its image is missing."""
        if card not in self._faces:
            self._faces[card] = self._load(face_filename(card))
        return self._faces[card]

    def back(self) -> pygame.Surface | None:
        """The shared card-back surface, or `None` if "back.png" is missing."""
        if not self._back_loaded:
            self._back = self._load(config.CARD_BACK_FILE)
            self._back_loaded = True
        return self._back
