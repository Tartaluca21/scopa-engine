"""Derived-view helpers over `ScopaEngine`: action masks and the state invariant.

Read-only query/debug utilities kept out of the transactional core so
`engine.core` holds only state mutation and hashing. `legal_action_mask` and
`capture_mask` expose the playable/capturing cards as (40,) uint8 vectors (used
by the UI and tests); `is_consistent` checks the one-card-per-zone invariant.
"""

from __future__ import annotations

import numpy as np

from engine.cards import HAND_ZONES, N_CARDS
from engine.core import CardArray, ScopaEngine


def legal_action_mask(engine: ScopaEngine, player: int) -> CardArray:
    """uint8 vector (40,): 1 for each card `player` may legally play (= hand)."""
    mask: CardArray = engine.state[HAND_ZONES[player]].copy()
    return mask


def capture_mask(engine: ScopaEngine, player: int) -> CardArray:
    """uint8 vector (40,): 1 for hand cards that trigger a capture."""
    mask = np.zeros(N_CARDS, dtype=np.uint8)
    for idx in engine.cards_in(HAND_ZONES[player]):
        if engine.legal_captures(int(idx)) != [[]]:
            mask[idx] = 1
    return mask


def is_consistent(engine: ScopaEngine) -> bool:
    """Every card is in exactly one zone (column sum == 1)."""
    return bool(np.all(engine.state.sum(axis=0) == 1))
