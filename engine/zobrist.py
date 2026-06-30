"""Zobrist hashing (Phase 2): deterministic 64-bit keys for the full game state.

The state hash is the XOR of the keys of all active facts:
  * one key per active (zone, card) pair (ZOBRIST),
  * one key for the side to move (TURN_KEYS),
  * one key per (player, scopa-count) pair (SCOPA_KEYS).

Every state change updates the hash with two XORs (O(1)). Key uniqueness across
all three tables is asserted at import time to rule out trivial collisions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engine.cards import N_CARDS, N_ZONES

ZOBRIST_SEED = 0x5C09A_B0B  # fixed seed: tables reproducible across sessions
MAX_SCOPE = 40  # upper bound on scope per player within one deal

_rng = np.random.default_rng(ZOBRIST_SEED)

ZOBRIST: npt.NDArray[np.uint64] = _rng.integers(
    1, 2**64, size=(N_ZONES, N_CARDS), dtype=np.uint64
)
TURN_KEYS: npt.NDArray[np.uint64] = _rng.integers(1, 2**64, size=2, dtype=np.uint64)
SCOPA_KEYS: npt.NDArray[np.uint64] = _rng.integers(
    1, 2**64, size=(2, MAX_SCOPE + 1), dtype=np.uint64
)


def _assert_unique() -> None:
    keys = np.concatenate(
        [ZOBRIST.ravel(), TURN_KEYS.ravel(), SCOPA_KEYS.ravel()]
    )
    if np.unique(keys).size != keys.size:
        raise RuntimeError("Zobrist key collision detected at startup")


_assert_unique()
