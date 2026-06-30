"""Core Engine — Scopa game state as binary NumPy uint8 matrices.

State is a (N_ZONES, N_CARDS) uint8 matrix (0/1): 1 = card present in zone.
The engine also tracks the side to move, the last capturer, and per-player
scopa counts; all of these feed the incremental Zobrist hash.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engine.cards import (
    HAND_ZONES,
    N_CARDS,
    N_ZONES,
    PRESE_ZONES,
    Zone,
    card_value,
    subsets_summing,
)
from engine.zobrist import SCOPA_KEYS, TURN_KEYS, ZOBRIST

CardArray = npt.NDArray[np.uint8]


class ScopaEngine:
    """Vectorized state engine for a Scopa deal with O(1) Zobrist hashing."""

    state: CardArray
    zhash: int
    current_player: int
    last_capturer: int
    scopa_counts: npt.NDArray[np.int64]

    def __init__(self) -> None:
        self.state = np.zeros((N_ZONES, N_CARDS), dtype=np.uint8)
        self.reset()

    def reset(self) -> None:
        """Reset to the start of a deal: all 40 cards in the deck."""
        self.state.fill(0)
        self.state[Zone.MAZZO, :] = 1
        self.current_player = 0
        self.last_capturer = -1
        self.scopa_counts = np.zeros(2, dtype=np.int64)
        self.zhash = self._recompute_hash()

    def _recompute_hash(self) -> int:
        """Recompute the full Zobrist hash from scratch (debug/invariant check)."""
        keys = ZOBRIST[self.state.astype(bool)]
        h = int(np.bitwise_xor.reduce(keys)) if keys.size else 0
        h ^= int(TURN_KEYS[self.current_player])
        h ^= int(SCOPA_KEYS[0, int(self.scopa_counts[0])])
        h ^= int(SCOPA_KEYS[1, int(self.scopa_counts[1])])
        return h

    # --- zone accessors --------------------------------------------------

    def cards_in(self, z: Zone) -> npt.NDArray[np.intp]:
        """Indices of cards present in zone `z`."""
        return np.flatnonzero(self.state[z])

    def count(self, z: Zone) -> int:
        """Number of cards in zone `z`."""
        return int(self.state[z].sum())

    # --- atomic transitions ---------------------------------------------

    def move(self, idx: int, src: Zone, dst: Zone) -> None:
        """Move card `idx` from `src` to `dst`, updating the hash incrementally."""
        if not 0 <= idx < N_CARDS:
            raise IndexError(f"card index out of bounds: {idx}")
        if self.state[src, idx] == 0:
            raise ValueError(f"card {idx} absent from zone {src.name}")
        self.state[src, idx] = 0
        self.state[dst, idx] = 1
        self.zhash ^= int(ZOBRIST[src, idx]) ^ int(ZOBRIST[dst, idx])

    def _set_turn(self, player: int) -> None:
        if player != self.current_player:
            self.zhash ^= int(TURN_KEYS[self.current_player]) ^ int(TURN_KEYS[player])
            self.current_player = player

    def _add_scopa(self, player: int) -> None:
        old = int(self.scopa_counts[player])
        self.zhash ^= int(SCOPA_KEYS[player, old]) ^ int(SCOPA_KEYS[player, old + 1])
        self.scopa_counts[player] = old + 1

    # --- rules: captures and action masking -----------------------------

    def captures_for(self, idx: int) -> list[npt.NDArray[np.intp]]:
        """Legal capture options when playing card `idx`, given the table.

        Official rule: if a single table card of equal value exists, capturing
        by combined sum is forbidden; only a single card may be taken.
        Empty list = no capture (card is laid on the table).
        """
        v = card_value(idx)
        table = self.cards_in(Zone.TAVOLO)
        singles = [c for c in table if card_value(int(c)) == v]
        if singles:
            return [np.array([c], dtype=np.intp) for c in singles]
        return [
            np.array(combo, dtype=np.intp)
            for combo in subsets_summing([int(c) for c in table], v)
        ]

    def legal_action_mask(self, player: int) -> CardArray:
        """uint8 vector (40,): 1 for each card `player` may legally play (= hand)."""
        return self.state[HAND_ZONES[player]].copy()

    def capture_mask(self, player: int) -> CardArray:
        """uint8 vector (40,): 1 for hand cards that trigger a capture."""
        mask = np.zeros(N_CARDS, dtype=np.uint8)
        for idx in self.cards_in(HAND_ZONES[player]):
            if self.captures_for(int(idx)):
                mask[idx] = 1
        return mask

    # --- transactional move ---------------------------------------------

    def execute_move(self, card_idx: int, capture_indices: list[int]) -> bool:
        """Play `card_idx` for the current player, capturing `capture_indices`.

        Atomically moves the played card and captured cards to the player's
        PRESE pile (or lays the card on the table when not capturing), updates
        the hash, flags a Scopa when the table clears (invalidated on the very
        last play of the game), and toggles the turn. Returns True iff a valid
        Scopa was scored. Raises on illegal card or illegal capture set.
        """
        player = self.current_player
        hand = HAND_ZONES[player]
        if self.state[hand, card_idx] == 0:
            raise ValueError(f"card {card_idx} not in player {player} hand")
        options = self.captures_for(card_idx)
        cap = sorted(int(c) for c in capture_indices)
        if options:
            legal = [sorted(int(c) for c in o) for o in options]
            if cap not in legal:
                raise ValueError(f"illegal capture {cap} for card {card_idx}")
        elif cap:
            raise ValueError("no capture available; capture set must be empty")

        scopa = False
        if cap:
            self.move(card_idx, hand, PRESE_ZONES[player])
            for c in cap:
                self.move(c, Zone.TAVOLO, PRESE_ZONES[player])
            self.last_capturer = player
            if self.count(Zone.TAVOLO) == 0 and not self._is_last_play():
                self._add_scopa(player)
                scopa = True
        else:
            self.move(card_idx, hand, Zone.TAVOLO)
        self._set_turn(1 - player)
        return scopa

    # --- round lifecycle -------------------------------------------------

    def _is_last_play(self) -> bool:
        return (
            self.count(Zone.MAZZO) == 0
            and self.count(Zone.MANO_P1) == 0
            and self.count(Zone.MANO_P2) == 0
        )

    def is_game_over(self) -> bool:
        """True when the deck and both hands are empty (deal finished)."""
        return self._is_last_play()

    def deal_round(self, rng: np.random.Generator | None = None) -> None:
        """Deal 3 cards to each player (and 4 to the table on the first deal)."""
        start = self.count(Zone.MAZZO) == N_CARDS
        deck = self.cards_in(Zone.MAZZO)
        order = [int(c) for c in (rng.permutation(deck) if rng is not None else deck)]
        need = 6 + (4 if start else 0)
        if len(order) < need:
            raise ValueError("not enough cards in deck to deal a round")
        i = 0
        if start:
            for _ in range(4):
                self.move(order[i], Zone.MAZZO, Zone.TAVOLO)
                i += 1
        for hand in HAND_ZONES:
            for _ in range(3):
                self.move(order[i], Zone.MAZZO, hand)
                i += 1

    def end_of_deal_sweep(self) -> None:
        """Award all remaining table cards to the last capturer."""
        if self.last_capturer not in (0, 1):
            return
        dst = PRESE_ZONES[self.last_capturer]
        for c in [int(x) for x in self.cards_in(Zone.TAVOLO)]:
            self.move(c, Zone.TAVOLO, dst)

    # --- invariants ------------------------------------------------------

    def is_consistent(self) -> bool:
        """Every card is in exactly one zone (column sum == 1)."""
        return bool(np.all(self.state.sum(axis=0) == 1))
