"""Core Engine — Scopa game state as binary NumPy uint8 matrices.

State is a (N_ZONES, N_CARDS) uint8 matrix (0/1): 1 = card present in zone.
The engine also tracks the side to move, the last capturer, and per-player
scopa counts. The incremental Zobrist hash covers the zones, the side to move,
and the scopa counts -- but NOT `last_capturer`. Since the end-of-deal sweep
depends on `last_capturer`, cache keys built on `zhash` must fold it in
separately (see `search.alphabeta._tt_key`).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engine.cards import (
    CARD_VALUES,
    HAND_ZONES,
    N_CARDS,
    N_ZONES,
    PRESE_ZONES,
    Zone,
    subsets_summing,
)
from engine.zobrist import (
    SCOPA_KEY_INTS,
    SCOPA_KEYS,
    TURN_KEY_INTS,
    TURN_KEYS,
    ZOBRIST,
    ZOBRIST_INTS,
)

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

    def rehash(self) -> None:
        """Recompute and store the full Zobrist hash from the current state."""
        self.zhash = self._recompute_hash()

    def clone(self) -> ScopaEngine:
        """Deep, independent copy: state matrix, scalars, scopa counts, hash."""
        other = ScopaEngine.__new__(ScopaEngine)
        other.state = self.state.copy()
        other.zhash = self.zhash
        other.current_player = self.current_player
        other.last_capturer = self.last_capturer
        other.scopa_counts = self.scopa_counts.copy()
        return other

    # --- zone accessors --------------------------------------------------

    def cards_in(self, z: Zone) -> npt.NDArray[np.intp]:
        """Indices of cards present in zone `z`."""
        # `row.nonzero()[0]` == `np.flatnonzero(row)` for a 1-D row but skips the
        # ravel/_wrapfunc Python layers flatnonzero adds — this is one of the
        # hottest calls in the search (the `type: ignore` is comment-only).
        return self.state[z].nonzero()[0]  # type: ignore[no-any-return]

    def count(self, z: Zone) -> int:
        """Number of cards in zone `z`."""
        return int(np.count_nonzero(self.state[z]))

    # --- atomic transitions ---------------------------------------------

    def move(self, idx: int, src: Zone, dst: Zone) -> None:
        """Move card `idx` from `src` to `dst`, updating the hash incrementally."""
        if not 0 <= idx < N_CARDS:
            raise IndexError(f"card index out of bounds: {idx}")
        if self.state[src, idx] == 0:
            raise ValueError(f"card {idx} absent from zone {src.name}")
        self.state[src, idx] = 0
        self.state[dst, idx] = 1
        self.zhash ^= ZOBRIST_INTS[src][idx] ^ ZOBRIST_INTS[dst][idx]

    def _set_turn(self, player: int) -> None:
        if player != self.current_player:
            self.zhash ^= TURN_KEY_INTS[self.current_player] ^ TURN_KEY_INTS[player]
            self.current_player = player

    def _add_scopa(self, player: int) -> None:
        old = int(self.scopa_counts[player])
        self.zhash ^= SCOPA_KEY_INTS[player][old] ^ SCOPA_KEY_INTS[player][old + 1]
        self.scopa_counts[player] = old + 1

    # --- rules: captures and action masking -----------------------------

    def legal_captures(self, idx: int) -> list[list[int]]:
        """Legal capture sets for playing card `idx`, as plain-int lists.

        The single source of the capture rule: an equal-value single on the table
        forbids sum-captures (only that card may be taken); otherwise every subset
        summing to the card's value qualifies. Returns `[[]]` when the card can
        only be laid. Plain ints, no per-option NumPy arrays, so the search's
        legal-move enumeration -- its hottest allocation site -- stays cheap.
        """
        v = CARD_VALUES[idx]
        table = self.cards_in(Zone.TAVOLO).tolist()
        singles = [c for c in table if CARD_VALUES[c] == v]
        if singles:
            return [[c] for c in singles]
        subs = subsets_summing(tuple(table), v)
        if subs:
            return [list(combo) for combo in subs]
        return [[]]

    def captures_for(self, idx: int) -> list[npt.NDArray[np.intp]]:
        """NumPy view of `legal_captures` ([] = no capture / lay the card).

        Thin wrapper over the rule primitive for callers/tests wanting arrays.
        """
        caps = self.legal_captures(idx)
        if caps == [[]]:
            return []
        return [np.array(c, dtype=np.intp) for c in caps]

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

    def apply_legal_move(self, card_idx: int, cap: list[int]) -> bool:
        """Apply an already-legal `(card, cap)` without re-deriving captures.

        Behaviourally identical to `execute_move` for a legal move, but skips the
        capture re-derivation, `sorted` canonicalization, and legality check --
        pure overhead when the move comes straight from `legal_moves` (the search
        hot path). Capture order is irrelevant (independent zone writes, XOR
        hash), so state and hash match `execute_move` exactly. Callers MUST pass
        a move produced from this state's legal options.
        """
        player = self.current_player
        hand = HAND_ZONES[player]
        scopa = False
        if cap:
            prese = PRESE_ZONES[player]
            self.move(card_idx, hand, prese)
            for c in cap:
                self.move(c, Zone.TAVOLO, prese)
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
