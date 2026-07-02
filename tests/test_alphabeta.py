"""Phase 4 tests: alpha-beta negamax over perfect-information states."""

from __future__ import annotations

import math

import numpy as np

from engine.cards import Suit, Zone, card_index
from engine.core import ScopaEngine
from engine.transposition import TranspositionTable
from search.alphabeta import (
    SearchConfig,
    alphabeta,
    capture_options,
    legal_moves,
)


def _place(eng: ScopaEngine, suit: Suit, value: int, dst: Zone) -> int:
    idx = card_index(suit, value)
    eng.move(idx, Zone.MAZZO, dst)
    return idx


def test_capture_options_lay_when_no_capture() -> None:
    eng = ScopaEngine()
    card = _place(eng, Suit.DENARI, 1, Zone.MANO_P1)
    _place(eng, Suit.COPPE, 5, Zone.TAVOLO)
    assert capture_options(eng, card) == [[]]


def test_capture_options_single_equal_value() -> None:
    eng = ScopaEngine()
    card = _place(eng, Suit.DENARI, 7, Zone.MANO_P1)
    target = _place(eng, Suit.COPPE, 7, Zone.TAVOLO)
    assert capture_options(eng, card) == [[target]]


def _one_card_each() -> ScopaEngine:
    """A near-terminal state: one card per hand, deck empty."""
    eng = ScopaEngine()
    eng.state[Zone.MAZZO, :] = 0
    _place_from_void(eng, Suit.DENARI, 7, Zone.MANO_P1)
    _place_from_void(eng, Suit.COPPE, 7, Zone.TAVOLO)
    _place_from_void(eng, Suit.BASTONI, 3, Zone.MANO_P2)
    eng.rehash()
    return eng


def _place_from_void(eng: ScopaEngine, suit: Suit, value: int, dst: Zone) -> int:
    idx = card_index(suit, value)
    eng.state[dst, idx] = 1
    return idx


def test_alphabeta_prefers_settebello_capture() -> None:
    eng = _one_card_each()
    tt = TranspositionTable()
    cfg = SearchConfig(max_depth=6)
    value = alphabeta(eng, cfg.max_depth, -math.inf, math.inf, tt, cfg)
    # Player 0 captures the settebello -> strictly positive margin.
    assert value > 0


def test_alphabeta_is_deterministic() -> None:
    eng = _one_card_each()
    cfg = SearchConfig(max_depth=6)
    v1 = alphabeta(eng, cfg.max_depth, -math.inf, math.inf, TranspositionTable(), cfg)
    v2 = alphabeta(eng, cfg.max_depth, -math.inf, math.inf, TranspositionTable(), cfg)
    assert v1 == v2


def test_alphabeta_tt_reuse_matches_fresh() -> None:
    eng = _one_card_each()
    cfg = SearchConfig(max_depth=6)
    shared = TranspositionTable()
    first = alphabeta(eng, cfg.max_depth, -math.inf, math.inf, shared, cfg)
    second = alphabeta(eng, cfg.max_depth, -math.inf, math.inf, shared, cfg)
    assert first == second
    assert len(shared) > 0


def _leftover_sweep(last_capturer: int) -> ScopaEngine:
    """Deck empty, one card each, no captures possible: all cards end on the
    table and are swept by `last_capturer`. Board-identical for either value, so
    the two states share a zhash but have opposite values."""
    eng = ScopaEngine()
    eng.state[Zone.MAZZO, :] = 0
    _place_from_void(eng, Suit.COPPE, 9, Zone.MANO_P1)
    _place_from_void(eng, Suit.BASTONI, 10, Zone.MANO_P2)
    _place_from_void(eng, Suit.DENARI, 2, Zone.TAVOLO)
    eng.last_capturer = last_capturer
    eng.rehash()
    return eng


def test_tt_not_confused_by_last_capturer() -> None:
    # Regression: zhash omits last_capturer, but the sweep depends on it. A TT
    # shared across the two states must NOT serve one's value for the other.
    s0, s1 = _leftover_sweep(0), _leftover_sweep(1)
    assert s0.zhash == s1.zhash  # identical Zobrist hash...
    cfg = SearchConfig(max_depth=4)
    shared = TranspositionTable()
    v0 = alphabeta(s0, 4, -math.inf, math.inf, shared, cfg)
    v1 = alphabeta(s1, 4, -math.inf, math.inf, shared, cfg)  # would reuse s0 pre-fix
    f0 = alphabeta(_leftover_sweep(0), 4, -math.inf, math.inf, TranspositionTable(), cfg)
    f1 = alphabeta(_leftover_sweep(1), 4, -math.inf, math.inf, TranspositionTable(), cfg)
    assert v0 == f0 and v1 == f1  # shared-TT result matches a clean search
    assert v0 == -v1 and v0 > 0  # ...but the two values genuinely differ


def test_legal_moves_enumerates_hand() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(0))
    moves = legal_moves(eng, 0)
    assert len(moves) >= 3  # at least one move per hand card
