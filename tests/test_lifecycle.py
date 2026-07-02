"""Phase 3 lifecycle tests: execute_move, dealing, scopa, end-of-deal sweep."""

from __future__ import annotations

import numpy as np
import pytest

from engine.cards import N_CARDS, Suit, Zone, card_index
from engine.core import ScopaEngine
from engine.masks import is_consistent


def _drain_deck(eng: ScopaEngine) -> None:
    """Move every deck card into PRESE_P2 to build custom end-game states."""
    for c in [int(x) for x in eng.cards_in(Zone.MAZZO)]:
        eng.move(c, Zone.MAZZO, Zone.PRESE_P2)


def test_deal_round_start_layout() -> None:
    eng = ScopaEngine()
    eng.deal_round()
    assert eng.count(Zone.TAVOLO) == 4
    assert eng.count(Zone.MANO_P1) == 3
    assert eng.count(Zone.MANO_P2) == 3
    assert eng.count(Zone.MAZZO) == N_CARDS - 10
    assert is_consistent(eng)


def test_deal_round_replenish_no_table() -> None:
    eng = ScopaEngine()
    eng.deal_round()
    # consume both hands by moving their cards away, then replenish
    for hz in (Zone.MANO_P1, Zone.MANO_P2):
        for c in [int(x) for x in eng.cards_in(hz)]:
            eng.move(c, hz, Zone.PRESE_P1)
    table_before = eng.count(Zone.TAVOLO)
    eng.deal_round()
    assert eng.count(Zone.TAVOLO) == table_before  # no new table cards
    assert eng.count(Zone.MANO_P1) == 3
    assert eng.count(Zone.MANO_P2) == 3


def test_execute_move_capture_to_prese_and_toggle() -> None:
    eng = ScopaEngine()
    eng.move(card_index(Suit.DENARI, 7), Zone.MAZZO, Zone.TAVOLO)
    eng.move(card_index(Suit.SPADE, 7), Zone.MAZZO, Zone.MANO_P1)
    scopa = eng.execute_move(card_index(Suit.SPADE, 7), [card_index(Suit.DENARI, 7)])
    assert scopa is True
    assert eng.count(Zone.TAVOLO) == 0
    assert eng.count(Zone.PRESE_P1) == 2
    assert eng.current_player == 1
    assert eng.last_capturer == 0
    assert eng.scopa_counts[0] == 1
    assert eng.zhash == eng._recompute_hash()


def test_execute_move_lay_down_when_no_capture() -> None:
    eng = ScopaEngine()
    eng.move(card_index(Suit.COPPE, 2), Zone.MAZZO, Zone.TAVOLO)
    eng.move(card_index(Suit.SPADE, 7), Zone.MAZZO, Zone.MANO_P1)
    scopa = eng.execute_move(card_index(Suit.SPADE, 7), [])
    assert scopa is False
    assert eng.count(Zone.TAVOLO) == 2
    assert eng.current_player == 1


def test_illegal_capture_raises() -> None:
    eng = ScopaEngine()
    eng.move(card_index(Suit.DENARI, 7), Zone.MAZZO, Zone.TAVOLO)
    eng.move(card_index(Suit.SPADE, 7), Zone.MAZZO, Zone.MANO_P1)
    with pytest.raises(ValueError):
        eng.execute_move(card_index(Suit.SPADE, 7), [])  # capture is mandatory


def test_scopa_invalidated_on_last_play() -> None:
    eng = ScopaEngine()
    _drain_deck(eng)
    eng.move(card_index(Suit.SPADE, 4), Zone.PRESE_P2, Zone.MANO_P1)
    eng.move(card_index(Suit.DENARI, 4), Zone.PRESE_P2, Zone.TAVOLO)
    # deck empty, P2 hand empty, P1 about to play its last card -> last play
    scopa = eng.execute_move(card_index(Suit.SPADE, 4), [card_index(Suit.DENARI, 4)])
    assert scopa is False
    assert eng.scopa_counts[0] == 0
    assert eng.is_game_over()


def test_end_of_deal_sweep_to_last_capturer() -> None:
    eng = ScopaEngine()
    eng.move(card_index(Suit.COPPE, 5), Zone.MAZZO, Zone.TAVOLO)
    eng.move(card_index(Suit.BASTONI, 9), Zone.MAZZO, Zone.TAVOLO)
    eng.last_capturer = 1
    eng.end_of_deal_sweep()
    assert eng.count(Zone.TAVOLO) == 0
    assert eng.count(Zone.PRESE_P2) == 2
    assert is_consistent(eng)


def test_move_bounds_checking() -> None:
    eng = ScopaEngine()
    with pytest.raises(IndexError):
        eng.move(N_CARDS, Zone.MAZZO, Zone.TAVOLO)
    with pytest.raises(IndexError):
        eng.move(-1, Zone.MAZZO, Zone.TAVOLO)


def test_full_deal_conserves_cards() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(7))
    assert is_consistent(eng)
    assert int(eng.state.sum()) == N_CARDS
