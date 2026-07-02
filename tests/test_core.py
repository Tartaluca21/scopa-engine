"""Phase 1 unit tests: state, card dealing, multi-capture cases, action mask."""

from __future__ import annotations

import numpy as np

from engine.cards import N_CARDS, Suit, Zone, card_index
from engine.core import ScopaEngine
from engine.masks import capture_mask, is_consistent, legal_action_mask


def _place(eng: ScopaEngine, suit: Suit, value: int, dst: Zone) -> int:
    idx = card_index(suit, value)
    eng.move(idx, Zone.MAZZO, dst)
    return idx


# --- initialization ------------------------------------------------------


def test_init_all_cards_in_deck() -> None:
    eng = ScopaEngine()
    assert eng.count(Zone.MAZZO) == N_CARDS
    assert is_consistent(eng)
    for z in (Zone.TAVOLO, Zone.MANO_P1, Zone.MANO_P2, Zone.PRESE_P1, Zone.PRESE_P2):
        assert eng.count(z) == 0


def test_state_shape_and_dtype() -> None:
    eng = ScopaEngine()
    assert eng.state.shape == (len(Zone), N_CARDS)
    assert eng.state.dtype == np.uint8


# --- dealing -------------------------------------------------------------


def test_deal_preserves_consistency_and_counts() -> None:
    eng = ScopaEngine()
    for v in range(1, 4):
        _place(eng, Suit.DENARI, v, Zone.MANO_P1)
    for v in range(1, 4):
        _place(eng, Suit.COPPE, v, Zone.MANO_P2)
    for v in (5, 6, 7, 10):
        _place(eng, Suit.SPADE, v, Zone.TAVOLO)
    assert eng.count(Zone.MANO_P1) == 3
    assert eng.count(Zone.MANO_P2) == 3
    assert eng.count(Zone.TAVOLO) == 4
    assert eng.count(Zone.MAZZO) == N_CARDS - 10
    assert is_consistent(eng)


def test_move_rejects_absent_card() -> None:
    eng = ScopaEngine()
    idx = _place(eng, Suit.SPADE, 7, Zone.TAVOLO)
    try:
        eng.move(idx, Zone.MAZZO, Zone.TAVOLO)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for card absent from zone")


# --- captures: single-card priority rule ---------------------------------


def test_single_capture_has_priority_over_combination() -> None:
    eng = ScopaEngine()
    _place(eng, Suit.DENARI, 7, Zone.TAVOLO)  # single 7
    _place(eng, Suit.COPPE, 3, Zone.TAVOLO)
    _place(eng, Suit.COPPE, 4, Zone.TAVOLO)  # 3+4 = 7 (forbidden while single exists)
    played = card_index(Suit.SPADE, 7)
    opts = eng.captures_for(played)
    assert len(opts) == 1
    assert opts[0].tolist() == [card_index(Suit.DENARI, 7)]


def test_multiple_singles_are_distinct_options() -> None:
    eng = ScopaEngine()
    _place(eng, Suit.DENARI, 5, Zone.TAVOLO)
    _place(eng, Suit.COPPE, 5, Zone.TAVOLO)
    opts = eng.captures_for(card_index(Suit.SPADE, 5))
    assert sorted(o.tolist()[0] for o in opts) == sorted(
        [card_index(Suit.DENARI, 5), card_index(Suit.COPPE, 5)]
    )


def test_combination_capture_when_no_single() -> None:
    eng = ScopaEngine()
    _place(eng, Suit.COPPE, 3, Zone.TAVOLO)
    _place(eng, Suit.COPPE, 4, Zone.TAVOLO)
    opts = eng.captures_for(card_index(Suit.SPADE, 7))
    assert len(opts) == 1
    assert sorted(int(c) for c in opts[0]) == sorted(
        [card_index(Suit.COPPE, 3), card_index(Suit.COPPE, 4)]
    )


def test_multiple_combinations_all_legal() -> None:
    eng = ScopaEngine()
    _place(eng, Suit.COPPE, 3, Zone.TAVOLO)
    _place(eng, Suit.COPPE, 4, Zone.TAVOLO)
    _place(eng, Suit.DENARI, 2, Zone.TAVOLO)
    _place(eng, Suit.DENARI, 5, Zone.TAVOLO)
    opts = eng.captures_for(card_index(Suit.SPADE, 7))
    sums = {tuple(sorted(int(c) for c in o)) for o in opts}
    assert sums == {
        (card_index(Suit.DENARI, 2), card_index(Suit.DENARI, 5)),
        (card_index(Suit.COPPE, 3), card_index(Suit.COPPE, 4)),
    }


def test_no_capture_returns_empty() -> None:
    eng = ScopaEngine()
    _place(eng, Suit.COPPE, 2, Zone.TAVOLO)
    assert eng.captures_for(card_index(Suit.SPADE, 7)) == []


# --- action mask ---------------------------------------------------------


def test_action_mask_equals_hand() -> None:
    eng = ScopaEngine()
    hand = [card_index(Suit.DENARI, v) for v in (1, 7, 10)]
    for idx in hand:
        eng.move(idx, Zone.MAZZO, Zone.MANO_P1)
    mask = legal_action_mask(eng, 0)
    assert mask.dtype == np.uint8
    assert int(mask.sum()) == 3
    assert all(mask[i] == 1 for i in hand)


def test_capture_mask_flags_only_capturing_cards() -> None:
    eng = ScopaEngine()
    _place(eng, Suit.DENARI, 7, Zone.TAVOLO)
    take = card_index(Suit.SPADE, 7)
    miss = card_index(Suit.SPADE, 2)
    eng.move(take, Zone.MAZZO, Zone.MANO_P1)
    eng.move(miss, Zone.MAZZO, Zone.MANO_P1)
    mask = capture_mask(eng, 0)
    assert mask[take] == 1
    assert mask[miss] == 0


# --- trusted fast-path equivalence ---------------------------------------


def test_apply_legal_move_matches_execute_move() -> None:
    """`apply_legal_move` must reproduce `execute_move` state/hash/scopa exactly.

    Fuzz across many random self-play deals: for every legal move, apply it via
    both paths on independent clones and assert byte-identical state, hash,
    scalars, scopa counts, and the returned Scopa flag.
    """
    from search.alphabeta import legal_moves

    rng = np.random.default_rng(20260702)
    for _ in range(200):
        eng = ScopaEngine()
        eng.deal_round(rng)
        while not eng.is_game_over():
            if eng.count(Zone.MANO_P1) == 0 and eng.count(Zone.MANO_P2) == 0:
                eng.deal_round(rng)
                continue
            player = eng.current_player
            moves = legal_moves(eng, player)
            card, cap = moves[int(rng.integers(len(moves)))]
            a, b = eng.clone(), eng.clone()
            scopa_ref = a.execute_move(card, list(cap))
            scopa_fast = b.apply_legal_move(card, list(cap))
            assert scopa_fast == scopa_ref
            assert np.array_equal(a.state, b.state)
            assert a.zhash == b.zhash
            assert a.current_player == b.current_player
            assert a.last_capturer == b.last_capturer
            assert np.array_equal(a.scopa_counts, b.scopa_counts)
            eng.apply_legal_move(card, list(cap))
