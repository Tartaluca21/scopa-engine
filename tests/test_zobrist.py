"""Phase 2 tests: incremental Zobrist hashing + Transposition Table."""

from __future__ import annotations

import numpy as np

from engine.cards import Suit, Zone, card_index
from engine.core import ScopaEngine
from engine.transposition import NodeType, TranspositionTable
from engine.zobrist import ZOBRIST


def test_zobrist_table_shape_dtype_unique() -> None:
    assert ZOBRIST.shape == (len(Zone), 40)
    assert ZOBRIST.dtype == np.uint64
    assert np.unique(ZOBRIST).size == ZOBRIST.size  # no duplicate keys


def test_initial_hash_is_deterministic() -> None:
    assert ScopaEngine().zhash == ScopaEngine().zhash
    assert ScopaEngine().zhash != 0


def test_incremental_matches_full_recompute() -> None:
    eng = ScopaEngine()
    idx = card_index(Suit.DENARI, 7)
    eng.move(idx, Zone.MAZZO, Zone.TAVOLO)
    assert eng.zhash == eng._recompute_hash()


def test_move_changes_hash() -> None:
    eng = ScopaEngine()
    before = eng.zhash
    eng.move(card_index(Suit.SPADE, 3), Zone.MAZZO, Zone.MANO_P1)
    assert eng.zhash != before


def test_hash_is_path_independent() -> None:
    a = ScopaEngine()
    a.move(card_index(Suit.DENARI, 7), Zone.MAZZO, Zone.TAVOLO)
    a.move(card_index(Suit.COPPE, 2), Zone.MAZZO, Zone.MANO_P1)
    b = ScopaEngine()
    b.move(card_index(Suit.COPPE, 2), Zone.MAZZO, Zone.MANO_P1)
    b.move(card_index(Suit.DENARI, 7), Zone.MAZZO, Zone.TAVOLO)
    assert a.zhash == b.zhash


def test_inverse_move_restores_hash() -> None:
    eng = ScopaEngine()
    start = eng.zhash
    idx = card_index(Suit.BASTONI, 10)
    eng.move(idx, Zone.MAZZO, Zone.TAVOLO)
    eng.move(idx, Zone.TAVOLO, Zone.MAZZO)
    assert eng.zhash == start


def test_distinct_states_have_distinct_hashes() -> None:
    seen: set[int] = set()
    for v in range(1, 11):
        eng = ScopaEngine()
        eng.move(card_index(Suit.DENARI, v), Zone.MAZZO, Zone.TAVOLO)
        seen.add(eng.zhash)
    assert len(seen) == 10  # no trivial collisions


# --- Transposition Table -------------------------------------------------


def test_tt_store_and_get() -> None:
    tt = TranspositionTable()
    tt.store(123, 0.5, depth=3)
    entry = tt.get(123)
    assert entry is not None
    assert entry.value == 0.5
    assert entry.depth == 3
    assert 123 in tt
    assert tt.get(999) is None


def test_tt_keeps_deeper_entry() -> None:
    tt = TranspositionTable()
    tt.store(7, 0.1, depth=2)
    tt.store(7, 0.9, depth=5)
    tt.store(7, -1.0, depth=1)  # shallower: ignored
    entry = tt.get(7)
    assert entry is not None
    assert entry.depth == 5
    assert entry.value == 0.9


def test_tt_stores_node_type() -> None:
    tt = TranspositionTable()
    tt.store(1, 0.3, depth=4, node_type=NodeType.LOWER)
    entry = tt.get(1)
    assert entry is not None
    assert entry.node_type == NodeType.LOWER


def test_tt_evicts_when_full() -> None:
    tt = TranspositionTable(capacity=2)
    tt.store(1, 0.0, depth=1)
    tt.store(2, 0.0, depth=1)
    tt.store(3, 0.0, depth=1)  # evicts oldest (key 1)
    assert len(tt) == 2
    assert 1 not in tt
    assert 3 in tt


# --- turn / scopa hashing ------------------------------------------------


def test_turn_toggle_changes_hash_reversibly() -> None:
    eng = ScopaEngine()
    base = eng.zhash
    eng._set_turn(1)
    assert eng.zhash != base
    assert eng.zhash == eng._recompute_hash()
    eng._set_turn(0)
    assert eng.zhash == base


def test_scopa_increment_changes_hash() -> None:
    eng = ScopaEngine()
    base = eng.zhash
    eng._add_scopa(0)
    assert eng.zhash != base
    assert eng.zhash == eng._recompute_hash()


def test_same_layout_different_turn_distinct_hash() -> None:
    a = ScopaEngine()
    b = ScopaEngine()
    b._set_turn(1)
    assert a.zhash != b.zhash  # side-to-move disambiguated
