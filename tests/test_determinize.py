"""Phase 4 tests: determinization preserves known info and card counts."""

from __future__ import annotations

import numpy as np

from engine.cards import HAND_ZONES, N_CARDS, Zone
from engine.core import ScopaEngine
from search.determinize import determinize


def _dealt(seed: int = 0) -> ScopaEngine:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(seed))
    return eng


def test_determinize_preserves_known_zones() -> None:
    eng = _dealt()
    rng = np.random.default_rng(123)
    world = determinize(eng, player=0, rng=rng)
    # Player 0's hand, the table, and both piles are untouched.
    for z in (Zone.MANO_P1, Zone.TAVOLO, Zone.PRESE_P1, Zone.PRESE_P2):
        assert np.array_equal(world.state[z], eng.state[z])
    assert world.current_player == eng.current_player
    assert world.last_capturer == eng.last_capturer
    assert np.array_equal(world.scopa_counts, eng.scopa_counts)


def test_determinize_preserves_counts_and_consistency() -> None:
    eng = _dealt(7)
    world = determinize(eng, player=0, rng=np.random.default_rng(1))
    assert world.is_consistent()
    assert world.count(Zone.MAZZO) == eng.count(Zone.MAZZO)
    assert world.count(HAND_ZONES[1]) == eng.count(HAND_ZONES[1])
    assert world.state.sum() == N_CARDS


def test_determinize_recomputes_hash() -> None:
    eng = _dealt(3)
    world = determinize(eng, player=1, rng=np.random.default_rng(9))
    assert world.zhash == world._recompute_hash()


def test_determinize_is_independent_copy() -> None:
    eng = _dealt(5)
    world = determinize(eng, player=0, rng=np.random.default_rng(2))
    world.state[Zone.MAZZO, :] = 0
    assert eng.count(Zone.MAZZO) > 0  # original untouched


def test_determinize_varies_hidden_cards() -> None:
    eng = _dealt(11)
    a = determinize(eng, player=0, rng=np.random.default_rng(0))
    b = determinize(eng, player=0, rng=np.random.default_rng(1))
    # Hidden assignment differs across seeds (overwhelmingly likely).
    assert not np.array_equal(a.state[HAND_ZONES[1]], b.state[HAND_ZONES[1]])


def test_determinize_respects_belief_weights() -> None:
    # Zero-weight hidden cards must never be dealt into the opponent's hand.
    eng = _dealt(4)
    pool = np.concatenate([eng.cards_in(Zone.MAZZO), eng.cards_in(HAND_ZONES[1])])
    weights = np.zeros(N_CARDS, dtype=np.float64)
    forbidden = int(pool[0])
    weights[pool] = 1.0
    weights[forbidden] = 0.0  # believed impossible to be in the opp hand
    for seed in range(20):
        world = determinize(eng, 0, np.random.default_rng(seed), weights)
        assert world.state[HAND_ZONES[1], forbidden] == 0
        assert world.count(HAND_ZONES[1]) == eng.count(HAND_ZONES[1])
