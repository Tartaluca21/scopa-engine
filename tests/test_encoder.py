"""Tests for the POV-normalized state encoder (Fase 9, stage 1)."""

from __future__ import annotations

import numpy as np

from engine.cards import HAND_ZONES, N_CARDS, PRESE_ZONES, Suit, Zone, card_index
from engine.core import ScopaEngine
from learning.encoder import FEATURE_DIM, encode_state


def _dealt(seed: int = 0) -> ScopaEngine:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(seed))
    return eng


def test_shape_and_dtype() -> None:
    vec = encode_state(_dealt(), 0)
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_rejects_bad_player() -> None:
    import pytest

    with pytest.raises(ValueError, match="player"):
        encode_state(_dealt(), 2)


def test_is_deterministic_and_readonly() -> None:
    eng = _dealt(5)
    before = eng.state.copy()
    v1 = encode_state(eng, 0)
    v2 = encode_state(eng, 0)
    assert np.array_equal(v1, v2)
    assert np.array_equal(eng.state, before)  # encoder never mutates the engine


def test_pov_swaps_me_and_opp_planes() -> None:
    # Encoding for player 0 vs player 1 must swap the my/opp card planes.
    eng = _dealt(2)
    v0 = encode_state(eng, 0)
    v1 = encode_state(eng, 1)
    my0, opp0 = v0[:N_CARDS], v0[N_CARDS : 2 * N_CARDS]
    my1, opp1 = v1[:N_CARDS], v1[N_CARDS : 2 * N_CARDS]
    assert np.array_equal(my0, opp1)
    assert np.array_equal(opp0, my1)
    # Shared planes (table, deck) are POV-invariant.
    assert np.array_equal(v0[2 * N_CARDS : 4 * N_CARDS], v1[2 * N_CARDS : 4 * N_CARDS])


def test_hand_plane_matches_engine() -> None:
    eng = _dealt(3)
    vec = encode_state(eng, 0)
    my_hand_plane = vec[:N_CARDS]
    for c in range(N_CARDS):
        expected = 1.0 if eng.state[HAND_ZONES[0], c] else 0.0
        assert my_hand_plane[c] == expected


def test_scalar_components_reflect_captures() -> None:
    # Give player 0 the settebello + another denaro; scalars should register it.
    eng = ScopaEngine()
    eng.state[Zone.MAZZO, :] = 0
    for c in (card_index(Suit.DENARI, 7), card_index(Suit.DENARI, 3)):
        eng.state[PRESE_ZONES[0], c] = 1
    eng.rehash()
    v0 = encode_state(eng, 0)
    v1 = encode_state(eng, 1)
    scal0 = v0[6 * N_CARDS :]
    scal1 = v1[6 * N_CARDS :]
    # Index 10 = my settebello flag; 8 = my denari (normalized by 10).
    assert scal0[10] == 1.0 and scal1[11] == 1.0  # settebello is p0's from both POVs
    assert np.isclose(scal0[8], 2 / 10.0)  # two denari captured by p0
    assert scal1[9] == scal0[8]  # opp-denari for p1 == my-denari for p0
