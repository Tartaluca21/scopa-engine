"""Stress tests for BeliefSystem and the PIMC determinization weighting.

Covers the mathematically hostile corners: impossible states, endgame
certainty, weight exhaustion in the sampler, and mandatory-capture isolation.
"""

from __future__ import annotations

import numpy as np

from cognitive.belief import BeliefSystem
from engine.cards import HAND_ZONES, N_CARDS, Suit, Zone, card_index
from engine.core import ScopaEngine
from engine.masks import is_consistent
from search.determinize import _sample_opp_hand, determinize


def _drain_deck(eng: ScopaEngine) -> None:
    """Move every remaining deck card into a pile so the talon hits zero."""
    for c in [int(x) for x in eng.cards_in(Zone.MAZZO)]:
        eng.move(c, Zone.MAZZO, Zone.PRESE_P1)


# --- 1. Impossible state recovery ---------------------------------------


def test_impossible_played_card_triggers_epsilon_fallback() -> None:
    # Impossible state: the distribution has collapsed to all-zero mass while a
    # hand is still expected, and the opponent plays a card the belief rated
    # 0.0. The normalizer must avoid a divide-by-zero and restore a uniform
    # prior over the surviving candidates instead of producing NaNs.
    bs = BeliefSystem(bot_player=0)
    played = card_index(Suit.SPADE, 3)
    b = card_index(Suit.DENARI, 5)
    c = card_index(Suit.COPPE, 6)
    bs.candidates = np.zeros(N_CARDS, dtype=bool)
    bs.candidates[[played, b, c]] = True
    bs.opp_hand_size = 2
    bs.probs = np.zeros(N_CARDS, dtype=np.float64)  # total mass has vanished
    assert bs.probs[played] == 0.0  # the "impossible" card

    bs.update_on_opponent_play(played, table_before_play=[])

    assert np.all(np.isfinite(bs.probs))  # no NaN/inf from a 0/0 division
    assert bs.probs[played] == 0.0
    assert np.isclose(bs.probs.sum(), bs.opp_hand_size)
    # Fallback redistributes the target mass uniformly over live candidates.
    assert np.isclose(bs.probs[b], 0.5)
    assert np.isclose(bs.probs[c], 0.5)


def test_normalize_survives_total_wipeout() -> None:
    # Directly force every probability to zero while a hand is still expected.
    bs = BeliefSystem(bot_player=0)
    bs.candidates = np.zeros(N_CARDS, dtype=bool)
    bs.candidates[[10, 20, 30]] = True
    bs.opp_hand_size = 1
    bs.probs = np.zeros(N_CARDS, dtype=np.float64)
    bs._normalize()
    assert np.all(np.isfinite(bs.probs))
    assert np.isclose(bs.probs.sum(), 1)
    assert np.allclose(bs.probs[[10, 20, 30]], 1 / 3)


# --- 2. Endgame certainty stability -------------------------------------


def test_talon_exhaustion_snaps_to_certainty() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(1))
    _drain_deck(eng)
    assert eng.count(Zone.MAZZO) == 0

    bs = BeliefSystem(bot_player=0)
    bs.update_on_deal(eng)
    assert bs.certain
    opp = [int(c) for c in eng.cards_in(HAND_ZONES[1])]
    for c in opp:
        assert bs.probs[c] == 1.0
    assert np.isclose(bs.probs.sum(), len(opp))


def test_certainty_persists_across_opponent_plays() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(2))
    _drain_deck(eng)
    bs = BeliefSystem(bot_player=0)
    bs.update_on_deal(eng)
    opp = [int(c) for c in eng.cards_in(HAND_ZONES[1])]
    table = eng.cards_in(Zone.TAVOLO)

    # Play the certain cards one by one; the rest must stay pinned at 1.0.
    for i, played in enumerate(opp[:-1]):
        bs.update_on_opponent_play(played, table_before_play=table)
        remaining = opp[i + 1 :]
        assert bs.certain
        assert bs.probs[played] == 0.0
        for c in remaining:
            assert bs.probs[c] == 1.0
        assert np.isclose(bs.probs.sum(), len(remaining))
        assert np.isclose(bs.probs.sum(), bs.opp_hand_size)


# --- 3. Determinization weight exhaustion -------------------------------


def test_sample_opp_hand_falls_back_when_support_too_small() -> None:
    pool = np.array([3, 7, 11, 15, 19], dtype=np.intp)
    n_opp = 3
    weights = np.zeros(N_CARDS, dtype=np.float64)
    weights[3] = 9.0  # only ONE positive weight, but three cards are needed
    for seed in range(30):
        picked = _sample_opp_hand(pool, n_opp, np.random.default_rng(seed), weights)
        assert picked.size == n_opp
        assert len(set(int(x) for x in picked)) == n_opp  # no duplicates
        assert set(int(x) for x in picked).issubset(set(int(x) for x in pool))


def test_sample_opp_hand_handles_all_zero_weights() -> None:
    pool = np.array([0, 1, 2, 3], dtype=np.intp)
    weights = np.zeros(N_CARDS, dtype=np.float64)  # zero mass everywhere
    picked = _sample_opp_hand(pool, 2, np.random.default_rng(0), weights)
    assert picked.size == 2
    assert set(int(x) for x in picked).issubset({0, 1, 2, 3})


def test_determinize_recovers_from_starved_belief() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(4))
    n_opp = eng.count(HAND_ZONES[1])
    hidden = np.concatenate([eng.cards_in(Zone.MAZZO), eng.cards_in(HAND_ZONES[1])])
    weights = np.zeros(N_CARDS, dtype=np.float64)
    weights[int(hidden[0])] = 1.0  # a single believable card, but n_opp == 3
    for seed in range(15):
        world = determinize(eng, 0, np.random.default_rng(seed), weights)
        assert is_consistent(world)
        assert world.count(HAND_ZONES[1]) == n_opp
        assert world.state.sum() == N_CARDS


# --- 4. Liscio carries no hand-wide information --------------------------


def test_liscio_does_not_eliminate_capturing_cards() -> None:
    # Table: 7, 4 and 3 (each a single). The opponent plays a 2 -- a liscio.
    # In Scopa the mandatory-capture rule binds ONLY the card actually played;
    # keeping a 7 in hand while playing a waiting 2 is a legal bluff. So the
    # belief must NOT zero any capturing card: after the move every remaining
    # candidate stays possible and uniform (pure hand-size renormalization).
    bs = BeliefSystem(bot_player=0)
    table = [
        card_index(Suit.DENARI, 7),
        card_index(Suit.COPPE, 4),
        card_index(Suit.BASTONI, 3),
    ]
    sevens = [card_index(s, 7) for s in (Suit.COPPE, Suit.BASTONI, Suit.SPADE)]
    safe = [card_index(Suit.DENARI, v) for v in (5, 6, 8, 9)]
    played = card_index(Suit.SPADE, 2)

    bs.candidates = np.zeros(N_CARDS, dtype=bool)
    bs.candidates[sevens + safe + [played]] = True
    bs.opp_hand_size = 3
    n = int(bs.candidates.sum())
    bs.probs = np.zeros(N_CARDS, dtype=np.float64)
    bs.probs[bs.candidates] = bs.opp_hand_size / n

    bs.update_on_opponent_play(played, table_before_play=table)

    survivors = sevens + safe
    for s in sevens:  # capturing cards remain fully plausible -> NOT zeroed
        assert bs.probs[s] > 0.0
    # No card is privileged: capturers and neutrals share one uniform value.
    assert np.allclose(bs.probs[survivors], bs.probs[survivors[0]])
    assert bs.probs[played] == 0.0
    assert np.isclose(bs.probs.sum(), bs.opp_hand_size)
