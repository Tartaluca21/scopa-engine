"""Tests for the gated soft (rational-play) belief inference."""

from __future__ import annotations

import numpy as np

from cognitive.belief import BeliefSystem
from engine.cards import CARD_VALUES, Suit, Zone, card_index
from engine.core import ScopaEngine


def _dealt(seed: int = 0) -> ScopaEngine:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(seed))
    return eng


def test_soft_off_matches_hard_facts() -> None:
    # Default (soft=False) must be byte-identical to the hard-facts belief.
    eng = _dealt(3)
    hard = BeliefSystem(bot_player=0)
    soft_off = BeliefSystem(bot_player=0, soft=False)
    for bs in (hard, soft_off):
        bs.update_on_deal(eng)
    played = int(eng.cards_in(Zone.MANO_P2)[0])
    table = [int(c) for c in eng.cards_in(Zone.TAVOLO)]
    hard.update_on_opponent_play(played, table_before_play=table)
    soft_off.update_on_opponent_play(played, table_before_play=table)
    assert np.array_equal(hard.probs, soft_off.probs)


def test_declined_capture_downweights_matching_values_but_never_zeroes() -> None:
    eng = _dealt(1)
    bs = BeliefSystem(bot_player=0, soft=True, alpha=0.25, declined_penalty=0.5)
    bs.update_on_deal(eng)
    table = [int(c) for c in eng.cards_in(Zone.TAVOLO)]
    table_vals = {CARD_VALUES[c] for c in table}
    played = int(eng.cards_in(Zone.MANO_P2)[0])  # treat as a lay (captured empty)
    before = bs.probs.copy()
    bs.update_on_opponent_play(played, table_before_play=table, captured=[])
    cands = np.flatnonzero(bs.candidates)
    matching = [c for c in cands if CARD_VALUES[c] in table_vals]
    nonmatching = [c for c in cands if CARD_VALUES[c] not in table_vals]
    assert matching, "test needs at least one matching-value candidate"
    # Matching candidates drop relative to non-matching, yet stay strictly > 0.
    assert all(bs.probs[c] > 0.0 for c in cands)
    if nonmatching:
        assert max(bs.probs[c] for c in matching) < max(bs.probs[c] for c in nonmatching)
    assert np.isclose(bs.probs.sum(), bs.opp_hand_size)
    assert before[played] > 0.0 and bs.probs[played] == 0.0  # hard fact still applied


def test_capture_move_skips_declined_penalty() -> None:
    eng = _dealt(2)
    bs = BeliefSystem(bot_player=0, soft=True, alpha=0.25, declined_penalty=0.5)
    bs.update_on_deal(eng)
    table = [int(c) for c in eng.cards_in(Zone.TAVOLO)]
    played = int(eng.cards_in(Zone.MANO_P2)[0])
    bs.update_on_opponent_play(played, table_before_play=table, captured=[table[0]])
    # A capture reveals nothing about declined captures: still uniform over cands.
    vals = bs.probs[np.flatnonzero(bs.candidates)]
    assert np.allclose(vals, vals[0])


def test_goal_pref_upweights_denari() -> None:
    eng = _dealt(4)
    bs = BeliefSystem(bot_player=0, soft=True, alpha=0.25, goal_pref=0.5)
    bs.update_on_deal(eng)
    played = int(eng.cards_in(Zone.MANO_P2)[0])
    bs.update_on_opponent_play(played, table_before_play=[], captured=[])
    cands = np.flatnonzero(bs.candidates)
    denari = [c for c in cands if c < 10]
    other = [c for c in cands if c >= 10]
    if denari and other:
        assert min(bs.probs[c] for c in denari) >= max(bs.probs[c] for c in other) - 1e-9
    assert np.isclose(bs.probs.sum(), bs.opp_hand_size)


def test_soft_requires_valid_alpha() -> None:
    import pytest

    with pytest.raises(ValueError, match="alpha"):
        BeliefSystem(bot_player=0, soft=True, alpha=0.0)


def test_settebello_index_is_denari() -> None:
    # Guard the denari-mask assumption used by goal_pref (indices 0..9).
    assert card_index(Suit.DENARI, 7) < 10
