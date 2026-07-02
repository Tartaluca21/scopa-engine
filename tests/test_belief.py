"""Tests for the Bayesian belief system over the opponent's hidden hand."""

from __future__ import annotations

import numpy as np

from cognitive.belief import BeliefSystem
from engine.cards import HAND_ZONES, Suit, Zone, card_index
from engine.core import ScopaEngine


def _dealt_engine(seed: int = 0) -> ScopaEngine:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(seed))
    return eng


def test_prior_sums_to_opponent_hand_size() -> None:
    eng = _dealt_engine()
    bs = BeliefSystem(bot_player=0)
    bs.update_on_deal(eng)
    assert np.isclose(bs.probs.sum(), eng.count(Zone.MANO_P2))


def test_bot_own_cards_and_table_have_zero_probability() -> None:
    eng = _dealt_engine()
    bs = BeliefSystem(bot_player=0)
    bs.update_on_deal(eng)
    for z in (Zone.MANO_P1, Zone.TAVOLO):
        for c in eng.cards_in(z):
            assert bs.probs[int(c)] == 0.0


def test_prior_is_uniform_over_hidden_cards() -> None:
    eng = _dealt_engine()
    bs = BeliefSystem(bot_player=0)
    bs.update_on_deal(eng)
    hidden = np.concatenate([eng.cards_in(Zone.MAZZO), eng.cards_in(Zone.MANO_P2)])
    vals = bs.probs[hidden]
    assert np.allclose(vals, vals[0])
    assert vals[0] > 0.0


def test_played_card_zeroed_and_sum_decremented() -> None:
    eng = _dealt_engine()
    bs = BeliefSystem(bot_player=0)
    bs.update_on_deal(eng)
    before = bs.opp_hand_size
    played = int(eng.cards_in(Zone.MANO_P2)[0])
    bs.update_on_opponent_play(played, table_before_play=[])
    assert bs.probs[played] == 0.0
    assert np.isclose(bs.probs.sum(), before - 1)


def test_liscio_does_not_zero_forced_capturers() -> None:
    # Table has a single 7; opponent lays a 3 (a liscio). Holding a 7 and
    # playing a waiting card is legal, so 7s must NOT be eliminated.
    bs = BeliefSystem(bot_player=0)
    bs.candidates = np.ones(40, dtype=bool)
    bs.probs = np.ones(40, dtype=np.float64)
    bs.opp_hand_size = 2
    three_b = card_index(Suit.BASTONI, 3)
    bs.candidates[three_b] = True
    table = [card_index(Suit.DENARI, 7)]
    bs.update_on_opponent_play(three_b, table_before_play=table)
    for suit in range(4):
        assert bs.probs[card_index(Suit(suit), 7)] > 0.0
    assert np.isclose(bs.probs.sum(), bs.opp_hand_size)


def test_liscio_keeps_capturers_uniform() -> None:
    # A liscio carries no info about the rest of the hand: capturing cards keep
    # the same probability as neutral cards (pure hand-size renormalization).
    bs = BeliefSystem(bot_player=0)
    bs.candidates = np.ones(40, dtype=bool)
    bs.probs = np.ones(40, dtype=np.float64)
    bs.opp_hand_size = 3
    table = [card_index(Suit.DENARI, 3), card_index(Suit.DENARI, 4)]  # combo sums to 7
    laid = card_index(Suit.COPPE, 1)
    seven = card_index(Suit.BASTONI, 7)  # combo capturer
    neutral = card_index(Suit.BASTONI, 9)  # captures nothing
    bs.update_on_opponent_play(laid, table_before_play=table)
    assert np.isclose(bs.probs[seven], bs.probs[neutral])
    assert bs.probs[seven] > 0.0


def test_played_card_becomes_visible_regardless_of_capture() -> None:
    # The only certain effect of an opponent move: the played card leaves hand.
    bs = BeliefSystem(bot_player=0)
    bs.candidates = np.ones(40, dtype=bool)
    bs.probs = np.ones(40, dtype=np.float64)
    bs.opp_hand_size = 2
    table = [card_index(Suit.DENARI, 7)]
    played_seven = card_index(Suit.COPPE, 7)
    bs.update_on_opponent_play(played_seven, table_before_play=table)
    assert bs.probs[played_seven] == 0.0
    assert bs.probs[card_index(Suit.BASTONI, 7)] > 0.0  # other 7s untouched


def test_endgame_certainty_on_talon_exhaustion() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(1))
    # Drain the deck so only the two hands remain hidden/unrevealed.
    for c in [int(x) for x in eng.cards_in(Zone.MAZZO)]:
        eng.move(c, Zone.MAZZO, Zone.PRESE_P1)
    bs = BeliefSystem(bot_player=0)
    bs.update_on_deal(eng)
    assert bs.certain
    for c in eng.cards_in(HAND_ZONES[1]):
        assert bs.probs[int(c)] == 1.0
    assert np.isclose(bs.probs.sum(), eng.count(HAND_ZONES[1]))


def test_endgame_opponent_play_keeps_others_certain() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(2))
    for c in [int(x) for x in eng.cards_in(Zone.MAZZO)]:
        eng.move(c, Zone.MAZZO, Zone.PRESE_P1)
    bs = BeliefSystem(bot_player=0)
    bs.update_on_deal(eng)
    opp = [int(c) for c in eng.cards_in(HAND_ZONES[1])]
    bs.update_on_opponent_play(opp[0], table_before_play=eng.cards_in(Zone.TAVOLO))
    assert bs.probs[opp[0]] == 0.0
    for c in opp[1:]:
        assert bs.probs[c] == 1.0


def test_get_probabilities_sorted_descending() -> None:
    bs = BeliefSystem(bot_player=0)
    bs.probs = np.zeros(40, dtype=np.float64)
    bs.probs[5] = 0.2
    bs.probs[9] = 0.9
    bs.probs[1] = 0.5
    out = bs.get_opponent_hand_probabilities()
    assert list(out.keys()) == [9, 1, 5]
    assert list(out.values()) == sorted(out.values(), reverse=True)


def test_impossible_state_fallback_is_uniform() -> None:
    bs = BeliefSystem(bot_player=0)
    bs.candidates = np.zeros(40, dtype=bool)
    bs.candidates[[10, 11, 12]] = True
    bs.probs = np.zeros(40, dtype=np.float64)  # all mass gone
    bs.opp_hand_size = 2
    bs._normalize()
    assert np.isclose(bs.probs.sum(), 2)
    assert np.allclose(bs.probs[[10, 11, 12]], 2 / 3)


def test_belief_does_not_mutate_engine_state() -> None:
    eng = _dealt_engine()
    snapshot = eng.state.copy()
    zhash = eng.zhash
    bs = BeliefSystem(bot_player=0)
    bs.update_on_deal(eng)
    bs.get_opponent_hand_probabilities()
    assert np.array_equal(eng.state, snapshot)
    assert eng.zhash == zhash
