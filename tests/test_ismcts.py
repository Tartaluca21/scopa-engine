"""Phase 6 tests: ISMCTS stability, non-mutation, and obvious-mate selection."""

from __future__ import annotations

import numpy as np

from cognitive.belief import BeliefSystem
from engine.cards import HAND_ZONES, Suit, Zone, card_index
from engine.core import ScopaEngine
from search.alphabeta import legal_moves
from search.ismcts import ismcts_decide


def _belief_for(engine: ScopaEngine, player: int) -> BeliefSystem:
    bs = BeliefSystem(bot_player=player)
    bs.update_on_deal(engine)
    return bs


def _place(engine: ScopaEngine, suit: Suit, value: int, dst: Zone) -> int:
    idx = card_index(suit, value)
    engine.state[dst, idx] = 1
    return idx


def test_ismcts_returns_legal_move_without_mutating_state() -> None:
    # Mid-game state: deal, then play a couple of plies to add captures/piles.
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(7))
    for _ in range(2):
        player = eng.current_player
        move = legal_moves(eng, player)[0]
        eng.execute_move(move[0], move[1])

    snapshot = eng.state.copy()
    zhash = eng.zhash
    turn = eng.current_player
    belief = _belief_for(eng, eng.current_player)

    card, cap = ismcts_decide(eng, belief, max_time_ms=500, rng=np.random.default_rng(2))

    # Legal move: card is in hand, and every captured card sits on the table.
    assert eng.state[HAND_ZONES[turn], card] == 1
    for c in cap:
        assert eng.state[Zone.TAVOLO, c] == 1
    # The root engine is untouched (state matrix, hash, and turn).
    assert np.array_equal(eng.state, snapshot)
    assert eng.zhash == zhash
    assert eng.current_player == turn


def test_ismcts_finds_obvious_scopa_mate() -> None:
    # Deterministic endgame (deck empty -> belief is certain). The bot (player 0)
    # can either clear the table for a Scopa or throw a useless ace. Only the
    # Scopa move wins the deal, so ISMCTS must reliably pick it.
    eng = ScopaEngine()
    eng.state[Zone.MAZZO, :] = 0
    four_table = _place(eng, Suit.COPPE, 4, Zone.TAVOLO)
    four_hand = _place(eng, Suit.DENARI, 4, Zone.MANO_P1)
    _place(eng, Suit.BASTONI, 1, Zone.MANO_P1)  # ace: captures nothing
    _place(eng, Suit.SPADE, 4, Zone.MANO_P2)
    eng.current_player = 0
    eng.rehash()

    belief = _belief_for(eng, 0)
    assert belief.certain  # talon exhausted -> opponent hand fully known

    card, cap = ismcts_decide(
        eng, belief, max_time_ms=10_000, max_iter=1000, rng=np.random.default_rng(0)
    )

    assert card == four_hand
    assert cap == [four_table]


def test_ismcts_single_move_is_returned_immediately() -> None:
    # Only one legal move -> returned without spending the search budget.
    eng = ScopaEngine()
    eng.state[Zone.MAZZO, :] = 0
    ace = _place(eng, Suit.BASTONI, 1, Zone.MANO_P1)
    _place(eng, Suit.SPADE, 9, Zone.TAVOLO)  # no value-1 capture available
    _place(eng, Suit.COPPE, 5, Zone.MANO_P2)
    eng.current_player = 0
    eng.rehash()

    card, cap = ismcts_decide(eng, _belief_for(eng, 0), max_iter=1)
    assert card == ace
    assert cap == []
