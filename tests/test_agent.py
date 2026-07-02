"""Phase 5 tests: SearchAgent legality and baseline dominance over RandomBot."""

from __future__ import annotations

import numpy as np

from engine.cards import Suit, Zone, card_index
from engine.core import ScopaEngine
from engine.heuristic import Weights, simulate_match
from search.agent import RandomBot, SearchAgent


def test_search_agent_returns_legal_move() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(0))
    agent = SearchAgent(Weights(), np.random.default_rng(1), n_worlds=4, max_depth=4)
    card, cap = agent.select(eng, 0)
    assert eng.state[Zone.MANO_P1, card] == 1
    for c in cap:
        assert eng.state[Zone.TAVOLO, c] == 1


def test_random_bot_returns_legal_move() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(0))
    card, cap = RandomBot(np.random.default_rng(3)).select(eng, 0)
    assert eng.state[Zone.MANO_P1, card] == 1


def test_search_agent_takes_obvious_capture() -> None:
    eng = ScopaEngine()
    eng.state[Zone.MAZZO, :] = 0
    seven = card_index(Suit.DENARI, 7)
    table_seven = card_index(Suit.COPPE, 7)
    for idx in (seven, card_index(Suit.BASTONI, 3), card_index(Suit.SPADE, 2)):
        eng.state[Zone.MANO_P1, idx] = 1
    eng.state[Zone.TAVOLO, table_seven] = 1
    for idx in (card_index(Suit.BASTONI, 5), card_index(Suit.SPADE, 6), card_index(Suit.COPPE, 9)):
        eng.state[Zone.MANO_P2, idx] = 1
    eng.rehash()
    agent = SearchAgent(Weights(), np.random.default_rng(2), n_worlds=6, max_depth=8)
    card, cap = agent.select(eng, 0)
    assert card == seven
    assert cap == [table_seven]


def test_search_agent_beats_random_baseline() -> None:
    """Over several seeded deals, PIMC with basic weights outscores random."""
    search_total = 0.0
    random_total = 0.0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        agent = SearchAgent(Weights(), rng, n_worlds=6, max_depth=4)
        baseline = RandomBot(rng)
        sa, sb = simulate_match(agent, baseline, rng)
        search_total += sa
        random_total += sb
    assert search_total > random_total
