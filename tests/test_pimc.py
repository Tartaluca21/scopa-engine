"""Phase 4 tests: PIMC coordinator move selection."""

from __future__ import annotations

import numpy as np

from engine.cards import Suit, Zone, card_index
from engine.core import ScopaEngine
from search.alphabeta import SearchConfig
from search.pimc import PimcConfig, pimc_decide


def _place_from_void(eng: ScopaEngine, suit: Suit, value: int, dst: Zone) -> int:
    idx = card_index(suit, value)
    eng.state[dst, idx] = 1
    return idx


def test_pimc_returns_legal_move() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(0))
    cfg = PimcConfig(n_worlds=4, search=SearchConfig(max_depth=4))
    card, cap = pimc_decide(eng, 0, cfg, np.random.default_rng(1))
    assert eng.state[Zone.MANO_P1, card] == 1
    for c in cap:
        assert eng.state[Zone.TAVOLO, c] == 1


def test_pimc_takes_available_capture() -> None:
    eng = ScopaEngine()
    eng.state[Zone.MAZZO, :] = 0
    seven = _place_from_void(eng, Suit.DENARI, 7, Zone.MANO_P1)
    table_seven = _place_from_void(eng, Suit.COPPE, 7, Zone.TAVOLO)
    _place_from_void(eng, Suit.BASTONI, 3, Zone.MANO_P1)
    _place_from_void(eng, Suit.SPADE, 2, Zone.MANO_P1)
    _place_from_void(eng, Suit.BASTONI, 5, Zone.MANO_P2)
    _place_from_void(eng, Suit.SPADE, 6, Zone.MANO_P2)
    _place_from_void(eng, Suit.COPPE, 9, Zone.MANO_P2)
    eng.rehash()
    cfg = PimcConfig(n_worlds=6, search=SearchConfig(max_depth=8))
    card, cap = pimc_decide(eng, 0, cfg, np.random.default_rng(2))
    assert card == seven
    assert cap == [table_seven]


def test_pimc_rejects_wrong_turn() -> None:
    eng = ScopaEngine()
    eng.deal_round(np.random.default_rng(0))
    cfg = PimcConfig(n_worlds=2)
    try:
        pimc_decide(eng, 1, cfg, np.random.default_rng(0))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong turn")
