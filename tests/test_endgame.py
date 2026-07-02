"""Tests for the exact deck-empty endgame solver and its PIMC integration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.cards import HAND_ZONES, N_CARDS, Suit, Zone, card_index
from engine.core import ScopaEngine
from engine.heuristic import score_deal
from engine.transposition import TranspositionTable
from search.alphabeta import SearchConfig, alphabeta
from search.endgame import solve_endgame


def _empty_deck_engine() -> ScopaEngine:
    eng = ScopaEngine()
    eng.state[Zone.MAZZO, :] = 0
    return eng


def _put(eng: ScopaEngine, suit: Suit, value: int, dst: Zone) -> int:
    idx = card_index(suit, value)
    eng.state[dst, idx] = 1
    return idx


def _brute(engine: ScopaEngine) -> float:
    """Independent, un-memoized negamax to terminal (cross-check for solver)."""
    if engine.is_game_over():
        final = engine.clone()
        final.end_of_deal_sweep()
        p0, p1 = score_deal(final)
        return (p0 - p1) if engine.current_player == 0 else (p1 - p0)
    best = -math.inf
    player = engine.current_player
    for card in engine.cards_in(HAND_ZONES[player]):
        c = int(card)
        options = engine.captures_for(c)
        caps = [[int(x) for x in o] for o in options] if options else [[]]
        for cap in caps:
            child = engine.clone()
            child.execute_move(c, cap)
            best = max(best, -_brute(child))
    return best


def test_forced_line_exact_margin_and_last_play_no_scopa() -> None:
    # Deck empty, table empty, one card each. P0 must lay the settebello (7D);
    # P1 is then forced to capture it (single equal-value rule). That capture
    # clears the table but is the LAST play of the deal -> NO scopa. P1 ends with
    # {7D, 7C}: carte + denari + settebello + primiera = 4 pts, scope 0. Margin
    # from P0's POV = 0 - 4 = -4. A wrongly-awarded last-play scopa would give -5.
    eng = _empty_deck_engine()
    _put(eng, Suit.DENARI, 7, Zone.MANO_P1)
    _put(eng, Suit.COPPE, 7, Zone.MANO_P2)
    eng.rehash()
    assert solve_endgame(eng) == -4.0


def test_solver_matches_full_alphabeta_on_endgame() -> None:
    # An exhaustive alpha-beta (no depth cutoff reached) must equal the solver.
    eng = _empty_deck_engine()
    _put(eng, Suit.DENARI, 7, Zone.MANO_P1)
    _put(eng, Suit.COPPE, 7, Zone.TAVOLO)
    _put(eng, Suit.BASTONI, 3, Zone.MANO_P2)
    eng.rehash()
    tt = TranspositionTable()
    full = alphabeta(eng, 8, -math.inf, math.inf, tt, SearchConfig(max_depth=8))
    assert solve_endgame(eng) == full == _brute(eng)


def test_requires_empty_deck() -> None:
    eng = ScopaEngine()  # full deck
    with pytest.raises(ValueError, match="empty deck"):
        solve_endgame(eng)


def test_memoization_is_consistent() -> None:
    eng = _empty_deck_engine()
    _put(eng, Suit.DENARI, 5, Zone.MANO_P1)
    _put(eng, Suit.COPPE, 3, Zone.MANO_P1)
    _put(eng, Suit.BASTONI, 5, Zone.TAVOLO)
    _put(eng, Suit.SPADE, 3, Zone.MANO_P2)
    _put(eng, Suit.DENARI, 2, Zone.MANO_P2)
    eng.rehash()
    shared: dict[tuple[int, int], float] = {}
    first = solve_endgame(eng, shared)
    second = solve_endgame(eng, shared)  # served from memo
    fresh = solve_endgame(eng, {})
    assert first == second == fresh == _brute(eng)


def test_last_capturer_changes_sweep_value() -> None:
    # Board-identical terminal states (deck + both hands empty, 7D left on the
    # table) that differ ONLY in last_capturer: the sweep awards 7D to that
    # player -> +4 vs -4 from P0's POV. Since zhash ignores last_capturer, this
    # is exactly the collision that must NOT be memoized together.
    def _leftover(last_capturer: int) -> ScopaEngine:
        eng = _empty_deck_engine()
        _put(eng, Suit.DENARI, 7, Zone.TAVOLO)
        eng.last_capturer = last_capturer
        eng.rehash()
        return eng

    p0, p1 = _leftover(0), _leftover(1)
    assert p0.zhash == p1.zhash  # identical board hash...
    assert solve_endgame(p0) == 4.0  # ...but different sweep outcomes
    assert solve_endgame(p1) == -4.0


def _random_endgame(rng: np.random.Generator, hand_size: int) -> ScopaEngine:
    """A random legal-shaped deck-empty state: balanced hands, rest split."""
    eng = _empty_deck_engine()
    cards = list(rng.permutation(N_CARDS))
    i = 0
    for _ in range(hand_size):
        eng.state[Zone.MANO_P1, cards[i]] = 1
        eng.state[Zone.MANO_P2, cards[i + 1]] = 1
        i += 2
    n_table = int(rng.integers(0, 4))
    for _ in range(n_table):
        eng.state[Zone.TAVOLO, cards[i]] = 1
        i += 1
    # Split a few already-captured cards so scoring is non-trivial.
    for _ in range(int(rng.integers(0, 6))):
        zone = Zone.PRESE_P1 if rng.integers(2) == 0 else Zone.PRESE_P2
        eng.state[zone, cards[i]] = 1
        i += 1
    eng.last_capturer = int(rng.integers(2))
    eng.rehash()
    return eng


@pytest.mark.parametrize("seed", range(40))
def test_solver_matches_brute_force_random_states(seed: int) -> None:
    rng = np.random.default_rng(seed)
    hand_size = int(rng.integers(1, 4))  # 1..3 cards per hand
    eng = _random_endgame(rng, hand_size)
    assert solve_endgame(eng) == _brute(eng)


def test_alphabeta_uses_solver_at_leaf_when_enabled() -> None:
    # At depth 0 on a deck-empty node, the plain search returns the heuristic;
    # the solver-enabled search returns the exact endgame value instead.
    eng = _empty_deck_engine()
    _put(eng, Suit.DENARI, 7, Zone.MANO_P1)
    _put(eng, Suit.COPPE, 7, Zone.TAVOLO)
    _put(eng, Suit.BASTONI, 3, Zone.MANO_P2)
    eng.rehash()
    exact = solve_endgame(eng)
    on = alphabeta(
        eng,
        0,
        -math.inf,
        math.inf,
        TranspositionTable(),
        SearchConfig(max_depth=0, use_endgame_solver=True),
    )
    assert on == exact
