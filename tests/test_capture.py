"""Tests for read-only per-decision capture from the live engine."""

from __future__ import annotations

import numpy as np

from capture import cards_seen, decision_record, legal_moves
from engine.cards import HAND_ZONES, Zone
from engine.core import ScopaEngine


def _fresh_engine() -> ScopaEngine:
    engine = ScopaEngine()
    engine.deal_round(np.random.default_rng(42))
    return engine


def test_legal_moves_cover_hand() -> None:
    engine = _fresh_engine()
    moves = legal_moves(engine, 0)
    hand = {int(c) for c in engine.cards_in(HAND_ZONES[0])}
    assert {m[0] for m in moves} == hand  # every hand card has at least one move
    assert all(isinstance(m[1], list) for m in moves)  # captures are JSON lists


def test_cards_seen_excludes_opponent_hand() -> None:
    engine = _fresh_engine()
    seen = set(cards_seen(engine, 0))
    opp_hand = {int(c) for c in engine.cards_in(HAND_ZONES[1])}
    own_hand = {int(c) for c in engine.cards_in(HAND_ZONES[0])}
    table = {int(c) for c in engine.cards_in(Zone.TAVOLO)}
    assert own_hand <= seen and table <= seen
    assert seen.isdisjoint(opp_hand)  # hidden information stays hidden


def test_decision_record_snapshot_is_pre_move() -> None:
    engine = _fresh_engine()
    hash_before = engine.zhash
    card, cap = legal_moves(engine, 0)[0][0], legal_moves(engine, 0)[0][1]
    rec = decision_record(engine, "human", 0, 3, card, cap)
    assert engine.zhash == hash_before  # capture never mutates the engine
    assert rec["player"] == "human"
    assert rec["turn"] == 3
    assert rec["chosen"] == [card, cap]
    assert card in rec["hand"]  # type: ignore[operator]
    assert len(rec["partial_score"]) == 2  # type: ignore[arg-type]
    assert set(rec.keys()) == {
        "turn",
        "player",
        "hand",
        "table",
        "legal_moves",
        "chosen",
        "partial_score",
        "cards_seen",
    }


def test_bot_decision_marked_bot() -> None:
    engine = _fresh_engine()
    rec = decision_record(engine, "bot", 1, 0, int(engine.cards_in(HAND_ZONES[1])[0]), [])
    assert rec["player"] == "bot"
