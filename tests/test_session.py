"""Tests for the shared non-UI session logic (session.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.cards import PRESE_ZONES, Suit, card_index
from engine.core import ScopaEngine
from gamelog import DealRecord, read_deals, read_matches
from session import (
    MatchSession,
    SessionConfig,
    finalize_deal,
    match_decided,
    named_winner,
    pimc_bot_name,
)

SUITS = (Suit.DENARI, Suit.COPPE, Suit.BASTONI, Suit.SPADE)


def _suit_cards(suit: Suit) -> list[int]:
    return [card_index(suit, v) for v in range(1, 11)]


def _end_state(human: list[int], bot: list[int], scope_human: int = 0) -> ScopaEngine:
    """A finished deal with captures placed straight into the PRESE zones."""
    eng = ScopaEngine()
    eng.state[:] = 0
    eng.state[PRESE_ZONES[0], human] = 1
    eng.state[PRESE_ZONES[1], bot] = 1
    eng.scopa_counts = np.array([scope_human, 0], dtype=np.int64)
    eng.current_player = 0
    eng.last_capturer = 0
    eng.rehash()
    return eng


def _deal(human: float, bot: float, deal_id: int) -> DealRecord:
    return DealRecord(
        time="t",
        human=human,
        bot=bot,
        margin=human - bot,
        result="win" if human > bot else "loss" if bot > human else "tie",
        deal_id=deal_id,
    )


# --- helpers ---------------------------------------------------------------


def test_named_winner_maps_to_human_bot_vocab() -> None:
    assert named_winner("p0") == "human"
    assert named_winner("p1") == "bot"
    assert named_winner("none") == "none"


def test_pimc_bot_name_format() -> None:
    assert pimc_bot_name(12, 5) == "PIMC(n_worlds=12,max_depth=5)"


def test_session_config_defaults() -> None:
    cfg = SessionConfig()
    assert cfg.target is None
    assert cfg.record_moves is False


# --- match termination rule ------------------------------------------------


def test_match_decided_needs_target_and_no_tie() -> None:
    assert match_decided(12.0, 7.0, 11.0) is True
    assert match_decided(7.0, 12.0, 11.0) is True
    assert match_decided(6.0, 4.0, 11.0) is False  # nobody at target
    assert match_decided(11.0, 11.0, 11.0) is False  # tied at target -> play on
    assert match_decided(6.0, 6.0, 6.0) is False


# --- deal finalization + logging (GUI single-deal path) --------------------


def test_finalize_deal_logs_human_win(tmp_path: Path) -> None:
    log = tmp_path / "deals.jsonl"
    human = _suit_cards(Suit.DENARI)  # 10 denari incl. settebello
    bot = _suit_cards(Suit.COPPE) + _suit_cards(Suit.BASTONI) + _suit_cards(Suit.SPADE)
    record = finalize_deal(
        _end_state(human, bot, scope_human=2), deal_id=99, bot_name="botX", path=log
    )
    assert record.result == "win"
    assert record.human_scope == 2
    assert record.human > record.bot
    assert record.settebello_winner == "human"
    assert record.denari_winner == "human"
    assert record.cards_winner == "bot"  # 30 vs 10 cards
    assert record.deal_id == 99
    assert record.bot_name == "botX"
    (loaded,) = read_deals(log)
    assert loaded == record


def test_finalize_deal_does_not_mutate_engine(tmp_path: Path) -> None:
    eng = _end_state([card_index(Suit.DENARI, 7)], [card_index(Suit.COPPE, 3)])
    before = eng.state.copy()
    finalize_deal(eng, deal_id=1, bot_name="b", path=tmp_path / "d.jsonl")
    assert np.array_equal(eng.state, before)  # scoring ran on a clone


def test_finalize_deal_passes_moves_through(tmp_path: Path) -> None:
    log = tmp_path / "deals.jsonl"
    moves = [{"turn": 0, "player": "human", "chosen": [1, []]}]
    finalize_deal(
        _end_state([card_index(Suit.DENARI, 3)], [card_index(Suit.COPPE, 3)]),
        deal_id=5,
        bot_name="b",
        moves=moves,
        path=log,
    )
    (loaded,) = read_deals(log)
    assert loaded.moves == moves


# --- match accumulation + logging (GUI match path) -------------------------


def test_match_session_accumulates_and_logs(tmp_path: Path) -> None:
    log = tmp_path / "matches.jsonl"
    ms = MatchSession(target=11.0, bot_name="botX", match_id=7)
    ms.add_deal(_deal(6.0, 4.0, 101))
    assert ms.is_decided() is False
    ms.add_deal(_deal(6.0, 3.0, 102))
    assert ms.is_decided() is True
    record = ms.finish(log)
    assert record.human_match_score == 12.0
    assert record.bot_match_score == 7.0
    assert record.winner == "human"
    assert record.n_deals == 2
    assert record.deal_ids == [101, 102]
    (loaded,) = read_matches(log)
    assert loaded == record


def test_match_session_plays_through_tie_at_target(tmp_path: Path) -> None:
    ms = MatchSession(target=6.0, bot_name="b", match_id=1)
    ms.add_deal(_deal(6.0, 6.0, 1))  # tied at target -> undecided
    assert ms.is_decided() is False
    ms.add_deal(_deal(0.0, 4.0, 2))  # bot pulls ahead
    assert ms.is_decided() is True
    record = ms.finish(tmp_path / "m.jsonl")
    assert record.winner == "bot"
