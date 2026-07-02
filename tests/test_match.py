"""Tests for match mode: accumulation, logging, CLI parsing, and reporting."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from gamelog import DealRecord, MatchRecord, log_match, match_winner, read_matches
from match_stats import summarize
from play import parse_args, run_match


def _deal(human: float, bot: float, deal_id: int) -> DealRecord:
    return DealRecord(
        time="t",
        human=human,
        bot=bot,
        margin=human - bot,
        result="win" if human > bot else "loss" if bot > human else "tie",
        deal_id=deal_id,
    )


def _provider(deals: list[DealRecord]) -> Callable[[], DealRecord]:
    it = iter(deals)

    def provide() -> DealRecord:
        return next(it)  # raises StopIteration if run_match over-pulls

    return provide


# --- default one-deal mode unchanged ---------------------------------------


def test_parse_args_default_is_one_deal() -> None:
    args = parse_args([])
    assert args.match_to is None
    assert args.record_moves is False


def test_parse_args_match_to() -> None:
    assert parse_args(["--match-to", "11"]).match_to == 11
    assert parse_args(["--match-to", "21"]).match_to == 21


def test_parse_args_record_moves_flag() -> None:
    assert parse_args(["--record-moves"]).record_moves is True


def test_parse_args_rejects_non_positive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--match-to", "0"])


# --- match accumulation to target ------------------------------------------


def test_match_normal_human_win() -> None:
    deals = [_deal(6.0, 4.0, 1), _deal(6.0, 3.0, 2), _deal(9.0, 9.0, 99)]
    played = run_match(11.0, _provider(deals))
    assert [r.deal_id for r in played] == [1, 2]  # stops at 12-7, third unused


def test_match_normal_bot_win() -> None:
    deals = [_deal(2.0, 6.0, 1), _deal(3.0, 6.0, 2)]
    played = run_match(11.0, _provider(deals))
    assert len(played) == 2
    assert sum(r.bot for r in played) == 12.0
    assert sum(r.human for r in played) == 5.0


def test_match_single_deal_exceeding_target() -> None:
    played = run_match(6.0, _provider([_deal(7.0, 1.0, 1)]))
    assert len(played) == 1


def test_match_continues_when_tied_at_target() -> None:
    # Deal 1 hits 6-6 (both >= target, tied) -> must NOT end; deal 2 breaks it.
    deals = [_deal(6.0, 6.0, 1), _deal(3.0, 0.0, 2)]
    played = run_match(6.0, _provider(deals))
    assert len(played) == 2
    assert sum(r.human for r in played) == 9.0
    assert sum(r.bot for r in played) == 6.0


def test_match_continues_through_repeated_ties() -> None:
    # 5-5 (tie, neither at target), 6-6 (tie, both past target), 8-6 decides.
    deals = [_deal(5.0, 5.0, 1), _deal(1.0, 1.0, 2), _deal(2.0, 0.0, 3)]
    played = run_match(5.0, _provider(deals))
    assert [r.deal_id for r in played] == [1, 2, 3]


def test_match_tie_resolves_for_bot() -> None:
    deals = [_deal(11.0, 11.0, 1), _deal(0.0, 4.0, 2)]
    played = run_match(11.0, _provider(deals))
    assert len(played) == 2
    assert sum(r.bot for r in played) == 15.0
    assert sum(r.human for r in played) == 11.0


# --- match logging ----------------------------------------------------------


def test_log_and_read_match_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "matches.jsonl"
    rec = log_match(
        match_id=42,
        target_score=11.0,
        human_match_score=12.0,
        bot_match_score=7.0,
        deal_ids=[101, 102],
        bot_name="PIMC(n_worlds=15,max_depth=6)",
        path=log,
    )
    (loaded,) = read_matches(log)
    assert loaded == rec
    assert loaded.winner == "human"
    assert loaded.final_margin == 5.0
    assert loaded.n_deals == 2
    assert loaded.deal_ids == [101, 102]


def test_match_winner_ties() -> None:
    assert match_winner(11.0, 11.0) == "tie"
    assert match_winner(11.0, 8.0) == "human"
    assert match_winner(5.0, 11.0) == "bot"


def test_completed_match_logs_decisive_winner(tmp_path: Path) -> None:
    # A match that ran through a tie must log a human/bot winner, never a tie.
    deals = [_deal(6.0, 6.0, 1), _deal(0.0, 4.0, 2)]
    played = run_match(6.0, _provider(deals))
    rec = log_match(
        match_id=7,
        target_score=6.0,
        human_match_score=sum(r.human for r in played),
        bot_match_score=sum(r.bot for r in played),
        deal_ids=[r.deal_id for r in played if r.deal_id is not None],
        path=tmp_path / "matches.jsonl",
    )
    assert rec.winner == "bot"
    assert rec.winner != "tie"


def test_read_missing_match_log_is_empty(tmp_path: Path) -> None:
    assert read_matches(tmp_path / "nope.jsonl") == []


# --- match reporting --------------------------------------------------------


def _match(winner: str, margin: float, deals: int, bot: str) -> MatchRecord:
    human = 11.0 if winner != "bot" else 11.0 - abs(margin)
    return MatchRecord("t", 0, 11.0, deals, human, human - margin, winner, margin, bot, [])


def test_summarize_empty() -> None:
    assert "No matches logged" in summarize([])


def test_summarize_counts_and_winrate() -> None:
    matches = [
        _match("human", 4.0, 3, "botA"),
        _match("bot", -2.0, 5, "botA"),
        _match("tie", 0.0, 4, "botA"),
    ]
    report = summarize(matches)
    assert "Matches      : 3" in report
    assert "1W - 1L - 1T" in report
    assert "50.0%" in report  # (1 + 0.5) / 3
    assert "Avg deals    : 4.00" in report  # (3 + 5 + 4) / 3


def test_summarize_breakdown_by_bot() -> None:
    matches = [
        _match("human", 3.0, 3, "botA"),
        _match("bot", -1.0, 6, "botB"),
    ]
    report = summarize(matches)
    assert "By bot configuration:" in report
    assert "botA:" in report
    assert "botB:" in report
