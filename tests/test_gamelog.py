"""Tests for deal logging (with rich metadata) and the win-rate report."""

from __future__ import annotations

import json
from pathlib import Path

from gamelog import DealRecord, log_deal, read_deals
from stats import summarize


def _write_lines(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_log_and_read_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "deals.jsonl"
    log_deal(6.0, 4.0, log)
    log_deal(3.0, 5.0, log)
    deals = read_deals(log)
    assert [d.result for d in deals] == ["win", "loss"]
    assert deals[0].margin == 2.0
    assert deals[1].margin == -2.0


def test_rich_row_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "deals.jsonl"
    rec = log_deal(
        6.0,
        4.0,
        log,
        deal_id=12345,
        human_scope=1,
        bot_scope=0,
        settebello_winner="human",
        denari_winner="bot",
        primiera_winner="none",
        cards_winner="human",
        bot_name="PIMC(n_worlds=15,max_depth=6)",
    )
    (loaded,) = read_deals(log)
    assert loaded == rec
    assert loaded.deal_id == 12345
    assert loaded.settebello_winner == "human"
    assert loaded.bot_name == "PIMC(n_worlds=15,max_depth=6)"
    assert loaded.moves is None  # optional move history not captured yet


def test_reads_legacy_minimal_rows(tmp_path: Path) -> None:
    log = tmp_path / "deals.jsonl"
    _write_lines(log, [{"time": "t", "human": 5.0, "bot": 3.0, "margin": 2.0, "result": "win"}])
    (loaded,) = read_deals(log)
    assert loaded.result == "win"
    assert loaded.human_scope is None
    assert loaded.settebello_winner is None


def test_unknown_extra_keys_are_ignored(tmp_path: Path) -> None:
    log = tmp_path / "deals.jsonl"
    _write_lines(
        log,
        [{"time": "t", "human": 5.0, "bot": 3.0, "margin": 2.0, "result": "win", "future": 1}],
    )
    (loaded,) = read_deals(log)
    assert loaded.result == "win"


def test_result_classification(tmp_path: Path) -> None:
    log = tmp_path / "deals.jsonl"
    assert log_deal(5.0, 5.0, log).result == "tie"
    assert log_deal(7.0, 2.0, log).result == "win"
    assert log_deal(1.0, 9.0, log).result == "loss"


def test_read_missing_is_empty(tmp_path: Path) -> None:
    assert read_deals(tmp_path / "nope.jsonl") == []


def test_summarize_empty() -> None:
    assert "No deals logged" in summarize([])


def test_summarize_counts() -> None:
    deals = [
        DealRecord("t", 6.0, 4.0, 2.0, "win"),
        DealRecord("t", 3.0, 5.0, -2.0, "loss"),
        DealRecord("t", 5.0, 5.0, 0.0, "tie"),
    ]
    report = summarize(deals)
    assert "1W - 1L - 1T" in report
    assert "50.0%" in report  # (1 win + 0.5 tie) / 3
    assert "+0.00" in report  # margins 2, -2, 0 average to 0


def test_summarize_mixed_old_and_new_rows() -> None:
    legacy = DealRecord("t", 6.0, 4.0, 2.0, "win")  # no rich fields
    rich = DealRecord(
        "t",
        5.0,
        6.0,
        -1.0,
        "loss",
        human_scope=0,
        bot_scope=1,
        settebello_winner="human",
        denari_winner="bot",
        primiera_winner="none",
        cards_winner="human",
    )
    report = summarize([legacy, rich])
    # Overall metrics span both rows.
    assert "Deals played : 2" in report
    assert "1W - 1L - 0T" in report
    # Scope and components computed only over the one rich row.
    assert "you 0.00  vs  bot 1.00  (over 1 deals)" in report
    assert "Settebello :" in report and "(over 1 deals)" in report


def test_summarize_all_legacy_rows_marks_components_na() -> None:
    report = summarize([DealRecord("t", 6.0, 4.0, 2.0, "win")])
    assert "Avg scope    : n/a" in report
    assert "n/a  (no rich rows)" in report
