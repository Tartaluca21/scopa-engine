"""Tests for the human-decision dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

from decision_dataset import (
    dataset_stats,
    human_decision_rows,
    report_text,
    sample_text,
    write_jsonl,
)
from gamelog import DealRecord, MatchRecord, log_deal, read_deals


def _move(player: str, turn: int, chosen: list[object]) -> dict[str, object]:
    return {
        "turn": turn,
        "player": player,
        "hand": [1, 2, 3],
        "table": [4, 5],
        "legal_moves": [[1, []], [2, [4]]],
        "chosen": chosen,
        "partial_score": [0.0, 0.0],
        "cards_seen": [1, 2, 3, 4, 5],
    }


def _deal_with_moves(deal_id: int, moves: list[dict[str, object]]) -> DealRecord:
    return DealRecord(
        time="t",
        human=6.0,
        bot=4.0,
        margin=2.0,
        result="win",
        deal_id=deal_id,
        bot_name="PIMC(n_worlds=15,max_depth=6)",
        moves=moves,
    )


def test_deal_without_moves_yields_nothing() -> None:
    legacy = DealRecord("t", 6.0, 4.0, 2.0, "win")  # moves is None
    empty = _deal_with_moves(1, [])
    assert human_decision_rows([legacy, empty]) == []


def test_extracts_only_human_decisions() -> None:
    deal = _deal_with_moves(
        7,
        [_move("human", 0, [1, []]), _move("bot", 1, [2, [4]]), _move("human", 2, [2, [4]])],
    )
    rows = human_decision_rows([deal])
    assert len(rows) == 2
    assert all(r["deal_id"] == 7 for r in rows)
    assert [r["turn"] for r in rows] == [0, 2]
    assert rows[0]["bot_name"] == "PIMC(n_worlds=15,max_depth=6)"
    assert rows[0]["final_result"] == "win" and rows[0]["final_margin"] == 2.0


def test_match_id_is_linked_when_available() -> None:
    deal = _deal_with_moves(101, [_move("human", 0, [1, []])])
    match = MatchRecord("t", 555, 11.0, 3, 12.0, 7.0, "human", 5.0, None, [101, 102])
    (row,) = human_decision_rows([deal], [match])
    assert row["match_id"] == 555


def test_match_id_none_without_link() -> None:
    deal = _deal_with_moves(9, [_move("human", 0, [1, []])])
    (row,) = human_decision_rows([deal])
    assert row["match_id"] is None


def test_backward_compatible_with_logged_moves(tmp_path: Path) -> None:
    log = tmp_path / "deals.jsonl"
    log_deal(6.0, 4.0, log, deal_id=3, moves=[_move("human", 0, [1, []])])
    log_deal(5.0, 5.0, log)  # legacy-style row, no moves
    deals = read_deals(log)
    rows = human_decision_rows(deals)
    assert len(rows) == 1 and rows[0]["deal_id"] == 3


def test_report_empty() -> None:
    text = report_text([], n_deals=4)
    assert "Deals read                 : 4" in text
    assert "Human decisions            : 0" in text


def test_dataset_stats_empty() -> None:
    stats = dataset_stats([])
    assert stats == {"decisions": 0, "deals_with_history": 0, "matches": 0}


def test_dataset_stats_full() -> None:
    # Two deals; deal 1 has a match link, deal 2 does not. _move() -> hand size 3,
    # table size 2, 2 legal moves. Chosen: one lay ([1,[]]) + one capture ([2,[4]]).
    d1 = _deal_with_moves(1, [_move("human", 0, [1, []]), _move("human", 1, [2, [4]])])
    d2 = _deal_with_moves(2, [_move("human", 0, [3, [5]])])  # capture
    match = MatchRecord("t", 9, 11.0, 2, 12.0, 7.0, "human", 5.0, None, [1])
    rows = human_decision_rows([d1, d2], [match])
    s = dataset_stats(rows)
    assert s["decisions"] == 3
    assert s["deals_with_history"] == 2
    assert s["matches"] == 1  # only deal 1 linked
    assert s["avg_decisions_per_deal"] == 1.5
    assert s["avg_legal_moves"] == 2.0
    assert s["capture_pct"] == 2 / 3 and s["lay_pct"] == 1 / 3
    assert s["avg_table_size"] == 2.0 and s["avg_hand_size"] == 3.0
    assert s["result_breakdown"] == {"win": 3}
    assert s["by_bot"]["PIMC(n_worlds=15,max_depth=6)"]["decisions"] == 3  # type: ignore[index]


def test_report_includes_new_fields() -> None:
    deal = _deal_with_moves(
        1,
        [_move("human", 0, [1, []]), _move("human", 1, [2, [4]]), _move("human", 2, [3, [5]])],
    )
    text = report_text(human_decision_rows([deal]), n_deals=1)
    assert "Human decisions            : 3" in text
    assert "Deals with move history    : 1" in text
    assert "Avg legal moves/decision   : 2.00" in text
    assert "Avg table size before move : 2.00" in text
    assert "Avg hand size before move  : 3.00" in text
    assert "66.7% capture" in text and "33.3% lay" in text
    assert "win   : 3" in text
    assert "PIMC(n_worlds=15,max_depth=6): 3 decisions" in text


def _scopa_move(
    chosen: list[object], table: list[int], legal: list[list[object]]
) -> dict[str, object]:
    return {
        "turn": 0,
        "player": "human",
        "hand": [1, 2, 3],
        "table": table,
        "legal_moves": legal,
        "chosen": chosen,
        "partial_score": [0.0, 0.0],
        "cards_seen": [],
    }


def test_avoidable_scopa_flag() -> None:
    # Card idx 6 has value 7; a lone value-7 on the table is scopable. Laying the
    # 6 (idx 5, value 6) instead keeps a two-card table summing to 13 (>10, safe).
    # Chosen leaves [idx6] alone -> scopable; alternative [idx0] avoids it.
    move = _scopa_move(
        chosen=[5, [0]],  # capture the Asso, leaving just idx6 (value 7) on table
        table=[0, 6],  # values 1 and 7
        legal=[[5, [0]], [16, [6]]],  # alt: capture idx6, leaving idx0 (value 1) alone
    )
    deal = _deal_with_moves(1, [move])
    (row,) = human_decision_rows([deal])
    assert row["left_table_scopable"] is True
    # both leftover tables are single cards -> both scopable, so NOT avoidable here
    assert row["avoidable_scopa"] is False


def test_avoidable_scopa_true_when_safe_alternative_exists() -> None:
    # Chosen lays value-3 onto empty-ish table leaving a lone card (scopable);
    # alternative lays value-9 building a two-card table summing to >10 (safe).
    move = _scopa_move(
        chosen=[2, []],  # lay idx2 (value 3) on empty table -> lone card, scopable
        table=[],
        legal=[[2, []], [8, []]],  # laying either leaves one card; still scopable
    )
    # With an empty table any lay leaves a single card -> always scopable, no escape.
    deal = _deal_with_moves(1, [move])
    (row,) = human_decision_rows([deal])
    assert row["left_table_scopable"] is True and row["avoidable_scopa"] is False


def test_stats_and_report_include_scopa_fields() -> None:
    move = _scopa_move(chosen=[5, [0]], table=[0, 6], legal=[[5, [0]], [16, [6]]])
    rows = human_decision_rows([_deal_with_moves(1, [move])])
    s = dataset_stats(rows)
    assert s["left_scopable_count"] == 1 and s["avoidable_scopa_count"] == 0
    text = report_text(rows, n_deals=1)
    assert "Left table scopable        : 1 (100.0%)" in text
    assert "Avoidable scopa left       : 0 (0.0%)" in text


def test_sample_text_renders_card_names() -> None:
    # Card 0 = Asso di Denari; card 4 = 5 di Denari (indices are value-major).
    deal = _deal_with_moves(7, [_move("human", 0, [0, [4]])])
    text = sample_text(human_decision_rows([deal]), k=5)
    assert "=== Sample of 1 decision(s) ===" in text
    assert "deal 7 turn 0" in text
    assert "di Denari" in text  # human-readable names, not raw indices
    assert "chosen:" in text and "capture" in text.lower()


def test_sample_text_zero_is_empty() -> None:
    deal = _deal_with_moves(1, [_move("human", 0, [1, []])])
    assert sample_text(human_decision_rows([deal]), k=0) == ""


def test_write_jsonl_roundtrip(tmp_path: Path) -> None:
    deal = _deal_with_moves(1, [_move("human", 0, [1, []])])
    rows = human_decision_rows([deal])
    out = tmp_path / "ds.jsonl"
    write_jsonl(rows, out)
    loaded = [json.loads(line) for line in out.read_text().splitlines()]
    assert loaded == rows
