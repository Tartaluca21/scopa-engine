"""Turn captured deal logs into a flat human-decision dataset.

Reads `DealRecord`s (via `gamelog`) and emits one row per *human* decision that
was captured under `play.py --record-moves`. Deals without a `moves` history --
including every legacy row where `moves` is missing or null -- are simply
skipped, so this is safe to run over a mixed log. The CLI lives in
`scripts/build_decision_dataset.py`; the logic here is import-friendly for tests.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import cast

from engine.cards import card_value
from gamelog import DealRecord, MatchRecord
from ui import card_name, describe_move

DEFAULT_OUTPUT = Path(__file__).with_name("logs") / "human_decisions.jsonl"

Row = dict[str, object]


def link_matches(matches: list[MatchRecord]) -> dict[int, int]:
    """Map each deal_id to the match_id that contains it (empty if none)."""
    link: dict[int, int] = {}
    for match in matches:
        for deal_id in match.deal_ids:
            link[deal_id] = match.match_id
    return link


def _table_after(table: list[int], move: list[object]) -> list[int]:
    """Table state after playing `move = [card, [captures]]` on `table`."""
    played = int(cast("int", move[0]))
    captured = [int(c) for c in cast("list[int]", move[1])]
    if captured:
        gone = set(captured)
        return [c for c in table if c not in gone]
    return [*table, played]


def _is_scopable(cards: list[int]) -> bool:
    """True if the opponent could clear this whole table with one played card.

    A single played value clears the table iff there is exactly one card (any
    matching value takes it), or the values sum to 1..10 with no single card
    equal to that sum (else the forced-single-capture rule blocks the sweep).
    """
    if not cards:
        return False
    if len(cards) == 1:
        return True
    values = [card_value(c) for c in cards]
    total = sum(values)
    return 1 <= total <= 10 and total not in values


def _scopa_flags(table: object, legal_moves: object, chosen: object) -> tuple[bool, bool]:
    """`(left_table_scopable, avoidable_scopa)` from human-visible info only.

    `left_table_scopable`: the chosen move leaves a table the opponent could
    sweep in one move. `avoidable_scopa`: it did, yet some legal alternative
    would have left a non-scopable table.
    """
    if not (isinstance(table, list) and isinstance(chosen, list) and len(chosen) == 2):
        return False, False
    left = _is_scopable(_table_after(table, chosen))
    if not left or not isinstance(legal_moves, list):
        return left, False
    for move in legal_moves:
        if not (isinstance(move, list) and len(move) == 2 and move != chosen):
            continue
        if not _is_scopable(_table_after(table, move)):
            return left, True
    return left, False


def human_decision_rows(
    deals: list[DealRecord], matches: list[MatchRecord] | None = None
) -> list[Row]:
    """One dataset row per captured human decision across all `deals`."""
    link = link_matches(matches or [])
    rows: list[Row] = []
    for deal in deals:
        if not deal.moves:  # None or empty: nothing captured, skip
            continue
        for move in deal.moves:
            if move.get("player") != "human":
                continue  # bot decisions are captured but not part of this set
            left_scopable, avoidable = _scopa_flags(
                move.get("table"), move.get("legal_moves"), move.get("chosen")
            )
            rows.append(
                {
                    "deal_id": deal.deal_id,
                    "turn": move.get("turn"),
                    "hand": move.get("hand"),
                    "table": move.get("table"),
                    "legal_moves": move.get("legal_moves"),
                    "chosen": move.get("chosen"),
                    "left_table_scopable": left_scopable,
                    "avoidable_scopa": avoidable,
                    "bot_name": deal.bot_name,
                    "final_result": deal.result,
                    "final_margin": deal.margin,
                    "match_id": link.get(deal.deal_id) if deal.deal_id is not None else None,
                }
            )
    return rows


def _chosen_is_capture(chosen: object) -> bool:
    """A chosen move `[card, [captures]]` is a capture iff it takes any card."""
    return isinstance(chosen, list) and len(chosen) == 2 and bool(chosen[1])


def _len(value: object) -> int:
    """Length of a stored list field, tolerating a missing/None value."""
    return len(value) if isinstance(value, list) else 0


def _bot_breakdown(rows: list[Row]) -> dict[str, dict[str, float]]:
    """Per-bot decision counts and capture rate."""
    by_bot: dict[str, list[Row]] = {}
    for row in rows:
        by_bot.setdefault(str(row.get("bot_name")), []).append(row)
    out: dict[str, dict[str, float]] = {}
    for name, group in by_bot.items():
        captures = sum(1 for r in group if _chosen_is_capture(r.get("chosen")))
        out[name] = {"decisions": len(group), "capture_pct": captures / len(group)}
    return out


def dataset_stats(rows: list[Row]) -> dict[str, object]:
    """Structured summary of the human-decision dataset (see `report_text`)."""
    n = len(rows)
    deals = len({r.get("deal_id") for r in rows})
    matches = len({r.get("match_id") for r in rows if r.get("match_id") is not None})
    stats: dict[str, object] = {"decisions": n, "deals_with_history": deals, "matches": matches}
    if n == 0:
        return stats
    captures = sum(1 for r in rows if _chosen_is_capture(r.get("chosen")))
    left_scopable = sum(1 for r in rows if r.get("left_table_scopable"))
    avoidable = sum(1 for r in rows if r.get("avoidable_scopa"))
    stats.update(
        {
            "avg_decisions_per_deal": n / deals,
            "avg_legal_moves": sum(_len(r.get("legal_moves")) for r in rows) / n,
            "capture_pct": captures / n,
            "lay_pct": (n - captures) / n,
            "avg_table_size": sum(_len(r.get("table")) for r in rows) / n,
            "avg_hand_size": sum(_len(r.get("hand")) for r in rows) / n,
            "left_scopable_count": left_scopable,
            "left_scopable_pct": left_scopable / n,
            "avoidable_scopa_count": avoidable,
            "avoidable_scopa_pct": avoidable / n,
            "result_breakdown": dict(Counter(r.get("final_result") for r in rows)),
            "by_bot": _bot_breakdown(rows),
        }
    )
    return stats


def report_text(rows: list[Row], n_deals: int) -> str:
    """Human-readable validation summary of the dataset."""
    s = dataset_stats(rows)
    if s["decisions"] == 0:
        return f"Deals read                 : {n_deals}\nHuman decisions            : 0"
    lines = [
        f"Deals read                 : {n_deals}",
        f"Deals with move history    : {s['deals_with_history']}",
        f"Matches represented        : {s['matches']}",
        f"Human decisions            : {s['decisions']}",
        f"Avg decisions/deal         : {s['avg_decisions_per_deal']:.2f}",
        f"Avg legal moves/decision   : {s['avg_legal_moves']:.2f}",
        f"Capture vs lay             : {s['capture_pct']:.1%} capture / {s['lay_pct']:.1%} lay",
        f"Avg table size before move : {s['avg_table_size']:.2f}",
        f"Avg hand size before move  : {s['avg_hand_size']:.2f}",
        f"Left table scopable        : {s['left_scopable_count']} ({s['left_scopable_pct']:.1%})",
        f"Avoidable scopa left       : {s['avoidable_scopa_count']} "
        f"({s['avoidable_scopa_pct']:.1%})",
        "Final result of decisions  :",
    ]
    result: dict[str, int] = s["result_breakdown"]  # type: ignore[assignment]
    lines.extend(f"  {name:<6}: {count}" for name, count in sorted(result.items()))
    lines.append("By bot configuration       :")
    bots: dict[str, dict[str, float]] = s["by_bot"]  # type: ignore[assignment]
    for name in sorted(bots):
        info = bots[name]
        lines.append(
            f"  {name}: {int(info['decisions'])} decisions, {info['capture_pct']:.1%} capture"
        )
    return "\n".join(lines)


def _render_move(move: object) -> str:
    """Render a `[card, [captures]]` move with human-readable card names."""
    if isinstance(move, list) and len(move) == 2:
        return describe_move(int(move[0]), [int(c) for c in move[1]])
    return str(move)


def sample_text(rows: list[Row], k: int) -> str:
    """Pretty-print the first `k` decisions with human-readable card names."""
    if k <= 0 or not rows:
        return ""
    shown = rows[:k]
    lines = [f"=== Sample of {len(shown)} decision(s) ==="]
    for i, r in enumerate(shown, start=1):
        hand = cast("list[int]", r.get("hand") or [])
        table = cast("list[int]", r.get("table") or [])
        legal = cast("list[object]", r.get("legal_moves") or [])
        header = f"[{i}] deal {r.get('deal_id')} turn {r.get('turn')}"
        lines.append(f"{header}  ({r.get('final_result')})")
        lines.append("  hand  : " + (", ".join(card_name(int(c)) for c in hand) or "(empty)"))
        lines.append("  table : " + (", ".join(card_name(int(c)) for c in table) or "(empty)"))
        lines.append("  legal :")
        lines.extend("    - " + _render_move(m) for m in legal)
        lines.append("  chosen: " + _render_move(r.get("chosen")))
    return "\n".join(lines)


def write_jsonl(rows: list[Row], path: Path = DEFAULT_OUTPUT) -> None:
    """Write one JSON object per row, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
