"""Persistent JSONL log of human-vs-bot deal results.

One record per finished deal, appended to `LOG_PATH`. Shared by `play.py`
(which writes) and `stats.py` (which reports). Kept dependency-free so both
the CLI and the report can import it without pulling in the engine.

The schema grows by *appending optional fields only*: every field added after
`result` defaults to `None`, so old rows carrying just
`{time, human, bot, margin, result}` still load. `human`/`bot` are the deal
scores (i.e. human_score / bot_score); component winners use the labels
"human" | "bot" | "none".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

LOG_PATH = Path(__file__).with_name("logs") / "human_vs_bot.jsonl"
MATCH_LOG_PATH = Path(__file__).with_name("logs") / "human_vs_bot_matches.jsonl"

Result = str  # "win" | "loss" | "tie", from the human's perspective
Winner = str  # "human" | "bot" | "none"
MatchWinner = str  # "human" | "bot" | "tie"
# One captured decision (see `capture.decision_record`), stored in the optional
# `moves` history when `--record-moves` (CLI) or the GUI record toggle is on.
Move = dict[str, object]


@dataclass(slots=True, frozen=True)
class DealRecord:
    """One finished deal: scores, margin, outcome, and optional rich metadata."""

    time: str
    human: float
    bot: float
    margin: float
    result: Result
    deal_id: int | None = None
    human_scope: int | None = None
    bot_scope: int | None = None
    settebello_winner: Winner | None = None
    denari_winner: Winner | None = None
    primiera_winner: Winner | None = None
    cards_winner: Winner | None = None
    bot_name: str | None = None
    moves: list[Move] | None = None


def _result(human: float, bot: float) -> Result:
    if human > bot:
        return "win"
    if bot > human:
        return "loss"
    return "tie"


def log_deal(
    human: float,
    bot: float,
    path: Path = LOG_PATH,
    *,
    deal_id: int | None = None,
    human_scope: int | None = None,
    bot_scope: int | None = None,
    settebello_winner: Winner | None = None,
    denari_winner: Winner | None = None,
    primiera_winner: Winner | None = None,
    cards_winner: Winner | None = None,
    bot_name: str | None = None,
    moves: list[Move] | None = None,
) -> DealRecord:
    """Append one deal result (human's perspective) and return the record.

    Only `human` and `bot` (the scores) are required; every richer field is
    optional so callers with less context still produce a valid, readable row.
    """
    record = DealRecord(
        time=datetime.now(UTC).isoformat(timespec="seconds"),
        human=float(human),
        bot=float(bot),
        margin=float(human) - float(bot),
        result=_result(human, bot),
        deal_id=deal_id,
        human_scope=human_scope,
        bot_scope=bot_scope,
        settebello_winner=settebello_winner,
        denari_winner=denari_winner,
        primiera_winner=primiera_winner,
        cards_winner=cards_winner,
        bot_name=bot_name,
        moves=moves,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")
    return record


def read_deals(path: Path = LOG_PATH) -> list[DealRecord]:
    """Load every logged deal; empty list if the log does not exist yet.

    Rows are tolerated as long as they are a *subset* of the current schema:
    missing optional keys fall back to their `None` defaults, so minimal legacy
    rows and rich new rows coexist. Unknown extra keys are dropped rather than
    crashing a report built by a newer writer than reader.
    """
    if not path.exists():
        return []
    fields = DealRecord.__dataclass_fields__
    records: list[DealRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(DealRecord(**{k: v for k, v in data.items() if k in fields}))
    return records


@dataclass(slots=True, frozen=True)
class MatchRecord:
    """One finished match: several deals accumulated to a target score."""

    time: str
    match_id: int
    target_score: float
    n_deals: int
    human_match_score: float
    bot_match_score: float
    winner: MatchWinner
    final_margin: float
    bot_name: str | None
    deal_ids: list[int]


def match_winner(human: float, bot: float) -> MatchWinner:
    """Match outcome from the human's perspective; equal totals are a tie."""
    if human > bot:
        return "human"
    if bot > human:
        return "bot"
    return "tie"


def log_match(
    *,
    match_id: int,
    target_score: float,
    human_match_score: float,
    bot_match_score: float,
    deal_ids: list[int],
    bot_name: str | None = None,
    path: Path = MATCH_LOG_PATH,
) -> MatchRecord:
    """Append one finished match and return the record."""
    record = MatchRecord(
        time=datetime.now(UTC).isoformat(timespec="seconds"),
        match_id=match_id,
        target_score=float(target_score),
        n_deals=len(deal_ids),
        human_match_score=float(human_match_score),
        bot_match_score=float(bot_match_score),
        winner=match_winner(human_match_score, bot_match_score),
        final_margin=float(human_match_score) - float(bot_match_score),
        bot_name=bot_name,
        deal_ids=list(deal_ids),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")
    return record


def read_matches(path: Path = MATCH_LOG_PATH) -> list[MatchRecord]:
    """Load every logged match; empty list if the log does not exist yet."""
    if not path.exists():
        return []
    fields = MatchRecord.__dataclass_fields__
    records: list[MatchRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(MatchRecord(**{k: v for k, v in data.items() if k in fields}))
    return records
