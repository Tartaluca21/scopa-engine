"""Integration tests for the GUI session controller (no pygame, no threads).

A synchronous bot double replaces the off-thread `AsyncBot` so a whole deal (or
match) can be driven deterministically, exercising the exact production path:
`SessionController.tick` -> `GameController` -> `finalize_deal`/`MatchSession`.
Card-slide timings are skipped by ticking with a large `dt`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from decision_dataset import human_decision_rows
from engine.core import ScopaEngine
from gamelog import read_deals, read_matches
from gui.async_bot import Move
from gui.game import HUMAN
from gui.session import SessionController, gui_config
from match_stats import summarize
from search.alphabeta import legal_moves

_BIG_DT = 10_000.0  # ms; collapses every card animation in a single tick


class _SimpleAgent:
    """Deterministic agent: always take the first legal move."""

    def select(self, engine: ScopaEngine, player: int) -> Move:
        return legal_moves(engine, player)[0]


class _SyncBot:
    """Synchronous stand-in for `AsyncBot`: computes the move inline, no thread."""

    def __init__(self) -> None:
        self._agent = _SimpleAgent()
        self._move: Move | None = None

    @property
    def thinking(self) -> bool:
        return False

    @property
    def ready(self) -> bool:
        return self._move is not None

    def start(self, engine: ScopaEngine, player: int) -> None:
        self._move = self._agent.select(engine, player)

    def take(self) -> Move:
        assert self._move is not None
        move, self._move = self._move, None
        return move


def _sync_factory(rng: np.random.Generator) -> _SyncBot:
    return _SyncBot()


def _play_first_human_move(sess: SessionController) -> None:
    ctrl = sess.controller
    card, capture = legal_moves(ctrl.engine, HUMAN)[0]
    ctrl.begin_human_move(int(card))
    if ctrl.pending is not None:  # ambiguous capture: pick the matching option
        for c in capture:
            ctrl.toggle_table_card(int(c))


def _drive_deal(sess: SessionController) -> None:
    """Play the current deal to completion, human always taking a legal move."""
    for _ in range(2000):
        sess.tick(_BIG_DT)
        if sess.is_over():
            return
        ctrl = sess.controller
        if ctrl.is_human_turn() and ctrl.pending is None and ctrl.capture_anim is None:
            _play_first_human_move(sess)
    raise AssertionError("deal did not finish within the tick budget")


def _new_session(tmp_path: Path, **cfg: object) -> SessionController:
    return SessionController(
        gui_config(**cfg),  # type: ignore[arg-type]
        np.random.default_rng(0),
        bot_factory=_sync_factory,
        deal_log_path=tmp_path / "deals.jsonl",
        match_log_path=tmp_path / "matches.jsonl",
    )


# --- single deal -----------------------------------------------------------


def test_gui_single_deal_logs_one_record(tmp_path: Path) -> None:
    sess = _new_session(tmp_path)
    _drive_deal(sess)
    deals = read_deals(tmp_path / "deals.jsonl")
    assert len(deals) == 1
    rec = deals[0]
    assert rec.result in ("win", "loss", "tie")
    assert rec.human + rec.bot > 0  # some points were scored
    assert rec.bot_name == "PIMC(n_worlds=12,max_depth=5)"
    assert rec.deal_id == sess.controller.deal_id
    assert sess.match is None
    assert rec.moves is None  # recording off by default


def test_gui_advance_starts_a_fresh_logged_deal(tmp_path: Path) -> None:
    sess = _new_session(tmp_path)
    _drive_deal(sess)
    first_id = sess.controller.deal_id
    sess.advance()
    assert sess.controller.deal_id != first_id
    _drive_deal(sess)
    assert len(read_deals(tmp_path / "deals.jsonl")) == 2


# --- move recording --------------------------------------------------------


def test_gui_move_recording_captures_both_players(tmp_path: Path) -> None:
    sess = _new_session(tmp_path, record_moves=True)
    _drive_deal(sess)
    (rec,) = read_deals(tmp_path / "deals.jsonl")
    assert rec.moves is not None and len(rec.moves) > 0
    players = {m["player"] for m in rec.moves}
    assert players == {"human", "bot"}
    # The dataset builder yields exactly the human decisions from this deal.
    rows = human_decision_rows(read_deals(tmp_path / "deals.jsonl"))
    assert rows and all(r["deal_id"] == rec.deal_id for r in rows)
    assert len(rows) == sum(1 for m in rec.moves if m["player"] == "human")


# --- match mode ------------------------------------------------------------


def test_gui_match_logs_and_appears_in_stats(tmp_path: Path) -> None:
    sess = _new_session(tmp_path, target=11.0)
    assert sess.match is not None
    for _ in range(200):
        _drive_deal(sess)
        if sess.match.is_decided():
            break
        assert sess.awaiting_next_deal()
        sess.advance()
    assert sess.match.is_decided()
    assert sess.last_match is not None

    matches = read_matches(tmp_path / "matches.jsonl")
    assert len(matches) == 1
    m = matches[0]
    assert m.winner in ("human", "bot")
    assert m.n_deals == len(m.deal_ids)
    assert max(m.human_match_score, m.bot_match_score) >= 11.0
    # Every deal of the match was logged individually, too.
    assert len(read_deals(tmp_path / "deals.jsonl")) == m.n_deals
    assert "PIMC(n_worlds=12,max_depth=5)" in summarize(matches)


def test_gui_match_status_line_reflects_progress(tmp_path: Path) -> None:
    sess = _new_session(tmp_path, target=11.0)
    _drive_deal(sess)
    line = sess.match_status_line()
    assert line is not None and line.startswith("Match to 11")
