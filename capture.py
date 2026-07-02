"""Optional per-decision capture for building an opponent-modeling dataset.

Reads the live engine state and packs one decision into a plain JSON-friendly
dict (the shape stored in `DealRecord.moves`). Everything here is *read-only* on
the engine -- it never mutates state or influences the move actually played, so
enabling capture cannot change game rules, scoring, or bot behavior.
"""

from __future__ import annotations

from engine.cards import HAND_ZONES, PRESE_ZONES, Zone
from engine.core import ScopaEngine
from engine.features import deal_breakdown

# A captured decision: JSON-serializable dict. Kept as a loose alias (matching
# `gamelog.Move`) so the log schema and this producer stay decoupled.
Decision = dict[str, object]


def _cards(engine: ScopaEngine, zone: Zone) -> list[int]:
    return [int(c) for c in engine.cards_in(zone)]


def legal_moves(engine: ScopaEngine, player: int) -> list[list[object]]:
    """Every legal move as ``[card, [capture...]]`` (JSON-friendly)."""
    moves: list[list[object]] = []
    for card in _cards(engine, HAND_ZONES[player]):
        opts = engine.captures_for(card)
        if opts:
            moves.extend([int(card), [int(c) for c in opt]] for opt in opts)
        else:
            moves.append([int(card), []])
    return moves


def cards_seen(engine: ScopaEngine, player: int) -> list[int]:
    """Cards observable to `player`: table, both capture piles, own hand.

    Deliberately excludes the opponent's hidden hand, so a downstream model
    trains only on information the decider actually had.
    """
    seen: set[int] = set(_cards(engine, Zone.TAVOLO))
    seen.update(_cards(engine, PRESE_ZONES[0]))
    seen.update(_cards(engine, PRESE_ZONES[1]))
    seen.update(_cards(engine, HAND_ZONES[player]))
    return sorted(seen)


def decision_record(
    engine: ScopaEngine,
    player_label: str,
    player: int,
    turn: int,
    card: int,
    capture: list[int],
) -> Decision:
    """Snapshot one decision (pre-move) as a JSON-serializable dict.

    Must be called *before* `execute_move` so hand/table reflect the state the
    player faced. `player_label` is "human" or "bot"; `player` is the 0/1 index.
    """
    bd = deal_breakdown(engine)  # provisional score if the deal ended now
    return {
        "turn": turn,
        "player": player_label,
        "hand": _cards(engine, HAND_ZONES[player]),
        "table": _cards(engine, Zone.TAVOLO),
        "legal_moves": legal_moves(engine, player),
        "chosen": [int(card), [int(c) for c in capture]],
        "partial_score": [bd.p0_score, bd.p1_score],
        "cards_seen": cards_seen(engine, player),
    }
