"""Negamax alpha-beta search over perfect-information Scopa states.

Operates on a determinized `ScopaEngine` (all cards known). Values are signed
margins from the perspective of the side to move, so a single symmetric search
serves both players. The TranspositionTable caches results by Zobrist hash with
EXACT / LOWER / UPPER bounds to prune repeated look-aheads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from engine.cards import HAND_ZONES, Zone
from engine.core import ScopaEngine
from engine.heuristic import Weights, evaluate, score_deal
from engine.transposition import NodeType, TranspositionTable
from search.endgame import solve_endgame

Move = tuple[int, list[int]]

# The Zobrist hash encodes zones + side to move + scopa counts, but NOT
# `last_capturer` -- which the end-of-deal sweep (and thus the terminal value)
# depends on. Two board-identical states with different last capturers therefore
# share a zhash yet have different values, so caching them under the bare zhash
# returns wrong entries (~1% of deck-empty leaves at depth 5). Salting the TT key
# by last_capturer (-1/0/1 -> index +1) disambiguates them without changing the
# engine hash. Salt 0 leaves the "no capture yet" states at their plain zhash.
_LAST_CAPTURER_KEYS: tuple[int, int, int] = (
    0x0,
    0x9E3779B97F4A7C15,
    0xD1B54A32D192ED03,
)


def _tt_key(engine: ScopaEngine) -> int:
    """TT key: the Zobrist hash disambiguated by `last_capturer`."""
    return engine.zhash ^ _LAST_CAPTURER_KEYS[engine.last_capturer + 1]


@dataclass(slots=True)
class SearchConfig:
    """Search parameters: depth cap and the leaf-evaluation weights.

    `use_endgame_solver` opts a search into the exact deck-empty solver
    (`search.endgame`): when the deck is empty the node is solved exactly instead
    of being cut off with the heuristic. Defaults off so the deployed bot's
    behaviour is unchanged unless a caller explicitly enables it.
    """

    max_depth: int = 12
    weights: Weights = field(default_factory=Weights)
    use_endgame_solver: bool = False


def capture_options(engine: ScopaEngine, card: int) -> list[list[int]]:
    """Legal capture sets for `card` ([] = lay it down), one entry per move."""
    options = engine.captures_for(card)
    if options:
        return [[int(c) for c in opt] for opt in options]
    return [[]]


def legal_moves(engine: ScopaEngine, player: int) -> list[Move]:
    """Every (card, capture_set) the player may play from the current state."""
    moves: list[Move] = []
    for card in engine.cards_in(HAND_ZONES[player]):
        for cap in capture_options(engine, int(card)):
            moves.append((int(card), cap))
    return moves


def _terminal_value(engine: ScopaEngine) -> float:
    """Final score margin from the perspective of the side to move."""
    final = engine.clone()
    final.end_of_deal_sweep()
    p0, p1 = score_deal(final)
    pov = engine.current_player
    return (p0 - p1) if pov == 0 else (p1 - p0)


def _leaf_value(engine: ScopaEngine, weights: Weights) -> float:
    """Heuristic margin from the perspective of the side to move at a cutoff."""
    pov = engine.current_player
    opp = 1 - pov
    own = evaluate(engine, pov, weights, int(engine.scopa_counts[pov]))
    other = evaluate(engine, opp, weights, int(engine.scopa_counts[opp]))
    return own - other


def alphabeta(
    engine: ScopaEngine,
    depth: int,
    alpha: float,
    beta: float,
    tt: TranspositionTable,
    cfg: SearchConfig,
) -> float:
    """Negamax value (margin for the side to move) with alpha-beta and TT."""
    h = _tt_key(engine)
    entry = tt.get(h)
    if entry is not None and entry.depth >= depth:
        if entry.node_type == NodeType.EXACT:
            return entry.value
        if entry.node_type == NodeType.LOWER:
            alpha = max(alpha, entry.value)
        else:
            beta = min(beta, entry.value)
        if alpha >= beta:
            return entry.value

    if engine.is_game_over():
        return _terminal_value(engine)

    # Deck exhausted: the rest of the deal is a small perfect-information
    # subgame. Solve it exactly (bounded, memoized) rather than cutting off with
    # the heuristic. Gated so the default bot is unaffected.
    if cfg.use_endgame_solver and engine.count(Zone.MAZZO) == 0:
        return solve_endgame(engine)

    empty_hands = engine.count(Zone.MANO_P1) == 0 and engine.count(Zone.MANO_P2) == 0
    if empty_hands and engine.count(Zone.MAZZO) > 0:
        dealt = engine.clone()
        dealt.deal_round()
        return alphabeta(dealt, depth, alpha, beta, tt, cfg)

    if depth == 0:
        return _leaf_value(engine, cfg.weights)

    alpha_orig = alpha
    best = -math.inf
    player = engine.current_player
    for card, cap in legal_moves(engine, player):
        child = engine.clone()
        child.execute_move(card, cap)
        value = -alphabeta(child, depth - 1, -beta, -alpha, tt, cfg)
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best <= alpha_orig:
        node_type = NodeType.UPPER
    elif best >= beta:
        node_type = NodeType.LOWER
    else:
        node_type = NodeType.EXACT
    tt.store(h, best, depth, node_type)
    return best
