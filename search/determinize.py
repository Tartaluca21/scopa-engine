"""Determinization: sample a perfect-information instance from a player's view.

The deciding player knows: their own hand, the table, both capture piles, and
the scopa counts. The deck and the opponent's hand are hidden. Determinization
keeps every known fact fixed and randomly redistributes the hidden cards across
the deck and the opponent's hand, preserving their respective counts so the
resulting deal stays structurally legal.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engine.cards import HAND_ZONES, Zone
from engine.core import ScopaEngine

_EPSILON = 1e-9


def _sample_opp_hand(
    pool: npt.NDArray[np.intp],
    n_opp: int,
    rng: np.random.Generator,
    weights: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.intp]:
    """Pick `n_opp` cards from `pool` for the opponent's hand.

    With no `weights`, sampling is uniform (a plain permutation). With a belief
    vector, cards are drawn without replacement proportionally to their per-card
    probability of being held; if the belief lacks enough positive support to
    fill the hand it falls back to a uniform draw so the world stays legal.
    """
    if n_opp <= 0:
        return pool[:0]
    if weights is None:
        return rng.permutation(pool)[:n_opp]
    w = np.clip(weights[pool].astype(np.float64), 0.0, None)
    if int(np.count_nonzero(w)) < n_opp or float(w.sum()) <= _EPSILON:
        w = np.ones(pool.size, dtype=np.float64)
    w /= w.sum()
    return rng.choice(pool, size=n_opp, replace=False, p=w)


def determinize(
    engine: ScopaEngine,
    player: int,
    rng: np.random.Generator,
    weights: npt.NDArray[np.float64] | None = None,
) -> ScopaEngine:
    """Return a perfect-information clone of `engine` from `player`'s viewpoint.

    The opponent's hand and the deck are re-sampled from their pooled hidden
    cards; counts, the player's hand, the table, both piles, scopa counts,
    turn, and last capturer are all preserved. The hash is recomputed.

    `weights` is an optional (40,) belief vector (see cognitive.belief); when
    given, the opponent's hand is drawn proportionally to it instead of
    uniformly. Only pooled hidden cards can be picked, so visible or discarded
    cards are never sampled regardless of their weight.
    """
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")
    clone = engine.clone()
    opp_hand = HAND_ZONES[1 - player]
    n_opp = clone.count(opp_hand)
    pool = np.concatenate([clone.cards_in(Zone.MAZZO), clone.cards_in(opp_hand)])
    new_opp = _sample_opp_hand(pool, n_opp, rng, weights)
    new_deck = np.setdiff1d(pool, new_opp)
    clone.state[Zone.MAZZO, :] = 0
    clone.state[opp_hand, :] = 0
    clone.state[opp_hand, new_opp] = 1
    clone.state[Zone.MAZZO, new_deck] = 1
    clone.rehash()
    return clone
