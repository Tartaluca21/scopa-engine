"""Bayesian belief system for estimating the opponent's hidden hand.

The bot observes its own hand, the table and both capture piles; the deck and
the opponent's hand stay hidden. `BeliefSystem` maintains a (40,) NumPy vector
of P(card in opponent's hand) and refines it with rational-play inference after
each opponent move, exporting sorted probabilities to the PIMC / ISMCTS
determinization layers.

Cards are integer indices 0..39 (see engine.cards). The system only reads the
engine state; it never mutates it, so Zobrist hashes and transposition tables
are untouched.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from engine.cards import CARD_VALUES, HAND_ZONES, N_CARDS, N_VALUES, Zone
from engine.core import ScopaEngine

EPSILON = 1e-9

ProbArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
TableInput = npt.NDArray[np.intp] | list[int]


class BeliefSystem:
    """Vectorized posterior over which cards the opponent is holding."""

    bot_player: int
    probs: ProbArray
    candidates: BoolArray
    opp_hand_size: int
    certain: bool
    soft: bool
    alpha: float
    declined_penalty: float
    goal_pref: float

    def __init__(
        self,
        bot_player: int = 0,
        *,
        soft: bool = False,
        alpha: float = 0.5,
        declined_penalty: float = 0.0,
        goal_pref: float = 0.0,
    ) -> None:
        """`soft` enables rational-play inference beyond hard facts (default off,
        so the deployed bot is unchanged). `alpha` is the safety weight blended
        toward a uniform prior on every soft update (always > 0 so worlds are
        never hard-filtered); `declined_penalty` downweights candidates that
        could have captured when the opponent instead laid; `goal_pref` mildly
        upweights the opponent retaining denari. All are no-ops when `soft`."""
        if bot_player not in (0, 1):
            raise ValueError(f"bot_player must be 0 or 1, got {bot_player}")
        if soft and not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1] so a uniform floor is always mixed in")
        self.bot_player = bot_player
        self.soft = soft
        self.alpha = alpha
        self.declined_penalty = declined_penalty
        self.goal_pref = goal_pref
        self.probs = np.zeros(N_CARDS, dtype=np.float64)
        self.candidates = np.zeros(N_CARDS, dtype=bool)
        self.opp_hand_size = 0
        self.certain = False

    @property
    def opp_hand_zone(self) -> Zone:
        """Engine zone holding the opponent's (hidden) hand."""
        return HAND_ZONES[1 - self.bot_player]

    def _hidden_mask(self, state: ScopaEngine) -> BoolArray:
        """Boolean (40,) mask of cards invisible to the bot: deck + opp hand."""
        deck = state.state[Zone.MAZZO].astype(bool)
        opp = state.state[self.opp_hand_zone].astype(bool)
        mask: BoolArray = deck | opp
        return mask

    # --- public updates -------------------------------------------------

    def update_on_deal(self, state: ScopaEngine) -> None:
        """Reset the prior after a fresh deal of 3 cards per player.

        Unknown cards receive a uniform prior scaled so the vector sums to the
        opponent's current hand count. When the talon is exhausted every hidden
        card is snapped to certainty (1.0) — it can only be in the opp hand.
        """
        self.opp_hand_size = int(state.state[self.opp_hand_zone].sum())
        hidden = self._hidden_mask(state)
        self.candidates = hidden.copy()
        self.probs = np.zeros(N_CARDS, dtype=np.float64)
        deck_size = int(state.state[Zone.MAZZO].sum())
        if deck_size == 0:
            self.certain = True
            self.probs[hidden] = 1.0
            return
        self.certain = False
        n_hidden = int(hidden.sum())
        if n_hidden > 0:
            self.probs[hidden] = self.opp_hand_size / n_hidden

    def update_on_opponent_play(
        self,
        played_card: int,
        table_before_play: TableInput | None = None,
        legal_moves: TableInput | None = None,
        captured: TableInput | None = None,
    ) -> None:
        """Update after the opponent plays `played_card`.

        Hard facts always apply: the played card becomes visible (probability 0),
        the opponent's hand shrinks by one, and the vector is renormalized to the
        new hand size (or held at certainty in the endgame). No hard inference is
        drawn from the *kind* of move -- a liscio never eliminates any candidate.

        When `soft` is enabled, a bounded, uniform-mixed reweighting is layered on
        top (see `_apply_soft`); `captured` (the cards taken, empty for a lay)
        drives the declined-capture signal. This never zeroes a candidate, so the
        determinizer can always still sample a legal world.
        """
        if not 0 <= played_card < N_CARDS:
            raise IndexError(f"card index out of bounds: {played_card}")
        self.probs[played_card] = 0.0
        self.candidates[played_card] = False
        self.opp_hand_size = max(0, self.opp_hand_size - 1)
        self._normalize()
        if self.soft:
            self._apply_soft(table_before_play, captured)

    def _apply_soft(
        self, table_before_play: TableInput | None, captured: TableInput | None
    ) -> None:
        """Bounded rational-play reweighting, always mixed with a uniform floor.

        Declined-capture: if the opponent laid (no capture) while cards were on
        the table, candidates whose value could have captured a table card are
        downweighted (a rational holder would usually have captured). Goal-card:
        denari candidates are mildly upweighted. The result is blended
        `(1 - alpha) * inferred + alpha * uniform`, so no candidate is ever
        eliminated and the posterior cannot drift far from the hard-facts prior.
        """
        n = int(self.candidates.sum())
        if self.opp_hand_size <= 0 or self.certain or n == 0:
            return
        inferred = self.probs.copy()
        laid = captured is None or len(captured) == 0
        if laid and self.declined_penalty > 0.0 and table_before_play is not None:
            table_vals = {int(CARD_VALUES[int(c)]) for c in table_before_play}
            if table_vals:
                for c in np.flatnonzero(self.candidates):
                    if int(CARD_VALUES[c]) in table_vals:
                        inferred[c] *= 1.0 - self.declined_penalty
        if self.goal_pref > 0.0:
            denari = self.candidates.copy()
            denari[N_VALUES:] = False  # denari are card indices 0..N_VALUES-1
            inferred[denari] *= 1.0 + self.goal_pref
        uniform = np.zeros(N_CARDS, dtype=np.float64)
        uniform[self.candidates] = self.opp_hand_size / n
        inf_sum = float(inferred.sum())
        if inf_sum > EPSILON:
            inferred *= self.opp_hand_size / inf_sum
        self.probs = (1.0 - self.alpha) * inferred + self.alpha * uniform
        self._normalize()

    @overload
    def get_opponent_hand_probabilities(
        self, as_array: Literal[False] = ...
    ) -> dict[int, float]: ...

    @overload
    def get_opponent_hand_probabilities(self, as_array: Literal[True]) -> ProbArray: ...

    def get_opponent_hand_probabilities(
        self, as_array: bool = False
    ) -> dict[int, float] | ProbArray:
        """Export estimated probabilities, sorted high→low (dict) or as a copy.

        `as_array=True` yields the raw (40,) float64 vector (for determinization
        weights); the default yields a descending {card: probability} mapping.
        """
        if as_array:
            probs: ProbArray = self.probs.copy()
            return probs
        idx = np.flatnonzero(self.probs > EPSILON)
        order = idx[np.argsort(-self.probs[idx], kind="stable")]
        return {int(c): float(self.probs[c]) for c in order}

    # --- internals ------------------------------------------------------

    def _normalize(self) -> None:
        """Scale probs so their sum equals the opponent's hand count exactly."""
        if self.opp_hand_size <= 0:
            self.probs[:] = 0.0
            return
        total = float(self.probs.sum())
        if total < EPSILON:
            # Impossible state: fall back to a uniform prior over candidates.
            n = int(self.candidates.sum())
            self.probs[:] = 0.0
            if n > 0:
                self.probs[self.candidates] = self.opp_hand_size / n
            return
        self.probs *= self.opp_hand_size / total
        if self.certain:
            self.probs[self.probs > EPSILON] = 1.0
