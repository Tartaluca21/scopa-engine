"""POV-normalized feature encoding of a Scopa state for learned evaluation.

`encode_state(engine, player)` maps a (determinized) `ScopaEngine` to a fixed
-length float32 vector, always from `player`'s point of view: block "me" is
`player`, block "opp" is the other side. This mirrors the symmetric convention of
`search.alphabeta` (values are margins for the side to move), so one model serves
both seats and the training label is the `score_deal` margin from `player`'s POV.

Layout (FEATURE_DIM = 6 * N_CARDS + N_SCALARS):
  * 6 one-hot card planes, in POV order: my hand, opp hand, table, deck,
    my captures, opp captures.
  * N_SCALARS engineered aggregates (counts + the four scoring components per
    side), normalized to roughly [0, 1] for stable training.

Pure numpy; no ML dependency. The encoder reads the engine and never mutates it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engine.cards import HAND_ZONES, N_CARDS, PRESE_ZONES, Zone
from engine.core import ScopaEngine
from engine.features import capture_features

FloatArray = npt.NDArray[np.float32]

_N_PLANES = 6
_N_SCALARS = 14
FEATURE_DIM = _N_PLANES * N_CARDS + _N_SCALARS

# Scalar normalizers: keep every engineered feature roughly in [0, 1].
_CARDS_NORM = 40.0  # a deal has 40 cards
_HAND_NORM = 3.0  # max cards in hand
_SCOPE_NORM = 4.0  # scope per deal is small
_DENARI_NORM = 10.0  # 10 denari in the deck
_PRIMIERA_NORM = 84.0  # max primiera sum ~ 4 suits * 21


def _plane(engine: ScopaEngine, zone: Zone) -> FloatArray:
    plane: FloatArray = engine.state[zone].astype(np.float32)
    return plane


def encode_state(engine: ScopaEngine, player: int) -> FloatArray:
    """Fixed-length POV feature vector for `player` (see module docstring)."""
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")
    opp = 1 - player
    out = np.empty(FEATURE_DIM, dtype=np.float32)

    planes = (
        _plane(engine, HAND_ZONES[player]),
        _plane(engine, HAND_ZONES[opp]),
        _plane(engine, Zone.TAVOLO),
        _plane(engine, Zone.MAZZO),
        _plane(engine, PRESE_ZONES[player]),
        _plane(engine, PRESE_ZONES[opp]),
    )
    out[: _N_PLANES * N_CARDS] = np.concatenate(planes)

    my_cap = capture_features(engine.cards_in(PRESE_ZONES[player]))
    opp_cap = capture_features(engine.cards_in(PRESE_ZONES[opp]))
    scalars = [
        engine.count(HAND_ZONES[player]) / _HAND_NORM,
        engine.count(HAND_ZONES[opp]) / _HAND_NORM,
        engine.count(Zone.TAVOLO) / _CARDS_NORM,
        engine.count(Zone.MAZZO) / _CARDS_NORM,
        my_cap.captures / _CARDS_NORM,
        opp_cap.captures / _CARDS_NORM,
        int(engine.scopa_counts[player]) / _SCOPE_NORM,
        int(engine.scopa_counts[opp]) / _SCOPE_NORM,
        my_cap.denari / _DENARI_NORM,
        opp_cap.denari / _DENARI_NORM,
        float(my_cap.settebello),
        float(opp_cap.settebello),
        my_cap.primiera / _PRIMIERA_NORM,
        opp_cap.primiera / _PRIMIERA_NORM,
    ]
    out[_N_PLANES * N_CARDS :] = np.asarray(scalars, dtype=np.float32)
    return out
