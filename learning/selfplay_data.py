"""Self-play dataset generation for the learned value function (Fase 9, stage 2).

Plays full PIMC self-play deals and records, at every decision, the POV-encoded
state (`learning.encoder`) paired with the eventual deal margin from that side's
point of view -- a Monte-Carlo value target `V(s) = E[final margin | s, policy]`.
Both seats use the deployed `default_pimc_config()`, so the targets reflect the
policy a learned leaf evaluator would be assisting.

Deterministic and reproducible: deal `i` is driven entirely by `seed + i`, so the
dataset is identical for any worker count. Numpy-only; no ML dependency.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.typing as npt

from botconfig import default_pimc_config
from engine.cards import Zone
from engine.core import ScopaEngine
from engine.heuristic import score_deal
from learning.encoder import FEATURE_DIM, encode_state
from search.pimc import pimc_decide

FloatArray = npt.NDArray[np.float32]


def _play_and_record(seed: int) -> tuple[list[FloatArray], list[float]]:
    """Play one seeded PIMC self-play deal; return (per-ply features, POV labels)."""
    cfg = default_pimc_config()
    rng = np.random.default_rng(seed)
    engine = ScopaEngine()
    engine.deal_round(rng)
    feats: list[FloatArray] = []
    players: list[int] = []
    while not engine.is_game_over():
        if engine.count(Zone.MANO_P1) == 0 and engine.count(Zone.MANO_P2) == 0:
            engine.deal_round(rng)
            continue
        p = engine.current_player
        feats.append(encode_state(engine, p))
        players.append(p)
        card, cap = pimc_decide(engine, p, cfg, rng)
        engine.execute_move(card, cap)
    engine.end_of_deal_sweep()
    p0, p1 = score_deal(engine)
    margin0 = p0 - p1
    labels = [float(margin0 if pl == 0 else -margin0) for pl in players]
    return feats, labels


def generate_dataset(
    n_deals: int, seed: int = 0, workers: int = 8
) -> tuple[FloatArray, FloatArray]:
    """Return `(X, y)`: features `(M, FEATURE_DIM)` and POV margins `(M,)`.

    `M` is the total number of decisions across `n_deals` deals (~36/deal). Set
    `workers <= 1` to run sequentially (used in tests).
    """
    seeds = [seed + i for i in range(n_deals)]
    if workers <= 1:
        results = [_play_and_record(s) for s in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_play_and_record, seeds))
    feats: list[FloatArray] = []
    labels: list[float] = []
    for f, y in results:
        feats.extend(f)
        labels.extend(y)
    if not feats:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.float32)
