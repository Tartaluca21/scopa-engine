"""Tests for the self-play dataset generator (Fase 9, stage 2)."""

from __future__ import annotations

import numpy as np

from learning.encoder import FEATURE_DIM
from learning.selfplay_data import _play_and_record, generate_dataset


def test_shapes_and_finiteness() -> None:
    x, y = generate_dataset(n_deals=2, seed=1, workers=1)
    assert x.ndim == 2 and x.shape[1] == FEATURE_DIM
    assert x.shape[0] == y.shape[0] > 0
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert x.dtype == np.float32 and y.dtype == np.float32


def test_labels_are_pov_margin_pair() -> None:
    # Within one deal every label is +margin0 (p0 states) or -margin0 (p1 states).
    feats, labels = _play_and_record(seed=3)
    assert len(feats) == len(labels) > 0
    uniq = set(np.round(labels, 6))
    assert len(uniq) <= 2
    if len(uniq) == 2:
        a, b = uniq
        assert np.isclose(a, -b)


def test_reproducible_across_worker_counts() -> None:
    x1, y1 = generate_dataset(n_deals=3, seed=5, workers=1)
    x2, y2 = generate_dataset(n_deals=3, seed=5, workers=1)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)


def test_empty_when_no_deals() -> None:
    x, y = generate_dataset(n_deals=0, workers=1)
    assert x.shape == (0, FEATURE_DIM) and y.shape == (0,)
