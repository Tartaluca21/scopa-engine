"""Geometry for the staged capture animation's unified card pile.

Given a `CaptureAnim` and the cards' static home centers, this computes where
every card in the capture group should be drawn this frame. The played card
slides onto the matched table cards, the table cards then snap into one stack
beneath it, and finally the whole pile travels together to the capture deck.

The captured cards are listed first and the played card last, so the renderer
(which draws moving cards in order) keeps the played card on top of the pile.
"""

from __future__ import annotations

from gui import config
from gui.animation import ease_out
from gui.moves import CaptureAnim

Point = tuple[float, float]


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _centroid(points: list[Point], fallback: Point) -> Point:
    if not points:
        return fallback
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def capture_group(
    anim: CaptureAnim,
    home: dict[int, Point],
    pile_center: Point,
) -> list[tuple[int, Point]]:
    """Per-card draw centers for the capture group, captured cards first.

    `home` maps each live card to its static slot center; the played card sits in
    a hand and the captured cards on the table. The result feeds the moving-card
    renderer so the whole group always draws above the static board.
    """
    played_home = home.get(anim.card, pile_center)
    captured = [c for c in anim.capture if c in home]
    landing = _centroid([home[c] for c in captured], played_home)

    if not anim.stacked:
        # Approach: table cards stay put while the played card glides onto them.
        group: list[tuple[int, Point]] = [(c, home[c]) for c in captured]
        group.append((anim.card, _lerp(played_home, landing, ease_out(anim.approach_t))))
        return group

    # Stacked: cards form one offset pile, then travel together to the deck. The
    # shared target collapses the offsets to zero as the pile reaches the deck.
    travel = ease_out(anim.travel_t)
    group = []
    for i, card in enumerate([*captured, anim.card]):
        off = i * config.CAPTURE_STACK_OFFSET
        stacked = (landing[0] + off, landing[1] - off)
        group.append((card, _lerp(stacked, pile_center, travel)))
    return group
