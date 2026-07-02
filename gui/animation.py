"""Delta-time eased interpolation for sliding cards to their new positions.

`AnimationManager` keeps one in-flight slide per card. Each frame the app feeds
it the latest per-card target centers; any card whose home jumped starts a short
ease-out glide from its previous spot. `update(dt)` advances every glide by the
elapsed milliseconds, and the renderer draws animating cards at `position(card)`
instead of their static slot.
"""

from __future__ import annotations

from dataclasses import dataclass

from gui import config

Point = tuple[float, float]


def _distance(a: Point, b: Point) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def ease_out(t: float) -> float:
    """Cubic ease-out: brisk start, gentle deceleration into the target."""
    return 1.0 - (1.0 - t) ** 3


@dataclass(slots=True)
class Slide:
    """A single card's eased glide from `start` to `end` over `duration` ms."""

    start: Point
    end: Point
    elapsed: float
    duration: float

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration

    def position(self) -> Point:
        t = 1.0 if self.duration <= 0 else min(1.0, self.elapsed / self.duration)
        e = ease_out(t)
        return (
            self.start[0] + (self.end[0] - self.start[0]) * e,
            self.start[1] + (self.end[1] - self.start[1]) * e,
        )


class AnimationManager:
    """Tracks active card slides and advances them by real elapsed time."""

    _slides: dict[int, Slide]

    def __init__(self) -> None:
        self._slides = {}

    def sync(self, targets: dict[int, Point], previous: dict[int, Point]) -> None:
        """Start a glide for every card whose target moved since last frame."""
        for card, end in targets.items():
            start = previous.get(card)
            if start is None or _distance(start, end) < config.ANIM_MIN_TRAVEL:
                continue
            active = self._slides.get(card)
            if active is not None and active.end == end:
                continue
            origin = active.position() if active is not None else start
            self._slides[card] = Slide(origin, end, 0.0, config.ANIM_DURATION_MS)

    def update(self, dt: float) -> None:
        """Advance every slide by `dt` ms and drop the ones that finished."""
        for card in list(self._slides):
            slide = self._slides[card]
            slide.elapsed += dt
            if slide.done:
                del self._slides[card]

    def active_cards(self) -> set[int]:
        """Cards currently mid-slide; the renderer skips their static slot."""
        return set(self._slides)

    def position(self, card: int) -> Point:
        """The interpolated center for an animating card."""
        return self._slides[card].position()
