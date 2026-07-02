"""Move-in-progress value types for the GUI controller.

Split out of `gui.game` so the controller can carry the extra session/logging
state without exceeding the file-size budget. `PendingMove` models an ambiguous
human capture awaiting table-card selection; `CaptureAnim` is a capture whose
staged animation is a pure function of elapsed time (never restarted or looped).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gui import config


@dataclass(slots=True)
class PendingMove:
    """An ambiguous human capture awaiting table-card selection."""

    card: int
    options: list[list[int]]
    selected: set[int] = field(default_factory=set)

    def candidates(self) -> set[int]:
        """Every table card that appears in some legal option (highlightable)."""
        return {c for option in self.options for c in option}

    def matched_option(self) -> list[int] | None:
        """The option exactly equal to the current selection, if any."""
        chosen = sorted(self.selected)
        for option in self.options:
            if sorted(option) == chosen:
                return option
        return None


@dataclass(slots=True)
class CaptureAnim:
    """A capture mid-animation, advanced purely by `elapsed` so its visual state
    is a deterministic function of time (never restarted or looped).

    Three back-to-back phases: the played card slides onto the matched table
    cards (approach), the table cards snap into one stack beneath it (pause),
    then the unified pile travels to `player`'s capture deck (travel).
    """

    card: int
    capture: list[int]
    player: int
    elapsed: float = 0.0

    @property
    def total(self) -> float:
        return config.CAPTURE_APPROACH_MS + config.CAPTURE_PAUSE_MS + config.CAPTURE_TRAVEL_MS

    @property
    def done(self) -> bool:
        return self.elapsed >= self.total

    @property
    def stacked(self) -> bool:
        """True once the approach finished and the cards form one pile."""
        return self.elapsed >= config.CAPTURE_APPROACH_MS

    @property
    def approach_t(self) -> float:
        """Played-card travel progress onto the table cards, clamped to [0, 1]."""
        return min(1.0, self.elapsed / config.CAPTURE_APPROACH_MS)

    @property
    def travel_t(self) -> float:
        """Unified-pile travel progress toward the capture deck, clamped to [0, 1]."""
        start = config.CAPTURE_APPROACH_MS + config.CAPTURE_PAUSE_MS
        return max(0.0, min(1.0, (self.elapsed - start) / config.CAPTURE_TRAVEL_MS))
