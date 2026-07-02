"""Pre-game setup screen: pick a single deal or a match, and move recording.

The selection *state* (`MenuState`) is pure and pygame-free so it can be unit
tested; it turns straight into a `SessionConfig` via `gui.session.gui_config`.
Rendering and hit-testing are thin helpers over that state. Bot selection is not
offered: the GUI ships a single trained champion opponent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from botconfig import default_bot_name
from gui import config
from gui.session import gui_config
from session import SessionConfig

# (label, match target) -- target None means a single logged deal.
MODES: list[tuple[str, float | None]] = [
    ("Single deal", None),
    ("Match to 11", 11.0),
    ("Match to 21", 21.0),
]

_BUTTON_WIDTH = 320
_BUTTON_HEIGHT = 48
_BUTTON_GAP = 16
_FIRST_Y = 170
_BUTTON_FILL: config.Color = (24, 90, 24)
_BUTTON_SELECTED: config.Color = (60, 200, 90)

# Action strings returned by hit-testing: one per clickable button.
_ACTIONS = [f"mode:{i}" for i in range(len(MODES))] + ["record", "start"]


@dataclass(slots=True)
class MenuState:
    """Current menu selection: which mode, and whether to record moves."""

    mode_index: int = 0
    record_moves: bool = False

    def set_mode(self, index: int) -> None:
        self.mode_index = index % len(MODES)

    def toggle_record(self) -> None:
        self.record_moves = not self.record_moves

    @property
    def target(self) -> float | None:
        return MODES[self.mode_index][1]

    def to_config(self) -> SessionConfig:
        """Turn the current selection into a logging session config."""
        return gui_config(self.target, self.record_moves)

    def apply(self, action: str) -> bool:
        """Apply a hit-test action; return True if it was the Start button."""
        if action == "start":
            return True
        if action == "record":
            self.toggle_record()
        elif action.startswith("mode:"):
            self.set_mode(int(action.split(":")[1]))
        return False


def _button_rect(index: int) -> pygame.Rect:
    x = (config.WINDOW_WIDTH - _BUTTON_WIDTH) // 2
    y = _FIRST_Y + index * (_BUTTON_HEIGHT + _BUTTON_GAP)
    return pygame.Rect(x, y, _BUTTON_WIDTH, _BUTTON_HEIGHT)


def menu_buttons() -> list[tuple[pygame.Rect, str]]:
    """Every clickable button as ``(rect, action)`` in draw order."""
    return [(_button_rect(i), action) for i, action in enumerate(_ACTIONS)]


def menu_action_at(pos: tuple[int, int]) -> str | None:
    """The action of the button under `pos`, or None if the click missed."""
    for rect, action in menu_buttons():
        if rect.collidepoint(pos):
            return action
    return None


def _button_label(action: str, state: MenuState) -> tuple[str, bool]:
    """(caption, selected) for a button given the current state."""
    if action == "start":
        return "Start", False
    if action == "record":
        return f"Record moves: {'ON' if state.record_moves else 'OFF'}", state.record_moves
    index = int(action.split(":")[1])
    return MODES[index][0], state.mode_index == index


def draw_menu(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    text_font: pygame.font.Font,
    state: MenuState,
) -> None:
    """Draw the setup screen: title, mode buttons, record toggle, Start."""
    surface.fill(config.FELT_GREEN)
    center_x = config.WINDOW_WIDTH // 2
    title = title_font.render(config.WINDOW_TITLE, True, config.TITLE_TEXT)
    surface.blit(title, title.get_rect(center=(center_x, 100)))
    opponent = text_font.render(f"Opponent: {default_bot_name()}", True, config.HINT_TEXT)
    surface.blit(opponent, opponent.get_rect(center=(center_x, 138)))
    for rect, action in menu_buttons():
        caption, selected = _button_label(action, state)
        pygame.draw.rect(surface, _BUTTON_FILL, rect, border_radius=config.CARD_RADIUS)
        border = _BUTTON_SELECTED if selected else config.CARD_BORDER
        pygame.draw.rect(surface, border, rect, width=3, border_radius=config.CARD_RADIUS)
        label = text_font.render(caption, True, config.LABEL_TEXT)
        surface.blit(label, label.get_rect(center=rect.center))
    hint = text_font.render("Click Start to play", True, config.HINT_TEXT)
    surface.blit(hint, hint.get_rect(center=(center_x, config.WINDOW_HEIGHT - 45)))
