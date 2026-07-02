"""End-game overlay: dim the felt and draw the winner and point breakdown.

Stateless drawing split out of `gui.render` so the live-board renderer stays
focused on cards. Reads the precomputed `Scoreboard` and lays its rows out in
three columns (label, you, bot).
"""

from __future__ import annotations

import pygame

from gui import config
from gui.scoreboard import Scoreboard, scoreboard_rows

# Overlay column anchors (label right-edge, human center, bot center).
_COL_LABEL_X = 360
_COL_HUMAN_X = 470
_COL_BOT_X = 600
_ROW_TOP = 190
_ROW_STEP = 40


def _draw_row(
    surface: pygame.Surface, font: pygame.font.Font, y: int, cells: tuple[str, str, str]
) -> None:
    label, human, bot = cells
    lbl = font.render(label, True, config.OVERLAY_TEXT)
    surface.blit(lbl, lbl.get_rect(midright=(_COL_LABEL_X, y)))
    for text, x in ((human, _COL_HUMAN_X), (bot, _COL_BOT_X)):
        cell = font.render(text, True, config.OVERLAY_TEXT)
        surface.blit(cell, cell.get_rect(center=(x, y)))


def draw_end_game_overlay(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    text_font: pygame.font.Font,
    board: Scoreboard,
    match_line: str | None = None,
    hint: str = "Space / Enter: new game     Esc: quit",
) -> None:
    """Dim the board and draw the winner declaration and point breakdown.

    In match mode `match_line` carries the running match score (and, once
    decided, the match winner); `hint` describes what Space does next.
    """
    veil = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
    veil.fill(config.OVERLAY_RGBA)
    surface.blit(veil, (0, 0))

    center_x = config.WINDOW_WIDTH // 2
    title = title_font.render(board.winner, True, config.TITLE_TEXT)
    surface.blit(title, title.get_rect(center=(center_x, 110)))

    if match_line is not None:
        line = text_font.render(match_line, True, config.TITLE_TEXT)
        surface.blit(line, line.get_rect(center=(center_x, 155)))

    _draw_row(surface, text_font, _ROW_TOP, ("", "You", "Bot"))
    for i, row in enumerate(scoreboard_rows(board), start=1):
        _draw_row(surface, text_font, _ROW_TOP + i * _ROW_STEP, row)

    rendered = text_font.render(hint, True, config.HINT_TEXT)
    surface.blit(rendered, rendered.get_rect(center=(center_x, config.WINDOW_HEIGHT - 45)))
