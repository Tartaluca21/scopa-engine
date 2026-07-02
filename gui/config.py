"""GUI constants: window dimensions, colors, framerate, and card geometry.

Single source of truth for every magic number used by the rendering layer, so
layout math (gui.layout) and drawing (gui.render) never hard-code values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

WINDOW_WIDTH: Final = 800
WINDOW_HEIGHT: Final = 600
FPS: Final = 60
WINDOW_TITLE: Final = "Scopa Bot"

# Card placeholder geometry (pixels).
CARD_WIDTH: Final = 70
CARD_HEIGHT: Final = 100
CARD_GAP: Final = 18
CARD_RADIUS: Final = 8

# RGB palette.
Color = tuple[int, int, int]
FELT_GREEN: Final[Color] = (34, 139, 34)
CARD_FACE: Final[Color] = (245, 245, 235)
CARD_BACK: Final[Color] = (120, 30, 40)
CARD_BORDER: Final[Color] = (20, 20, 20)
LABEL_TEXT: Final[Color] = (235, 235, 235)
CARD_TEXT: Final[Color] = (20, 20, 20)
CARD_HIGHLIGHT: Final[Color] = (240, 210, 60)  # capture candidate outline
CARD_SELECTED: Final[Color] = (60, 200, 90)  # chosen capture-card outline
STATUS_TEXT: Final[Color] = (210, 210, 210)  # subtle "Bot is thinking..." cue
HIGHLIGHT_WIDTH: Final = 4
STATUS_POS: Final = (16, 12)

# Glowing outline drawn around table cards being captured (approach + pause).
CAPTURE_GLOW_COLOR: Final[Color] = (255, 215, 0)  # high-contrast gold
CAPTURE_GLOW_WIDTH: Final = 4
CAPTURE_GLOW_PADS: Final[tuple[int, ...]] = (8, 5, 2)  # concentric halo insets

# Vertical band centers for each zone, as a fraction of window height.
OPPONENT_BAND_Y: Final = 0.13
TABLE_BAND_Y: Final = 0.47
PLAYER_BAND_Y: Final = 0.83

LABEL_FONT_SIZE: Final = 22
CARD_FONT_SIZE: Final = 20
TITLE_FONT_SIZE: Final = 48

# End-game overlay.
OVERLAY_RGBA: Final[tuple[int, int, int, int]] = (0, 0, 0, 200)
TITLE_TEXT: Final[Color] = (245, 240, 210)
OVERLAY_TEXT: Final[Color] = (235, 235, 235)
HINT_TEXT: Final[Color] = (180, 180, 180)

# Traditional regional-deck (Del Negro style) image assets. Drop PNGs named
# "<value>_<suit>.png" (e.g. "7_denari.png") plus "back.png" here; a missing
# file falls back to the drawn rectangle so the game never crashes.
ASSET_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "assets" / "cards"
CARD_BACK_FILE: Final = "back.png"

# Capture-pile geometry: a small stacked deck on each side of the felt.
PILE_SCALE: Final = 0.6
PILE_CARD_WIDTH: Final = int(CARD_WIDTH * PILE_SCALE)
PILE_CARD_HEIGHT: Final = int(CARD_HEIGHT * PILE_SCALE)
PILE_STACK_OFFSET: Final = 1  # px shift per stacked card, for a 3D look
PILE_MAX_VISIBLE: Final = 16  # cap drawn cards so deep piles stay tidy
HUMAN_PILE_CENTER: Final[tuple[int, int]] = (740, 470)
BOT_PILE_CENTER: Final[tuple[int, int]] = (740, 130)
PILE_LABEL_DY: Final = 70  # card-count caption offset below the pile center
PILE_TEXT: Final[Color] = (235, 235, 235)

# Scopa marker: a captured card laid face-up and rotated 90 deg (di traverso),
# sticking out of the owner's pile so scope are visible at a glance.
SCOPA_ROTATION: Final = 90
SCOPA_MARKER_DX: Final = -52  # offset left of the pile center
SCOPA_MARKER_DY: Final = 22  # vertical gap between stacked scopa markers

# Eased delta-time slide animation when a card travels to a new home.
ANIM_DURATION_MS: Final = 650.0
ANIM_MIN_TRAVEL: Final = 4.0  # px; below this a move snaps instead of sliding

# Capture staging: the played card slides onto the matched table cards and holds
# for this pause so the human sees the match before the lot flies to the pile.
# The three phases run back to back: approach (played card -> table), pause (the
# table cards snap into one stack under it), travel (the whole pile -> capture deck).
CAPTURE_PAUSE_MS: Final = 600.0
CAPTURE_APPROACH_MS: Final = ANIM_DURATION_MS
CAPTURE_TRAVEL_MS: Final = ANIM_DURATION_MS
CAPTURE_STACK_OFFSET: Final = 6  # px shift per card in the unified capture stack
