"""The Pygame application shell: window, game loop, and input handling.

`ScopaApp` owns the window, clock, fonts, sprite cache, a setup `MenuState`, and
a `SessionController`. It runs two phases: the menu (pick a single deal or match,
toggle move recording) and play. Each play frame it ticks the session (deals +
async bot + result logging), processes input, advances the delta-time slide
animations, and redraws the live board. The render loop stays at 60 FPS while the
bot thinks on a background thread.
"""

from __future__ import annotations

import pygame

from engine.cards import HAND_ZONES
from gui import capture, config, menu, overlay, render
from gui.animation import AnimationManager, Point
from gui.game import GameController
from gui.layout import (
    OPPONENT,
    CapturePile,
    Zone,
    build_piles,
    build_zones,
    card_positions,
    hit_test,
    hit_test_table,
)
from gui.menu import MenuState
from gui.session import SessionController, gui_config
from gui.sprites import SpriteCache


class ScopaApp:
    """Owns the pygame window and runs the main render/input loop."""

    screen: pygame.Surface
    clock: pygame.time.Clock
    label_font: pygame.font.Font
    card_font: pygame.font.Font
    status_font: pygame.font.Font
    title_font: pygame.font.Font
    session: SessionController
    menu: MenuState
    phase: str
    sprites: SpriteCache
    anim: AnimationManager
    zones: list[Zone]
    piles: list[CapturePile]
    prev_pos: dict[int, Point]
    running: bool

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.label_font = pygame.font.SysFont("arial", config.LABEL_FONT_SIZE, bold=True)
        self.card_font = pygame.font.SysFont("arial", config.CARD_FONT_SIZE, bold=True)
        self.status_font = pygame.font.SysFont("arial", config.LABEL_FONT_SIZE)
        self.title_font = pygame.font.SysFont("arial", config.TITLE_FONT_SIZE, bold=True)
        self.menu = MenuState()
        self.phase = "menu"
        self.session = SessionController(gui_config())
        self.sprites = SpriteCache()
        self.anim = AnimationManager()
        self.running = True
        self._rebuild_layout()
        self.prev_pos = card_positions(self.zones, self.piles)

    @property
    def game(self) -> GameController:
        """The live single-deal controller the renderer reads each frame."""
        return self.session.controller

    def _rebuild_layout(self) -> None:
        """Recompute zones and piles from the live engine for this frame."""
        self.zones = build_zones(self.game.engine)
        self.piles = build_piles(self.game.engine)

    def _snap_to_board(self) -> None:
        """Snap animations to the current fresh board, with no sliding."""
        self.anim = AnimationManager()
        self._rebuild_layout()
        self.prev_pos = card_positions(self.zones, self.piles)

    def _advance(self) -> None:
        """Continue the session (next deal or a fresh game) and resnap the board."""
        self.session.advance()
        self._snap_to_board()

    def _start_from_menu(self) -> None:
        """Leave the menu: build the chosen session and enter play."""
        self.session = SessionController(self.menu.to_config())
        self.phase = "play"
        self._snap_to_board()

    def _to_menu(self) -> None:
        """Return to the setup screen (only offered once a game is over)."""
        self.phase = "menu"

    def _on_left_click(self, pos: tuple[int, int]) -> None:
        """Route a left-click: pick a capture card, or play a hand card."""
        if self.game.is_over():
            return
        if self.game.pending is not None:
            card = hit_test_table(self.zones, pos)
            if card is not None:
                self.game.toggle_table_card(card)
        elif self.game.is_human_turn():
            card = hit_test(self.zones, pos)
            if card is not None:
                self.game.begin_human_move(card)

    def _on_escape(self) -> None:
        """Multi-level Escape: cancel a pending capture, else quit the game."""
        if self.game.pending is not None:
            self.game.cancel_pending()
        else:
            self.running = False

    def _on_keydown(self, key: int) -> None:
        """Escape cancels/quits; once over, Space starts the next, M opens menu."""
        if key == pygame.K_ESCAPE:
            self._on_escape()
        elif self.session.is_over() and key in (pygame.K_SPACE, pygame.K_RETURN):
            self._advance()
        elif self.session.is_over() and key == pygame.K_m:
            self._to_menu()

    def handle_events(self) -> None:
        """Process the event queue: quit, keys, left-click, right-click."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._on_keydown(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._on_left_click(event.pos)
                elif event.button == 3:
                    self.game.cancel_pending()

    def handle_menu_events(self) -> None:
        """Process the setup screen: quit, Escape, and button clicks."""
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                action = menu.menu_action_at(event.pos)
                if action is not None and self.menu.apply(action):
                    self._start_from_menu()

    def _step_animations(self, dt: float, group: set[int]) -> set[int]:
        """Advance the diff-based slides, leaving the capture group to `capture`.

        Cards in the staged capture are driven deterministically by the
        `CaptureAnim` clock, so they are kept out of the slide animator entirely
        to avoid a target diff restarting or looping their motion mid-flight.
        """
        targets = card_positions(self.zones, self.piles)
        for card in group:
            targets.pop(card, None)
            self.prev_pos.pop(card, None)
        self.anim.sync(targets, self.prev_pos)
        self.anim.update(dt)
        self.prev_pos = targets
        return self.anim.active_cards()

    def _capture_moving(self) -> tuple[list[tuple[int, bool, Point]], set[int], set[int]]:
        """Moving entries, the full group, and the glowing-target set, if any.

        The glow set holds the captured table cards while the played card is still
        approaching or paused on top of them (before the pile flies to the deck).
        """
        anim = self.game.capture_anim
        if anim is None:
            return [], set(), set()
        home = card_positions(self.zones, self.piles)
        pile = self.piles[anim.player]
        center = (float(pile.center[0]), float(pile.center[1]))
        group = capture.capture_group(anim, home, center)
        glow = set(anim.capture) if anim.travel_t == 0.0 else set()
        return [(card, True, pos) for card, pos in group], {card for card, _ in group}, glow

    def _slide_moving(self, sliding: set[int]) -> list[tuple[int, bool, Point]]:
        """Moving entries for plain diff-driven slides, keeping hands face-down."""
        hidden = {int(c) for c in self.game.engine.cards_in(HAND_ZONES[OPPONENT])}
        return [(card, card not in hidden, self.anim.position(card)) for card in sliding]

    def draw(self, dt: float) -> None:
        """Render one frame: felt, zones, piles, sliding cards, status, overlay."""
        self._rebuild_layout()
        capture_moving, group, glow = self._capture_moving()
        sliding = self._step_animations(dt, group)
        skip = sliding | group
        highlight, selected = self.game.selection_highlight()
        moving = self._slide_moving(sliding) + capture_moving
        render.draw_background(self.screen)
        render.draw_capture_piles(self.screen, self.piles, self.sprites, self.card_font, skip)
        render.draw_zones(
            self.screen,
            self.zones,
            self.label_font,
            self.card_font,
            self.sprites,
            highlight,
            selected,
            skip,
        )
        # Moving cards paint last so the sliding/capture pile always wins z-order.
        render.draw_moving_cards(self.screen, moving, self.sprites, self.card_font, glow)
        if self.game.bot_thinking:
            render.draw_status(self.screen, self.status_font, "Bot is thinking...")
        if self.game.is_over():
            hint = (
                "Space: next deal     M: menu     Esc: quit"
                if self.session.awaiting_next_deal()
                else "Space: new game     M: menu     Esc: quit"
            )
            overlay.draw_end_game_overlay(
                self.screen,
                self.title_font,
                self.status_font,
                self.game.scoreboard(),
                self.session.match_status_line(),
                hint,
            )
        pygame.display.flip()

    def run(self) -> None:
        """Run the main loop at a steady framerate until the window is closed."""
        while self.running:
            dt = self.clock.tick(config.FPS)
            if self.phase == "menu":
                self.handle_menu_events()
                menu.draw_menu(self.screen, self.title_font, self.status_font, self.menu)
                pygame.display.flip()
            else:
                self.session.tick(dt)
                self.handle_events()
                self.draw(dt)
        pygame.quit()
