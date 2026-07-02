"""Drawing helpers: paint the felt, zones, capture piles, and sliding cards.

Stateless functions that turn `gui.layout` geometry into pixels. Each card is
painted from its loaded sprite when available, falling back to a rounded
rectangle with a text code so the board renders even without image assets. The
`skip` set lets the app hide cards that are mid-animation from their static home
so only the moving copy is drawn.
"""

from __future__ import annotations

import pygame

from gui import config
from gui.cards import card_code
from gui.layout import CapturePile, CardSlot, Zone
from gui.sprites import SpriteCache

Point = tuple[float, float]


def draw_background(surface: pygame.Surface) -> None:
    """Fill the whole surface with the classic card-table felt green."""
    surface.fill(config.FELT_GREEN)


def _outline_color(card: int, highlight: set[int], selected: set[int]) -> config.Color | None:
    if card in selected:
        return config.CARD_SELECTED
    if card in highlight:
        return config.CARD_HIGHLIGHT
    return None


def _paint_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    card: int,
    face_up: bool,
    sprites: SpriteCache,
    font: pygame.font.Font,
) -> None:
    """Blit a card's sprite into `rect`, or draw a labeled rectangle fallback."""
    image = sprites.face(card) if face_up else sprites.back()
    if image is not None:
        surface.blit(pygame.transform.smoothscale(image, (rect.width, rect.height)), rect)
        return
    fill = config.CARD_FACE if face_up else config.CARD_BACK
    pygame.draw.rect(surface, fill, rect, border_radius=config.CARD_RADIUS)
    pygame.draw.rect(surface, config.CARD_BORDER, rect, width=2, border_radius=config.CARD_RADIUS)
    if face_up:
        text = font.render(card_code(card), True, config.CARD_TEXT)
        surface.blit(text, text.get_rect(center=rect.center))


def _draw_slot(
    surface: pygame.Surface,
    slot: CardSlot,
    font: pygame.font.Font,
    sprites: SpriteCache,
    highlight: set[int],
    selected: set[int],
) -> None:
    rect = pygame.Rect(slot.rect.x, slot.rect.y, slot.rect.width, slot.rect.height)
    _paint_card(surface, rect, slot.card, slot.face_up, sprites, font)
    outline = _outline_color(slot.card, highlight, selected)
    if outline is not None:
        pygame.draw.rect(
            surface, outline, rect, width=config.HIGHLIGHT_WIDTH, border_radius=config.CARD_RADIUS
        )


def draw_zone(
    surface: pygame.Surface,
    zone: Zone,
    label_font: pygame.font.Font,
    card_font: pygame.font.Font,
    sprites: SpriteCache,
    highlight: set[int],
    selected: set[int],
    skip: set[int],
) -> None:
    """Draw a zone's title and its row of live card slots with overlays."""
    title = label_font.render(zone.label, True, config.LABEL_TEXT)
    title_y = zone.band_y - config.CARD_HEIGHT // 2 - config.LABEL_FONT_SIZE - 6
    surface.blit(title, title.get_rect(center=(config.WINDOW_WIDTH // 2, title_y)))
    for slot in zone.slots:
        if slot.card not in skip:
            _draw_slot(surface, slot, card_font, sprites, highlight, selected)


def draw_zones(
    surface: pygame.Surface,
    zones: list[Zone],
    label_font: pygame.font.Font,
    card_font: pygame.font.Font,
    sprites: SpriteCache,
    highlight: set[int] | None = None,
    selected: set[int] | None = None,
    skip: set[int] | None = None,
) -> None:
    """Draw every zone onto the surface, outlining highlight/selected cards."""
    highlight = highlight or set()
    selected = selected or set()
    skip = skip or set()
    for zone in zones:
        draw_zone(surface, zone, label_font, card_font, sprites, highlight, selected, skip)


def _draw_scopa_markers(surface: pygame.Surface, pile: CapturePile, sprites: SpriteCache) -> None:
    cx, cy = pile.center
    w, h = config.PILE_CARD_WIDTH, config.PILE_CARD_HEIGHT
    top = cy - h // 2
    for i in range(pile.scope):
        card = pile.cards[-(i + 1)] if i < len(pile.cards) else None
        center = (cx + config.SCOPA_MARKER_DX, top + i * config.SCOPA_MARKER_DY)
        face = sprites.face(card) if card is not None else None
        if face is not None:
            scaled = pygame.transform.smoothscale(face, (w, h))
            img = pygame.transform.rotate(scaled, config.SCOPA_ROTATION)
            surface.blit(img, img.get_rect(center=center))
        else:
            rect = pygame.Rect(0, 0, h, w)
            rect.center = (int(center[0]), int(center[1]))
            pygame.draw.rect(surface, config.CARD_FACE, rect, border_radius=config.CARD_RADIUS)
            pygame.draw.rect(surface, config.CARD_BORDER, rect, width=2)


def _draw_pile(
    surface: pygame.Surface,
    pile: CapturePile,
    sprites: SpriteCache,
    font: pygame.font.Font,
    skip: set[int],
) -> None:
    cards = [c for c in pile.cards if c not in skip]
    cx, cy = pile.center
    w, h = config.PILE_CARD_WIDTH, config.PILE_CARD_HEIGHT
    visible = min(len(cards), config.PILE_MAX_VISIBLE)
    for i in range(visible):
        off = i * config.PILE_STACK_OFFSET
        rect = pygame.Rect(cx - w // 2 + off, cy - h // 2 - off, w, h)
        _paint_card(surface, rect, cards[i], False, sprites, font)
    _draw_scopa_markers(surface, pile, sprites)
    if pile.cards:
        label = font.render(str(len(pile.cards)), True, config.PILE_TEXT)
        surface.blit(label, label.get_rect(center=(cx, cy + config.PILE_LABEL_DY)))


def draw_capture_piles(
    surface: pygame.Surface,
    piles: list[CapturePile],
    sprites: SpriteCache,
    font: pygame.font.Font,
    skip: set[int] | None = None,
) -> None:
    """Draw both players' stacked capture piles and their scopa markers."""
    skip = skip or set()
    for pile in piles:
        _draw_pile(surface, pile, sprites, font, skip)


def _draw_glow(surface: pygame.Surface, rect: pygame.Rect) -> None:
    """Paint a layered translucent gold halo around `rect` for a glowing edge."""
    color = config.CAPTURE_GLOW_COLOR
    for i, pad in enumerate(config.CAPTURE_GLOW_PADS):
        halo = rect.inflate(pad * 2, pad * 2)
        layer = pygame.Surface((halo.width, halo.height), pygame.SRCALPHA)
        alpha = 80 + i * 70
        pygame.draw.rect(
            layer,
            (*color, alpha),
            layer.get_rect(),
            width=config.CAPTURE_GLOW_WIDTH,
            border_radius=config.CARD_RADIUS,
        )
        surface.blit(layer, halo.topleft)


def draw_moving_cards(
    surface: pygame.Surface,
    moving: list[tuple[int, bool, Point]],
    sprites: SpriteCache,
    font: pygame.font.Font,
    glow: set[int] | None = None,
) -> None:
    """Draw cards mid-slide as full-size sprites centered on their lerp point.

    Cards in `glow` (the table cards being captured during the approach and pause
    phases) get a thick gold halo so the human sees exactly what is being taken.
    """
    glow = glow or set()
    for card, face_up, (x, y) in moving:
        rect = pygame.Rect(0, 0, config.CARD_WIDTH, config.CARD_HEIGHT)
        rect.center = (int(x), int(y))
        _paint_card(surface, rect, card, face_up, sprites, font)
        if card in glow:
            _draw_glow(surface, rect)


def draw_status(surface: pygame.Surface, font: pygame.font.Font, text: str) -> None:
    """Render a subtle status cue (e.g. the bot-thinking indicator)."""
    label = font.render(text, True, config.STATUS_TEXT)
    surface.blit(label, config.STATUS_POS)
