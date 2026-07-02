# Card Assets (traditional 40-card regional deck, Del Negro style)

Drop PNG images here to replace the drawn-rectangle fallback. Missing files are
tolerated: any card without an image renders as a labeled rounded rectangle, so
the game never crashes on absent assets.

## Naming convention

- Faces: `<value>_<suit>.png` where `value` is `1`–`10` and `suit` is one of
  `denari`, `coppe`, `bastoni`, `spade`. Example: `7_denari.png`.
- Card back: `back.png` (shared by both capture piles and face-down hands).

Images are scaled to the slot size at draw time, so any consistent resolution
works (a portrait aspect ratio close to 70x100 looks best).
