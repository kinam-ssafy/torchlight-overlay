# Localized round font

`round-font.json` contains normalized digit vectors derived from `round-font-atlas.png`.
All ten digits share the same extractor and acceptance thresholds; there are no round-specific rules.

The installed game's `Art/UI/Fonts/tl_kon.ufont` identifies itself as Nowar Warcraft
Rounded CN Bold 0.2.2, licensed under SIL OFL 1.1. Its extracted font SHA256 is
`72e8419f30836848517249bf01b9f8ef17b115e407ac172f781bd8f7dbc3a6a5`.
Copyright 2018-2019 Cyano Hao; portions Google and Adobe. Reserved names and the full
license are in `../../../licenses/Nowar-Rounded-OFL.txt`.
Project: https://github.com/nowar-fonts/Nowar-Rounded

The metadata identifies an available game font, not a proven UI asset binding. A local
pixel comparison of the captured round 4 gave normalized correlation 0.954 for this font,
versus 0.724 for Noto Sans KR SemiBold. The production glyph matcher scores it 0.976.
Korean heading font candidates in the game include Noto Sans KR Regular and SemiBold;
the heading reader still uses OCR, not a hard-coded font assumption.

Reproduce with Pillow and the legally obtained input font:

```
python vorax/scripts/render_round_font.py path/to/tl_kon.ttf
cargo run --release --locked --manifest-path vorax/Cargo.toml -p vorax-scan --example round_font_refs -- vorax/data/scan/round-font-atlas.png vorax/data/scan/round-font.json
```

The small raster sheets and derived vectors are distributed with the OFL notice.
No game archive, full font, extraction key, or decompression binary is included.
`round-font-counters.png` renders whole numbers 1..13 separately to test multi-digit
spacing, five scales and five brightness levels. These are synthetic tests, not claims
of having played all 13 rounds live.
