"""Render OFL round-font references and an independent 1..13 counter test sheet.

Usage: python vorax/scripts/render_round_font.py path/to/tl_kon.ttf
Requires Pillow. The input font is not distributed with the application.
"""
from pathlib import Path
import argparse
import hashlib
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('font', type=Path)
args = parser.parse_args()
assert hashlib.sha256(args.font.read_bytes()).hexdigest() == '72e8419f30836848517249bf01b9f8ef17b115e407ac172f781bd8f7dbc3a6a5'
root = Path(__file__).resolve().parents[1]
font = ImageFont.truetype(str(args.font), 51)
background, foreground = (24, 20, 16), (188, 158, 100)

def draw_centered(image, text, x, y, width):
    box = font.getbbox(text)
    ImageDraw.Draw(image).text((x + (width - box[2] + box[0]) / 2 - box[0],
                               y + (80 - box[3] + box[1]) / 2 - box[1]),
                              text, font=font, fill=foreground)

atlas = Image.new('RGB', (640, 80), background)
for digit in range(10):
    draw_centered(atlas, str(digit), digit * 64, 0, 64)
atlas.save(root / 'data/scan/round-font-atlas.png')
# Use the font's actual kerning/spacing for whole numbers, not concatenated reference images.
sheet = Image.new('RGB', (96, 13 * 80), background)
for number in range(1, 14):
    draw_centered(sheet, str(number), 0, (number - 1) * 80, 96)
sheet.save(root / 'fixtures/frames/round-font-counters.png')
