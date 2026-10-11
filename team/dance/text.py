"""Text for the video HUD: real fonts through Pillow, rendered once per (text, size, colour) into a cached BGRA sprite and
alpha-blended into the frame (only the sprite's rectangle is touched, so a full HUD costs a few milliseconds).

Fonts (OFL, team/dance/assets/fonts), as the Brodie character kit specifies: Archivo (Black) for display, Plus Jakarta Sans for body. Falls back to Windows fonts,
then to OpenCV's Hershey font when Pillow or the files are missing.

  put_text(img, text, org, px, color, font='body', anchor='ls', stroke=0, stroke_color=(0,0,0), alpha=1.0) -> right x
  text_size(text, px, font='body') -> (w, h)
anchor: two letters like Pillow's ('l'/'m'/'r' horizontal, 'a'/'m'/'s'/'d' vertical); 'ls' = left baseline, the way cv2.putText
places text, so old call sites keep their layout.
"""
import os
from functools import lru_cache
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, 'assets', 'fonts')
CANDIDATES = dict(
    display=[os.path.join(FONT_DIR, 'ArchivoBlack-Regular.ttf'), r'C:\Windows\Fonts\ariblk.ttf', '/System/Library/Fonts/Supplemental/Arial Black.ttf'],
    body=[os.path.join(FONT_DIR, 'PlusJakartaSans.ttf'), os.path.join(FONT_DIR, 'SpaceGrotesk.ttf'), r'C:\Windows\Fonts\segoeuib.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'],
)
try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:                       # pragma: no cover
    HAVE_PIL = False


@lru_cache(maxsize=64)
def _font(kind, px):
    if not HAVE_PIL: return None
    for p in CANDIDATES.get(kind, CANDIDATES['body']):
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, int(px))
                if 'SpaceGrotesk' in p or 'PlusJakarta' in p:
                    try: f.set_variation_by_axes([650])
                    except Exception: pass
                return f
            except Exception:
                continue
    return None


def available():
    return _font('body', 20) is not None


@lru_cache(maxsize=1024)
def _sprite(text, px, color, kind, stroke, stroke_color):
    """(BGRA uint8 sprite, (ascent offset x, y of the baseline inside the sprite))."""
    f = _font(kind, px)
    l, t, r, b = f.getbbox(text, stroke_width=stroke, anchor='ls')
    w, h = max(1, r - l), max(1, b - t)
    im = Image.new('RGBA', (w + 2, h + 2), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    srgb = (int(stroke_color[2]), int(stroke_color[1]), int(stroke_color[0]))
    d.text((1 - l, 1 - t), text, font=f, fill=rgb + (255,), anchor='ls', stroke_width=stroke, stroke_fill=srgb + (255,))
    a = np.asarray(im, np.uint8)
    bgra = a[:, :, [2, 1, 0, 3]].copy()
    return bgra, (1 - l, 1 - t)


@lru_cache(maxsize=2048)
def _text_size(text, px, font):
    f = _font(font, int(px))
    if f is None:
        sc = px / 34.0; (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, 2); return w, h
    l, t, r, b = f.getbbox(text or ' ', anchor='ls')
    return r - l, b - t


def text_size(text, px, font='body'):
    return _text_size(str(text), int(px), font)


def _blend(img, spr, x, y, alpha=1.0):
    H, W = img.shape[:2]; h, w = spr.shape[:2]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (s[:, :, 3:4].astype(np.float32) / 255.0) * float(alpha)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1.0 - a) + s[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def put_text(img, text, org, px, color=(245, 245, 245), font='body', anchor='ls', stroke=0, stroke_color=(0, 0, 0), alpha=1.0, scale=1.0):
    """Draw text; returns the x just right of it. scale: extra multiplier for animations (sprites are cached per size)."""
    text = str(text); px = max(6, int(round(px * scale)))
    if not text: return org[0]
    f = _font(font, px)
    if f is None:
        sc = px / 34.0; th = max(1, int(px / 14))
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
        x, y = org
        if anchor[0] == 'm': x -= w // 2
        elif anchor[0] == 'r': x -= w
        if anchor[1] == 'm': y += h // 2
        elif anchor[1] == 'a': y += h
        if stroke: cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, stroke_color, th + 2 * stroke, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)
        return x + w
    spr, (bx, by) = _sprite(text, px, tuple(int(c) for c in color), font, int(stroke), tuple(int(c) for c in stroke_color))
    h, w = spr.shape[:2]
    x, y = int(org[0]), int(org[1])
    # place so that the baseline-left point (bx, by) of the sprite lands on org, then shift by the anchor
    left = x - bx; top = y - by
    tw, th = w - 2, h - 2
    if anchor[0] == 'm': left -= tw // 2
    elif anchor[0] == 'r': left -= tw
    if anchor[1] == 'm': top += int(round((by - 1) * 0.5))          # centre of the glyphs above the baseline on org's y
    elif anchor[1] == 'a': top += by - 1                              # top of the glyphs on org's y
    _blend(img, spr, left, top, alpha)
    return left + w
