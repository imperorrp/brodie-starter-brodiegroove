"""Brodie's avatar on the call, from the official character kit (team/dance/assets/brodie, seraphimlabs-dev/brodie-character-kit).

Only kit art is used: reference/brodie-sit-wave.png (flat sticker style, transparent background), cropped to head, waving arm and
the BRODIE tank (the lettering always stays visible). The kit allows two mouths, the big grin with outlined white teeth and a small
closed smile, so speaking alternates the two; the closed-smile and sleepy frames are made by painting canon cream #EFE7DA over the
grin or the open eye and drawing an ink #1A1919 arc in the kit's line weight. No other colours touch his body (cream check:
tools/check_cream.py, run by the tests).

States (driven by real signals in live.py): idle (slow breathing bob), listening (volt ring pulses while the caller talks),
speaking (mouth alternates about 7 times a second while his audio plays), thinking (three dots while a line is being prepared),
offline (both eyes closed, "sleepy", when the voice isn't available). This is a speaking animation, not phoneme lip-sync.

  av = avatar()                                  # one shared instance
  av.draw(img_bgr, cx, cy, height_px, state, t)  # draws on a dark bubble with the state ring
"""
import math, os
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, 'assets', 'brodie', 'brodie-sit-wave.png')
CREAM = (239, 231, 218); INK = (26, 25, 25)                 # RGB, from the kit
VOLT_BGR = (61, 255, 217); PANEL_BGR = (21, 18, 18); GREY_BGR = (152, 145, 142)
CROP = (110, 0, 1000, 1110)                                 # head + waving arm + BRODIE tank
MOUTH = (546, 512, 192, 102)                                 # centre x, y and radii over the grin (source pixels)
LEFT_EYE = (389, 367, 52, 70)
LINE = 15                                                   # the kit's outline weight at this scale


def _variants():
    base = Image.open(ART).convert('RGBA')
    closed = base.copy(); d = ImageDraw.Draw(closed)
    cx, cy, rx, ry = MOUTH
    d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=CREAM + (255,))
    d.arc((cx - 82, cy - 70, cx + 82, cy + 40), start=25, end=155, fill=INK + (255,), width=LINE)
    sleepy = closed.copy(); d2 = ImageDraw.Draw(sleepy)
    ex, ey, erx, ery = LEFT_EYE
    d2.ellipse((ex - erx, ey - ery, ex + erx, ey + ery), fill=CREAM + (255,))
    d2.arc((ex - 46, ey - 30, ex + 46, ey + 40), start=20, end=160, fill=INK + (255,), width=LINE)
    out = {}
    for name, im in (('grin', base), ('closed', closed), ('sleepy', sleepy)):
        out[name] = im.crop(CROP)
    return out


_VAR = None


def _get():
    global _VAR
    if _VAR is None: _VAR = _variants()
    return _VAR


@lru_cache(maxsize=32)
def sprite(variant, height):
    """BGRA uint8 sprite of a variant scaled to `height` pixels."""
    im = _get()[variant]; w = max(1, int(round(im.width * height / im.height)))
    a = np.asarray(im.resize((w, int(height)), Image.LANCZOS), np.uint8)
    return a[:, :, [2, 1, 0, 3]].copy()


def _blend(img, spr, x, y):
    H, W = img.shape[:2]; h, w = spr.shape[:2]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]; a = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + s[:, :, :3].astype(np.float32) * a).astype(np.uint8)


@lru_cache(maxsize=64)
def _circle_mask(h, w, cx, cy, r):
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.uint8)


def _circle_clip(spr, cx, cy, r):
    m = _circle_mask(spr.shape[0], spr.shape[1], int(cx), int(cy), int(r))
    out = spr.copy(); out[:, :, 3] = out[:, :, 3] * m
    return out


class BrodieAvatar:
    STATES = ('idle', 'listening', 'speaking', 'thinking', 'offline')

    def variant(self, state, t):
        if state == 'offline': return 'sleepy'
        if state == 'speaking': return 'grin' if int(t * 7.0) % 2 == 0 else 'closed'
        if state in ('listening', 'thinking'): return 'closed'
        return 'grin'

    def draw(self, img, cx, cy, height, state='idle', t=0.0, bubble=True, still=False):
        """Brodie centred on (cx, cy), `height` px tall, on a round dark bubble whose ring shows the state."""
        h = int(height); r = int(h * 0.56)
        if bubble:
            cv2.circle(img, (int(cx), int(cy)), r, PANEL_BGR, -1, cv2.LINE_AA)
            ring = VOLT_BGR if state in ('listening', 'speaking') else GREY_BGR
            wdt = max(3, h // 28)
            if state == 'listening' and not still:
                rr = int(r + wdt + (h * 0.05) * (0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.6)))
                cv2.circle(img, (int(cx), int(cy)), rr, VOLT_BGR, max(2, wdt // 2), cv2.LINE_AA)
            cv2.circle(img, (int(cx), int(cy)), r, ring, wdt, cv2.LINE_AA)
        bob = 0 if still else int(round(h * 0.012 * math.sin(t * 2 * math.pi / 3.6)))
        spr = sprite(self.variant(state, t), int(h * 0.9))
        x, y = int(cx - spr.shape[1] / 2), int(cy - spr.shape[0] * 0.52) + bob
        if bubble:                                               # keep him inside the round bubble (profile-picture look)
            spr = _circle_clip(spr, cx - x, cy - y, r - max(3, h // 28))
        _blend(img, spr, x, y)
        if state == 'thinking' and not still:
            for i in range(3):
                a = 0.5 + 0.5 * math.sin(t * 6 - i * 0.9)
                c = (int(cx + r * 0.55 + i * h * 0.09), int(cy - r * 0.95))
                cv2.circle(img, c, max(3, h // 26), tuple(int(v * (0.5 + 0.5 * a)) for v in VOLT_BGR), -1, cv2.LINE_AA)
        return img


@lru_cache(maxsize=16)
def _round_mask(h, w, r):
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 1, -1); cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 1, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)): cv2.circle(m, (cx, cy), r, 1, -1)
    return m


@lru_cache(maxsize=32)
def _tile_pixels(variant, w, h, bob):
    """The tile's pixels (cream background + the kit sprite), cached per variant, size and bob offset (a few pixels)."""
    tile = np.empty((h, w, 3), np.uint8); tile[:] = CREAM[::-1]
    spr = sprite(variant, int(h * 1.05))
    sx = (w - spr.shape[1]) // 2 + int(w * 0.06); sy = int(h * 0.04) + bob
    _blend(tile, spr, sx, sy)
    tile.setflags(write=False)
    return tile


def _draw_tile(self, img, x0, y0, w, h, state='idle', t=0.0, still=False):
    """Brodie in a FaceTime-style picture-in-picture tile (the site's .pip): kit cream background, rounded corners, a thin light
    border; the border turns volt and pulses while he talks or listens; three dots while he is thinking."""
    H, W = img.shape[:2]; x0, y0, w, h = int(x0), int(y0), int(w), int(h)
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H: return img
    r = max(6, w // 7); m = _round_mask(h, w, r)
    bob = 0 if still else int(round(h * 0.01 * math.sin(t * 2 * math.pi / 3.6)))
    tile = _tile_pixels(self.variant(state, t), w, h, bob)
    roi = img[y0:y0 + h, x0:x0 + w]; np.copyto(roi, tile, where=(m == 1)[:, :, None])
    live = state in ('listening', 'speaking') and not still
    pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.6)
    col = VOLT_BGR if live else (200, 200, 200)
    wd = max(2, w // 40) + (int(round(pulse * 2)) if live else 0)
    for (a, b) in (((x0 + r, y0), (x0 + w - r, y0)), ((x0 + r, y0 + h - 1), (x0 + w - r, y0 + h - 1)), ((x0, y0 + r), (x0, y0 + h - r)), ((x0 + w - 1, y0 + r), (x0 + w - 1, y0 + h - r))):
        cv2.line(img, a, b, col, wd, cv2.LINE_AA)
    for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x0 + w - r - 1, y0 + r), 270), ((x0 + w - r - 1, y0 + h - r - 1), 0), ((x0 + r, y0 + h - r - 1), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, col, wd, cv2.LINE_AA)
    if state == 'thinking' and not still:
        for i in range(3):
            a = 0.5 + 0.5 * math.sin(t * 6 - i * 0.9)
            cv2.circle(img, (x0 + w // 2 - w // 6 + i * w // 6, y0 + h - max(8, h // 12)), max(3, w // 24), tuple(int(v * (0.5 + 0.5 * a)) for v in VOLT_BGR), -1, cv2.LINE_AA)
    return img


BrodieAvatar.draw_tile = _draw_tile


_AV = None


def avatar():
    global _AV
    if _AV is None: _AV = BrodieAvatar()
    return _AV
