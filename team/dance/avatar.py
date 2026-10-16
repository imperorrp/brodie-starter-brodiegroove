"""Brodie's avatar on the call, from the official character kit (team/dance/assets/brodie, seraphimlabs-dev/brodie-character-kit).

Only kit art is used: reference/brodie-sit-wave.png (flat sticker style, transparent background), cropped to head, waving arm and
the BRODIE tank (the lettering always stays visible). The kit allows two mouths, the big grin with outlined white teeth and a small
closed smile, so speaking alternates the two; the closed-smile and sleepy frames are made by painting canon cream #EFE7DA over the
grin or the open eye and drawing an ink #1A1919 arc in the kit's line weight. No other colours touch his body (cream check:
tools/check_cream.py, run by the tests).

States (driven by real signals in live.py): idle (slow breathing bob), listening (volt ring pulses while the caller talks),
speaking (mouth alternates about 7 times a second while his audio plays), thinking (a typing bubble, three hopping dots, from the
end of the caller's line or our ask until his audio starts),
offline (both eyes closed, "sleepy", when the voice isn't available). This is a speaking animation, not phoneme lip-sync.

Moods (live.py sets one from the last verdict, score, grade and coach style; the kit's "yours to change: expression, pose"):
  happy      the sit-wave art as drawn: wink + grin (the default)
  pumped     the kit's own flex pose (brodie-flex.png): PERFECT streaks, big scores, S/A, a K.O.
  focus      both eyes open, brow bars lowered toward the middle, closed smile: watching you dance, the COACH style
  encourage  both eyes open, brow bars raised in the middle, closed smile: after misses and low scores ("you've got this")
Open eyes are solid ink ovals the size of his open eye; brows stay thick ink bars; mouths stay the grin or the closed smile. Every
feature edit erases one ink stroke (found as the dark component nearest its measured centre) with canon cream and redraws it in ink.

  av = avatar()                                        # one shared instance
  av.draw(img_bgr, cx, cy, height_px, state, t, mood)  # draws on a dark bubble with the state ring
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
# The sit-wave face, measured on the kit file (dark components): brow bars, the open eye, the wink (his other eye).
FACE_BOX = (300, 180, 780, 450)
L_BROW_C, R_BROW_C, WINK_C = (375, 228), (686, 301), (688, 368)
OPEN_EYE = (688, 370, 44, 64)                               # an open eye where the wink was: the left eye's size (88 x 128)
BROW_W = 32                                                 # the brow bars' thickness
BROWS = dict(focus=(((326, 252), (430, 270)), ((640, 270), (744, 252))),       # inner ends down: locked in
             encourage=(((326, 270), (430, 250)), ((640, 250), (744, 270))))   # inner ends up: soft, "you've got this"
FLEX_ART = os.path.join(HERE, 'assets', 'brodie', 'brodie-flex.png')
FLEX_CROP = (150, 0, 1470, 1240)                            # head, both flexed arms, the BRODIE lettering
FLEX_MOUTH = (840, 570, 210, 110)                           # over the flex grin (outline 677..1004 x 501..649, corners included)
MOODS = ('happy', 'pumped', 'focus', 'encourage')


def _closed_mouth(im, mouth, k=1.0):
    """The kit's small closed smile in place of the grin: cream over the grin, one ink arc in the kit's line weight."""
    d = ImageDraw.Draw(im); cx, cy, rx, ry = mouth
    d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=CREAM + (255,))
    d.arc((cx - 82 * k, cy - 70 * k, cx + 82 * k, cy + 40 * k), start=25, end=155, fill=INK + (255,), width=LINE)
    return im


def _erase_stroke(im, centre, box=FACE_BOX, grow=6):
    """Paint canon cream over the one ink stroke nearest `centre` (grown a few pixels for its anti-aliased edge), never over a
    neighbouring stroke (the head or hand outline)."""
    a = np.array(im); x0, y0, x1, y1 = box; sub = a[y0:y1, x0:x1]
    dark = ((sub[:, :, :3].max(axis=2) < 90) & (sub[:, :, 3] > 200)).astype(np.uint8)
    n, lab, _, cen = cv2.connectedComponentsWithStats(dark, 8)
    if n < 2: return im
    i = 1 + int(np.argmin([(cx + x0 - centre[0]) ** 2 + (cy + y0 - centre[1]) ** 2 for cx, cy in cen[1:]]))
    m = cv2.dilate((lab == i).astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)
    m &= ~((lab > 0) & (lab != i))
    sub[m] = CREAM + (255,)
    return Image.fromarray(a)


def _bar(d, p, q, w=BROW_W):
    """A thick ink bar with round ends (the kit's brow)."""
    d.line((p, q), fill=INK + (255,), width=w)
    for x, y in (p, q): d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=INK + (255,))


def _open_face(base, brows):
    """Both eyes open (the wink becomes an oval like the other eye) with the given brow bars."""
    im = base
    lx, ly = LEFT_EYE[:2]                                    # the new eye takes the open eye's own ink, so the pair matches
    eye_ink = tuple(int(v) for v in np.median(np.asarray(base)[ly - 12:ly + 12, lx - 12:lx + 12, :3].reshape(-1, 3), axis=0))
    for c in (L_BROW_C, R_BROW_C, WINK_C): im = _erase_stroke(im, c)
    d = ImageDraw.Draw(im); ex, ey, rx, ry = OPEN_EYE
    d.ellipse((ex - rx, ey - ry, ex + rx, ey + ry), fill=eye_ink + (255,))
    for p, q in brows: _bar(d, p, q)
    return im


def _variants():
    base = Image.open(ART).convert('RGBA')
    closed = _closed_mouth(base.copy(), MOUTH)
    sleepy = closed.copy(); d2 = ImageDraw.Draw(sleepy)
    ex, ey, erx, ery = LEFT_EYE
    d2.ellipse((ex - erx, ey - ery, ex + erx, ey + ery), fill=CREAM + (255,))
    d2.arc((ex - 46, ey - 30, ex + 46, ey + 40), start=20, end=160, fill=INK + (255,), width=LINE)
    out = {}
    for name, im in (('grin', base), ('closed', closed), ('sleepy', sleepy)):
        out[name] = im.crop(CROP)
    for mood in ('focus', 'encourage'):
        face = _open_face(base.copy(), BROWS[mood])
        out[f'{mood}_grin'] = face.crop(CROP); out[f'{mood}_closed'] = _closed_mouth(face.copy(), MOUTH).crop(CROP)
    flex = Image.open(FLEX_ART).convert('RGBA')
    out['pumped_grin'] = flex.crop(FLEX_CROP); out['pumped_closed'] = _closed_mouth(flex.copy(), FLEX_MOUTH, k=1.3).crop(FLEX_CROP)
    return out


def variant_for(mood, mouth):
    """The sprite name for a mood and a mouth ('grin' | 'closed'); 'happy' is the original sit-wave art."""
    if mood not in MOODS or mood == 'happy': return mouth
    return f'{mood}_{mouth}'


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

    def variant(self, state, t, mood='happy'):
        if state == 'offline': return 'sleepy'
        if state == 'speaking': mouth = 'grin' if int(t * 7.0) % 2 == 0 else 'closed'
        elif state in ('listening', 'thinking'): mouth = 'closed'
        else: mouth = 'closed' if mood in ('focus', 'encourage') else 'grin'   # resting: watchful moods keep the small smile
        return variant_for(mood, mouth)

    def draw(self, img, cx, cy, height, state='idle', t=0.0, bubble=True, still=False, mood='happy'):
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
        spr = sprite(self.variant(state, t, mood), int(h * 0.9))
        x, y = int(cx - spr.shape[1] / 2), int(cy - spr.shape[0] * 0.52) + bob
        if bubble:                                               # keep him inside the round bubble (profile-picture look)
            spr = _circle_clip(spr, cx - x, cy - y, r - max(3, h // 28))
        _blend(img, spr, x, y)
        if state == 'thinking' and not still:
            typing_bubble(img, cx + r * 0.75, cy - r * 0.95, h * 0.36, t)
        return img


def typing_bubble(img, cx, cy, w, t, alpha=0.9):
    """A chat app's "typing" bubble centred on (cx, cy), w px wide: a dark pill with three volt dots hopping in turn. Brodie is
    working on an answer (the second or two before his voice starts, which on a call otherwise reads as dead air)."""
    H, W = img.shape[:2]; w = int(w); h = max(14, int(w * 0.42)); r = h // 2
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H: return img
    roi = img[y0:y0 + h, x0:x0 + w]; over = roi.copy()
    cv2.rectangle(over, (r, 0), (w - r - 1, h - 1), PANEL_BGR, -1)
    for x in (r, w - r - 1): cv2.circle(over, (x, r), r, PANEL_BGR, -1, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)
    dr = max(2, h // 7); gap = w / 4.0
    for i in range(3):
        hop = max(0.0, math.sin(((t * 1.8 - i * 0.16) % 1.0) * 2 * math.pi))      # each dot hops in turn, then rests
        c = (int(x0 + gap * (i + 1)), int(y0 + r - hop * h * 0.16))
        cv2.circle(img, c, dr, tuple(int(v * (0.55 + 0.45 * hop)) for v in VOLT_BGR), -1, cv2.LINE_AA)
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


def _draw_tile(self, img, x0, y0, w, h, state='idle', t=0.0, still=False, mood='happy'):
    """Brodie in a FaceTime-style picture-in-picture tile (the site's .pip): kit cream background, rounded corners, a thin light
    border; the border turns volt and pulses while he talks or listens; a typing bubble on its bottom edge while he is thinking."""
    H, W = img.shape[:2]; x0, y0, w, h = int(x0), int(y0), int(w), int(h)
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H: return img
    r = max(6, w // 7); m = _round_mask(h, w, r)
    bob = 0 if still else int(round(h * 0.01 * math.sin(t * 2 * math.pi / 3.6)))
    tile = _tile_pixels(self.variant(state, t, mood), w, h, bob)
    roi = img[y0:y0 + h, x0:x0 + w]; np.copyto(roi, tile, where=(m == 1)[:, :, None])
    live = state in ('listening', 'speaking') and not still
    pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.6)
    col = VOLT_BGR if live else (200, 200, 200)
    wd = max(2, w // 40) + (int(round(pulse * 2)) if live else 0)
    for (a, b) in (((x0 + r, y0), (x0 + w - r, y0)), ((x0 + r, y0 + h - 1), (x0 + w - r, y0 + h - 1)), ((x0, y0 + r), (x0, y0 + h - r)), ((x0 + w - 1, y0 + r), (x0 + w - 1, y0 + h - r))):
        cv2.line(img, a, b, col, wd, cv2.LINE_AA)
    for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x0 + w - r - 1, y0 + r), 270), ((x0 + w - r - 1, y0 + h - r - 1), 0), ((x0 + r, y0 + h - r - 1), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, col, wd, cv2.LINE_AA)
    if state == 'thinking' and not still:                   # the typing bubble just under the tile (the BRODIE lettering stays visible)
        bw = int(w * 0.66); typing_bubble(img, x0 + w // 2, y0 + h + max(14, int(bw * 0.42)) // 2 + max(4, h // 30), bw, t)
    return img


BrodieAvatar.draw_tile = _draw_tile


_AV = None


def avatar():
    global _AV
    if _AV is None: _AV = BrodieAvatar()
    return _AV
