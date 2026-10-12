"""Render the BrodieGroove pitch video: timeline.py's edit drawn frame by frame (1080x1920, 30 fps) and muxed with mix.wav.

Run:  python -m team.video.music && python -m team.video.make_pitch [--from BEAT --to BEAT] [--stills]
--stills writes one PNG per segment to data/tmp/pitch/stills/ instead of the video (fast look at the edit).
Output: data/tmp/pitch/brodie_pitch.mp4
"""
import os, sys, math, subprocess, functools
import av, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from team.video.timeline import (PITCH, SRC, HUD, WEB, FONTS, BEAT, FPS, W, H, TOTAL_BEATS, DURATION, SEGMENTS, VOICE,
                                 FLASH, CAPTIONS, t)

C = dict(ink=(10, 10, 11), ink2=(18, 18, 21), ink3=(26, 27, 31), line=(42, 44, 49), mut=(154, 157, 164),
         paper=(244, 244, 240), cream=(239, 231, 218), volt=(217, 255, 61), pink=(255, 61, 168), cyan=(61, 224, 255),
         amber=(255, 177, 61), red=(255, 77, 94))
CROP = (16, 52, 704, 1278)                         # the call capture's frame: grey band on top, black edges left/right
CROP_TOP = {'flow.mp4': 12}                        # flow.mp4 has the older HUD that sits in the top band: keep it
K = W / (CROP[2] - CROP[0])                        # source px -> output px (about 1.57)
Y_OFF = ((CROP[3] - CROP[1]) * K - H) / 2


# ---------------- drawing helpers ----------------
@functools.lru_cache(None)
def font(name, size, wght=None):
    path = os.path.join(FONTS, dict(display='ArchivoBlack-Regular.ttf', grotesk='SpaceGrotesk.ttf', body='PlusJakartaSans.ttf')[name])
    f = ImageFont.truetype(path, size)
    if wght:
        try:
            f.set_variation_by_axes([wght])
        except Exception:
            pass
    return f


def ease_out(x):
    x = min(1.0, max(0.0, x)); return 1 - (1 - x) ** 3


def ease_inout(x):
    x = min(1.0, max(0.0, x)); return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2


def spring(x):
    """0 -> 1 with a little overshoot (pop-ins)."""
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    return 1 - math.exp(-6 * x) * math.cos(9 * x)


def blend(dst, rgba, x, y):
    """alpha-composite an RGBA PIL image or array onto the RGB frame at (x, y) (top-left, may be off-screen)."""
    a = np.asarray(rgba) if not isinstance(rgba, np.ndarray) else rgba
    h, w = a.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    src = a[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    al = src[..., 3:4] / 255.0
    d = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (src[..., :3] * al + d * (1 - al)).astype(np.uint8)


def blend_center(dst, im, cx, cy, scale=1.0, angle=0.0, alpha=1.0):
    if scale <= 0.01 or alpha <= 0.01:
        return
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if angle:
        im = im.rotate(angle, Image.BICUBIC, expand=True)
    if alpha < 1.0:
        a = np.asarray(im).copy(); a[..., 3] = (a[..., 3] * alpha).astype(np.uint8); im = a
        blend(dst, im, int(cx - im.shape[1] / 2), int(cy - im.shape[0] / 2)); return
    blend(dst, im, int(cx - im.width / 2), int(cy - im.height / 2))


def text_size(f, s, spacing=0):
    b = f.getbbox(s)
    return b[2] - b[0] + spacing * max(0, len(s) - 1), b[3] - b[1], b


def draw_spaced(d, xy, s, f, fill, spacing=0, **kw):
    x, y = xy
    for ch in s:
        d.text((x, y), ch, font=f, fill=fill, **kw)
        x += f.getlength(ch) + spacing


@functools.lru_cache(None)
def pill(label, color, size=46, pad=(34, 18), stroke=False):
    """a site-style chip: rounded pill, ink text on an accent (or glass with accent text)."""
    f = font('grotesk', size, 700)
    tw, th, b = text_size(f, label, 2)
    w, h = tw + pad[0] * 2, size + pad[1] * 2
    im = Image.new('RGBA', (w + 12, h + 12), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((6 + 4, 6 + 6, 6 + w + 2, 6 + h + 4), h // 2, fill=(0, 0, 0, 110))      # drop shadow
    if stroke:
        d.rounded_rectangle((6, 6, 6 + w, 6 + h), h // 2, fill=(10, 10, 11, 210), outline=C[color] + (255,), width=4)
        fg = C[color]
    else:
        d.rounded_rectangle((6, 6, 6 + w, 6 + h), h // 2, fill=C[color] + (255,))
        fg = C['ink']
    draw_spaced(d, (6 + pad[0] - b[0], 6 + (h - size) / 2 - b[1] + size * 0.1), label, f, fg + (255,), 2)
    return im


@functools.lru_cache(None)
def circle_img(path, size, ring=0, ring_color='paper'):
    im = Image.open(path).convert('RGBA').resize((size, size), Image.LANCZOS)
    mask = Image.new('L', (size * 4, size * 4), 0); ImageDraw.Draw(mask).ellipse((0, 0, size * 4, size * 4), fill=255)
    mask = mask.resize((size, size), Image.LANCZOS)
    bg = Image.new('RGBA', (size, size), C['ink3'] + (255,)); bg.alpha_composite(im)
    bg.putalpha(mask)
    if not ring:
        return bg
    out = Image.new('RGBA', (size + ring * 2, size + ring * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(out); d.ellipse((0, 0, size + ring * 2 - 1, size + ring * 2 - 1), fill=C[ring_color] + (255,))
    out.alpha_composite(bg, (ring, ring))
    return out


@functools.lru_cache(None)
def sticker(path, height, border=14):
    """the mascot as a die-cut sticker: a cream border around the line art so it reads on ink."""
    im = Image.open(path).convert('RGBA')
    im = im.resize((int(im.width * height / im.height), height), Image.LANCZOS)
    pad = border * 2
    big = Image.new('RGBA', (im.width + pad * 2, im.height + pad * 2), (0, 0, 0, 0)); big.alpha_composite(im, (pad, pad))
    a = np.asarray(big)[..., 3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border * 2 + 1, border * 2 + 1))
    grown = cv2.GaussianBlur(cv2.dilate((a > 30).astype(np.uint8) * 255, k), (5, 5), 0)
    out = np.zeros((*a.shape, 4), np.uint8); out[..., :3] = C['paper']; out[..., 3] = grown
    shadow = np.zeros_like(out); shadow[..., 3] = (cv2.GaussianBlur(grown, (31, 31), 0) * 0.45).astype(np.uint8)
    res = Image.new('RGBA', big.size, (0, 0, 0, 0))
    res.alpha_composite(Image.fromarray(shadow).crop((0, 0, big.width, big.height - 12)), (0, 12))
    res.alpha_composite(Image.fromarray(out)); res.alpha_composite(big)
    return res


# ---------------- backgrounds (call screen + end card) ----------------
def blobs(tsec, strength=1.0):
    """the site's blurred colour blobs drifting on ink, rendered small and upscaled."""
    s = 8; w, h = W // s, H // s
    im = np.zeros((h, w, 3), np.float32); im[:] = C['ink']
    for color, (bx, by), (ax, ay), r, sp in (('pink', (0.18, 0.22), (0.10, 0.06), 0.34, 0.31),
                                            ('cyan', (0.85, 0.42), (0.08, 0.10), 0.30, 0.23),
                                            ('volt', (0.35, 0.86), (0.12, 0.05), 0.26, 0.19)):
        cx = (bx + ax * math.sin(tsec * sp * 2 * math.pi / 3)) * w
        cy = (by + ay * math.cos(tsec * sp * 2 * math.pi / 4)) * h
        layer = np.zeros((h, w), np.float32); cv2.circle(layer, (int(cx), int(cy)), int(r * w), 1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), r * w * 0.55) * 0.42 * strength
        im += layer[..., None] * (np.array(C[color], np.float32) - im) * 0.9
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 255).astype(np.uint8)


def grid_floor(frame, beat, horizon=1240, glow=1.0):
    """volt perspective grid floor that pulses on every beat (the site's hero floor)."""
    pulse = math.exp(-((beat % 1.0) * 5.0))
    layer = np.zeros((H - horizon, W), np.uint8)
    vx = W // 2; depth = H - horizon
    for k in range(-12, 13):                                   # converging lines
        x_bot = vx + k * 150
        cv2.line(layer, (vx + k * 14, 0), (x_bot, depth), 255, 2, cv2.LINE_AA)
    scroll = (beat % 1.0)
    for j in range(14):                                        # horizontal lines rushing toward camera on the beat
        z = (j + scroll) / 14
        y = int(depth * z ** 2.2)
        cv2.line(layer, (0, y), (W, y), 255, 2, cv2.LINE_AA)
    fade = np.linspace(0, 1, depth, dtype=np.float32)[:, None] ** 1.3
    a = layer.astype(np.float32) / 255 * fade * (0.28 + 0.5 * pulse) * glow
    region = frame[horizon:].astype(np.float32)
    frame[horizon:] = (region + a[..., None] * (np.array(C['volt'], np.float32) - region)).astype(np.uint8)


# ---------------- footage ----------------
class Footage:
    """decodes the frames a segment needs, once; returns the frame on screen at a source time."""
    def __init__(self):
        self.key = None; self.frames = []

    def load(self, src, t0, t1):
        key = (src, round(t0, 3), round(t1, 3))
        if key == self.key:
            return
        self.key = key; self.frames = []
        with av.open(os.path.join(SRC, src)) as c:
            s = c.streams.video[0]
            c.seek(int(max(0, t0 - 1.0) / s.time_base), stream=s)
            for f in c.decode(s):
                ft = float(f.time)
                if ft < t0 - 0.2:
                    continue
                self.frames.append((ft, f.to_ndarray(format='rgb24')))
                if ft > t1 + 0.2:
                    break

    def at(self, ts):
        best = self.frames[0][1]
        for ft, im in self.frames:
            if ft <= ts + 1e-3:
                best = im
            else:
                break
        return best


@functools.lru_cache(None)
def still(name):
    return np.asarray(Image.open(os.path.join(HUD, name)).convert('RGB'))


def place(src_img, scale=1.0, cx=None, cy=None, top=CROP[1]):
    """crop the capture frame and scale it to 1080x1920 in one warp; optional zoom about a point (output px)."""
    m = np.array([[K, 0, -CROP[0] * K], [0, K, -top * K - Y_OFF]], np.float64)
    if scale != 1.0:
        cx = W / 2 if cx is None else cx; cy = H / 2 if cy is None else cy
        z = np.array([[scale, 0, cx - scale * cx], [0, scale, cy - scale * cy]], np.float64)
        m = np.hstack([z[:, :2] @ m[:, :2], z[:, :2] @ m[:, 2:] + z[:, 2:]])
    return cv2.warpAffine(src_img, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def src_to_out(x, y, top=CROP[1]):
    return (x - CROP[0]) * K, (y - top) * K - Y_OFF


# ---------------- overlays ----------------
def chip_overlay(frame, label, color, beat, start):
    if beat < start:
        return
    p = spring((beat - start) / 0.5)
    im = pill(label, color, 50)
    blend_center(frame, im, W / 2, 490, scale=0.4 + 0.6 * p, angle=-3 * p)


def ring_overlay(frame, ring, beat, start, top):
    x, y, w, h = ring
    cx, cy = src_to_out(x, y, top); w, h = w * K, h * K
    p = ease_out((beat - start) / 0.5)
    if p <= 0:
        return
    pulse = 1 + 0.06 * math.exp(-((beat % 1.0) * 6))
    ww, hh = w * pulse + 30 * (1 - p), h * pulse + 30 * (1 - p)
    layer = Image.new('RGBA', (int(ww) + 40, int(hh) + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    d.rounded_rectangle((20, 20, 20 + ww, 20 + hh), int(hh / 2), outline=C['volt'] + (int(255 * p),), width=7)
    blend_center(frame, layer, cx, cy)


@functools.lru_cache(None)
def burst_word(word, color):
    f = font('display', 190)
    tw, th, b = text_size(f, word)
    im = Image.new('RGBA', (tw + 120, th + 120), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((60 - b[0] + 10, 60 - b[1] + 12), word, font=f, fill=(0, 0, 0, 160), stroke_width=12, stroke_fill=(0, 0, 0, 160))
    d.text((60 - b[0], 60 - b[1]), word, font=f, fill=C[color] + (255,), stroke_width=10, stroke_fill=C['ink'] + (255,))
    return im


def burst_overlay(frame, word, color, age_beats, cy=760):
    """the PERFECT pop, boosted: starburst + big word, one and a half beats."""
    if age_beats < 0 or age_beats > 1.6:
        return
    cx = W / 2
    rays = np.zeros((H, W), np.uint8)
    r0, r1 = 180 + 500 * ease_out(age_beats / 0.8), 260 + 800 * ease_out(age_beats / 0.8)
    for k in range(16):
        a = k * math.pi / 8 + 0.2
        p0 = (int(cx + r0 * math.cos(a)), int(cy + r0 * math.sin(a))); p1 = (int(cx + r1 * math.cos(a)), int(cy + r1 * math.sin(a)))
        cv2.line(rays, p0, p1, 255, 14, cv2.LINE_AA)
    al = (rays.astype(np.float32) / 255 * max(0, 1 - age_beats / 0.9))[..., None]
    frame[:] = (frame * (1 - al) + np.array(C[color], np.float32) * al).astype(np.uint8)
    s = 0.5 + 0.6 * spring(age_beats / 0.5)
    fade = 1.0 if age_beats < 1.2 else max(0, 1 - (age_beats - 1.2) / 0.4)
    blend_center(frame, burst_word(word, color), cx, cy, scale=s * 0.92, angle=-4, alpha=fade)


@functools.lru_cache(None)
def caption_img(text, side, shown):
    """Brodie: glass pill with his face; user: pink bubble on the right. `shown` = number of words revealed."""
    words = text.split(' ')
    f = font('body', 50, 750)
    maxw = 760
    lines, cur = [], ''
    for w in words:                                  # wrap on the full text so the layout never jumps
        trial = (cur + ' ' + w).strip()
        if f.getlength(trial) > maxw and cur:
            lines.append(cur); cur = w
        else:
            cur = trial
    lines.append(cur)
    lh = 64
    tw = int(max(f.getlength(l) for l in lines))
    av_size = 92 if side == 'b' else 0
    pad_x, pad_y = 34, 26
    w = tw + pad_x * 2 + (av_size + 18 if av_size else 0)
    h = max(len(lines) * lh + pad_y * 2 - 8, av_size + 24)
    im = Image.new('RGBA', (w + 16, h + 20), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((8, 14, 8 + w, 14 + h), 40, fill=(0, 0, 0, 90))
    if side == 'b':
        d.rounded_rectangle((8, 6, 8 + w, 6 + h), 40, fill=(14, 14, 17, 225), outline=C['volt'] + (255,), width=4)
        im.alpha_composite(circle_img(os.path.join(WEB, 'brodie-pfp-square.png'), av_size, 0), (8 + 18, 6 + (h - av_size) // 2))
        x0, fg = 8 + 18 + av_size + 18, C['paper']
    else:
        d.rounded_rectangle((8, 6, 8 + w, 6 + h), 40, fill=C['pink'] + (255,))
        x0, fg = 8 + pad_x, C['ink']
    n = 0
    y = 6 + (h - len(lines) * lh) / 2 + 2
    for line in lines:                               # words appear as they are spoken
        x = x0
        for word in line.split(' '):
            if n < shown:
                d.text((x, y), word, font=f, fill=fg + (255,))
            x += f.getlength(word + ' '); n += 1
        y += lh
    return im


def caption_overlay(frame, tsec):
    for key, beat, max_s, side in VOICE:
        text = CAPTIONS.get(key)
        if not text:
            continue
        dur = VO_DUR[key] if not max_s else min(max_s, VO_DUR[key])
        a = t(beat)
        if not (a - 0.05 <= tsec <= a + dur + 0.45):
            continue
        nwords = len(text.split(' '))
        shown = min(nwords, 1 + int((tsec - a) / max(0.2, dur * 0.85) * nwords))
        im = caption_img(text, side, shown)
        p = spring((tsec - a) / 0.35)
        out = max(0.0, (tsec - (a + dur + 0.2)) / 0.25)
        alpha = 1 - min(1, out)
        y = 1270 if side == 'b' else 1030
        x = 60 + im.width / 2 if side == 'b' else W - 60 - im.width / 2
        blend_center(frame, im, x, y + 40 * (1 - p), scale=0.85 + 0.15 * p, alpha=alpha)


def flash_overlay(frame, beat):
    for fb in FLASH:
        age = (beat - fb) * BEAT * FPS                       # in frames
        if 0 <= age < 4:
            a = 0.55 * (1 - age / 4)
            frame[:] = (frame * (1 - a) + np.array((250, 255, 225)) * a).astype(np.uint8)


GRAIN = [np.random.default_rng(i).normal(0, 4, (H // 2, W // 2)).astype(np.float32) for i in range(6)]
VIGNETTE = None                                    # built on the first frame


def finish(frame, fi):
    global VIGNETTE
    if VIGNETTE is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2)
        VIGNETTE = (1 - 0.28 * np.clip(r - 0.35, 0, 1) ** 1.5)[..., None]
    g = cv2.resize(GRAIN[fi % len(GRAIN)], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    return np.clip(frame.astype(np.float32) * VIGNETTE + g, 0, 255).astype(np.uint8)


# ---------------- scenes ----------------
def call_screen(beat, tsec):
    f = blobs(tsec)
    grid_floor(f, beat, horizon=1420, glow=0.6)
    # kicker, the site's "NEW · A dance coach you FaceTime"
    blend_center(f, kicker_img(), W / 2, 300, scale=0.9 + 0.1 * spring(beat / 0.6))
    # pulsing rings + avatar
    cx, cy = W / 2, 790
    for k in range(3):
        age = (beat - k * 0.33) % 1.0
        r = 230 + 220 * ease_out(age)
        a = max(0, 1 - age) * 0.8
        layer = np.zeros((H, W), np.uint8); cv2.circle(layer, (int(cx), int(cy)), int(r), 255, 6, cv2.LINE_AA)
        al = (layer.astype(np.float32) / 255 * a)[..., None]
        f[:] = (f * (1 - al) + np.array(C['volt']) * al).astype(np.uint8)
    bump = 1 + 0.05 * math.exp(-((beat % 1.0) * 6))
    answer = ease_inout((beat - 3.0) / 0.8)
    blend_center(f, circle_img(os.path.join(WEB, 'brodie-pfp-square.png'), 420, 10, 'volt'), cx, cy, scale=bump * (1 + 0.25 * answer))
    name_f = font('display', 130)
    tw, th, b = text_size(name_f, 'BRODIE')
    layer = Image.new('RGBA', (tw + 40, th + 40), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((20 - b[0], 20 - b[1]), 'BRODIE', font=name_f, fill=C['paper'] + (255,))
    blend_center(f, layer, W / 2, 1130)
    dots = '.' * (1 + int(beat * 2) % 3)
    sub = Image.new('RGBA', (700, 80), (0, 0, 0, 0))
    sf = font('grotesk', 50, 500); s = 'is calling' + dots
    ImageDraw.Draw(sub).text((350 - sf.getlength('is calling...') / 2, 10), s, font=sf, fill=C['mut'] + (255,))
    blend_center(f, sub, W / 2, 1245)
    # slide to answer
    sw, sh = 780, 150
    track = Image.new('RGBA', (sw + 20, sh + 20), (0, 0, 0, 0)); d = ImageDraw.Draw(track)
    d.rounded_rectangle((10, 10, 10 + sw, 10 + sh), sh // 2, fill=(255, 255, 255, 28), outline=(255, 255, 255, 60), width=2)
    lf = font('grotesk', 44, 600); lab = 'slide to dance'
    d.text((10 + sw / 2 - lf.getlength(lab) / 2 + 50, 10 + sh / 2 - 28), lab, font=lf, fill=C['paper'] + (int(200 * (1 - answer)),))
    kx = 10 + 12 + (sw - sh) * answer
    d.ellipse((kx, 22, kx + sh - 24, 10 + sh - 12), fill=C['volt'] + (255,))
    # a little camera glyph on the knob
    gx, gy = kx + (sh - 24) / 2, 10 + sh / 2
    d.rounded_rectangle((gx - 30, gy - 20, gx + 12, gy + 20), 8, fill=C['ink'] + (255,))
    d.polygon([(gx + 14, gy - 4), (gx + 32, gy - 16), (gx + 32, gy + 16), (gx + 14, gy + 4)], fill=C['ink'] + (255,))
    blend_center(f, track, W / 2, 1560)
    if beat > 3.75:                                           # zoom-flash into the call
        a = (beat - 3.75) / 0.25
        f[:] = (f * (1 - a) + np.array((250, 255, 225)) * a).astype(np.uint8)
    return f


@functools.lru_cache(None)
def kicker_img():
    f = font('grotesk', 38, 600)
    lab = 'A dance coach you FaceTime'
    tag_f = font('grotesk', 30, 700)
    tw = f.getlength(lab); tag_w = tag_f.getlength('NEW') + 36
    w, h = int(tw + tag_w + 70), 76
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), h // 2, fill=(255, 255, 255, 18), outline=(255, 255, 255, 40), width=2)
    d.rounded_rectangle((12, 12, 12 + tag_w, h - 12), (h - 24) // 2, fill=C['pink'] + (255,))
    d.text((12 + 18, h / 2 - 17), 'NEW', font=tag_f, fill=C['ink'] + (255,))
    d.text((12 + tag_w + 22, h / 2 - 22), lab, font=f, fill=C['paper'] + (255,))
    return im


@functools.lru_cache(None)
def headline_letter(ch, filled):
    f = font('display', 205)
    tw, th, b = text_size(f, ch)
    im = Image.new('RGBA', (int(f.getlength(ch)) + 30, th + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    if filled:
        d.text((15, 20 - b[1]), ch, font=f, fill=C['volt'] + (255,), stroke_width=5, stroke_fill=C['volt'] + (255,))
    else:
        # the stroke draws the outline; the transparent fill punches the inside back out (ImageDraw pastes, it does not blend)
        d.text((15, 20 - b[1]), ch, font=f, fill=(0, 0, 0, 0), stroke_width=5, stroke_fill=C['paper'] + (255,))
    return im


@functools.lru_cache(None)
def marquee_strip():
    f = font('display', 54)
    s = '  PERFECT • NICE • COMBO • GHOST BATTLE • SQUAD UP • DAILY CHALLENGE •' * 3
    w = int(f.getlength(s)) + 40
    im = Image.new('RGBA', (w, 100), C['pink'] + (255,)); d = ImageDraw.Draw(im)
    d.text((20, 18), s, font=f, fill=C['ink'] + (255,))
    return im


def end_card(beat, tsec):
    b = beat - 64
    f = blobs(tsec, 1.1)
    grid_floor(f, beat, horizon=1350, glow=1.0)
    # tilted pink marquee
    m = marquee_strip(); off = int((tsec * 180) % (m.width / 3))
    strip = m.crop((off, 0, off + W + 300, 100))
    blend_center(f, strip, W / 2, 1790, angle=4)
    # BRODIE / GROOVE: outlined letters fill volt one per half beat (the site's hero)
    for row, (word, y) in enumerate((('BRODIE', 330), ('GROOVE', 560))):
        letters = [headline_letter(ch, False) for ch in word]
        total = sum(l.width for l in letters) - 30 * (len(word) - 1)
        x = W / 2 - total / 2
        for i, ch in enumerate(word):
            idx = row * 6 + i
            filled = b >= idx * 0.5
            im = headline_letter(ch, filled)
            age = b - idx * 0.5
            bump = 1 + 0.12 * math.exp(-max(0, age) * 6) if filled else 1
            appear = spring((b + 0.6 - row * 0.3) / 0.6)
            blend_center(f, im, x + im.width / 2, y + 60 * (1 - appear), scale=bump, alpha=min(1, appear))
            x += im.width - 30
    # Brodie flexes in and bobs on the beat
    pop = spring((b - 1.0) / 0.8)
    if pop > 0:
        sq = math.exp(-((beat % 1.0) * 5))
        st = sticker(os.path.join(WEB, 'brodie-flex.png'), 470)
        sx, sy = 1 + 0.03 * sq, 1 - 0.04 * sq
        im = st.resize((int(st.width * sx * pop), int(st.height * sy * pop)), Image.BICUBIC)
        blend(f, im, int(W / 2 - im.width / 2), int(1180 - im.height + 140 * (1 - pop)))
    # tagline with the voice, then the call-to-action pill
    tg = spring((b - 2.6) / 0.5)
    if tg > 0:
        blend_center(f, tagline_img(), W / 2, 1290 + 30 * (1 - tg), alpha=min(1, tg))
    cta = spring((b - 5.5) / 0.5)
    if cta > 0:
        pulse = 1 + 0.05 * math.exp(-((beat % 1.0) * 5))
        blend_center(f, cta_img(), W / 2, 1490, scale=cta * pulse)
        blend_center(f, tag_img(), W / 2, 1610, alpha=min(1, cta))
    return f


@functools.lru_cache(None)
def tagline_img():
    f = font('body', 54, 800)
    a, bb = 'Your first 8-count is ', 'one call away.'
    w = int(f.getlength(a + bb)) + 20
    im = Image.new('RGBA', (w, 100), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((10, 14), a, font=f, fill=C['paper'] + (255,)); d.text((10 + f.getlength(a), 14), bb, font=f, fill=C['volt'] + (255,))
    return im


@functools.lru_cache(None)
def cta_img():
    f = font('display', 68)
    lab = 'CALL BRODIE NOW'
    tw, th, b = text_size(f, lab)
    w, h = tw + 120, 150
    im = Image.new('RGBA', (w + 20, h + 24), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((10, 18, 10 + w, 18 + h), h // 2, fill=(0, 0, 0, 120))
    d.rounded_rectangle((10, 6, 10 + w, 6 + h), h // 2, fill=C['volt'] + (255,))
    d.text((10 + 60 - b[0], 6 + (h - th) / 2 - b[1]), lab, font=f, fill=C['ink'] + (255,))
    return im


@functools.lru_cache(None)
def tag_img():
    f = font('grotesk', 40, 700)
    s = '#BeatMyBrodie  ·  BrodieGroove'
    im = Image.new('RGBA', (int(f.getlength(s)) + 20, 60), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 6), s, font=f, fill=C['mut'] + (255,))
    return im


# ---------------- the frame ----------------
FOOT = Footage()
VO_DUR = {}


def seg_at(beat):
    for s in SEGMENTS:
        if s[0] <= beat < s[1]:
            return s
    return SEGMENTS[-1]


def render_frame(fi):
    tsec = fi / FPS
    beat = tsec / BEAT
    b0, b1, kind, p = seg_at(beat)
    local = tsec - t(b0)
    lb = beat - b0
    if kind == 'call':
        frame = call_screen(beat, tsec)
    elif kind == 'end':
        frame = end_card(beat, tsec)
    else:
        punch = p.get('punch', 1.0)
        scale = 1 + (punch - 1) * math.exp(-lb * 3.2) if punch != 1.0 else 1.0
        if kind == 'still':
            z0, z1 = p.get('zoom', (1.0, 1.0))
            scale *= z0 + (z1 - z0) * ease_inout(lb / (b1 - b0))
            frame = place(still(p['src']), scale=scale)
        else:
            rate = p.get('rate', 1.0)
            top = CROP_TOP.get(p['src'], CROP[1])
            if kind == 'freeze':
                FOOT.load(p['src'], p['at'], p['at']); src_t = p['at']
                zx, zy, zs = p['zoom_to']
                scale *= 1 + (zs - 1) * ease_out(lb / ((b1 - b0) * 0.5))
                cx, cy = src_to_out(zx, zy, top)
            else:
                FOOT.load(p['src'], p['at'], p['at'] + t(b1 - b0) * rate)
                src_t = p['at'] + local * rate
                cx = cy = None
                if 'zoom' in p:
                    zx, zy, zs = p['zoom']; cx, cy = src_to_out(zx, zy, top)
                    scale *= 1 + (zs - 1) * ease_inout(min(1, lb / 1.5))
            frame = place(FOOT.at(src_t), scale=scale, cx=cx, cy=cy, top=top)
            if kind == 'freeze':                         # freeze-frame tells: desaturate a touch + volt edge
                g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[..., None]
                frame = (frame * 0.75 + g * 0.25).astype(np.uint8)
                cv2.rectangle(frame, (8, 8), (W - 9, H - 9), C['volt'], 10)
            if 'burst' in p:
                bt, word, color = p['burst']
                age = ((tsec - t(b0)) * rate + p['at'] - bt) / rate / BEAT
                burst_overlay(frame, word, color, age)
        if 'ring' in p:
            ring_overlay(frame, p['ring'], beat, b0 + 0.3, CROP_TOP.get(p.get('src'), CROP[1]))
        if 'chip' in p:
            chip_overlay(frame, p['chip'][0], p['chip'][1], beat, p['chip'][2])
        if p.get('shutter') and beat >= p['shutter']:
            age = (beat - p['shutter']) * BEAT * FPS
            if age < 6:
                a = 1 - age / 6
                frame[:] = (frame * (1 - a) + 255 * a).astype(np.uint8)
        caption_overlay(frame, tsec)
    flash_overlay(frame, beat)
    return finish(frame, fi)


def load_vo_durations():
    import wave
    for key, *_ in VOICE:
        with wave.open(os.path.join(PITCH, 'vo', key + '.wav')) as w:
            VO_DUR[key] = w.getnframes() / w.getframerate()


def main():
    load_vo_durations()
    args = sys.argv[1:]
    fb = float(args[args.index('--from') + 1]) if '--from' in args else 0
    tb = float(args[args.index('--to') + 1]) if '--to' in args else TOTAL_BEATS
    nframes = int(round(DURATION * FPS))
    if '--stills' in args:
        out = os.path.join(PITCH, 'stills'); os.makedirs(out, exist_ok=True)
        for s in SEGMENTS:
            mid = (s[0] + s[1]) / 2
            for tag, b in (('a', s[0] + 0.35), ('m', mid)):
                fi = int(t(b) * FPS)
                Image.fromarray(render_frame(fi)).save(os.path.join(out, f'{s[0]:05.1f}_{tag}_{s[2]}.jpg'), quality=85)
        print('stills ->', out); return
    f0, f1 = int(t(fb) * FPS), min(nframes, int(t(tb) * FPS))
    dst = os.path.join(PITCH, 'brodie_pitch.mp4' if (fb, tb) == (0, TOTAL_BEATS) else f'preview_{fb:g}-{tb:g}.mp4')
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-ss', f'{f0 / FPS:.4f}', '-t', f'{(f1 - f0) / FPS:.4f}', '-i', os.path.join(PITCH, 'mix.wav'),
           '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-maxrate', '10M', '-bufsize', '20M', '-pix_fmt', 'yuv420p',
           '-profile:v', 'high', '-movflags', '+faststart',
           '-af', 'loudnorm=I=-14:TP=-1.0:LRA=11', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-shortest', dst]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        proc.stdin.write(render_frame(fi).tobytes())
        if (fi - f0) % 90 == 0:
            print(f'frame {fi}/{f1}  beat {fi / FPS / BEAT:.1f}', flush=True)
    proc.stdin.close(); proc.wait()
    print('->', dst)


if __name__ == '__main__':
    main()
