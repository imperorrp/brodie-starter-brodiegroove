"""Cut the pitch film: Veo shots on timeline.py's beats, the gags and product overlays drawn on top, muxed with mix.wav.

Run:  python -m team.video.film.audio && python -m team.video.film.make_film [--stills] [--from B --to B]
Output: data/tmp/film/brodie_film.mp4   (--stills: two JPGs per shot in data/tmp/film/stills/)
"""
import os, sys, math, subprocess
import av, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFilter
from team.video import make_pitch as P
from team.video.film.timeline import SHOTS, VOICE, FLASH, BEAT, FPS, W, H, TOTAL_BEATS, DURATION, CLIPS, VO, t
from team.video.film.gen import FILM
from team.dance import render as R

C = P.C


class Clip:
    """decodes the window of a clip one shot needs, once; frame at a clip time, scaled to 1080x1920 and sharpened."""
    def __init__(self):
        self.key = None; self.frames = []

    def load(self, name, t0, t1):
        if self.key == (name, t0, t1):
            return
        self.key = (name, t0, t1); self.frames = []
        with av.open(os.path.join(CLIPS, name + '.mp4')) as c:
            s = c.streams.video[0]
            for f in c.decode(s):
                ft = float(f.time)
                if ft < t0 - 0.1:
                    continue
                if ft > t1 + 0.1:
                    break
                im = cv2.resize(f.to_ndarray(format='rgb24'), (W, H), interpolation=cv2.INTER_LANCZOS4)
                blur = cv2.GaussianBlur(im, (0, 0), 1.6)
                self.frames.append((ft, cv2.addWeighted(im, 1.35, blur, -0.35, 0)))     # unsharp: Veo is 720p

    def at(self, ts):
        best = self.frames[0][1]
        for ft, im in self.frames:
            if ft <= ts + 1e-3:
                best = im
            else:
                break
        return best


CLIP = Clip()


def zoom(img, s, cx=W / 2, cy=H / 2, dx=0, dy=0):
    if s == 1.0 and not dx and not dy:
        return img
    m = np.array([[s, 0, cx - s * cx + dx], [0, s, cy - s * cy + dy]], np.float64)
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------- overlays ----------------
def labels_overlay(frame, beat, start, s):
    """the freeze-frame gag: both feet tagged LEFT."""
    for i, (fx, fy, text, col) in enumerate(((230, 1215, 'LEFT', 'volt'), (630, 1215, 'ALSO LEFT', 'pink'))):
        b0 = start + i * 0.5
        if beat < b0:
            continue
        p = P.spring((beat - b0) / 0.45)
        x = W / 2 + (fx * 1.5 - W / 2) * s; y = H / 2 + (fy * 1.5 - H / 2) * s
        ly = y - 300
        d = np.zeros((H, W), np.uint8)
        cv2.arrowedLine(d, (int(x), int(ly + 50)), (int(x), int(ly + 50 + (y - ly - 90) * p)), 255, 10, cv2.LINE_AA, tipLength=0.25)
        a = (d.astype(np.float32) / 255)[..., None]
        frame[:] = (frame * (1 - a) + np.array(C[col]) * a).astype(np.uint8)
        im = P.pill(text, col, 54)
        lx = min(max(x, 50 + im.width / 2), W - 50 - im.width / 2)          # keep the pill on screen; the arrow stays on the foot
        P.blend_center(frame, im, lx, ly, scale=p, angle=(-6 if i == 0 else 5) * p)


def call_banner(frame, beat, start):
    """the incoming call, drawn like the product's call screen: Brodie's face, name, and a pulsing answer button."""
    if beat < start:
        return
    p = P.spring((beat - start) / 0.6)
    w, h = 980, 190
    im = Image.new('RGBA', (w + 20, h + 26), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((10, 18, 10 + w, 18 + h), 48, fill=(0, 0, 0, 110))
    d.rounded_rectangle((10, 8, 10 + w, 8 + h), 48, fill=(16, 16, 19, 235), outline=(255, 255, 255, 40), width=2)
    im.alpha_composite(P.circle_img(os.path.join(P.WEB, 'brodie-pfp-square.png'), 130, 6, 'volt'), (36, 8 + (h - 142) // 2))
    d.text((210, 46), 'BRODIE', font=P.font('display', 58), fill=C['paper'] + (255,))
    dots = '.' * (1 + int(beat * 2) % 3)
    d.text((212, 118), 'BrodieGroove · video call' + dots, font=P.font('grotesk', 34, 500), fill=C['mut'] + (255,))
    pulse = 1 + 0.12 * math.exp(-((beat % 1.0) * 5))
    r = int(52 * pulse); cx, cy = 10 + w - 90, 8 + h // 2
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=C['volt'] + (255,))
    d.rounded_rectangle((cx - 26, cy - 17, cx + 8, cy + 17), 6, fill=C['ink'] + (255,))
    d.polygon([(cx + 10, cy - 4), (cx + 26, cy - 14), (cx + 26, cy + 14), (cx + 10, cy + 4)], fill=C['ink'] + (255,))
    P.blend_center(frame, im, W / 2, -120 + 400 * p)


def dots_overlay(frame, beat, b0):
    """the product's beat dots, filling volt count by count from the downbeat."""
    k = beat - (b0 + 2)
    verdicts = [('green' if i < int(k) else None) for i in range(8)] if k >= 0 else [None] * 8
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr = R.draw_beat_dots(bgr, verdicts, int(k) if k >= 0 else -1, 8, (W, H), (beat - b0) * BEAT)
    frame[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def count_overlay(frame, beat, counts):
    for i, cb in enumerate(counts):
        age = beat - cb
        if 0 <= age < 1.0:
            s = 0.6 + 0.6 * P.spring(age / 0.35)
            a = 1.0 if age < 0.7 else (1 - age) / 0.3
            P.blend_center(frame, P.burst_word(str(5 + i), 'volt'), 300, 640, scale=s * 0.9, angle=-6, alpha=a)


def tip_overlay(frame, beat, text, start):
    if beat < start:
        return
    p = P.spring((beat - start) / 0.4)
    P.blend_center(frame, P.pill(text, 'amber', 52), W / 2, 1180, scale=p)


def combo_overlay(frame, beat, start, text):
    if beat < start:
        return
    p = P.spring((beat - start) / 0.4)
    pulse = 1 + 0.06 * math.exp(-((beat % 1.0) * 5))
    P.blend_center(frame, P.pill(text, 'pink', 58), W / 2, 760, scale=p * pulse, angle=4)


def stamp_overlay(frame, beat, start, grade, title):
    if beat < start:
        return
    age = beat - start
    s = 2.4 - 1.4 * P.ease_out(age / 0.25)                        # slams down onto the frame
    a = min(1.0, age / 0.12)
    ring = Image.new('RGBA', (360, 360), (0, 0, 0, 0)); d = ImageDraw.Draw(ring)
    d.ellipse((10, 10, 350, 350), fill=(10, 10, 11, 230), outline=C['volt'] + (255,), width=18)
    f = P.font('display', 210); tw, th, b = P.text_size(f, grade)
    d.text((180 - tw / 2 - b[0], 180 - th / 2 - b[1]), grade, font=f, fill=C['volt'] + (255,))
    P.blend_center(frame, ring, 840, 400, scale=s * 0.62, angle=-8, alpha=a)      # top right, clear of both faces
    if age > 0.3:
        tp = P.spring((age - 0.3) / 0.4)
        tf = P.font('display', 92); tw, th, b = P.text_size(tf, title)
        im = Image.new('RGBA', (tw + 40, th + 40), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((20 - b[0], 20 - b[1]), title, font=tf, fill=C['pink'] + (255,), stroke_width=6, stroke_fill=C['ink'] + (255,))
        P.blend_center(frame, im, W / 2, 1150, scale=tp, angle=-3)


def phone_frame(frame_bg, screen):
    """the real call screen inside a phone, floating over a blurred copy of the room."""
    bg = cv2.GaussianBlur(frame_bg, (0, 0), 18)
    bg = (bg * 0.55).astype(np.uint8)
    sw, sh = 600, 1066
    scr = cv2.resize(screen, (sw, sh), interpolation=cv2.INTER_AREA)
    bez = 22
    ph = np.zeros((sh + 2 * bez, sw + 2 * bez, 4), np.uint8)
    m = Image.new('L', (sw + 2 * bez, sh + 2 * bez), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, sw + 2 * bez - 1, sh + 2 * bez - 1), 70, fill=255)
    ph[..., :3] = (14, 14, 16); ph[..., 3] = np.asarray(m)
    sm = Image.new('L', (sw, sh), 0); ImageDraw.Draw(sm).rounded_rectangle((0, 0, sw - 1, sh - 1), 52, fill=255)
    sma = (np.asarray(sm, np.float32) / 255)[..., None]
    ph[bez:bez + sh, bez:bez + sw, :3] = (scr * sma + ph[bez:bez + sh, bez:bez + sw, :3] * (1 - sma)).astype(np.uint8)
    im = Image.fromarray(ph, 'RGBA')
    return bg, im


# ---------------- frames ----------------
def shot_at(beat):
    for s in SHOTS:
        if s[0] <= beat < s[1]:
            return s
    return SHOTS[-1]


VO_DUR = {}


def render_frame(fi):
    tsec = fi / FPS; beat = tsec / BEAT
    b0, b1, clip, p = shot_at(beat)
    lb = beat - b0
    if clip == 'end':
        frame = P.end_card(64 + lb, tsec)
    else:
        rate = p.get('rate', 1.0)
        if p.get('freeze'):
            CLIP.load(clip, p['at'], p['at']); ct = p['at']
        else:
            CLIP.load(clip, p['at'], p['at'] + t(b1 - b0) * rate); ct = p['at'] + (tsec - t(b0)) * rate
        frame = CLIP.at(ct).copy()
        s = 1.0
        if 'punch' in p:
            s *= 1 + (p['punch'] - 1) * math.exp(-lb * 3.0)
        if 'zoom' in p:
            z0, z1 = p['zoom']; s *= z0 + (z1 - z0) * P.ease_inout(lb / (b1 - b0))
        dx = 0
        if p.get('freeze') and lb < 0.35:                          # the needle-scratch judder
            dx = (18 if int(lb * 20) % 2 else -18)
        if p.get('phone'):
            CLIP2.load('s09c', p['at'], p['at'] + t(b1 - b0))
            bg, ph = phone_frame(CLIP2.at(ct), CLIP.at(ct))
            frame = bg
            fl = math.sin(tsec * 2.2) * 8
            P.blend_center(frame, ph, W / 2, H / 2 + 30 + fl, scale=0.92 + 0.08 * P.spring(lb / 0.6), angle=-4)
        else:
            frame = zoom(frame, s, dx=dx)
        if p.get('desat'):
            g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[..., None]
            frame = (frame * (1 - p['desat']) + g * p['desat']).astype(np.uint8)
        if 'labels' in p:
            labels_overlay(frame, beat, p['labels'], s)
        if 'banner' in p:
            call_banner(frame, beat, p['banner'])
        if p.get('dots'):
            dots_overlay(frame, beat, b0)
        if 'counts' in p:
            count_overlay(frame, beat, p['counts'])
        if 'tip' in p:
            tip_overlay(frame, beat, p['tip'][0], p['tip'][1])
        if 'burst' in p:
            bb, word, col = p['burst']
            P.burst_overlay(frame, word, col, beat - bb, cy=1150)          # below the faces
        if 'combo' in p:
            combo_overlay(frame, beat, p['combo'][0], p['combo'][1])
        if 'stamp' in p:
            stamp_overlay(frame, beat, *p['stamp'])
        if 'chip' in p:
            P.chip_overlay(frame, p['chip'][0], p['chip'][1], beat, p['chip'][2])
        captions(frame, tsec)
    for fb in FLASH:
        age = (beat - fb) * BEAT * FPS
        if 0 <= age < 4:
            a = 0.6 * (1 - age / 4)
            frame = (frame * (1 - a) + np.array((250, 255, 225)) * a).astype(np.uint8)
    return P.finish(frame, fi)


CLIP2 = Clip()


def captions(frame, tsec):
    for key, beat, side, text in VOICE:
        if not text:
            continue
        a = t(beat); dur = VO_DUR[key]
        if not (a - 0.05 <= tsec <= a + dur + 0.25):
            continue
        n = len(text.split(' '))
        shown = min(n, 1 + int((tsec - a) / max(0.2, dur * 0.85) * n))
        im = P.caption_img(text, 'b' if side == 'b' else 'u', shown)
        pp = P.spring((tsec - a) / 0.35); out = max(0.0, (tsec - (a + dur + 0.05)) / 0.2)
        y = 1460 if side == 'b' else 1330
        x = 60 + im.width / 2 if side == 'b' else W - 60 - im.width / 2
        P.blend_center(frame, im, x, y + 40 * (1 - pp), scale=0.85 + 0.15 * pp, alpha=1 - min(1, out))


def main():
    import wave
    for key, *_ in VOICE:
        with wave.open(os.path.join(VO, key + '.wav')) as w:
            VO_DUR[key] = w.getnframes() / w.getframerate()
    args = sys.argv[1:]
    if '--stills' in args:
        out = os.path.join(FILM, 'stills'); os.makedirs(out, exist_ok=True)
        for s in SHOTS:
            for tag, b in (('a', s[0] + 0.3), ('m', (s[0] + s[1]) / 2)):
                Image.fromarray(render_frame(int(t(b) * FPS))).save(os.path.join(out, f'{s[0]:05.1f}_{tag}_{s[2]}.jpg'), quality=85)
        print('stills ->', out); return
    fb = float(args[args.index('--from') + 1]) if '--from' in args else 0
    tb = float(args[args.index('--to') + 1]) if '--to' in args else TOTAL_BEATS
    n = int(round(DURATION * FPS)); f0, f1 = int(t(fb) * FPS), min(n, int(t(tb) * FPS))
    dst = os.path.join(FILM, 'brodie_film.mp4' if (fb, tb) == (0, TOTAL_BEATS) else f'preview_{fb:g}-{tb:g}.mp4')
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-ss', f'{f0 / FPS:.4f}', '-t', f'{(f1 - f0) / FPS:.4f}', '-i', os.path.join(FILM, 'mix.wav'),
           '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-maxrate', '12M', '-bufsize', '24M',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-af', 'loudnorm=I=-14:TP=-1.0:LRA=11,alimiter=limit=0.85:level=false',   # one-pass loudnorm overshoots the peak
           '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-shortest', dst]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        proc.stdin.write(render_frame(fi).tobytes())
        if (fi - f0) % 150 == 0:
            print(f'frame {fi}/{f1}', flush=True)
    proc.stdin.close(); proc.wait()
    print('->', dst)


if __name__ == '__main__':
    main()
