"""The preview card for a result link (Open Graph / Twitter image, 1200x630 PNG): what iMessage, Instagram DMs, X, WhatsApp and
Slack show when someone pastes https://<backend>/r/<code>. Drawn with Pillow from the result row; no photo of the dancer (the
in-call scorecard has their finish pose, the public card never does).

  png = og.result_card(result_row, handle=os.environ.get('DANCE_CALL_HANDLE'))   # bytes

Brodie on the card is kit art in the grade's mood (team/dance/avatar.py: S/A pumped, B happy, C/D encourage). If the avatar module
can't load (an API-only install without OpenCV) the kit profile picture is used instead.
"""
import importlib.util, io, os, sys
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DANCE = os.path.join(os.path.dirname(HERE), 'dance')
FONTS = os.path.join(DANCE, 'assets', 'fonts')
KIT = os.path.join(DANCE, 'assets', 'brodie')
W, H = 1200, 630
INK = (10, 10, 11); PAPER = (244, 244, 240); VOLT = (217, 255, 61); PINK = (255, 61, 168); MUTE = (154, 157, 164); CREAM = (239, 231, 218)
GRADE_COL = dict(S=VOLT, A=(61, 224, 255), B=(201, 201, 196), C=(255, 177, 61), D=(255, 77, 94))
MOOD = dict(S='pumped', A='pumped', B='happy', C='encourage', D='encourage')
MODE_LABEL = dict(learn='LEARN', battle='GHOST BATTLE', daily='DAILY CHALLENGE', squad_battle='SQUAD BATTLE', telephone='DANCE TELEPHONE',
                  **{'class': 'CLASS'})


def grade(score):
    s = float(score or 0)
    return 'S' if s >= 92 else 'A' if s >= 80 else 'B' if s >= 65 else 'C' if s >= 50 else 'D'


def score100(r):
    """The 0-100 score of a result row (battles store a total out of 300 and score100 beside it)."""
    v = r.get('score100')
    return float(v) if v is not None else float(r.get('total') or 0)


@lru_cache(maxsize=16)
def _font(name, px):
    path = os.path.join(FONTS, dict(display='ArchivoBlack-Regular.ttf', body='SpaceGrotesk.ttf')[name])
    try: return ImageFont.truetype(path, px)
    except Exception: return ImageFont.load_default()


@lru_cache(maxsize=8)
def _brodie(mood, height):
    """Kit Brodie as an RGBA image `height` px tall, in a mood (avatar.py), else the kit profile picture."""
    try:
        if 'team_avatar' not in sys.modules:
            spec = importlib.util.spec_from_file_location('team_avatar', os.path.join(DANCE, 'avatar.py'))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.modules['team_avatar'] = m
        a = sys.modules['team_avatar']
        im = a._get()[a.variant_for(mood, 'grin')]
    except Exception:
        im = Image.open(os.path.join(KIT, 'brodie-pfp-square.png')).convert('RGBA')
    w = max(1, round(im.width * height / im.height))
    return im.resize((w, height), Image.LANCZOS)


def _text(d, xy, s, px, fill, font='display', anchor='ls', max_w=None):
    f = _font(font, px)
    while max_w and px > 12 and d.textlength(s, font=f) > max_w:
        px -= 2; f = _font(font, px)
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def result_card(r, handle=None):
    """PNG bytes of the preview card for one result row (db.get_result)."""
    code = str(r.get('code') or ''); s = score100(r); g = grade(s); name = (r.get('name') or 'A dancer').strip()[:24]
    im = Image.new('RGB', (W, H), INK); d = ImageDraw.Draw(im)
    # Brodie on a cream disc at the right, in the grade's mood
    cx, cy, rad = 960, 330, 236
    d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=CREAM)
    b = _brodie(MOOD.get(g, 'happy'), 470)
    disc = Image.new('L', (W, H), 0); ImageDraw.Draw(disc).ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=255)
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0)); layer.paste(b, (cx - b.width // 2, cy - rad + 20), b)
    mask = Image.composite(layer.split()[3], Image.new('L', (W, H), 0), disc)
    im.paste(layer.convert('RGB'), (0, 0), mask)
    d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), outline=GRADE_COL[g], width=10)
    # left: wordmark, mode, the score with its grade, the name, the code, how to beat it
    x = 64
    _text(d, (x, 92), 'BRODIE GROOVE', 40, VOLT)
    mode = MODE_LABEL.get(r.get('mode') or '', str(r.get('mode') or '').upper())
    if mode:
        f = _font('display', 22); tw = d.textlength(mode, font=f)
        d.rounded_rectangle((x, 118, x + tw + 32, 156), radius=19, fill=PINK); d.text((x + 16, 146), mode, font=f, fill=INK, anchor='ls')
    _text(d, (x, 322), f'{s:.0f}', 180, PAPER)
    sw = d.textlength(f'{s:.0f}', font=_font('display', 180))
    gx = x + sw + 70; d.ellipse((gx - 52, 210, gx + 52, 314), fill=GRADE_COL[g]); _text(d, (gx, 262), g, 70, INK, anchor='mm')
    _text(d, (x, 380), f'{name.upper()} SCORED {s:.0f}', 34, PAPER, max_w=640)
    if code:
        _text(d, (x, 452), 'BEAT MY SCORE', 26, VOLT)
        tx = x
        for ch in code:
            d.rounded_rectangle((tx, 470, tx + 74, 560), radius=14, fill=VOLT); _text(d, (tx + 37, 516), ch, 58, INK, anchor='mm'); tx += 86
        how = f"FaceTime {handle} and say 'battle {code}'" if handle else f"FaceTime Brodie and say 'battle {code}'"
        _text(d, (x, 600), how, 26, MUTE, font='body', max_w=W - 2 * x)
    buf = io.BytesIO(); im.save(buf, 'PNG', optimize=True)
    return buf.getvalue()
