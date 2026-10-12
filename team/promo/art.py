"""New Brodie poses for the launch posters (gemini-3-pro-image via team.video.film.gen, cached by request hash).

Run:  python -m team.promo.art [name ...] [name=TAKE ...]
Each pose is generated on white per the kit canon, cut out to a transparent PNG, and checked against canon cream.
Output: data/tmp/promo/art/<name>.png  (posters reference these paths)
"""
import os, sys, subprocess
import cv2, numpy as np
from PIL import Image
from team.video.film.gen import image, ROOT
from team.video.film.keyframes import BRODIE_REFS, CANON

ART = os.path.join(ROOT, 'data', 'tmp', 'promo', 'art')
CHECK = os.path.join(ROOT, 'team', 'dance', 'assets', 'brodie', 'tools', 'check_cream.py')

POSES = {
    'phone': 'Brodie holding a smartphone up beside his head like he is on a video call, the other hand waving hello, '
             'winking, big open grin.',
    'hype': 'HYPE Brodie: a backwards black snapback cap, one fist pumped high in the air, the other hand holding a small '
            'black microphone, mouth wide open in a huge hyped grin, eyebrows up.',
    'chill': 'CHILL Brodie: black over-ear headphones on his head, relaxed half-closed eyes, small closed calm smile, one '
             'hand holding a takeaway iced drink with a straw, the other hand giving a relaxed peace sign.',
    'coach': 'COACH Brodie: a black visor cap, a silver whistle on a black cord around his neck, one hand holding a '
             'clipboard against his chest, the other hand pointing forward at the viewer, focused serious eyebrows, '
             'small confident closed smile.',
    'point': 'Brodie pointing straight at the viewer with both index fingers, leaning forward, cheeky wink, big open '
             'grin, like he is saying "you, tag your crew". Keep his exact head from the references: a huge round ball '
             'head with a three-lobe cream hair tuft, NO ears, NO nose, chibi proportions, short stubby legs.',
    'judge': 'Brodie holding a stopwatch up in one hand and looking at it, one eyebrow raised skeptically, the other '
             'hand on his hip, small smirk.',
}


def cutout(src, dst):
    """white background -> transparent: flood the near-white region connected to the border, then feather the edge."""
    im = np.asarray(Image.open(src).convert('RGB'))
    h, w = im.shape[:2]
    white = (im.min(2) > 235).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    bg = white.copy()
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1)]:
        if bg[y, x] == 1:
            cv2.floodFill(bg, mask, (x, y), 2)
    alpha = np.where(bg == 2, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(cv2.erode(alpha, np.ones((3, 3), np.uint8)), (3, 3), 0)
    ys, xs = np.nonzero(alpha > 10)
    rgba = np.dstack([im, alpha])[max(0, ys.min() - 8):ys.max() + 9, max(0, xs.min() - 8):xs.max() + 9]
    Image.fromarray(rgba).save(dst)


def main(args):
    takes = dict(a.split('=') for a in args if '=' in a)
    names = [a.split('=')[0] for a in args] or list(POSES)
    os.makedirs(ART, exist_ok=True)
    for n in names:
        p = image(f'{POSES[n]} {CANON}', refs=BRODIE_REFS, aspect='1:1', name=f'promo_{n}', take=int(takes.get(n, 0)))
        dst = os.path.join(ART, n + '.png')
        cutout(p, dst)
        ok = subprocess.run([sys.executable, CHECK, dst], cwd=os.path.dirname(CHECK), capture_output=True, text=True)
        print(f'{n}: {dst}  cream: {"PASS" if ok.returncode == 0 else "FAIL"} {ok.stdout.strip()[-120:]}')


if __name__ == '__main__':
    main(sys.argv[1:])
