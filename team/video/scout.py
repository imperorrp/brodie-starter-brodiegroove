"""Contact sheets of candidate pitch-video footage, so in/out points can be picked by eye.

Run:  python -m team.video.scout [clip.mp4 ...]   (default: the candidate list below)
      python -m team.video.scout clip.mp4 FROM TO STEP   (close-up sheet of one time range)
Writes data/tmp/pitch/scout/<name>.jpg, one labelled thumbnail every STEP seconds.
"""
import os, sys, glob
import av, numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'tmp', 'pitch', 'scout')
EVAL = os.path.join(ROOT, 'data', 'tmp', 'eval')
CANDIDATES = [os.path.join(EVAL, f'{n}.mp4') for n in ('flow', 'battle', 'duel', 'walk_in', 'space_gesture', 'squad', 'daily', 'proactive')]
CANDIDATES += sorted(glob.glob(os.path.join(ROOT, 'data', 'tmp', 'dance_fake_room_*.mp4')))
CANDIDATES += [os.path.join(ROOT, 'data', 'tmp', 'sim_learn.mp4')]
COLS, THUMB_H = 10, 300


def sheet(path, t0=0.0, t1=None, step=None, tag=''):
    with av.open(path) as c:
        s = c.streams.video[0]
        dur = float(c.duration / av.time_base)
        step = step or (2.0 if dur <= 120 else 4.0 if dur <= 240 else 8.0)
        thumbs, next_t = [], t0
        if t0 > 1:
            c.seek(int((t0 - 1) / s.time_base), stream=s)
        for f in c.decode(s):
            t = float(f.time)
            if t1 is not None and t > t1:
                break
            if t + 1e-6 < next_t:
                continue
            im = f.to_image()
            w = round(im.width * THUMB_H / im.height)
            im = im.resize((w, THUMB_H))
            ImageDraw.Draw(im).text((6, 4), f'{t:.1f}s', fill=(217, 255, 61))
            thumbs.append(im); next_t += step
    w = thumbs[0].width
    rows = (len(thumbs) + COLS - 1) // COLS
    out = Image.new('RGB', (COLS * w, rows * THUMB_H), (10, 10, 11))
    for i, im in enumerate(thumbs):
        out.paste(im, ((i % COLS) * w, (i // COLS) * THUMB_H))
    name = os.path.splitext(os.path.basename(path))[0]
    dst = os.path.join(OUT, name + tag + '.jpg')
    out.save(dst, quality=80)
    print(f'{name}: {dur:.1f}s, step {step}s -> {dst}')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) == 5:                         # clip.mp4 from to step: a close-up sheet of one range
        a, b, st = map(float, sys.argv[2:])
        sheet(sys.argv[1], a, b, st, tag=f'_{a:g}-{b:g}')
        sys.exit()
    for p in sys.argv[1:] or CANDIDATES:
        try:
            sheet(p)
        except Exception as e:
            print(f'skip {p}: {e}')
