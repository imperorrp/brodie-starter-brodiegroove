"""Contact sheet of generated clips (frames every STEP s), to review Veo output by eye.

Run:  python -m team.video.film.peek [id ...] [--step 0.75]
Writes data/tmp/film/peek_<ids>.jpg
"""
import os, sys, subprocess, json
import av
from PIL import Image, ImageDraw
from team.video.film.gen import FILM

CLIPS = os.path.join(FILM, 'clips')


def frames(path, step):
    out = []
    with av.open(path) as c:
        s = c.streams.video[0]; nxt = 0.0
        has_audio = bool(c.streams.audio)
        for f in c.decode(s):
            if float(f.time) + 1e-6 >= nxt:
                out.append((float(f.time), f.to_image())); nxt += step
    return out, has_audio


def main():
    args = sys.argv[1:]
    step = float(args[args.index('--step') + 1]) if '--step' in args else 0.75
    ids = [a for a in args if not a.startswith('--') and a != str(step)] or sorted(p[:-4] for p in os.listdir(CLIPS))
    th = 360; rows = []
    for sid in ids:
        fr, aud = frames(os.path.join(CLIPS, sid + '.mp4'), step)
        tw = round(fr[0][1].width * th / fr[0][1].height)
        row = Image.new('RGB', (tw * len(fr), th))
        for i, (tt, im) in enumerate(fr):
            im = im.resize((tw, th)); ImageDraw.Draw(im).text((4, 4), f'{sid} {tt:.1f}s{" a" if aud else ""}', fill=(255, 0, 255))
            row.paste(im, (i * tw, 0))
        rows.append(row)
    W = max(r.width for r in rows)
    sheet = Image.new('RGB', (W, th * len(rows)))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * th))
    dst = os.path.join(FILM, f'peek_{"_".join(ids)[:60]}.jpg'); sheet.save(dst, quality=85); print(dst)


if __name__ == '__main__':
    main()
