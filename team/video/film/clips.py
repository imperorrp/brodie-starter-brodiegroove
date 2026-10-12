"""Animate each shot's keyframe with Veo (cached; budget-capped in gen.py).

Run:  python -m team.video.film.clips [id ...] [id=TAKE ...]
The chosen clip is copied to data/tmp/film/clips/<id>.mp4.
"""
import os, sys, shutil
from team.video.film.gen import veo, FILM, veo_seconds_used
from team.video.film.shots import SHOTS, veo_prompt

KEYS = os.path.join(FILM, 'keys')
CLIPS = os.path.join(FILM, 'clips')
NEGATIVE = 'subtitles, captions, text overlay, watermark, distorted hands, extra limbs, pink skin on Brodie, 3D shaded Brodie'


def main():
    args = sys.argv[1:]
    takes = dict(a.split('=') for a in args if '=' in a)
    ids = [a.split('=')[0] for a in args] or list(SHOTS)      # "s12=1" names the shot too, not just its take
    os.makedirs(CLIPS, exist_ok=True)
    for sid in ids:
        s = SHOTS[sid]
        path = veo(veo_prompt(sid), os.path.join(KEYS, sid + '.png'), s['secs'], name=sid,
                   take=int(takes.get(sid, 0)), negative=NEGATIVE)
        shutil.copy(path, os.path.join(CLIPS, sid + '.mp4'))
        print(f'{sid}: {path}', flush=True)
    print(f'Veo seconds used: {veo_seconds_used()}')


if __name__ == '__main__':
    main()
