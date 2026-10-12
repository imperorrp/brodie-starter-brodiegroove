"""The real BrodieGroove call screen over a generated shot: MediaPipe pose on Maya, drawn by the product's own renderer
(team/dance/render.py compose: live skeleton, ghost from a real AIST++ key, beat dots, PERFECT/NICE pops, combo, score).

Run:  python -m team.video.film.hud_insert [shot] [--bpm 105]
Writes data/tmp/film/clips/<shot>_hud.mp4 (720x1280, the shot's fps, no audio).
"""
import os, sys, json, subprocess
import av, cv2, numpy as np
from team.video.film.gen import FILM, ROOT
from team.dance.multipose import MultiPose
from team.dance import render as R
from team.dance.fx import FxQueue

CLIPS = os.path.join(FILM, 'clips')
KEY = os.path.join(ROOT, 'team', 'dance', 'library', 'aist_lh_110_ch01.json')      # LA-style hip-hop, a real dancer
# the verdict on each count of the eight (the story: she's got it now, with one NICE to keep it honest)
VERDICTS = ['green', 'green', 'amber', 'green', 'green', 'green', 'green', 'green']


def main():
    args = sys.argv[1:]
    shot = next((a for a in args if not a.startswith('--')), 's09c')
    bpm = float(args[args.index('--bpm') + 1]) if '--bpm' in args else 105.0
    key = json.load(open(KEY))
    skel = np.asarray(key['skeleton'], np.float32)          # (n, 33, 2), 4 samples per beat
    src = os.path.join(CLIPS, shot + '.mp4')
    dst = os.path.join(CLIPS, shot + '_hud.mp4')
    pose = MultiPose('full', video=True, max_poses=1)
    fx = FxQueue()
    spb = 60 / bpm
    with av.open(src) as c:
        s = c.streams.video[0]; fps = float(s.average_rate)
        w, h = s.codec_context.width, s.codec_context.height
        enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '720x1280',
                                '-r', f'{fps:g}', '-i', '-', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', dst],
                               stdin=subprocess.PIPE)
        scored = set(); score = 0; combo = 0; best = 0
        for f in c.decode(s):
            t = float(f.time)
            rgb = f.to_ndarray(format='rgb24')
            poses = pose(rgb, t * 1000)
            lm = poses[0][0] if poses else None
            beat = t / spb
            b = int(beat) % 8
            for k in range(int(beat) + 1):                    # judge each count as it lands
                if k not in scored and k < 8 * 4:
                    v = VERDICTS[k % 8]; scored.add(k)
                    fx.add('judgment', k * spb, verdict=v)
                    combo = combo + 1 if v == 'green' else 0; best = max(best, combo)
                    score += 100 if v == 'green' else 60
                    if combo in (4, 8):
                        fx.add('combo', k * spb, n=combo)
            verdicts = [VERDICTS[i] if i < (int(beat) % 8) or int(beat) >= 8 else None for i in range(8)]
            hud = dict(state='attempt', key=key['name'], title='Hip-Hop Starter', bpm=bpm, n_beats=8, verdicts=verdicts,
                       current_beat=b, t=t, fx=fx, stage='CHALLENGE', combo=combo, running_score=score,
                       difficulty='beginner', speaking=False)
            ghost = skel[int(((beat + 1) % (len(skel) / 4)) * 4) % len(skel)]      # the ghost runs one beat ahead
            out = R.compose(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), lm, hud, ghost=ghost, size=(720, 1280))
            enc.stdin.write(np.ascontiguousarray(out).tobytes())
        enc.stdin.close(); enc.wait()
    pose.close()
    print(f'{dst}: {w}x{h} at {fps:g} fps, combo best {best}, score {score}')


if __name__ == '__main__':
    main()
