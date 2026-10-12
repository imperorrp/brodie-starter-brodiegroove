"""Cut the 9:16 reels (TikTok / IG Reels) for days 0-7 from footage already on disk, each closing on that day's end card.

Run:  python -m team.promo.clips [d0 d1 ...]
Re-renders the end cards and title overlays from config.json first (team.promo.render), so filling in call_line and
re-running updates the videos too. Output: data/tmp/promo/<day>/<day>_reel.mp4 + <day>_reel_cover.jpg (1080x1920, 30 fps).

Sources: data/tmp/pitch/brodie_pitch.mp4 (46.2 s pitch, end card from 38.4 s), data/tmp/film/brodie_film.mp4 (54.8 s
story, end card from 49.7 s), data/tmp/eval/*.mp4 (720x1280 or 1280x720 call captures, 15 fps, silent),
data/tmp/hud/*.png, data/tmp/pitch/music.wav (the landing-page groove), data/tmp/promo/vo/*.wav (Day 4 voices).
"""
import sys, json, subprocess, tempfile, shutil
from pathlib import Path
from team.promo import render as R

ROOT = R.ROOT
OUT = R.OUT
PITCH = ROOT / 'data' / 'tmp' / 'pitch' / 'brodie_pitch.mp4'
FILM = ROOT / 'data' / 'tmp' / 'film' / 'brodie_film.mp4'
EVAL = ROOT / 'data' / 'tmp' / 'eval'
HUD = ROOT / 'data' / 'tmp' / 'hud'
MUSIC = ROOT / 'data' / 'tmp' / 'pitch' / 'music.wav'
VO = OUT / 'vo'
W, H, FPS = 1080, 1920, 30
PITCH_END, FILM_END = 64 * 0.6, 87 * 60 / 105          # where each video's own end card starts (timeline beats)
CALL_CROP = (16, 52, 704, 1278)                         # the call capture's frame inside a 720x1280 eval render
ENC = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', str(FPS)]


def ff(*args):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])


def dur(path):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                                capture_output=True, text=True).stdout)


# ---------- segments: each one is a silent 1080x1920 30 fps H.264 file, so they concat without re-timing ----------
FW = 744                                                # framed footage width (the title overlay sits above it)
FRAMED = (f'split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:2,eq=brightness=-0.32[bg];'
          f'[b]scale={FW}:{FW * 16 // 9},pad={FW + 12}:{FW * 16 // 9 + 12}:6:6:color=0xD9FF3D[fg];[bg][fg]overlay=(W-w)/2:500')


def _run(dst, inputs, base, overlay, out_args=()):
    """base: filter chain from [0:v] to a full 1080x1920 frame. With an overlay (a title PNG) the footage is framed
    smaller under it so the title never covers the product HUD."""
    if overlay:
        fc = f'[0:v]{base},{FRAMED}[v];[v][1:v]overlay=0:0'
        ff(*inputs, '-i', overlay, '-filter_complex', fc, *out_args, '-an', *ENC, dst)
    else:
        ff(*inputs, '-vf', base, *out_args, '-an', *ENC, dst)


def seg_video(dst, src, start, length, overlay=None):
    """a slice of a 1080x1920 video (pitch / film)."""
    _run(dst, ['-ss', start, '-t', length, '-i', src], 'null', overlay)


def seg_call(dst, src, start, length, overlay=None):
    """a slice of a portrait call capture: crop to the call frame, fill 1080x1920, 15 -> 30 fps."""
    x0, y0, x1, y1 = CALL_CROP
    _run(dst, ['-ss', start, '-t', length, '-i', src], f'crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={W}:-2,crop={W}:{H},fps={FPS}', overlay)


def seg_land(dst, src, start, length, overlay):
    """a slice of a landscape capture: full-width over a blurred, darkened fill of itself, overlay on top."""
    fc = (f'[0:v]fps={FPS},split[a][b];[a]scale=-2:{H},crop={W}:{H},boxblur=30:2,eq=brightness=-0.25[bg];'
          f'[b]scale={W}:-2[fg];[bg][fg]overlay=0:(H-h)/2+120[v];[v][1:v]overlay=0:0')
    ff('-ss', start, '-t', length, '-i', src, '-i', overlay, '-filter_complex', fc, '-an', *ENC, dst)


def seg_still(dst, img, length, zoom=0.05, crop_top=None, overlay=None):
    """a still with a slow push-in. crop_top keeps only the top fraction (hides the demo QR on scorecards)."""
    n = int(round(length * FPS))
    fit = (f"crop=iw:ih*{crop_top}:0:0,scale={W * 2}:{H * 2}:force_original_aspect_ratio=decrease,"
           f"pad={W * 2}:{H * 2}:(ow-iw)/2:(oh-ih)/2:color=0x0A0A0B," if crop_top else
           f"scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},")
    base = (f"{fit}"
            f"zoompan=z='1+{zoom}*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS}")
    _run(dst, ['-i', img], base, overlay, ['-frames:v', n])


def concat(parts, dst):
    lst = Path(dst).with_suffix('.txt')
    lst.write_text(''.join(f"file '{Path(p).as_posix()}'\n" for p in parts), encoding='utf-8')
    ff('-f', 'concat', '-safe', 0, '-i', lst, '-c', 'copy', dst)
    lst.unlink()


def mux(video, dst, audio_filter_inputs, fc_audio):
    """video + an audio graph that must output [a]; loudness to -14 LUFS (make_pitch's master), fade out the tail."""
    d = dur(video)
    ins = []
    for a in audio_filter_inputs:
        ins += a
    fc = f'{fc_audio};[a]atrim=0:{d:.3f},afade=t=out:st={max(0, d - 0.8):.3f}:d=0.8,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[ao]'
    ff('-i', video, *ins, '-filter_complex', fc, '-map', '0:v', '-map', '[ao]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
       '-movflags', '+faststart', '-shortest', dst)


def end_overlay(video, card, at, dst):
    """fade the day's end card in over a finished video from `at` seconds (keeps the video's own audio)."""
    fc = f"[1:v]format=rgba,fade=t=in:st={at:.3f}:d=0.35:alpha=1[c];[0:v][c]overlay=0:0:enable='gte(t,{at:.3f})'[v]"
    ff('-i', video, '-loop', 1, '-t', dur(video), '-i', card, '-filter_complex', fc, '-map', '[v]', '-map', '0:a',
       *ENC, '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest', dst)


def cover(video, at, dst):
    ff('-ss', at, '-i', video, '-frames:v', 1, '-q:v', 2, dst)


# ---------- the reels ----------
def assets():
    """end cards + overlays from the current config (so the CTA in the videos matches the posters)."""
    R.write_fonts_css(); R.write_config_js(R.load_config())
    exe = R.browser()
    cards = {d: OUT / 'end' / f'endcard_9x16_{d}.png' for d in range(8)}
    for d, p in cards.items():
        R.shoot('endcard.html', p, '9x16', {'slide': d}, exe=exe)
    frames = {}
    for k in ['d3', 'd4h', 'd4c', 'd4k', 'd5', 'd6', 'd7']:
        frames[k] = R.shoot('reelframe.html', OUT / 'frames' / f'{k}.png', '9x16', {'slide': k, 'alpha': 1}, exe=exe)
    return cards, frames


def reel(day, cards, frames, tmp):
    out_dir = OUT / f'd{day}'; out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f'd{day}_reel.mp4'
    t = lambda name: Path(tmp) / f'd{day}_{name}.mp4'
    music = [['-i', MUSIC]]

    if day == 0:          # the incoming call, then "Missed call. He'll call back MON 10.12."
        seg_video(t('a'), PITCH, 0, 2.3); seg_still(t('b'), cards[0], 4.7, 0.04)
        concat([t('a'), t('b')], t('v'))
        mux(t('v'), dst, [['-i', PITCH], ['-i', MUSIC]],
            '[1:a]atrim=0:2.3,afade=t=out:st=2.0:d=0.3[x];[2:a]atrim=2.3,asetpts=PTS-STARTPTS,afade=t=in:d=0.4,volume=0.8[y];[x][y]concat=n=2:v=0:a=1[a]')
        cover(dst, 0.5, out_dir / f'd{day}_reel_cover.jpg')
    elif day == 1:        # the full pitch video, new end card from beat 64
        end_overlay(PITCH, cards[1], PITCH_END, dst)
        cover(dst, 0.5, out_dir / f'd{day}_reel_cover.jpg')
    elif day == 2:        # the Maya film: two left feet -> S-rank
        end_overlay(FILM, cards[2], FILM_END, dst)
        cover(dst, 43.0, out_dir / f'd{day}_reel_cover.jpg')
    elif day == 3:        # scored on every count: the PERFECT / combo run from the battle capture
        seg_call(t('a'), EVAL / 'battle.mp4', 26.0, 11.0, frames['d3']); seg_still(t('b'), cards[3], 4.0, 0.03)
        concat([t('a'), t('b')], t('v'))
        mux(t('v'), dst, music, '[1:a]volume=0.9[a]')
        cover(dst, 4.0, out_dir / f'd{day}_reel_cover.jpg')
    elif day == 4:        # the same tip in HYPE, CHILL and COACH voices
        meta = json.loads((VO / 'lines.json').read_text())
        lens = [meta[k]['duration_s'] + 0.7 for k in ('hype', 'chill', 'coach')]
        starts = [44.0, 44.0 + lens[0], 44.0 + lens[0] + lens[1]]
        for i, (k, f) in enumerate(zip(('a', 'b', 'c'), ('d4h', 'd4c', 'd4k'))):
            seg_call(t(k), EVAL / 'duel.mp4', starts[i], lens[i], frames[f])
        seg_still(t('d'), cards[4], 4.0, 0.03)
        concat([t('a'), t('b'), t('c'), t('d')], t('v'))
        at = [0.35, lens[0] + 0.35, lens[0] + lens[1] + 0.35]
        ins = music + [['-i', VO / f'{k}.wav'] for k in ('hype', 'chill', 'coach')]
        fc = ('[1:a]volume=0.32[m];' + ''.join(f'[{i + 2}:a]adelay={int(a * 1000)}|{int(a * 1000)},volume=1.6[v{i}];' for i, a in enumerate(at)) +
              '[m][v0][v1][v2]amix=inputs=4:normalize=0[a]')
        mux(t('v'), dst, ins, fc)
        cover(dst, 1.0, out_dir / f'd{day}_reel_cover.jpg')
    elif day == 5:        # squad battle: two dancers, colour tags, results
        seg_land(t('a'), EVAL / 'squad.mp4', 30.0, 9.0, frames['d5']); seg_land(t('b'), EVAL / 'squad.mp4', 54.5, 3.0, frames['d5'])
        seg_still(t('c'), cards[5], 4.0, 0.03)
        concat([t('a'), t('b'), t('c')], t('v'))
        mux(t('v'), dst, music, '[1:a]volume=0.9[a]')
        cover(dst, 3.0, out_dir / f'd{day}_reel_cover.jpg')
    elif day == 6:        # grade reveal montage, one still per beat pair (0.6 s beat at 100 BPM)
        stills = [(HUD / '17_levelup.png', None), (HUD / '16_recap.png', None), (HUD / '19_scorecard.png', 0.6),
                  (ROOT / 'data' / 'tmp' / 'film' / 'stills' / '075.0_m_s11.jpg', None), (HUD / '20_photo_card.png', 0.6), (HUD / '10_challenge_fire.png', None)]
        parts = []
        for i, (img, ct) in enumerate(stills):
            seg_still(t(f's{i}'), img, 1.2, 0.08, ct, frames['d6']); parts.append(t(f's{i}'))
        seg_still(t('e'), cards[6], 4.0, 0.03)
        concat(parts + [t('e')], t('v'))
        mux(t('v'), dst, music, '[1:a]volume=0.9[a]')
        cover(dst, 3.0, out_dir / f'd{day}_reel_cover.jpg')
    elif day == 7:        # best-of the week: call, ghost, speed ramp, S-rank, scorecard
        seg_video(t('a'), PITCH, 0.0, 2.4, frames['d7']); seg_video(t('b'), PITCH, 9.6, 3.0, frames['d7'])
        seg_video(t('c'), PITCH, 26.4, 2.4, frames['d7']); seg_video(t('d'), FILM, 42.5, 3.4, frames['d7'])
        seg_video(t('e'), PITCH, 34.8, 2.4, frames['d7']); seg_still(t('f'), cards[7], 4.0, 0.03)
        concat([t(k) for k in 'abcdef'], t('v'))
        mux(t('v'), dst, music, '[1:a]volume=0.9[a]')
        cover(dst, 6.0, out_dir / f'd{day}_reel_cover.jpg')
    return dst


def main(argv):
    days = [int(a.lstrip('d')) for a in argv] or list(range(8))
    cards, frames = assets()
    tmp = tempfile.mkdtemp(prefix='promo_reels_')
    try:
        for d in days:
            p = reel(d, cards, frames, tmp)
            print(f'{p.relative_to(ROOT)}  {dur(p):.1f}s')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main(sys.argv[1:])
