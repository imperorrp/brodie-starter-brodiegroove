"""Offline harness: build keys from videos, run attempt videos through the engine, and self-test without any camera or phone.

  python -m team.dance.offline_test selftest
  python -m team.dance.offline_test key --video ref.mp4 --name wave --bpm 100 [--start 2 --end 12] [--no-mirror] [--source ... --licence ...]
  python -m team.dance.offline_test run --video attempt.mp4 --key wave [--phrase 0] [--tempo 1.0] [--offset 0.0] [--out out.mp4] [--report r.json]
  python -m team.dance.offline_test landmarks --video clip.mp4 --out clip.npz        # cache a pose extraction
  python -m team.dance.offline_test list

`run` treats the attempt video's clock as the music clock: the attempt's first beat is at --offset seconds into the video (the
dancer started on the music at that moment). It writes an annotated portrait MP4 (what the caller would see) and a JSON report.
"""
import argparse, json, os, sys, time
import numpy as np
import cv2
from . import LIBRARY_DIR
from .features import N_CHANNELS, CHANNELS, GROUPS
from .scoring import score_phrase, PER_BEAT
from .reference import build_key, load_key, list_keys, key_from_arrays, extract_landmarks, phrase_slice
from .engine import PhraseEngine
from .render import compose, draw_card, W, H


def synthetic_key(n_beats=8, bpm=100.0, per_beat=PER_BEAT, seed=0):
    """Smooth, plausible angle trajectories: each channel a sinusoid with its own phase, in realistic ranges."""
    rng = np.random.default_rng(seed); n = n_beats * per_beat; t = np.arange(n) / per_beat
    F = np.zeros((n, N_CHANNELS), np.float32)
    base = np.array([150, 150, 40, 40, 160, 160, 165, 165, 10, 20, 40, 40, 30, 30], np.float32)
    amp = np.array([40, 40, 30, 30, 15, 15, 20, 20, 8, 10, 50, 50, 15, 25], np.float32)
    for c in range(N_CHANNELS):
        F[:, c] = base[c] + amp[c] * np.sin(2 * np.pi * (t / 2.0 + rng.uniform(0, 1)))
    E = 0.8 + 0.6 * np.abs(np.sin(2 * np.pi * t))          # hip energy peaks on beats
    return key_from_arrays('synthetic', bpm, F, E=E, per_beat=per_beat, counts_per_phrase=n_beats)


def selftest(verbose=True):
    key = synthetic_key(); kF, kM, kE, kS = phrase_slice(key, 0); pb = key['per_beat']; n = len(kF); rng = np.random.default_rng(1)
    extra = pb
    def live_from(F, E):
        Fl = np.concatenate([F, np.repeat(F[-1:], extra, axis=0)]); El = np.concatenate([E, np.repeat(E[-1:], extra)])
        return Fl, np.ones_like(Fl, bool), El
    checks = []
    F, M, E = live_from(kF, kE); r_id = score_phrase(F, M, E, kF, kM, kE, pb); checks.append(('identity scores >= 95', r_id['score'] >= 95, r_id['score']))
    shift = 2                                                       # half a beat late
    Fs = np.concatenate([np.repeat(kF[:1], shift, axis=0), kF]); Es = np.concatenate([np.repeat(kE[:1], shift), kE])
    F, M, E = live_from(Fs[:n], Es[:n]); F = np.concatenate([Fs, np.repeat(kF[-1:], extra - shift, axis=0)]); E = np.concatenate([Es, np.repeat(kE[-1:], extra - shift)])
    r_sh = score_phrase(F, np.ones_like(F, bool), E, kF, kM, kE, pb)
    checks.append(('half-beat shift: timing term drops below identity', r_sh['timing'] < r_id['timing'] or r_sh['mean_offset_beats'] > 0.3, (r_sh['timing'], r_sh['mean_offset_beats'])))
    checks.append(('half-beat shift: shapes still mostly right (spatial > 0.6)', r_sh['spatial'] > 0.6, r_sh['spatial']))
    Fp = kF.copy(); Fp[:, GROUPS['arms']] += 40.0
    F, M, E = live_from(Fp, kE); r_arm = score_phrase(F, M, E, kF, kM, kE, pb)
    checks.append(('arms perturbed: arms group error > legs error', r_arm['per_group']['arms'] > r_arm['per_group']['legs'] + 20, r_arm['per_group']))
    checks.append(('arms perturbed: score below identity', r_arm['score'] < r_id['score'] - 10, r_arm['score']))
    Fn = kF + rng.normal(0, 45, kF.shape).astype(np.float32)
    F, M, E = live_from(Fn, rng.uniform(0, 2, len(kE)).astype(np.float32)); r_noise = score_phrase(F, M, E, kF, kM, kE, pb)
    checks.append(('random noise scores < 50', r_noise['score'] < 50, r_noise['score']))
    Mv = np.ones_like(kF, bool); Mv[:, :] = False; Mv[:, :3] = True
    F, _, E = live_from(kF, kE); r_vis = score_phrase(F, np.concatenate([Mv, Mv[-extra:]]), E, kF, kM, kE, pb)
    checks.append(('mostly invisible joints: unscored', not r_vis['scored'], r_vis['reason']))
    # the engine end to end on feature input, with a lag of 0.3 s compensated by rtt_s
    ev = []; eng = PhraseEngine(key, tempo=1.0, rtt_s=0.3, on_event=ev.append, skip_show=True); t0 = 100.0; eng.start(t0)
    beat_s = eng.beat_s; dt = 1 / 15.0; t = t0
    while eng.state not in ('done',):
        tp = eng.attempt_t(t)
        idx = int(round(tp / beat_s * pb))
        if 0 <= idx < n: eng.on_features(t, kF[idx], None, kE[idx])
        eng.tick(t); t += dt
        if t - t0 > 60: break
    r_eng = eng.result; checks.append(('engine replay of the key with lag compensation scores >= 90', r_eng is not None and r_eng['score'] >= 90, r_eng['score'] if r_eng else None))
    checks.append(('engine emitted phrase_scored and cue_decided', any(e['type'] == 'phrase_scored' for e in ev) and any(e['type'] == 'cue_decided' for e in ev), [e['type'] for e in ev][-4:]))
    # renderer on a blank frame with a fake person
    lm = np.zeros((33, 5), np.float32); lm[:, 3] = 1.0
    lm[11] = [0.4, 0.3, 0, 1, 1]; lm[12] = [0.6, 0.3, 0, 1, 1]; lm[23] = [0.42, 0.55, 0, 1, 1]; lm[24] = [0.58, 0.55, 0, 1, 1]
    lm[13] = [0.35, 0.42, 0, 1, 1]; lm[14] = [0.65, 0.42, 0, 1, 1]; lm[15] = [0.33, 0.52, 0, 1, 1]; lm[16] = [0.67, 0.52, 0, 1, 1]
    lm[25] = [0.43, 0.75, 0, 1, 1]; lm[26] = [0.57, 0.75, 0, 1, 1]; lm[27] = [0.43, 0.95, 0, 1, 1]; lm[28] = [0.57, 0.95, 0, 1, 1]; lm[0] = [0.5, 0.2, 0, 1, 1]
    frame = np.full((640, 360, 3), 40, np.uint8)
    img = compose(frame, lm, eng.hud(t), ghost=kS[0]); checks.append(('renderer returns 720x1280', img.shape == (H, W, 3), img.shape))
    card = draw_card(img, r_eng, key['name']); checks.append(('card renders', card.shape == (H, W, 3), card.shape))
    # the game HUD: every state in portrait and landscape, the scorecard QR decodes back to the FaceTime link, compose stays fast
    from . import render as R
    from .fx import FxQueue
    q = FxQueue(); q.add('judgment', t - 0.1, verdict='green'); q.add('tip', t - 0.2, short='LEFT ARM HIGHER'); q.add('confetti', t - 0.5, seed=1)
    q.add('levelup', t - 0.3, level_name='Groover', unlocks=['intermediate']); q.add('achievement', t - 0.1, name='First Steps')
    states_ok = True
    for fr in (np.full((640, 360, 3), 40, np.uint8), np.full((360, 640, 3), 40, np.uint8)):
        size = R.canvas_size(fr.shape[1], fr.shape[0])
        for st in ('show', 'countin', 'attempt', 'feedback'):
            h = eng.hud(t); h.update(state=st, t=t, t_state=t - 1.5, fx=q, countin_label=6, beat_pos=0.3, combo=5, running_score=80, step=dict(label='STEP 1/3'),
                                     battle=dict(ghost_hp=120, ghost_max=300, round=2, rounds=3), gesture=dict(kind='pause', p=0.5))
            out = R.compose(fr, lm, h, ghost=kS[0], size=size); states_ok &= out.shape[:2] == (size[1], size[0])
        out = R.compose_idle(fr, lm, 'Brodie', lines=['Say your name'], size=size, recap=dict(result=r_eng, title='Human Metronome', cue='Arms up', xp=20),
                             caption='Nice one.', fxq=q, t=t); states_ok &= out.shape[:2] == (size[1], size[0])
        out = R.compose_idle(fr, lm, 'Brodie', size=size, safety=True); states_ok &= out.shape[:2] == (size[1], size[0])
    checks.append(('every HUD state renders, portrait and landscape', bool(states_ok), states_ok))
    sc = R.draw_card(R.to_canvas(frame, (W, H))[0], r_eng, key['name'], code='4821', handle='brodie@example.com', xp=dict(gained=90, level_name='Groover', xp_after=120, level_floor=100, next_at=250),
                     achievements=[dict(name='First Steps')], mode_label='Learn')
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(sc)
    checks.append(('scorecard QR decodes to the FaceTime link', data == 'facetime:brodie@example.com', data))
    import time as _t
    h = eng.hud(t); h.update(state='attempt', t=t, fx=q, combo=5, running_score=80)
    for _ in range(3): R.compose(frame, lm, h, ghost=kS[0])
    ts = []
    for _ in range(10):
        t1 = _t.perf_counter(); R.compose(frame, lm, h, ghost=kS[0]); ts.append((_t.perf_counter() - t1) * 1000)
    checks.append(('compose median under 25 ms', float(np.median(ts)) < 25.0, round(float(np.median(ts)), 1)))
    ok = all(c[1] for c in checks)
    if verbose:
        for name, passed, val in checks: print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {val}")
        print('selftest', 'PASSED' if ok else 'FAILED')
    return ok, checks


def cmd_key(a):
    L = None
    if a.landmarks and os.path.exists(a.landmarks):
        z = np.load(a.landmarks, allow_pickle=True); L = dict(times=z['times'], img=list(z['img']), world=list(z['world']), fps=float(z['fps']), size=tuple(z['size']), n=int(z['n']))
        L['img'] = [None if (x is None or (isinstance(x, np.ndarray) and x.size == 0)) else x for x in L['img']]; L['world'] = [None if (x is None or (isinstance(x, np.ndarray) and x.size == 0)) else x for x in L['world']]
    t = time.time()
    key = build_key(a.video, a.name, bpm=a.bpm, start=a.start, end=a.end, counts_per_phrase=a.counts, mirror=not a.no_mirror, variant=a.pose,
                    source=a.source, licence=a.licence, style=a.style, difficulty=a.difficulty, landmarks=L, progress=lambda i: print(f'  frames {i}', flush=True))
    print(f"key {key['name']}: {key['n_beats']} beats at {key['bpm']:.1f} bpm, {len(key['phrases'])} phrases, pose in {key['pose_frames']}/{key['frames']} frames, "
          f"{time.time() - t:.1f} s -> {os.path.join(LIBRARY_DIR, key['name'] + '.json')}")


def cmd_landmarks(a):
    L = extract_landmarks(a.video, variant=a.pose, progress=lambda i: print(f'  frames {i}', flush=True))
    img = np.array([x if x is not None else np.zeros((0, 5), np.float32) for x in L['img']], dtype=object)
    world = np.array([x if x is not None else np.zeros((0, 3), np.float32) for x in L['world']], dtype=object)
    np.savez(a.out, times=L['times'], img=img, world=world, fps=L['fps'], size=np.array(L['size']), n=L['n'])
    print(f"saved {L['n']} frames ({sum(1 for x in L['img'] if x is not None)} with pose) to {a.out}")


def cmd_run(a):
    key = load_key(a.key)
    from rtcoach.pose import Pose
    from . import MODELS_DIR
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise SystemExit(f'cannot open {a.video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    pose = Pose(a.pose, video=True); ev = []
    eng = PhraseEngine(key, phrase_index=a.phrase, tempo=a.tempo, rtt_s=a.rtt, on_event=ev.append, skip_show=True, mirror_live=a.mirror_live)
    writer = None
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
        from .render import canvas_size
        size = canvas_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 720), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1280))
        writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'), min(fps, 30.0), size)
    i = 0; last_ms = -1; started = False; t_end = None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok: break
            t = i / fps; i += 1
            if t < a.offset - 0.5: continue
            if not started and t >= a.offset: eng.start(t); started = True
            s = min(bgr.shape[:2]); f = 480 / s if s > 480 else 1.0
            small = cv2.resize(bgr, (round(bgr.shape[1] * f), round(bgr.shape[0] * f))) if f < 1 else bgr
            ms = int(round(t * 1000)); ms = ms if ms > last_ms else last_ms + 1; last_ms = ms
            lm, wld = pose(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), ms)
            if started:
                eng.on_frame(t, lm, wld); eng.tick(t)
            if writer is not None:
                hud = eng.hud(t) if started else dict(state='idle', key=key['name'], phrase=a.phrase, n_beats=eng.n_beats, current_beat=-1, countdown=None, verdicts=[None] * eng.n_beats, running_score=None, combo=0, result=None, cue=None, bpm=eng.bpm, tempo=a.tempo)
                if eng.state == 'done':
                    from .render import to_canvas
                    img = draw_card(to_canvas(bgr, size)[0], eng.result, key['name'], extra_lines=[eng.cue['text']] if eng.cue else ())
                    if t_end is None: t_end = t
                else:
                    img = compose(bgr, lm, hud, ghost=eng.ghost_sample(t) if started else None, show_live=True, size=size)
                writer.write(img)
            if eng.state == 'done' and t_end is not None and t - t_end > 2.0: break
    finally:
        cap.release(); pose.close()
        if writer is not None: writer.release()
    r = eng.result
    if r is None:
        print('the attempt never finished inside the video (too short for the phrase at this tempo, or --offset too late)'); return 1
    print(f"score {r['score']}  shape {r['spatial']}  timing {r['timing']}  energy {r['energy']}  visible {r['visible_fraction']}  frames {r['frames']}")
    print(f"per group: {r['per_group']}   weak beats: {[b + 1 for b in r['weak_beats']]}   cue: {eng.cue['text']}")
    for row in r['per_beat']: print(f"  beat {row['beat'] + 1}: {row['verdict']:5} err {row['error_deg']:5.1f} deg  offset {row['offset_beats']:+.2f} beats  worst {row['worst_group']}")
    if a.report:
        with open(a.report, 'w', encoding='utf-8') as f: json.dump(dict(result=r, cue=eng.cue, events=ev, video=os.path.abspath(a.video), key=key['name']), f, indent=1)
        print(f'report -> {a.report}')
    if a.out: print(f'annotated video -> {a.out}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('selftest'); s.set_defaults(fn=lambda a: 0 if selftest()[0] else 1)
    s = sub.add_parser('key'); s.add_argument('--video', required=True); s.add_argument('--name', required=True); s.add_argument('--bpm', type=float)
    s.add_argument('--start', type=float, default=0.0); s.add_argument('--end', type=float); s.add_argument('--counts', type=int, default=8)
    s.add_argument('--no-mirror', action='store_true'); s.add_argument('--pose', default='full'); s.add_argument('--source', default=''); s.add_argument('--licence', default='')
    s.add_argument('--style', default=''); s.add_argument('--difficulty', type=int, default=2); s.add_argument('--landmarks', help='cached .npz from the landmarks command')
    s.set_defaults(fn=lambda a: cmd_key(a) or 0)
    s = sub.add_parser('landmarks'); s.add_argument('--video', required=True); s.add_argument('--out', required=True); s.add_argument('--pose', default='full'); s.set_defaults(fn=lambda a: cmd_landmarks(a) or 0)
    s = sub.add_parser('run'); s.add_argument('--video', required=True); s.add_argument('--key', required=True); s.add_argument('--phrase', type=int, default=0)
    s.add_argument('--tempo', type=float, default=1.0); s.add_argument('--offset', type=float, default=0.0, help='seconds into the video where the attempt starts on beat 1')
    s.add_argument('--rtt', type=float, default=0.0); s.add_argument('--out'); s.add_argument('--report'); s.add_argument('--pose', default='full')
    s.add_argument('--mirror-live', action='store_true', help='mirror the attempt landmarks (when the attempt video was filmed like the reference, not like a caller)')
    s.add_argument('--show-live', action='store_true', help='also draw the live skeleton'); s.set_defaults(fn=cmd_run)
    s = sub.add_parser('list'); s.set_defaults(fn=lambda a: print(json.dumps(list_keys(), indent=1)) or 0)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
