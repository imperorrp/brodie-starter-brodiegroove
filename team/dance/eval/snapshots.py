"""Render every HUD state to PNGs (data/tmp/hud/) for a visual check, and time compose().

  python -m team.dance.eval.snapshots [--video data/tmp/mac_handoff/segments/run3_jacks.mp4] [--landscape]
"""
import argparse, os, sys, time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from team.dance import render, choreo, fx as fxm, progression        # noqa: E402
from team.dance.engine import PhraseEngine                           # noqa: E402


def frame_and_pose(video, landscape=False, at=0.4):
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * at)); ok, bgr = cap.read(); cap.release()
    if not ok: bgr = np.full((1280, 720, 3), 40, np.uint8)
    if landscape:
        h, w = bgr.shape[:2]; canvas = np.full((h, int(h * 16 / 9), 3), 30, np.uint8); x0 = (canvas.shape[1] - w) // 2; canvas[:, x0:x0 + w] = bgr; bgr = canvas
    from rtcoach.pose import Pose
    lm, w = Pose('full', video=False)(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0)
    return bgr, lm


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--video', default=os.path.join(ROOT, 'data', 'tmp', 'mac_handoff', 'segments', 'run3_jacks.mp4'))
    ap.add_argument('--landscape', action='store_true'); ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'tmp', 'hud'))
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True); suf = '_land' if a.landscape else ''
    bgr, lm = frame_and_pose(a.video, a.landscape)
    sh, sw = bgr.shape[:2]; size = render.canvas_size(sw, sh)
    key = choreo.generate('hiphop', 'beginner', seed=11)
    eng = PhraseEngine(key, phrase_index=0, tempo=0.8); eng.step = dict(label='STEP 1/3')
    t0 = 100.0; eng.start(t0); q = fxm.FxQueue()
    out = {}

    def hud_at(t, **extra):
        h = eng.hud(t); h.update(fx=q, difficulty='beginner', step=dict(label='STEP 1/3'), speaking=False); h.update(extra); return h

    from team.dance.setup import SetupTracker
    out['01_explore_hints'] = render.compose_idle(bgr, lm, 'Brodie', size=size, hints=True, state='EXPLORE', avatar_state='listening', t=t0)
    out['02_explore_caption'] = render.compose_idle(bgr, lm, 'Brodie', size=size, caption="Hey, I'm Brodie! What's your name, and what kind of dancing are you into?", state='EXPLORE', avatar_state='speaking', t=t0 + 0.2)
    out['02b_offer'] = render.compose_idle(bgr, lm, 'Brodie', size=size, caption="Want to start a beginner hip-hop routine now? We'll do a quick camera check first.", state='OFFER', avatar_state='speaking', t=t0)
    out['02c_finish_pose'] = render.compose_idle(bgr, lm, 'Brodie', size=size, state='REVIEW', countdown=2, t=t0)
    lm_cut = None if lm is None else lm.copy()
    if lm_cut is not None: lm_cut[[27, 28, 29, 30, 31, 32], 1] = 1.08
    tr = SetupTracker(); res = None
    for i in range(16): res = tr.update(i / 15, lm_cut, 110)
    out['03_setup_feet'] = render.compose_idle(bgr, lm, 'Brodie', size=size, setup=res, state='PREPARE', t=t0)
    tr = SetupTracker()
    for i in range(12): res = tr.update(i / 15, lm, 110)
    out['04_setup_hold'] = render.compose_idle(bgr, lm, 'Brodie', size=size, setup=res, state='PREPARE', t=t0)
    out['05_confirm'] = render.compose_idle(bgr, lm, 'Brodie', size=size, setup=res, state='CONFIRM', t=t0)
    ghost = eng.ghost_sample(t0 + 1.0)
    out['06_watch'] = render.compose(bgr, lm, hud_at(t0 + 1.0, stage='WATCH', demo=True, tempo=0.6), ghost=ghost, size=size)
    tc = t0 + eng.show_s + 1.5 * eng.beat_s; eng.tick(tc)
    out['07_countin'] = render.compose(bgr, lm, hud_at(tc, stage='WITH ME', tempo=0.6), ghost=eng.ghost_sample(tc), size=size)
    ta = t0 + eng.show_s + eng.countin_s + 5.3 * eng.beat_s; eng.tick(ta)
    eng.live_verdicts = ['green', 'green', 'amber', 'green', 'green', None, None, None]
    h = hud_at(ta, stage='WITH ME', tempo=0.6); h['current_beat'] = 5; h['verdicts'] = eng.live_verdicts
    out['08_with_me'] = render.compose(bgr, lm, h, ghost=eng.ghost_sample(ta), size=size)
    h = hud_at(ta, stage='YOUR TURN', tempo=0.75, ghost_overlay=False); h['current_beat'] = 5; h['verdicts'] = eng.live_verdicts
    out['09_your_turn'] = render.compose(bgr, lm, h, ghost=eng.ghost_sample(ta), size=size)
    q.add('judgment', ta - 0.12, verdict='green'); q.add('tip', ta - 0.5, short='LEFT ARM HIGHER', text='Lift your left arm higher')
    h = hud_at(ta, stage='CHALLENGE', tempo=1.0, ghost_overlay=False, step=dict(label='CHALLENGE')); h['combo'] = 5; h['running_score'] = 88; h['current_beat'] = 5; h['verdicts'] = eng.live_verdicts
    out['10_challenge_fire'] = render.compose(bgr, lm, h, ghost=eng.ghost_sample(ta), size=size)
    h2 = dict(h); h2['gesture'] = dict(kind='pause', p=0.6)
    out['11_gesture_hold'] = render.compose(bgr, lm, h2, ghost=eng.ghost_sample(ta), size=size)
    h3 = dict(h); h3['paused'] = True
    out['12_paused'] = render.compose(bgr, lm, h3, ghost=None, size=size)
    q.clear()
    res_ = dict(scored=True, score=87.0, spatial=0.84, timing=0.92, energy=0.7, max_combo=6, per_group=dict(arms=12, legs=20, torso=5), partial=False)
    cue = dict(text='Lift your left arm higher on counts 5 to 7', short='LEFT ARM HIGHER', kind='correction')
    base = dict(state='feedback', key=key['name'], title=key['title'], phrase=0, n_beats=8, current_beat=-1, countdown=None, verdicts=['green'] * 8,
                running_score=None, combo=0, result=res_, cue=cue, bpm=88, tempo=0.75, t_state=t0, fx=q, difficulty='beginner', step=dict(label='STEP 1/3'))
    out['13_reveal_your_turn'] = render.compose(bgr, lm, dict(base, stage='YOUR TURN', t=t0 + 2.0), size=size)
    out['14_reveal_practice'] = render.compose(bgr, lm, dict(base, stage='WITH ME', practice=True, t=t0 + 0.5), size=size)
    q.add('confetti', t0 + 1.0, seed=4)
    out['15_reveal_challenge'] = render.compose(bgr, lm, dict(base, stage='CHALLENGE', tempo=1.0, t=t0 + 2.0), size=size)
    q.clear()
    out['16_recap'] = render.compose_idle(bgr, lm, key['title'], size=size, recap=dict(result=res_, title='Human Metronome', cue='Left arm higher on 5 to 7'),
                                          caption="Ninety on timing! Lift that left arm on five to seven.", state='REVIEW', t=t0)
    q.add('levelup', t0, level_name='Groover', unlocks=['intermediate']); q.add('achievement', t0 + 0.2, name='Eight Clean')
    out['17_levelup'] = render.compose_idle(bgr, lm, key['title'], size=size, fxq=q, t=t0 + 0.8, state='EXPLORE')
    q.clear()
    out['18_safety'] = render.compose_idle(bgr, lm, 'Brodie', size=size, safety=True)
    xp = dict(gained=96, level_name='Groover', xp_after=136, level_floor=100, next_at=250, streak=2)
    out['20_photo_card'] = render.draw_card(render.to_canvas(bgr, size)[0], res_, key['name'], code='4821', title=key['title'], award_title='Human Metronome', xp=xp,
                                            handle='brodie@example.com', mode_label='Learn', photo=render.to_canvas(bgr, size)[0], brag=['NEW PERSONAL BEST', 'TOP 12% TODAY'])
    out['19_scorecard'] = render.draw_card(render.to_canvas(bgr, size)[0], res_, key['name'], code='4821', title=key['title'], award_title='Human Metronome', xp=xp,
                                           achievements=[dict(name='First Steps'), dict(name='Eight Clean')], handle='brodie@example.com', mode_label='Learn', perfects=11)
    for k, img in out.items(): cv2.imwrite(os.path.join(a.out, f'{k}{suf}.png'), img)
    ts = []
    for i in range(30):
        q.clear(); q.add('judgment', ta - 0.1, verdict='green'); q.add('tip', ta - 0.3, short='LEFT ARM HIGHER')
        hh = hud_at(ta, stage='CHALLENGE'); t = time.perf_counter(); render.compose(bgr, lm, hh, ghost=eng.ghost_sample(ta), size=size); ts.append((time.perf_counter() - t) * 1000)
    print(f'{len(out)} snapshots -> {a.out}; compose median {np.median(ts):.1f} ms (attempt with fx), size {size}')


if __name__ == '__main__':
    main()
