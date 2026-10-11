"""Pure modules added for the game layer: progression, choreography, feedback, gestures, telemetry, multi-person tracking, text, fx."""
import os, sys, math
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from team.dance import progression as pg
from team.dance import choreo, feedback, gestures, telemetry, multipose, fx
from team.dance.scoring import score_phrase
from team.dance.reference import phrase_slice
from team.dance.features import CHANNELS


# ---- progression ----
def test_grades_and_levels():
    assert [pg.grade(s) for s in (99, 92, 91.9, 80, 65, 50, 49)] == ['S', 'S', 'A', 'A', 'B', 'C', 'D']
    assert pg.level_for(0)[:2] == (0, 'Rookie') and pg.level_for(100)[:2] == (1, 'Groover') and pg.level_for(99)[3] == 100
    assert pg.level_for(10 ** 6)[3] is None


def test_unlocks(monkeypatch):
    monkeypatch.delenv('DANCE_UNLOCK_ALL', raising=False)
    assert 'advanced' not in pg.unlocked(0) and 'intermediate' in pg.unlocked(1) and 'advanced' in pg.unlocked(2)
    assert pg.xp_to_unlock('advanced', 200) == 50
    monkeypatch.setenv('DANCE_UNLOCK_ALL', '1')
    assert 'advanced' in pg.unlocked(0) and pg.xp_to_unlock('advanced', 0) == 0


def test_titles_priority():
    assert pg.title_for(dict(scored=True, score=96, timing=1, spatial=1, energy=1, max_combo=8)) == 'Main Character'
    assert pg.title_for(dict(scored=True, score=75, timing=0.95, spatial=0.5, energy=0.2, max_combo=2)) == 'Human Metronome'
    assert pg.title_for(dict(scored=True, score=70, timing=0.5, spatial=0.5, energy=0.2, max_combo=9)) == 'Combo Menace'
    assert pg.title_for(dict(scored=True, score=30, timing=0.1, spatial=0.2, energy=0.1, max_combo=0)) == 'Certified Off-Beat (for now)'
    assert pg.title_for(dict(scored=False)) == 'Camera Shy (step back)'


def test_first_session_levels_up():
    rounds = [dict(scored=True, score=s, max_combo=3) for s in (55, 68, 74, 81)]
    xp, br = pg.xp_for_session(rounds, 'learn', 'beginner', dict(first_today=True))
    assert xp >= 100 and pg.level_for(xp)[0] >= 1, (xp, br)


def test_achievements_once():
    sess = dict(rounds=[dict(scored=True, score=93, max_combo=8)], styles={'hiphop'}, streak=1)
    got = pg.new_achievements([], sess); ids = [a['id'] for a in got]
    assert 'first_dance' in ids and 's_rank' in ids and 'perfect_8' in ids
    assert pg.new_achievements(ids, sess) == []


# ---- choreography ----
def test_choreo_deterministic_and_named():
    a = choreo.generate('hiphop', 'beginner', seed=42); b = choreo.generate('hiphop', 'beginner', seed=42)
    assert a['name'] == 'gen_hiphop_beginner_42' and a['features'] == b['features'] and len(a['phrases']) == 3
    c = choreo.from_name(a['name']); assert c['features'] == a['features'] and c['title'] == a['title']
    assert len({(m['src'], m['phrase']) for m in a['moves']}) == len(a['moves'])


def test_choreo_beginner_length_and_selfscore():
    k = choreo.generate('pop', 'beginner', seed=7)
    secs = choreo.routine_seconds(k, choreo.LEVEL_TEMPO['beginner'])
    assert 15.0 <= secs <= 20.0, secs
    F, M, E, _ = phrase_slice(k, 1); pb = k['per_beat']
    Fl = np.concatenate([F, np.repeat(F[-1:], pb, 0)]); Ml = np.concatenate([M, np.repeat(M[-1:], pb, 0)]); El = np.concatenate([E, np.repeat(E[-1:], pb)])
    assert score_phrase(Fl, Ml, El, F, M, E, pb)['score'] >= 95


def test_choreo_combined_and_pools():
    k = choreo.generate('house', 'intermediate', seed=3)
    k2, idx = choreo.with_combined(k, 1)
    F, _, _, _ = phrase_slice(k2, idx); assert len(F) == 16 * k['per_beat'] and len(k['phrases']) == 3
    assert choreo.pool('fitness') and choreo.daily('20261011')['name'] == 'gen_daily_20261011'
    assert choreo.from_name('gen_daily_20261011')['features'] == choreo.daily('20261011')['features']
    assert choreo.generate('krump', 'advanced', seed=1)['difficulty'] == 3


# ---- feedback ----
def _result_with(ch, signed, weak=(4, 5, 6)):
    cs = [[None] * len(CHANNELS) for _ in range(8)]
    for b in weak:
        row = [0.0] * len(CHANNELS); row[ch] = signed; cs[b] = row
    return dict(scored=True, timing=0.9, mean_offset_beats=0.0, weak_beats=list(weak), score=60, per_group=dict(arms=30.0, legs=8.0, torso=2.0), channel_signed=cs)


def test_directional_cues():
    c = feedback.structured_cue(_result_with(CHANNELS.index('arm_elev_l'), -35.0))
    assert c['kind'] == 'correction' and 'left' in c['text'] and 'higher' in c['text'] and '5 to 7' in c['text'] and c['group'] == 'arms'
    c = feedback.structured_cue(_result_with(CHANNELS.index('knee_r'), 30.0))
    assert 'Bend your right knee' in c['text'] and c['short'] == 'BEND RIGHT KNEE'
    late = dict(scored=True, timing=0.3, mean_offset_beats=0.5, weak_beats=[2, 3, 4], score=50, per_group={})
    assert feedback.structured_cue(late)['error_type'] == 'late'
    assert feedback.structured_cue(dict(scored=False))['kind'] == 'visibility'


def test_live_tip_needs_two_reds():
    row = np.zeros(len(CHANNELS)); row[CHANNELS.index('elbow_r')] = -40
    assert feedback.live_tip([row]) is None
    t = feedback.live_tip([row, row]); assert t and 'right' in t['text'].lower() and 'straighten' in t['text'].lower()
    small = np.zeros(len(CHANNELS)); small[0] = 8
    assert feedback.live_tip([small, small]) is None


def test_describe_phrase():
    k = choreo.generate('hiphop', 'beginner', seed=5)
    d = feedback.describe_phrase(k, 0); assert d.startswith('counts 1 to 2') and d.count(';') == 3


# ---- gestures ----
def _person(lw, rw, nose_y=0.3):
    lm = np.zeros((33, 5), np.float32); lm[:, 3] = 1.0
    lm[0, :2] = (0.5, nose_y); lm[11, :2] = (0.58, 0.4); lm[12, :2] = (0.42, 0.4); lm[23, :2] = (0.56, 0.65); lm[24, :2] = (0.44, 0.65)
    lm[15, :2] = lw; lm[16, :2] = rw
    return lm


def test_gesture_hold_and_cooldown():
    g = gestures.GestureDetector(); up = _person((0.75, 0.1), (0.25, 0.1)); fired = []
    for i in range(30):
        r = g.update(i / 15, up)
        if r: fired.append((i, r))
    assert fired and fired[0][1] == 'ready' and fired[0][0] >= 14 and len(fired) == 1
    g = gestures.GestureDetector(); cross = _person((0.47, 0.15), (0.53, 0.15)); out = [g.update(i / 15, cross) for i in range(25)]
    assert 'pause' in out
    g = gestures.GestureDetector(); out = []
    for i in range(30):                                   # crossed but swinging: no pause
        dx = 0.08 * math.sin(i)
        out.append(g.update(i / 15, _person((0.47 + dx, 0.15), (0.53 + dx, 0.15))))
    assert 'pause' not in out
    g = gestures.GestureDetector(); out = [g.update(i / 15, up) for i in range(10)]
    assert not any(out)                                   # short hold


# ---- telemetry ----
def test_telemetry_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, 'SESS_DIR', str(tmp_path))
    clock = dict(t=0.0); tm = telemetry.Telemetry('test', now=lambda: clock['t'])
    for i in range(10): tm.frame(i < 9, full_body=i < 8)
    tm.cue(1.0, 1.4, 'tip'); tm.cue(2.0, 2.2, 'tip'); tm.cue(3.0, 4.0, 'spoken')
    tm.tool_result('control_session', 3.0); tm.tool_result('compare_with', 5.0, ok=False, error='x')
    tm.score(dict(scored=True, score=70)); tm.caller = 'Maya'; clock['t'] = 12.0
    s = tm.close(completed=True)
    assert s['tracking_rate'] == 0.9 and s['median_cue_latency_ms'] == 400.0 and s['tool_errors'] == 1 and s['completed']
    assert s['caller'] != 'Maya' and os.path.exists(os.path.join(tmp_path, 'test.jsonl'))


# ---- multi-person tracking ----
def _body(cx, scale=1.0):
    lm = np.zeros((33, 5), np.float32); lm[:, 3] = 1.0
    for j, (dx, dy) in {11: (0.06, 0.35), 12: (-0.06, 0.35), 23: (0.05, 0.6), 24: (-0.05, 0.6), 0: (0, 0.25), 15: (0.08, 0.5), 16: (-0.08, 0.5)}.items():
        lm[j, :2] = (cx + dx * scale, 0.1 + dy * scale)
    return lm


def test_tracker_stable_ids_and_background():
    tr = multipose.PlayerTracker(); a, b, tiny = _body(0.3), _body(0.7), _body(0.5, scale=0.3)
    for i in range(40):
        poses = [(a, None), (b, None), (tiny, None)] if i % 2 else [(b, None), (tiny, None), (a, None)]
        tr.update(i / 15, poses)
    st = tr.stable_players(); assert len(st) == 2
    ids = {round(float(p.centre[0]), 1): p.pid for p in st}
    for i in range(40, 60): tr.update(i / 15, [(b, None), (a, None)])
    assert {round(float(p.centre[0]), 1): p.pid for p in tr.stable_players()} == ids
    assert any(e['type'] == 'extra_person_ignored' for e in tr.pop_events())
    cols = dict(tr.assign_colours(mirrored=True)); assert set(cols.values()) == {'volt', 'pink'}


def test_tracker_solo_lock_and_drop():
    tr = multipose.PlayerTracker(); me = _body(0.5)
    for i in range(30): tr.update(i / 15, [(me, None)])
    pid = tr.primary_id; assert pid is not None
    other = _body(0.85)
    for i in range(30, 60): tr.update(i / 15, [(other, None), (me, None)])
    assert tr.primary_id == pid and tr.solo_pose()[0] is me
    for i in range(60, 110): tr.update(i / 15, [(other, None)])
    assert pid not in tr.players


# ---- text and fx ----
def test_text_and_fx_render():
    from team.dance import text
    img = np.zeros((1280, 720, 3), np.uint8)
    w1 = text.text_size('PERFECT', 40, 'display')[0]; w2 = text.text_size('PERFECT', 80, 'display')[0]
    assert w2 > w1 > 0
    x = text.put_text(img, 'Hello', (20, 100), 40, (61, 255, 217), 'display', stroke=3); assert x > 20 and img.sum() > 0
    text.put_text(img, 'Centered', (360, 640), 30, anchor='mm'); text.put_text(img, 'edge', (700, 5), 60)   # clipped, no crash
    q = fx.FxQueue(); q.add('judgment', 1.0, verdict='green'); q.add('confetti', 1.0, seed=3)
    assert len(q.active(1.2)) == 2 and len(q.active(10.0)) == 0
    assert fx.confetti_particles(3, 50, (720, 1280), 0.5)
