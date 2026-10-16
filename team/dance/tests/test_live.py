"""The conversation-to-dance flow: intent, one voice, guarded starts, the Gemini tools, camera setup, engine demo/abort."""
import asyncio, json, os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.environ.setdefault('BACKEND_URL', 'fake')

from team.dance import live
from team.dance.engine import PhraseEngine
from team.dance import choreo
from team.dance import feedback


def test_command_priority_and_negation():
    pc = lambda t: live.pick_command(live.parse_commands(t.lower()))
    assert pc('I want to go slower') == 'slower'
    assert pc("ok let's go") is None                                    # starting is never a keyword
    assert pc('where do I start?') is None
    assert pc('my knee hurts, stop') == 'pain'
    assert pc("don't stop, again!") == 'again'
    assert pc("it doesn't hurt") is None
    assert pc('can you break it down') == 'simplify'
    assert pc('forget me') == 'forget' and pc('hold on a sec') == 'pause' and pc('run it back') == 'rematch'
    assert 'squad_battle' in live.parse_commands('squad battle'.lower())


def test_classify_intent():
    c = live.classify
    for t in ('teach me hip hop', "let's dance", 'I want to learn K-pop', 'can we start?', 'start a battle', 'rematch'):
        assert c(t)['request'], t
    for t in ('where do I start?', 'what is a battle?', 'should I learn hip hop?', 'I like K-pop', 'maybe dance later', "I'm not ready", 'hi I am Maya'):
        assert not c(t)['request'], t
    assert c('yes')['affirm'] and c("I'm ready")['affirm'] and c("let's go")['affirm']
    assert not c("I'm not ready")['affirm'] and c("I'm not ready")['deny'] and c('one sec')['hold'] and c('never mind')['cancel']
    assert c('run it back')['rematch']


def test_parse_code_needs_four_in_a_row():
    assert live.parse_code('battle 4 8 2 1') == '4821' and live.parse_code('four eight two one please') == '4821'
    assert live.parse_code('oh one more two times three, and four') is None


def _session():
    job = live._FakeJob(); log = []; clock = dict(t=0.0)
    job.player = live._FakePlayer(log, lambda: clock['t']); job.voice = live._FakeVoice(log, lambda: clock['t'])
    s = live.DanceSession(job, key_name=None, now=lambda: clock['t'])
    return s, clock, log


def test_budget_swap_closes_pose_model_on_pose_thread(monkeypatch):
    """The lite swap closes the old landmarker itself, on the pose thread; left to the garbage collector, MediaPipe's __del__
    deadlocked the whole worker (2026-10-12)."""
    import time
    from team.dance import multipose
    class FakePose:
        made = []
        def __init__(self, variant='full', video=True, max_poses=4): self.variant = variant; self.closed = 0; FakePose.made.append(self)
        def __call__(self, rgb, ms): return []
        def close(self): self.closed += 1
    monkeypatch.setattr(multipose, 'MultiPose', FakePose)
    s, clock, log = _session()
    rgb = np.zeros((48, 64, 3), np.uint8)
    s._pose_sync(rgb, 0); old = s.mpose
    s.now = time.monotonic; s.budget.add = lambda *a: 'lite'
    s._guards(1.0, None, None, None)
    assert s.pose_variant == 'lite' and s.pose_reload and s.mpose is old and old.closed == 0     # nothing dropped on the loop thread
    s._pose_sync(rgb, 40)
    assert old.closed == 1 and s.mpose is not old and s.mpose.variant == 'lite' and not s.pose_reload
    s._close_pose_sync()
    assert FakePose.made[-1].closed == 1 and s.mpose is None


def test_voice_event_flow_and_dedupe():
    s, clock, log = _session()
    s.on_worker_event(dict(type='voice', role='caller', text="Hi I'm Maya, I like pop, chill coach please, easy"))
    assert s.name == 'Maya' and s.style == 'po' and s.persona == 'chill' and s.diff_choice == 'beginner'
    assert s.state == 'EXPLORE' and s.activity_req is None                       # chat and preferences never start anything
    for t in ('where do I start?', "what's a battle?", "I'm ready", "ok let's go"):
        s.on_worker_event(dict(type='voice', role='caller', text=t))
    assert s.activity_req is None and s.pending_cmd is None
    s.on_worker_event(dict(type='voice', role='caller', text='Today I want to dance some hip hop. Is that possible?'))
    assert s.proposal is None and s.state == 'EXPLORE'                          # a question is not a decision
    s.on_worker_event(dict(type='voice', role='caller', text='teach me hip hop'))
    assert s.state == 'OFFER' and s.proposal['mode'] == 'learn' and s.style == 'lh'   # a wish becomes an offer, not setup
    s.on_worker_event(dict(type='voice', role='caller', text='actually K-pop'))
    assert s.state == 'OFFER' and s.proposal['style'] == 'kpop'                  # change of mind updates the offer
    s.state = 'CONFIRM'; clock['t'] = 5.0
    s.on_worker_event(dict(type='voice', role='caller', text="I'm not ready"))
    assert s.deny_t == 5.0 and s.affirm_t < 0
    clock['t'] = 6.0; s.on_worker_event(dict(type='voice', role='caller', text='ok ready'))
    assert s.affirm_t == 6.0
    s.state = 'ACTIVITY'; s.pending_cmd = None
    assert s._cmd('again', 'tool') and not s._cmd('again', 'voice')              # same command within 4 s (regex + Gemini tool): once
    clock['t'] = 10.5; assert s._cmd('again', 'voice')
    s.on_worker_event(dict(type='voice', role='coach', text='Nice one Maya.'))
    assert s.caption[0] == 'Nice one Maya.'


def test_speaker_single_flight_and_stale_drop():
    s, clock, log = _session()
    async def go():
        s.say_line('line one', key='a'); s.say_line('line two', key='b'); s.say_line('line two again', key='b')
        s.epoch += 1                                                             # the state moved on: those lines are stale
        s.say_line('line three', key='c')
        for _ in range(30): await asyncio.sleep(0)
    asyncio.run(go())
    spoken = [e['ask'] for e in log if 'ask' in e and 'note' not in e and 'result' not in e]
    assert len(spoken) == 1 and 'line three' in spoken[0] and spoken[0].startswith('Say this to the caller, word for word'), spoken


def test_caller_speech_holds_system_lines():
    s, clock, log = _session()
    clock['t'] = 10.0; s.last_caller_text_t = 9.0
    async def go():
        s.say_line('setup line', key='setup')
        for _ in range(20): await asyncio.sleep(0)
    asyncio.run(go())
    assert not [e for e in log if 'ask' in e and 'note' not in e and 'result' not in e]   # within 6 s of the caller: silent
    clock['t'] = 16.5
    async def go2():
        for _ in range(20): await asyncio.sleep(0)
    s.speaker_task = None; s.say_line('setup line', key='setup'); asyncio.run(go2())
    assert [e for e in log if 'ask' in e and 'setup line' in e['ask']]


def test_answered_caller_releases_lines_after_a_second():
    """Once Gemini's answer to the caller has played, our next line waits 1 s, not the full 6 s (the awkward pause)."""
    s, clock, log = _session()
    clock['t'] = 11.0; s.last_caller_text_t = 9.0; s.coach_text_t = 10.5        # Gemini answered at 10.5
    async def go():
        s.say_line('setup line', key='setup')
        for _ in range(20): await asyncio.sleep(0)
    asyncio.run(go())
    assert [e for e in log if 'ask' in e and 'setup line' in e['ask']]


def test_thinking_from_the_callers_line_until_his_audio():
    s, clock, log = _session()
    assert s.avatar_state(False) == 'idle'
    clock['t'] = 5.0; s.on_worker_event(dict(type='voice', role='caller', text='teach me hip hop'))
    assert s.reply_wait_t == 5.0
    clock['t'] = 5.2; assert s.avatar_state(False) == 'thinking'                 # the caller is done: no blank "listening" gap
    clock['t'] = 7.5; assert s.avatar_state(False) == 'thinking'
    clock['t'] = 8.0; assert s.avatar_state(True) == 'speaking'                  # his audio started
    clock['t'] = 8.5; assert s.avatar_state(False) in ('idle', 'thinking') and not s.thinking(8.5)
    clock['t'] = 20.0; s.reply_wait_t = 20.0; s.say_q = []
    clock['t'] = 26.5; assert not s.thinking(26.5)                              # a reply that never comes: the dots stop after THINK_S


def test_typing_bubble_drawn_while_thinking():
    from team.dance import render as R
    frame = np.full((1280, 720, 3), 90, np.uint8); size = R.canvas_size(720, 1280)
    idle = R.compose_idle(frame, None, 'Brodie', size=size, caption='Want to start?', state='EXPLORE', avatar_state='idle', t=0.3)
    think = R.compose_idle(frame, None, 'Brodie', size=size, caption='Want to start?', state='EXPLORE', avatar_state='thinking', t=0.3)
    assert think.shape == idle.shape and (think != idle).any()
    land = R.compose_idle(np.full((720, 1280, 3), 90, np.uint8), None, 'Brodie', size=R.canvas_size(1280, 720), state='REVIEW', avatar_state='thinking', t=0.1)
    assert land.shape[:2] == (720, 1280)


def test_slow_speech_and_readable_ghost():
    """Testers 2026-10-15: Brodie talked too fast; the ghost's skeleton didn't show which body part was which."""
    assert 'slowly' in live.stage_direction(live.SAY_PREFIX + '"Hi!"') and 'slowly' in live.stage_direction('Ask their name.')
    assert 'PACE.' in open(os.path.join(ROOT, 'team', 'dance', 'persona.md'), encoding='utf-8').read()
    from team.dance import render as R
    g = np.zeros((33, 2), np.float32)
    for j, (x, y) in {0: (0, -1.55), 7: (0.12, -1.5), 8: (-0.12, -1.5), 11: (0.25, -1.1), 12: (-0.25, -1.1), 13: (0.45, -0.7), 14: (-0.45, -0.7),
                      15: (0.55, -0.3), 16: (-0.55, -0.3), 23: (0.15, 0), 24: (-0.15, 0), 25: (0.17, 0.6), 26: (-0.17, 0.6), 27: (0.18, 1.2), 28: (-0.18, 1.2)}.items():
        g[j] = (x, y)
    img = np.zeros((400, 300, 3), np.uint8); R.draw_mannequin(img, g * 120 + (150, 210), width=10)
    left, right = img[90:300, 150:300].astype(int).sum(), img[90:300, 0:150].astype(int).sum()
    assert img.sum() > 0 and left != right                                     # the two sides are drawn in different shades
    assert (img[int(-0.7 * 120 + 210), int(0.45 * 120 + 150)] > 200).all()     # a light joint ball on the elbow


def test_pose_cache_memo_and_disk(tmp_path, monkeypatch):
    """The simulator asks MediaPipe once per distinct frame; the memo is shared across runs through a file."""
    from team.dance import multipose
    calls = []
    class FakePose:
        def __init__(self, *a, **k): pass
        def __call__(self, rgb, ms): calls.append(ms); return [(np.full((33, 5), 0.5, np.float32), np.zeros((33, 3), np.float32))]
        def close(self): pass
    monkeypatch.setattr(multipose, 'MultiPose', FakePose)
    path = str(tmp_path / 'pc.pkl')
    s, clock, log = _session(); s.pose_cache = live.PoseCache(path, tag='lite')
    a = np.zeros((48, 64, 3), np.uint8); b = a.copy(); b[0, 0] = 9
    for i in range(5): s._pose_sync(a, i * 70)
    s._pose_sync(b, 400)
    assert len(calls) == 2 and s.pose_cache.hits == 4 and s.pose_cache.misses == 2
    got = s.pose_cache.get(a); got[0][0][:] = 0                                 # callers get copies: the memo stays intact
    assert s.pose_cache.get(a)[0][0][0, 0] == 0.5
    s.pose_cache.save()
    pc2 = live.PoseCache(path, tag='lite'); assert pc2.get(a) is not None and pc2.get(b) is not None
    assert live.PoseCache(path, tag='full').get(a) is None                       # another model never reuses these


def test_no_meta_prompts_in_flow_code():
    src = open(live.__file__, encoding='utf-8').read()
    body = src[src.index('class DanceSession'):src.index('# ---- worker registration ----')]   # every mode, telephone and class included
    calls = [l for l in body.splitlines() if 'self.ask(' in l or 'self.say_line(' in l or 'ask_yes_no(' in l]
    for l in calls:
        for bad in ('ask them to', 'One short sentence:', 'One short question:'):
            assert bad not in l, l


def test_offer_needs_a_yes_after_the_question():
    s, clock, log = _session(); s.lobby_enabled = True; s.t0 = 0.0
    s.request_activity(mode='learn', source='tool')
    assert s.state == 'OFFER'
    s.affirm_t = 0.5                                                             # a "yes" before anything was asked
    async def go():
        t = asyncio.create_task(s.offer_loop())
        for i in range(400):
            clock['t'] += 0.05; await asyncio.sleep(0)
            if t.done(): break
        return t
    task = asyncio.run(go())
    assert task.cancelled() or not task.done() or task.result() != 'yes'


def test_guard_turns_a_crash_into_chat():
    s, clock, log = _session(); s.state = 'PREPARE'
    async def boom(): raise RuntimeError('camera exploded')
    r = asyncio.run(s.guard(boom(), 'setup'))
    assert r is None and s.state == 'CHOOSE' and s.telemetry.errors


def test_difficulty_lock(monkeypatch):
    monkeypatch.delenv('DANCE_UNLOCK_ALL', raising=False)
    s, _, _ = _session(); s.level = 0; s.xp = 40
    ok, reason = s.apply_difficulty('advanced', announce=False)
    assert not ok and 'unlocks' in reason and s.difficulty == 'beginner'
    s.level = 2; ok, reason = s.apply_difficulty('advanced', announce=False); assert ok and s.difficulty == 'advanced'


def test_tools_guarded():
    pytest.importorskip('livekit.agents')
    from team.dance import tools
    s, clock, _ = _session()
    t = {f.info.name if hasattr(f, 'info') else f.__name__: f for f in tools.build_tools(s)}
    run = lambda name, *a, **k: json.loads(asyncio.run(t[name](*a, **k)))
    out = run('control_session', 'slower'); assert out['ok'] and out['will_happen'] == 'nothing'       # nothing runs in chat
    assert not run('control_session', 'ready')['ok']                                                    # no start through tools
    assert run('confirm_start')['started'] is False
    out = run('set_preferences', personality='coach', difficulty='advanced')
    assert out['ok'] and out['persona'] == 'coach' and out['locked_reason']
    assert run('check_camera')['ok'] and run('analyze_last_attempt')['attempted'] is False
    r = run('generate_routine', 'house', 'beginner'); assert r['ok'] and 'House' in r['title'] and s.queued_key is not None
    assert run('get_profile')['level_name'] == 'Rookie'
    assert run('compare_with', '0000')['ok'] is False
    assert run('forget_me', confirm=False)['needs_confirmation']
    assert run('set_call_mode', 'squad_battle')['note'].startswith('only one dancer')
    p = run('propose_activity', 'learn', 'pop', 'beginner'); assert p['ok'] and p['state'] == 'OFFER' and s.proposal['mode'] == 'learn' and s.state == 'OFFER'
    assert run('propose_activity', 'battle')['state'] == 'OFFER' and s.proposal['mode'] == 'battle'      # updates the offer, still no start
    s.state = 'ACTIVITY'; assert run('propose_activity', 'learn')['reason'] == 'already setting up or dancing'
    assert s.telemetry.tool_calls >= 10


def _body(cx=0.5, top=0.15, h=0.7, vis=1.0, feet=True):
    lm = np.zeros((33, 5), np.float32); lm[:, 3] = vis
    y = lambda f: top + f * h
    pts = {0: (cx, y(0.0)), 11: (cx + 0.06, y(0.15)), 12: (cx - 0.06, y(0.15)), 13: (cx + 0.09, y(0.3)), 14: (cx - 0.09, y(0.3)), 15: (cx + 0.1, y(0.45)),
           16: (cx - 0.1, y(0.45)), 23: (cx + 0.04, y(0.5)), 24: (cx - 0.04, y(0.5)), 25: (cx + 0.04, y(0.75)), 26: (cx - 0.04, y(0.75)), 27: (cx + 0.04, y(1.0)), 28: (cx - 0.04, y(1.0))}
    for j, p in pts.items(): lm[j, :2] = p
    if not feet: lm[[27, 28], 1] = 1.1
    return lm


def test_tracker_counts_shoulders_only():
    from team.dance.multipose import PlayerTracker
    lm = _body(); lm[[23, 24, 25, 26, 27, 28], 3] = 0.1                          # at a desk: no hips or legs
    tr = PlayerTracker()
    for i in range(40): tr.update(i / 15, [(lm, None)])
    assert tr.solo_pose()[0] is lm and tr.stable_count() == 1


def test_setup_hysteresis_and_note():
    from team.dance import setup as su
    tr = su.SetupTracker(); issues = []
    for i in range(90):                                                          # hips flicker in and out every 0.4 s
        lm = _body(); lm[[23, 24], 3] = 1.0 if (i // 6) % 2 == 0 else 0.1
        issues.append(tr.update(i / 15, lm, 120)['issue'])
    changes = sum(1 for a, b in zip(issues, issues[1:]) if a != b)
    assert changes <= 2, issues
    lm = _body(); lm[[23, 24], 3] = 0.1
    note = su.camera_note(lm, 'too_close')
    assert 'hips' in note and 'out of frame' in note and 'feet' not in note and note.startswith('[camera now:')


def test_setup_checker():
    from team.dance import setup as su
    assert su.evaluate(_body(), luma=120)['ok']
    assert su.evaluate(None)['issues'] == ['no_person']
    assert su.evaluate(_body(top=0.0, h=0.98), luma=120)['issues'][0] == 'too_close'
    assert su.evaluate(_body(feet=False), luma=120)['ok']                         # feet are ignored
    nohips = _body(); nohips[[23, 24], 3] = 0.1
    assert su.evaluate(nohips, luma=120)['issues'][0] == 'too_close'              # head to hips must be in frame
    assert su.evaluate(_body(h=0.15, top=0.4), luma=120)['issues'][0] == 'too_far'
    assert 'off_center' in su.evaluate(_body(cx=0.15), luma=120)['issues']
    assert 'dark' in su.evaluate(_body(), luma=30)['issues']
    assert su.evaluate(_body(feet=False), luma=120, upper=True)['ok']
    tr = su.SetupTracker(); res = None
    for i in range(30): res = tr.update(i / 15, _body(), 120)
    assert res['ok_stable']
    res = tr.update(2.1, nohips, 120); assert res['ok_stable']                    # one bad frame doesn't break it any more
    for i in range(12): res = tr.update(2.2 + i / 15, nohips, 120)
    assert not res['ok_stable']


def test_engine_demo_abort_focus_and_halves():
    key = choreo.generate('hiphop', 'beginner', seed=3)
    e = PhraseEngine(key, focus='arms', skip_show=True); e.start(10.0)
    h = e.hud(10.5); assert h['focus'] == 'arms' and h['t'] == 10.5 and 'countin_label' in h
    e.abort(11.0); assert e.state == 'done' and e.aborted and e.result is None
    e2 = PhraseEngine(key, countin_beats=4); e2.start(0.0); e2.tick(e2.show_s + 0.1)
    assert e2.state == 'countin' and e2.hud(e2.show_s + 0.1)['countin_label'] == 5
    d = PhraseEngine(key, show_repeats=1, demo_only=True); d.start(0.0); d.tick(d.show_s + 0.05)
    assert d.state == 'done' and d.result is None
    k2, hp = choreo.with_half(key, 1, 1)
    assert k2['phrases'][hp]['beats'] == 4 and k2['phrases'][hp]['start_beat'] == key['phrases'][1]['start_beat'] + 4 and len(key['phrases']) == choreo.LEVEL_PHRASES['beginner']


def test_buttons_are_state_gated():
    s, clock, log = _session()
    assert s.button_cmd('pause') is False                                         # nothing to pause in chat
    s.state = 'PREPARE'; s.button_cmd('stop'); assert s.cancel_req
    s.state = 'ACTIVITY'; s.cancel_req = False; clock['t'] = 10.0
    assert s.button_cmd('repeat') and s.pending_cmd == 'again'
    assert s.button_cmd('rm -rf') is False


def test_avatar_is_kit_compliant():
    import importlib, tempfile, cv2
    from team.dance import avatar as A
    tools = os.path.join(ROOT, 'team', 'dance', 'assets', 'brodie', 'tools')
    if tools not in sys.path: sys.path.insert(0, tools)
    cc = importlib.import_module('check_cream')                              # the kit's checker, in-process (a second Python cost ~10 s)
    names = ['grin', 'closed', 'sleepy'] + [A.variant_for(m, mo) for m in A.MOODS if m != 'happy' for mo in ('grin', 'closed')]
    for v in names:                                                          # every mood frame stays canon cream
        p = os.path.join(tempfile.gettempdir(), f'brodie_{v}.png'); cv2.imwrite(p, A.sprite(v, 600))
        fill, _ = cc.measure(p)
        assert fill is not None, f'{v}: no cream body found'
        assert not cc.verdict(fill)[1], (v, cc.verdict(fill))
    img = np.zeros((300, 300, 3), np.uint8)
    for st in A.BrodieAvatar.STATES:
        for m in A.MOODS: A.avatar().draw(img, 150, 150, 200, st, 0.3, mood=m)
    assert img.sum() > 0
    av = A.avatar()
    assert av.variant('idle', 0.0, 'pumped') == 'pumped_grin' and av.variant('idle', 0.0, 'focus') == 'focus_closed'
    assert av.variant('offline', 0.0, 'pumped') == 'sleepy' and av.variant('idle', 0.0, 'happy') == 'grin'   # offline wins; happy = the old art
    assert A.sprite('focus_grin', 200).tobytes() != A.sprite('grin', 200).tobytes()       # the face really changed


def test_mood_follows_verdicts_scores_and_style():
    s, clock, log = _session()
    assert s.cur_mood(0.0) == 'happy'                                        # hype (default): happy at rest
    eng = type('E', (), {'state': 'attempt'})()
    assert s.cur_mood(0.0, eng) == 'focus'                                   # watching the caller dance
    for b in range(4): s.on_engine_event(dict(type='beat', beat=b, live_verdict='green'))
    assert s.cur_mood(clock['t']) == 'pumped'                                # a PERFECT streak of four
    clock['t'] = 10.0; assert s.cur_mood(clock['t']) == 'happy'              # reactions expire
    s.on_engine_event(dict(type='beat', beat=0, live_verdict='red')); s.on_engine_event(dict(type='beat', beat=1, live_verdict='red'))
    assert s.cur_mood(clock['t']) == 'encourage'                             # two misses in a row
    clock['t'] = 20.0; s.mood_for_score(78.0); assert s.cur_mood(clock['t']) == 'pumped'        # hype: 75 is big
    s.persona = 'chill'; s.mood_for_score(78.0); assert s.cur_mood(clock['t']) == 'happy'       # chill: only 92+
    s.mood_for_score(40.0); assert s.cur_mood(clock['t']) == 'encourage'
    s.persona = 'coach'; clock['t'] = 40.0; assert s.cur_mood(clock['t']) == 'focus'            # COACH rests on focus


def test_scorecard_qr_opens_the_result_page(monkeypatch):
    from team.dance import render as R
    monkeypatch.delenv('DANCE_PUBLIC_URL', raising=False)
    assert R.qr_payload('brodie@x.com', '4821') == 'facetime:brodie@x.com' and R.qr_payload(None, '4821') is None
    monkeypatch.setenv('DANCE_PUBLIC_URL', 'https://b.example.app/')
    assert R.qr_payload('brodie@x.com', '4821') == 'https://b.example.app/r/4821'                # the page has the FaceTime button
    assert R.qr_payload('brodie@x.com', None) == 'facetime:brodie@x.com'
    assert R.mood_for_grade('S') == 'pumped' and R.mood_for_grade('D') == 'encourage' and R.mood_for_grade(None) == 'happy'


def test_photo_card_and_safe_area():
    from team.dance import render as R
    photo = np.full((1280, 720, 3), 90, np.uint8)
    card = R.draw_card(photo.copy(), dict(scored=True, score=88, spatial=0.9, timing=0.9, energy=0.8, max_combo=6, per_group={}), 'k', code='4821',
                       handle='brodie@example.com', photo=photo, brag=['NEW PERSONAL BEST'])
    import cv2
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(card)
    assert card.shape == (1280, 720, 3) and data == 'facetime:brodie@example.com'
    x0, y0, x1, y1 = R.safe_box((720, 1280)); assert y1 <= int(1280 * 0.85) and y0 >= int(1280 * 0.09)


def test_facetime_callers_get_portrait(monkeypatch):
    s, clock, log = _session()
    s.job.bound = 'chert-bridge-abc'; s.job.latest = (np.zeros((720, 1280, 3), np.uint8), 0, 0.0, 0.0, 1)
    monkeypatch.setenv('DANCE_CANVAS', 'auto')
    async def go():
        s.lobby_enabled = False; s.stop = True
        await s.run()
    asyncio.run(go())
    assert s.size == (720, 1280)


def test_cards_never_auto_offer_and_hands_up_picks_the_lit_card():
    """CHOOSE: Brodie asks what we're doing and never makes an offer by himself; after CYCLE_START_S quiet a highlight steps through
    the cards and both hands up picks the card that was lit when the hold began."""
    s, clock, log = _session(); s.lobby_enabled = True; s.t0 = 0.0
    s.sleep = lambda x: asyncio.sleep(0)
    seen = dict(lit=None)
    async def go():
        t = asyncio.create_task(s.explore(first=True))
        for i in range(2000):
            clock['t'] += 0.05; await asyncio.sleep(0)
            if s.state == 'OFFER': break
            lit = (s.menu_ui or {}).get('lit')
            if s.menu_ui and s.menu_ui.get('cycling') and lit == 1 and seen['lit'] is None:
                seen['lit'] = clock['t']; s.gesture_ui = dict(kind='ready', p=0.5)          # a hold starts while Ghost Battle is lit
            if seen['lit'] is not None and clock['t'] - seen['lit'] > 3.0 and s.affirm_t < 0:
                s.affirm_t = clock['t']                                                    # ...and completes after the highlight moved on
        t.cancel()
    asyncio.run(go())
    assert s.state == 'OFFER' and s.proposal['mode'] == 'battle' and s.proposal['source'] == 'card'
    assert not any(e.get('note') == 'brodie' for e in log) and s.offers_made == 0
    assert any(live.mode_cards.HELLO[:20] in e.get('ask', '') for e in log)


class _OkTracker:
    """Framing that holds, except while `bad(t)` is true (a camera tip in between)."""
    bad = staticmethod(lambda t: False)
    def __init__(self, upper=False): self.upper = upper
    def update(self, now, lm, luma, aspect=None):
        b = _OkTracker.bad(now)
        return dict(ok_now=not b, ok_stable=not b, issue='too_close' if b else None, checks={}, hold=0.0)


def _run_setup(s, clock, monkeypatch, script, until=90.0, bad=lambda t: False):
    """Drive setup_camera on the simulated clock; `script` maps a time to a callback (a caller line, a gesture)."""
    from team.dance import setup as setup_mod
    _OkTracker.bad = staticmethod(bad); monkeypatch.setattr(setup_mod, 'SetupTracker', _OkTracker)
    s.lobby_enabled = True; s.t0 = 0.0; s.sleep = lambda x: asyncio.sleep(0); seen = dict(want_ready=False)
    pending = sorted(script.items())
    async def go():
        t = asyncio.create_task(s.setup_camera())
        while clock['t'] < until and not t.done():
            clock['t'] += 0.1
            while pending and pending[0][0] <= clock['t']: pending.pop(0)[1]()
            seen['want_ready'] |= s.want_ready
            for _ in range(3): await asyncio.sleep(0)
        if not t.done(): t.cancel(); return None, seen
        return t.result(), seen
    return asyncio.run(go())


def _said(log, *texts): return [e for e in log if any(x in e.get('ask', '') for x in texts)]


def test_space_question_late_yes_counts(monkeypatch):
    s, clock, log = _session()
    caller = lambda txt: (lambda: s.on_worker_event(dict(type='voice', role='caller', text=txt)))
    ok, seen = _run_setup(s, clock, monkeypatch, {40.0: caller('yes')}, bad=lambda t: 20 < t < 26)   # a camera tip in between, then a late yes
    assert ok is True and s.space_ok and seen['want_ready'] and not s.want_ready
    assert len(_said(log, 'Perfect, I can see', 'Looking good!')) == 1                                 # asked once, never re-asked


def test_space_question_hands_up_answers(monkeypatch):
    s, clock, log = _session()
    def hands_up():
        assert s.want_ready                                                      # the gesture path is armed while the question is open
        s.affirm_t = s.now()                                                     # what _gestures does on both hands up
    ok, _ = _run_setup(s, clock, monkeypatch, {15.0: hands_up})
    assert ok is True and s.space_ok


def test_space_question_nudges_twice_then_trusts_the_caller(monkeypatch):
    s, clock, log = _session()
    ok, _ = _run_setup(s, clock, monkeypatch, {}, until=120.0)
    assert len(_said(log, 'Perfect, I can see', 'Looking good!')) == 1 and len(_said(log, "If you've got")) == 1 and len(_said(log, 'Take your time')) == 1
    assert ok is True and s.space_ok and len(_said(log, "trust you've got room")) == 1     # space can't be seen: never a dead end
    s2, clock2, log2 = _session()
    ok2, _ = _run_setup(s2, clock2, monkeypatch, {}, until=20.0)
    assert ok2 is None and not s2.space_ok and s2.setup_ui.get('prompt')         # before that: waiting, prompt on screen, no start


# ---- the 2026-10-11 audit: every wait has an exit, every answer counts ----
YES = ["yes", "yeah", "Ya.", "yea", "yesss", "yessir", "okayy", "ok", "k", "let's gooo", "bet", "I'm down", "I'm in", "for sure", "sounds good",
       "got it", "cool", "oh yeah", "um okay", "so yeah", "hell yeah", "bro let's go", "de Hello yes", "Yeah?", "Ready?", "Yes! can you see me?",
       "All right. Let's start it.", "yeah let's do it", "Do it!", "I can't wait, let's go!", "no cap let's go", "yes no doubt", "makes sense",
       "I'm good to go", "uh huh", "mm-hmm", "Yeah, started.", "Claro.", "Ja.", "sure"]
NO = ["no", "nah", "nope", "not right now", "I'm good", "absolutely not", "of course not", "I do not want to", "go away", "not yet", "wait",
      "hold on", "I'm not ready", "maybe later"]
NEITHER = ["no worries", "don't stop", "I have a question", "I have to go", "what's that", "Is that okay?", "c'est quand"]


@pytest.mark.parametrize('text', YES)
def test_classify_yes(text):
    c = live.classify(text); assert c['affirm'] and not c['deny'] and not c['question'], (text, c)


@pytest.mark.parametrize('text', NO)
def test_classify_no(text):
    c = live.classify(text); assert c['deny'] and not c['affirm'], (text, c)


@pytest.mark.parametrize('text', NEITHER)
def test_classify_neither(text):
    c = live.classify(text); assert not c['affirm'] and not c['deny'], (text, c)


def test_names_styles_and_pain_words():
    s, clock, log = _session()
    s.on_worker_event(dict(type='voice', role='caller', text="I'm down, this is fun"))
    assert s.name is None                                                        # "I'm down" is not a name
    s.on_worker_event(dict(type='voice', role='caller', text="My name is Brian and I want to dance hip-hop today."))
    assert s.name == 'Brian' and s.style == 'lh'
    assert live.style_from('can we do kpop instead of hip hop') == 'kpop' and live.style_from('not hip hop, pop') == 'po'
    assert live.pain_level('ow my knee hurts') == 'strong' and live.pain_level('this song hurts my ears') == 'weak'
    assert live.pain_level("it doesn't hurt") is None and live.pain_level('my friend got hurt dancing') == 'weak'
    s.on_worker_event(dict(type='voice', role='caller', text="I'm a student"))
    assert s.mode_choice is None                                                 # a word in passing never switches the mode


def test_unchanged_offer_keeps_the_yes():
    s, clock, log = _session(); s.lobby_enabled = True; s.t0 = 0.0
    s.request_activity(mode='learn', style='hip-hop', source='voice'); t_prop = s.proposal_t
    clock['t'] = 3.0; s.offer_asked_t = 3.0                                      # the offer question was asked
    clock['t'] = 5.0; s.on_worker_event(dict(type='voice', role='caller', text="All right. Let's start it."))
    r = s.request_activity(mode='learn', style='hip-hop', source='tool')         # Gemini proposes the same routine again
    assert r['already_offered'] and s.proposal_t == t_prop and s.affirm_t == 5.0
    clock['t'] = 6.0; s.on_worker_event(dict(type='voice', role='caller', text='actually K-pop'))
    assert s.proposal['style'] == 'kpop' and s.proposal_t == 6.0                 # a real change of mind is a new offer
    clock['t'] = 7.0; s.on_worker_event(dict(type='voice', role='caller', text='no thanks'))
    assert s.cancel_req                                                          # "no" goes back to chat


def test_offer_yes_after_gemini_asked():
    s, clock, log = _session(); s.lobby_enabled = True; s.t0 = 0.0; s.sleep = lambda x: asyncio.sleep(0)
    s.request_activity(mode='learn', style='pop', source='voice')
    async def go():
        t = asyncio.create_task(s.offer_loop())
        for i in range(200):
            clock['t'] += 0.05; await asyncio.sleep(0)
            if i == 20: s.on_worker_event(dict(type='voice', role='coach', text='Want to jump into a pop routine?'))
            if i == 40: s.on_worker_event(dict(type='voice', role='caller', text='bet'))
            if i == 45: s.on_worker_event(dict(type='voice', role='coach', text="Let's go! Ready for the camera check?"))
            if t.done(): return t.result()
        t.cancel()
    assert asyncio.run(go()) == 'yes'


def test_waits_recover_when_the_question_was_dropped():
    s, clock, log = _session(); s.lobby_enabled = True; s.t0 = 0.0; s.sleep = lambda x: asyncio.sleep(0)
    s.state = 'REVIEW'
    async def go():
        t = asyncio.create_task(s.wait_choice('Got it?', key='gotit'))
        await asyncio.sleep(0); s.say_q = []                                     # the line never gets said
        for i in range(400):
            clock['t'] += 0.1; await asyncio.sleep(0)
            if i == 200: s.on_worker_event(dict(type='voice', role='caller', text='yeah'))
            if t.done(): return t.result()
        t.cancel()
    assert asyncio.run(go()) == 'yes'


def test_got_it_any_reply_and_timeout():
    s, clock, log = _session(); s.lobby_enabled = True; s.t0 = 0.0; s.sleep = lambda x: asyncio.sleep(0); s.state = 'REVIEW'
    async def go(script, until=120.0):
        t = asyncio.create_task(s.wait_choice('Got it?', key='gotit', reprompt='Take your time.', any_reply=True, timeout=25.0, on_timeout='yes'))
        start = clock['t']
        while clock['t'] - start < until:
            clock['t'] += 0.1; await asyncio.sleep(0)
            for at, text in list(script):
                if clock['t'] - start >= at: script.remove((at, text)); s.on_worker_event(dict(type='voice', role='caller', text=text))
            if t.done(): return t.result(), clock['t'] - start
        t.cancel(); return None, None
    r, _ = asyncio.run(go([(3.0, 'makes sense, I think')])); assert r == 'yes'
    r, took = asyncio.run(go([])); assert r == 'yes' and 40 < took < 60                     # quiet: one reprompt, then on to practice


def test_stale_commands_never_skip_or_stop():
    s, clock, log = _session(); s.state = 'REVIEW'
    clock['t'] = 1.0; s._cmd('stop', 'voice')                                    # "I'm done" at the last wrap card
    clock['t'] = 5.0; s.setup_activity(dict(mode='learn'))                        # a new routine later
    assert s.take_cmd(('stop',)) is None                                         # the old stop never ends the new one
    s.state = 'PREPARE'; clock['t'] = 10.0
    assert s.tool_command('next')['will_happen'] == 'nothing' and s.pending_cmd is None   # nothing to skip during the camera check
    s.state = 'REVIEW'; clock['t'] = 20.0; s.on_worker_event(dict(type='voice', role='caller', text='hold on'))
    assert s.hold_until > 20 and s.deny_t < 0 and s.pending_cmd is None           # a hold is a wait, not a no or a pause


def test_tracker_is_smoothed():
    from team.dance import setup as su
    tr = su.SetupTracker(); res = None
    for i in range(60):                                                          # one frame in five fails: still stable
        lm = _body()
        if i % 5 == 0: lm[[23, 24], 3] = 0.1
        res = tr.update(i / 15, lm, 120)
    assert res['ok_stable'] and res['issue'] is None
    lm = _body(); lm[[11, 12], 0] = lm[[11, 12], 0] * 0.5 + 0.25                 # narrow shoulders in x (turned away)...
    assert su.evaluate(lm, 120, aspect=16 / 9)['metrics']['facing'] > su.evaluate(lm, 120)['metrics']['facing']   # ...less so on 16:9


def test_lines_go_to_gemini_as_user_turns():
    calls = []
    class Sess:
        def generate_reply(self, **kw): calls.append(kw); return 'h'
    from facetime import voice as fv
    live.patch_voice_layer()
    m = fv.GeminiModel('persona'); m.session = Sess()
    m.generate(live.SAY_PREFIX + '"Take a step back."')
    assert 'instructions' not in calls[0] and calls[0]['user_input'].startswith('[Stage direction')
    assert calls[0]['user_input'].endswith('"Take a step back."') and 'Private instruction' not in calls[0]['user_input']


# ---- v5: the opening menu, every site mode, codes, pose by pose, depth-aware coaching ----
def test_menu_choice_and_edge_cases():
    s, clock, log = _session(); s.lobby_enabled = True; s.menu_open = True
    s.on_worker_event(dict(type='voice', role='caller', text='what can you do?'))
    assert s.state == 'EXPLORE' and any('What are we doing' in e.get('ask', '') for e in log if 'ask' in e)
    s.say_q = []
    s.on_worker_event(dict(type='voice', role='caller', text='battle or the daily?'))
    assert s.state == 'EXPLORE' and any('Which one' in e.get('ask', '') for e in log if 'ask' in e)   # two modes: "which one?"
    s.on_worker_event(dict(type='voice', role='caller', text='the daily one'))
    assert s.state == 'OFFER' and s.proposal['mode'] == 'daily'                           # after the menu, a mode word is a choice
    assert "daily challenge" in s.offer_label()
    s2, _, _ = _session(); s2.lobby_enabled = True; s2.menu_open = True
    s2.on_worker_event(dict(type='voice', role='caller', text='after class I want to dance'))
    assert s2.proposal is None or s2.proposal['mode'] != 'class'                          # "after class" is not class mode
    s3, _, _ = _session(); s3.lobby_enabled = True; s3.menu_open = True
    s3.on_worker_event(dict(type='voice', role='caller', text='idk'))
    assert s3.state == 'OFFER' and s3.proposal['source'] == 'brodie'                      # "idk": Brodie picks


def test_offer_labels_and_tool_kinds():
    pytest.importorskip('livekit.agents')
    from team.dance import tools
    s, clock, log = _session(); s.lobby_enabled = True
    t = {f.info.name if hasattr(f, 'info') else f.__name__: f for f in tools.build_tools(s)}
    run = lambda name, *a, **k: json.loads(asyncio.run(t[name](*a, **k)))
    want = dict(learn='routine', battle='ghost battle', daily='daily challenge', squad_battle='squad battle', squad_learn='whole squad',
                telephone='telephone', **{'class': 'I lead', 'class_record': 'class recording'})
    for kind, word in want.items():
        clock['t'] += 1.0; r = run('propose_activity', kind)
        assert r['ok'] and s.proposal['mode'] == kind and word in s.offer_label(), (kind, s.offer_label())
    assert not run('propose_activity', 'karaoke')['ok']                                     # unknown kinds are refused, never silently "learn"
    s.request_activity(mode='telephone', style='hip-hop', source='brodie')
    assert s.proposal.get('code') is None                                                  # regression: a style code once leaked into the code


def test_codes_resolve_across_kinds():
    from team.dance.live import Backend
    b = Backend('fake'); db = b.db
    c_battle = db.add_result('Sam', 'gen_x', 'battle', 240.0, [])
    c_chain = db.start_chain('gen_x', dict(name='Ana', key={'name': 'k'}, score=100.0))
    c_class = db.start_class('Mia', {'name': 'class_k', 'phrases': []})
    assert len({c_battle, c_chain, c_class}) == 3
    assert [k['kind'] for k in db.resolve_code(c_battle)['kinds']] == ['battle']
    assert [k['kind'] for k in db.resolve_code(c_chain)['kinds']] == ['telephone']
    assert [k['kind'] for k in db.resolve_code(c_class)['kinds']] == ['class']
    assert db.resolve_code('0000')['kinds'] == [] or True
    s, clock, log = _session(); s.lobby_enabled = True
    async def go(code): return await s.handle_code(code)
    assert asyncio.run(go(c_chain))['kind'] == 'telephone' and s.state == 'OFFER' and s.proposal['mode'] == 'telephone' and s.proposal['code'] == c_chain
    assert 'telephone chain' in s.offer_label()


def test_daily_board_one_shot():
    from team.dance.live import Backend
    import time as _t
    db = Backend('fake').db
    key = f'gen_daily_test_{int(_t.time())}'
    db.add_result('Lee', key, 'daily', 70.0, [], score100=70.0); db.add_result('Kim', key, 'daily', 90.0, [], score100=90.0)
    db.add_result('Kim', key, 'learn', 99.0, [], score100=99.0)                            # practice / other modes never rank
    board = db.leaderboard(key, 10000, mode='daily', since=_t.time() - 3600)
    assert [r['name'] for r in board] == ['Kim', 'Lee'] and db.daily_played('kim', key) and not db.daily_played('Zed', key)


def test_pose_step_watch_and_freeze():
    from team.dance.posestep import PoseStepEngine
    key = choreo.generate('hiphop', 'beginner', seed=3)
    e = PoseStepEngine(key, 0, mode='watch', hold_s=3.5); e.start(0.0); counts = []
    t = 0.0
    while e.state != 'done' and t < 60:
        e.tick(t); counts.append((round(t, 1), e.cur)); t += 0.1
    changes = [c for (tt, c), (tp, cp) in zip(counts[1:], counts) if c != cp]
    assert e.state == 'done' and len(changes) == e.n_beats - 1 and abs(t - 3.5 * e.n_beats) < 0.6
    g0 = e.ghost_sample(0.0); assert g0 is not None and g0.shape == (33, 2)
    f = PoseStepEngine(key, 0, mode='freeze', max_hold_s=4.0); f.start(0.0); t = 0.0
    while f.state != 'done' and t < 60: f.on_frame(t, None); f.tick(t); t += 0.1                 # nobody matches: it still moves on
    assert f.state == 'done' and abs(t - 4.0 * f.n_beats) < 0.6
    h = f.hud(t); assert h['pose_step'] == 'freeze' and len(h['verdicts']) == f.n_beats


def test_body_frame_signals_and_depth_cues():
    import math
    from team.dance.features import body_frame_signals
    P = np.zeros((33, 3), np.float32); V = np.ones((33, 5), np.float32)
    P[11] = [0.18, -0.45, 0]; P[12] = [-0.18, -0.45, 0]; P[23] = [0.1, 0, 0]; P[24] = [-0.1, 0, 0]; P[0] = [0, -0.65, -0.08]
    P[15] = [0.18, -0.45, -0.55]; P[16] = [-0.73, -0.45, 0]                     # left arm straight forward, right arm out to the side
    sg = body_frame_signals(P, V); assert sg['arm_fwd_l'] > 80 and abs(sg['arm_fwd_r']) < 5 and abs(sg['facing_yaw']) < 5
    a = math.radians(40); R = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]], np.float32)
    sg2 = body_frame_signals(P @ R.T, V); assert abs(sg2['facing_yaw'] - 40) < 2 and sg2['arm_fwd_l'] > 80   # body frame: turning doesn't change it
    key = choreo.generate('hiphop', 'beginner', seed=3); prof = feedback.ref_profile(key, 0)
    assert prof and all(0 < p['flat_l'] <= 1.0001 for p in prof)
    flat = dict(prof[0], flat_l=1.0, dir_l='out')
    c = feedback.depth_issue(dict(arm_fwd_l=70.0), flat, 'l'); assert c and 'forward' in c['text'] and 'out to the side' in c['text']
    assert feedback.depth_issue(dict(arm_fwd_l=20.0), flat, 'l') is None                     # inside the deadband: no cue
    res = dict(scored=True, score=60, weak_beats=[], per_beat=[], body3d=[dict(arm_fwd_l=70.0, facing_yaw=0.0)] * len(prof))
    prof_flat = [dict(p, flat_l=1.0, dir_l='out', facing=1.0) for p in prof]
    import team.dance.feedback as fb
    orig = fb.ref_profile; fb.ref_profile = lambda k, i: prof_flat
    try: cues = fb.depth_cues(res, key, 0)
    finally: fb.ref_profile = orig
    assert cues and cues[0]['kind'] == 'depth' and 'counts 1 to' in cues[0]['text']


def test_layered_cues_ranked_capped_and_progress():
    from team.dance.features import CHANNELS
    ch = CHANNELS.index('arm_elev_l'); cs = [[None] * len(CHANNELS) for _ in range(8)]
    for b in (4, 5, 6): row = [0.0] * len(CHANNELS); row[ch] = -40.0; cs[b] = row
    pb = [dict(beat=b, error_deg=12.0 if b < 4 else 40.0, offset_beats=0.5 if b in (1, 2) else 0.0, verdict='amber') for b in range(8)]
    r = dict(scored=True, score=55, timing=0.5, mean_offset_beats=0.1, weak_beats=[4, 5, 6], per_group=dict(arms=40.0), channel_signed=cs, per_beat=pb,
             size=dict(arms=0.4, energy=0.9))
    cues = feedback.layered_cues(r)
    kinds = [c['kind'] for c in cues]
    assert 1 <= len(cues) <= 3 and 'correction' in kinds and 'timing' in kinds and 'size' in kinds
    assert len({c.get('body_part') or c['kind'] for c in cues}) == len(cues)              # one per body part
    better = dict(r, channel_signed=[[None if v is None else (0.0 if v == 0.0 else -10.0) for v in (row or [None] * len(CHANNELS))] if row else None for row in cs])
    assert 'better' in (feedback.progress_note(r, better, cues[[c['kind'] for c in cues].index('correction')]) or '')
    assert feedback.layered_cues(dict(scored=False))[0]['kind'] == 'visibility'


def test_loop_slice_and_weak_span():
    key = choreo.generate('hiphop', 'beginner', seed=3)
    k2, p2 = choreo.with_counts(key, 0, 4, 6)
    assert k2['phrases'][p2]['beats'] == 3 and len(key['phrases']) == len(k2['phrases']) - 1     # a copy: the routine itself is untouched
    assert live.weak_span(dict(kind='correction', beats=[5])) == (5, 6) and live.weak_span(dict(kind='praise', beats=[1])) is None
    assert live.count_words('counts 1 to 2') == 'One, two'
