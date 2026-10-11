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
    assert s._cmd('again', 'tool') and not s._cmd('again', 'voice')              # same command within 2.5 s: once
    clock['t'] = 9.0; assert s._cmd('again', 'voice')
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


def test_no_meta_prompts_in_flow_code():
    src = open(live.__file__, encoding='utf-8').read()
    body = src[src.index('class DanceSession'):src.index('    async def record_phrase')]
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
    assert r is None and s.state == 'EXPLORE' and s.telemetry.errors


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
    out = run('control_session', 'slower'); assert out['ok'] and out['will_happen'] == 'nothing yet'   # nothing runs in chat
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
    res = tr.update(2.1, nohips, 120); assert not res['ok_stable']


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
    assert k2['phrases'][hp]['beats'] == 4 and k2['phrases'][hp]['start_beat'] == key['phrases'][1]['start_beat'] + 4 and len(key['phrases']) == 3


def test_buttons_are_state_gated():
    s, clock, log = _session()
    assert s.button_cmd('pause') is False                                         # nothing to pause in chat
    s.state = 'PREPARE'; s.button_cmd('stop'); assert s.cancel_req
    s.state = 'ACTIVITY'; s.cancel_req = False; clock['t'] = 10.0
    assert s.button_cmd('repeat') and s.pending_cmd == 'again'
    assert s.button_cmd('rm -rf') is False


def test_avatar_is_kit_compliant():
    import subprocess, tempfile, cv2
    from team.dance import avatar as A
    tools = os.path.join(ROOT, 'team', 'dance', 'assets', 'brodie', 'tools')
    paths = []
    for v in ('grin', 'closed', 'sleepy'):
        p = os.path.join(tempfile.gettempdir(), f'brodie_{v}.png'); cv2.imwrite(p, A.sprite(v, 600)); paths.append(p)
    r = subprocess.run([sys.executable, 'check_cream.py'] + paths, cwd=tools, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    img = np.zeros((300, 300, 3), np.uint8)
    for st in A.BrodieAvatar.STATES: A.avatar().draw(img, 150, 150, 200, st, 0.3)
    assert img.sum() > 0


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


def test_brodie_suggests_a_routine_himself():
    s, clock, log = _session(); s.lobby_enabled = True; s.t0 = 0.0
    s.sleep = lambda x: asyncio.sleep(0)                                         # simulated clock: no real waiting
    async def go():
        t = asyncio.create_task(s.explore(first=True))
        for i in range(600):
            clock['t'] += 0.05; await asyncio.sleep(0)
            if s.state == 'OFFER' and any('Want to start now' in e.get('ask', '') or 'Shall we start' in e.get('ask', '') or 'Up for it' in e.get('ask', '') for e in log):
                break
        t.cancel()
    asyncio.run(go())
    assert s.state == 'OFFER' and s.proposal['source'] == 'brodie' and s.offers_made == 1   # he leads, but nothing starts without a yes
