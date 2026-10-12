"""Phase 1 guards: codes vs counting, squad ready window, space re-asked per activity, setup hysteresis, the dead-air watchdog."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.environ.setdefault('BACKEND_URL', 'fake')

from team.dance import live, watchdog
from team.dance.tests.test_live import _session, _body


def test_counting_is_not_a_code():
    assert live.parse_code('one two three four', strict=True) is None             # a count-in
    assert live.parse_code('five six seven eight', strict=True) is None
    assert live.parse_code('my code is one two three four', strict=True) == '1234'   # said it's a code
    assert live.parse_code('4 4 7 1', strict=True) == '4471'
    assert live.parse_code('so we went there and it was like 4 4 7 1 people or something', strict=True) is None   # long, no "code"
    assert live.parse_code('one two three four') == '1234'                          # the menu (not strict) still takes it


class _FakeGesture:
    schedule = {}                                    # pid -> times the hold completes
    def __init__(self): self.pid = None; self.done = set()
    def update(self, now, lm):
        for t in _FakeGesture.schedule.get(self.pid, ()):
            if t <= now and t not in self.done: self.done.add(t); return 'ready'
        return None
    def progress(self, now): return (None, 0.0)
    def reset(self): pass


def test_squad_ready_counts_holds_within_a_window(monkeypatch):
    s, clock, log = _session()
    s.call_mode = 'squad'; s.players = [(1, None, None), (2, None, None), (3, None, None)]
    def make(pid):
        g = _FakeGesture(); g.pid = pid; return g
    for pid in (1, 2, 3): s.gest[pid] = make(pid)
    _FakeGesture.schedule = {1: [1.0], 2: [2.2]}                                     # two of three, 1.2 s apart (used to need the same frame)
    s.want_ready = True; s.affirm_t = -1.0
    for i in range(60):
        clock['t'] = i * 0.1; s._gestures(clock['t'], None, False)
    assert s.affirm_t > 0
    s2, clock2, _ = _session()
    s2.call_mode = 'squad'; s2.players = [(1, None, None), (2, None, None), (3, None, None)]
    for pid in (1, 2, 3): s2.gest[pid] = make(pid)
    _FakeGesture.schedule = {1: [1.0], 2: [6.0]}                                     # 5 s apart: not one "ready"
    s2.want_ready = True; s2.affirm_t = -1.0
    for i in range(80):
        clock2['t'] = i * 0.1; s2._gestures(clock2['t'], None, False)
    assert s2.affirm_t < 0


def test_space_is_asked_again_for_a_new_activity():
    s, clock, log = _session()
    s.space_ok = True; s.space_ok_t = 0.0; clock['t'] = 100.0
    s.setup_activity(dict(mode='learn', rematch=True))
    assert s.space_ok                                                                 # a rematch within 10 minutes keeps it
    s.space_ok = True; s.space_ok_t = 0.0; clock['t'] = 900.0
    s.setup_activity(dict(mode='learn', rematch=True))
    assert not s.space_ok
    s.space_ok = True; s.space_ok_t = clock['t']
    s.setup_activity(dict(mode='battle'))
    assert not s.space_ok


def test_setup_thresholds_have_hysteresis():
    from team.dance import setup as su
    lm = _body(top=0.05, h=1.14)                             # torso 0.35 h = 0.40 of the frame: between the on (0.42) and off (0.38) lines
    assert 'too_close' not in su.evaluate(lm, 120)['issues']
    assert 'too_close' in su.evaluate(lm, 120, prev='too_close')['issues']
    lm = _body(cx=0.265)
    assert 'off_center' not in su.evaluate(lm, 120)['issues']
    assert 'off_center' in su.evaluate(lm, 120, prev='off_center')['issues']


def test_dead_air_nudges_then_backs_off():
    clock = dict(t=0.0); wd = watchdog.DeadAir(lambda: clock['t'])
    said = []
    for i in range(int(400 / 0.5)):
        clock['t'] = i * 0.5; wd.seen(clock['t'])
        r = wd.check('REVIEW', clock['t'])
        if r: said.append((clock['t'], r[0]))
    times = [t for t, _ in said]
    assert said and said[0][0] <= watchdog.LIMITS['REVIEW'] + 0.5
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps[:2] == [watchdog.LIMITS['REVIEW']] * 2 and max(gaps) <= watchdog.MAX_GAP_S and gaps[-1] > gaps[0]
    assert all(k == 'nudge' for _, k in said)


def test_dead_air_off_while_dancing_and_ends_empty_calls():
    clock = dict(t=0.0); wd = watchdog.DeadAir(lambda: clock['t'])
    for i in range(200):
        clock['t'] = i; assert wd.check('ACTIVITY', clock['t']) is None
    clock = dict(t=0.0); wd = watchdog.DeadAir(lambda: clock['t']); out = None
    for i in range(400):
        clock['t'] = float(i); r = wd.check('EXPLORE', clock['t'])
        if r and r[0] == 'end': out = clock['t']; break
    assert out is not None and out >= watchdog.END_AFTER_S


def test_activity_resets_silence():
    clock = dict(t=0.0); wd = watchdog.DeadAir(lambda: clock['t'])
    for i in range(100):
        clock['t'] = i * 0.5
        if i % 10 == 0: wd.activity(clock['t'])
        assert wd.check('OFFER', clock['t']) is None
