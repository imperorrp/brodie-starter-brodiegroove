"""New modules: coaching cues, the ghost record, addressed speech, answer confidence, caller audio options, pacing."""
import os, sys, types
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from team.dance import cues, ghost, addressed, answers, audio_in, style_vocab, choreo


def test_cue_lines_are_short_and_deterministic():
    for style in ('hiphop', 'house', 'waack', 'pop', 'krump'):
        try: k = choreo.generate(style, 'beginner', seed=3)
        except Exception: continue
        for pi in range(len(k['phrases'])):
            a = cues.move_card(k, pi); b = cues.move_card(k, pi)
            assert a['name'] == b['name'] and cues.count_lines(a) == cues.count_lines(b)
            assert len(cues.intro_line(a, 'First move, counts 1 to 8').split()) <= 16
            assert all(len(l.replace(',', ' ').split()) <= 8 for l in cues.count_lines(a)), cues.count_lines(a)
            assert len(cues.card_note(a, '1').split()) <= 60
            assert len(a['feel'].split()) <= 12
            assert not any(w in ' '.join(cues.count_lines(a)).lower() for w in ('channel', 'degrees', 'flexion'))


def test_style_vocab_by_source():
    assert style_vocab.genre_of_src('aist_ho_110_ch01') == 'house' and style_vocab.genre_of_src('cc_huntrix') == 'kpop'
    assert 'whip' in style_vocab.feel_line('waack', ['sharp']).lower()


class _Eng:
    def __init__(self, n=8, beat_s=0.6, vis=1.0, samples=48):
        self.n_beats = n; self.beat_s = beat_s
        ts = np.linspace(0, n * beat_s, samples)
        self.live = [(t, np.full(14, 90.0, np.float32), np.full(14, vis >= 0.5), 0.1) for t in ts]
        self.live_skel = [(t, np.zeros((33, 2), np.float32)) for t in ts]


def test_ghost_round_trip_builds_a_key():
    r = ghost.live_to_arrays(_Eng())
    assert r is not None and r['F'].shape == (32, 14) and r['S'].shape == (32, 33, 2)
    data = ghost.pack([r, r]); back = ghost.unpack(data)
    k = ghost.ghost_key('4471', 'Maya', back, bpm=100.0)
    assert k['name'] == 'ghost_4471' and len(k['phrases']) == 2 and k['title'] == "Maya's ghost"
    assert ghost.live_to_arrays(_Eng(vis=0.0)) is None and ghost.live_to_arrays(_Eng(samples=5)) is None


def test_addressed_speech():
    a = lambda t, **k: addressed.addressed(t, **k)
    for t in ('yes', 'ghost battle', 'Brodie can you slow down', 'teach me hip hop', 'again', "what's the daily?"):
        assert a(t), t
    assert a('yeah', question_open=True)
    for t in ('so then she told him the whole story about the party and everybody laughed for ages and ages',
              'my mom said we should go to the store later', 'did you see this episode last night on tv'):
        assert not a(t), t
    assert addressed.score('yes', brodie_speaking=True) < addressed.score('yes')


def test_answer_confidence():
    c = answers.confidence
    assert answers.decide(c('gesture')) == 'accept' and answers.decide(c('strong')) == 'accept'
    assert answers.decide(c('gemini', addressed=False)) == 'ask' and answers.decide(c('weak')) == 'ask'
    assert answers.decide(c('weak'), already_asked=True) == 'accept'                 # never asks twice
    assert answers.decide(c('strong', addressed=False)) == 'ask'
    assert c('gemini', agrees=True) >= 0.9


def test_noise_cancellation_choice(monkeypatch):
    fake = types.ModuleType('livekit.plugins.noise_cancellation'); fake.BVC = lambda: 'bvc'; fake.BVCTelephony = lambda: 'tel'
    import livekit.plugins as lp
    monkeypatch.setitem(sys.modules, 'livekit.plugins.noise_cancellation', fake); monkeypatch.setattr(lp, 'noise_cancellation', fake, raising=False)
    assert audio_in.choose_nc('off')[0] is None
    assert audio_in.choose_nc('auto', 'wss://x.livekit.cloud')[0] == 'bvc'
    kind, why = audio_in.choose_nc('auto', 'wss://chert.example.com')
    assert kind is None and 'not LiveKit Cloud' in why
    assert audio_in.choose_nc('bvc_tel', 'wss://anything')[0] == 'bvc_tel'


def test_room_options_patch_restores_and_upgrades(monkeypatch):
    import asyncio
    import livekit.agents.voice.room_io as rio
    seen = {}
    class Dummy:
        async def start(self):
            from livekit.agents.voice.room_io import RoomOptions
            seen['opts'] = RoomOptions(audio_input=True)
    monkeypatch.setenv('DANCE_NC', 'off'); monkeypatch.setenv('DANCE_AGC', '1')
    orig = rio.RoomOptions
    asyncio.run(audio_in.wrap_start(Dummy.start)(Dummy()))
    assert rio.RoomOptions is orig
    ai = seen['opts'].audio_input
    assert ai is not True and getattr(ai, 'auto_gain_control', None) is True


def test_pacing_defaults_are_3_to_4_seconds():
    from team.dance import pacing
    assert 3.0 <= pacing.MOVE_GAP_S <= 4.0 and 3.0 <= pacing.POSE_HOLD_S <= 4.0 and 3.0 <= pacing.TURN_GAP_S <= 4.0 and 3.0 <= pacing.ROUND_GAP_S <= 4.0
