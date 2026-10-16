"""Synthetic tests for the dance engine (no MediaPipe, no camera). Run: .venv/Scripts/python -m pytest team/dance/tests -q"""
import numpy as np
from team.dance.features import FEET_CHANNELS
import pytest
from team.dance.features import features_from_landmarks, visibility_mask, mirror_landmarks, resample_to_grid, N_CHANNELS, GROUPS
from team.dance.scoring import score_phrase, dtw_band, PER_BEAT
from team.dance.reference import phrase_slice
from team.dance.engine import PhraseEngine, choose_cue
from team.dance.offline_test import synthetic_key, selftest


def test_selftest_passes():
    ok, checks = selftest(verbose=False)
    bad = [c for c in checks if not c[1] and ' ms' not in c[0]]                 # timings vary with machine load: the CLI selftest reports them
    assert not bad, bad


def test_identity_and_degradations():
    key = synthetic_key(); kF, kM, kE, _ = phrase_slice(key, 0); pb = key['per_beat']
    extra = pb
    def live(F, E): return np.concatenate([F, np.repeat(F[-1:], extra, axis=0)]), np.concatenate([E, np.repeat(E[-1:], extra)])
    F, E = live(kF, kE); r = score_phrase(F, None, E, kF, kM, kE, pb)
    assert r['scored'] and r['score'] >= 95 and all(row['verdict'] == 'green' for row in r['per_beat'])
    Fp = kF.copy(); Fp[:, GROUPS['legs']] += 35
    F, E = live(Fp, kE); r2 = score_phrase(F, None, E, kF, kM, kE, pb)
    assert r2['per_group']['legs'] > r2['per_group']['arms'] + 20 and r2['score'] < r['score']


def test_dtw_band_identity_path():
    A = np.random.default_rng(0).normal(size=(16, 4)).astype(np.float32)
    cost, path = dtw_band(A, A, band=4)
    assert cost == pytest.approx(0.0, abs=1e-5) and path[0] == (0, 0) and path[-1] == (15, 15)


def test_features_mirror_swaps_sides():
    lm = np.random.default_rng(2).uniform(0.2, 0.8, (33, 5)).astype(np.float32); lm[:, 3] = 1.0
    f = features_from_landmarks(lm); m, _ = mirror_landmarks(lm); fm = features_from_landmarks(m)
    assert fm[0] == pytest.approx(f[1], abs=1e-3) and fm[1] == pytest.approx(f[0], abs=1e-3)       # elbows swap
    assert fm[13] == pytest.approx(f[13], abs=1e-3)                                               # wrist distance is symmetric
    assert np.delete(visibility_mask(lm), FEET_CHANNELS).all()


def test_resample_holds_ends():
    out = resample_to_grid([0.0, 1.0], [[0.0], [10.0]], [-1.0, 0.5, 2.0])
    assert out[:, 0].tolist() == [0.0, 5.0, 10.0]


def test_engine_states_and_cue():
    key = synthetic_key(); kF, kM, kE, _ = phrase_slice(key, 0); pb = key['per_beat']; n = len(kF)
    ev = []; eng = PhraseEngine(key, show_repeats=1, countin_beats=2, on_event=ev.append); t = 0.0; eng.start(t)
    assert eng.state == 'show'
    seen = []
    while eng.state != 'done' and t < 120:
        if eng.state == 'attempt':
            idx = int(round(eng.attempt_t(t) / eng.beat_s * pb))
            if 0 <= idx < n: eng.on_features(t, kF[idx] + 50 * np.isin(np.arange(N_CHANNELS), GROUPS['arms']), None, kE[idx])
        eng.tick(t); seen.append(eng.state); t += 1 / 15.0
    assert 'show' in seen and 'countin' in seen and 'attempt' in seen and 'done' in seen
    assert eng.result['scored'] and eng.cue['kind'] in ('correction', 'timing') and eng.cue['group'] in ('arms', None)
    types = [e['type'] for e in ev]
    assert 'move_shown' in types and 'attempt_started' in types and 'phrase_scored' in types and 'cue_decided' in types


def test_engine_freestyle_round_scores_without_a_ghost():
    key = synthetic_key(); kF, kM, kE, _ = phrase_slice(key, 0); pb = key['per_beat']; n = len(kF)
    eng = PhraseEngine(key, freestyle=True); t = 0.0; eng.start(t)
    assert eng.state == 'attempt' and eng.ghost_sample(t) is None
    while eng.state != 'done' and t < 60:
        idx = int(round(eng.attempt_t(t) / eng.beat_s * pb))
        if 0 <= idx < n: eng.on_features(t, kF[idx], None, kE[idx])
        eng.tick(t); t += 1 / 15.0
    r = eng.result
    assert r['freestyle'] and r['scored'] and 0 <= r['score'] <= 100 and 'groove' in r and eng.cue['kind'] in ('timing', 'correction', 'praise')


def test_choose_cue_branches():
    assert choose_cue(dict(scored=False))['kind'] == 'visibility'
    pb = [dict(beat=b, error_deg=10.0, offset_beats=0.6 if b in (1, 2) else 0.0, verdict='amber') for b in range(8)]
    r = dict(scored=True, timing=0.3, mean_offset_beats=0.6, weak_beats=[1, 2], score=50, per_group=dict(arms=10.0, legs=5.0, torso=2.0), per_beat=pb)
    assert choose_cue(r)['kind'] == 'timing'
    r = dict(scored=True, timing=0.9, mean_offset_beats=0.0, weak_beats=[], score=90, per_group=dict(arms=5.0, legs=5.0, torso=2.0))
    assert choose_cue(r)['kind'] == 'praise'
    r = dict(scored=True, timing=0.9, mean_offset_beats=0.0, weak_beats=[4, 5, 6], score=60, per_group=dict(arms=30.0, legs=8.0, torso=2.0))
    c = choose_cue(r); assert c['kind'] == 'correction' and c['group'] == 'arms' and c['beats'] == [4, 5, 6] and '5 to 7' in c['text']
