"""Score an attempt against a key, beat by beat.

Inputs are feature sequences sampled on a beat grid (PER_BEAT samples per beat, see features.resample_to_grid): the key's
samples for one phrase (n_key, 14) and the live samples over a slightly longer window (n_live >= n_key, 14), with per-sample
visibility masks and a hip-energy scalar per sample.

Alignment: a banded dynamic time warp (Sakoe-Chiba band of about one beat) between key and live. Spatial error is measured on
the aligned pairs (so a late dancer with perfect shapes is not punished twice); timing offsets come from the warp path (live
index minus key index, in beats). Energy compares the hip-speed envelope so "marking" a move scores lower than dancing it.

Score (0 to 100) = 60 * spatial + 30 * timing + 10 * energy, where
  spatial = exp(-ln2 * mean_error_deg / 15)      (15 degrees of mean error = half marks)
  timing  = fraction of beats within +-0.25 beat whose shape error is also under 30 degrees
  energy  = correlation of the per-beat hip-energy envelopes, clipped to [0, 1] (or a level match when the key barely moves)
Per-beat verdicts: green (error < 15 and |offset| <= 0.25), amber (error < 30 and |offset| <= 0.5), red otherwise.
Combo: longest run of green beats; a display multiplier of up to 1.5x rewards it without touching the raw score.
"""
import math
import os as _os
import numpy as np
from .features import GROUPS, N_CHANNELS

PER_BEAT = 4
# 2026-10-11, after the first real webcam takes (a person roughly following scored 24 to 36): the 15-degree half-marks were
# tuned on clean clips and sit inside MediaPipe's own jitter on a webcam. Loosened so "roughly following" lands around 50 to 70
# while the clean reference still scores 100 and random noise stays under 50. Sweep on the self-test (identity / arms perturbed /
# noise): 15-30-15 -> 100/68/35 (old), 20-32-18 -> 100/70/41 (chosen), 22-40-20 -> 100/72/62 (too loose). Env overrides below.
GREEN_DEG, AMBER_DEG = float(_os.environ.get('DANCE_GREEN_DEG', '18') or 18), float(_os.environ.get('DANCE_AMBER_DEG', '32') or 32)
GREEN_OFF, AMBER_OFF = 0.35, 0.6
HALF_MARKS_DEG = float(_os.environ.get('DANCE_HALF_MARKS_DEG', '20') or 20)
W_SPATIAL, W_TIMING, W_ENERGY = 0.6, 0.3, 0.1
MIN_VISIBLE_FRACTION = float(_os.environ.get('DANCE_MIN_VISIBLE', '0.6') or 0.6)   # whole-body gate
PARTIAL_ARMS_FRACTION = float(_os.environ.get('DANCE_PARTIAL_ARMS', '0.7') or 0.7)  # upper-body mode: arm channels visible this much -> score the arms alone


def dtw_band(A, B, band, mask_a=None, mask_b=None):
    """Banded DTW between A (n, C) and B (m, C). Cost per pair = mean |A_i - B_j| over channels visible in both.
    Returns (total_cost, path) where path is a list of (i, j) from (0, 0) to (n-1, m-1)."""
    A = np.asarray(A, np.float32); B = np.asarray(B, np.float32); n, m = len(A), len(B)
    if n == 0 or m == 0: return float('inf'), []
    ma = np.ones((n, A.shape[1]), bool) if mask_a is None else np.asarray(mask_a, bool)
    mb = np.ones((m, B.shape[1]), bool) if mask_b is None else np.asarray(mask_b, bool)
    band = max(int(band), abs(n - m) + 1)
    D = np.full((n + 1, m + 1), np.inf, np.float64); D[0, 0] = 0.0
    step = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        jc = int(round((i - 1) * (m - 1) / max(n - 1, 1)))
        lo = max(1, jc - band + 1); hi = min(m, jc + band + 1)
        for j in range(lo, hi + 1):
            both = ma[i - 1] & mb[j - 1]
            if both.any(): c = float(np.mean(np.abs(A[i - 1][both] - B[j - 1][both])))
            else: c = 0.0
            cands = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = int(np.argmin(cands)); D[i, j] = c + cands[k]; step[i, j] = k
    if not np.isfinite(D[n, m]): return float('inf'), []
    i, j = n, m; path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1)); k = step[i, j]
        if k == 0: i -= 1; j -= 1
        elif k == 1: i -= 1
        else: j -= 1
    path.reverse()
    return float(D[n, m]), path


def _per_beat(values_per_sample, n_beats, per_beat=PER_BEAT, reduce=np.mean):
    out = np.zeros(n_beats, np.float32)
    for b in range(n_beats):
        seg = values_per_sample[b * per_beat:(b + 1) * per_beat]
        out[b] = reduce(seg) if len(seg) else 0.0
    return out


def energy_term(live_e, key_e):
    live_e = np.asarray(live_e, np.float32); key_e = np.asarray(key_e, np.float32)
    n = min(len(live_e), len(key_e))
    if n < 2: return 1.0
    live_e = live_e[:n]; key_e = key_e[:n]
    if key_e.std() < 1e-4 or live_e.std() < 1e-4:
        lvl = abs(float(live_e.mean()) - float(key_e.mean())) / (float(key_e.mean()) + 1e-3)
        return float(max(0.0, 1.0 - min(1.0, lvl)))
    r = float(np.corrcoef(live_e, key_e)[0, 1])
    return float(max(0.0, min(1.0, r)))


def score_phrase(live, live_mask, live_energy, key, key_mask, key_energy, per_beat=PER_BEAT, band_beats=1.0):
    """live: (n_live, 14) samples on the beat grid (window may be longer than the key); key: (n_key, 14).
    Returns a dict with score, the three terms, per-group errors, per-beat rows and the weak beats."""
    live = np.asarray(live, np.float32); key = np.asarray(key, np.float32)
    n_key = len(key); n_beats = n_key // per_beat
    live_mask = np.ones_like(live, bool) if live_mask is None else np.asarray(live_mask, bool)
    key_mask = np.ones_like(key, bool) if key_mask is None else np.asarray(key_mask, bool)
    visible_fraction = float(live_mask.mean()) if live_mask.size else 0.0
    arms_fraction = float(live_mask[:, GROUPS['arms']].mean()) if live_mask.size else 0.0
    # upper-body mode: a desk webcam never sees the legs; the arms alone are scored when they are well visible and the key uses them
    partial = visible_fraction < MIN_VISIBLE_FRACTION and arms_fraction >= PARTIAL_ARMS_FRACTION
    if n_key == 0 or len(live) == 0 or (visible_fraction < MIN_VISIBLE_FRACTION and not partial):
        return dict(score=0.0, spatial=0.0, timing=0.0, energy=0.0, scored=False, reason='not enough visible joints',
                    visible_fraction=visible_fraction, per_group={g: None for g in GROUPS}, per_beat=[], weak_beats=[], max_combo=0,
                    display_score=0.0, mean_error_deg=None, mean_offset_beats=None, partial=False)
    if partial:                                   # legs and hips are guesses when out of frame: drop them from the comparison
        live_mask = live_mask.copy(); live_mask[:, GROUPS['legs'] + GROUPS['torso']] = False
    cost, path = dtw_band(key, live, band=int(round(band_beats * per_beat)), mask_a=key_mask, mask_b=live_mask)
    # aligned error per key sample (mean over channels visible in both) and matched live index (median over the path)
    err_s = np.zeros(n_key, np.float32); match = np.full(n_key, np.nan, np.float32)
    ch_err = np.zeros((n_key, N_CHANNELS), np.float32); ch_cnt = np.zeros((n_key, N_CHANNELS), np.float32)
    ch_sgn = np.zeros((n_key, N_CHANNELS), np.float32)                 # live minus key (signed), for directional cues
    buckets = [[] for _ in range(n_key)]
    for i, j in path: buckets[i].append(j)
    for i in range(n_key):
        js = buckets[i] or [min(i, len(live) - 1)]
        match[i] = float(np.median(js))
        both = key_mask[i] & live_mask[int(round(match[i]))]
        d = np.abs(key[i] - live[int(round(match[i]))])
        ch_err[i][both] = d[both]; ch_cnt[i][both] = 1.0
        ch_sgn[i][both] = (live[int(round(match[i]))] - key[i])[both]
        err_s[i] = float(d[both].mean()) if both.any() else 0.0
    off_s = (match - np.arange(n_key)) / float(per_beat)            # live index minus key index, in beats (positive = late)
    err_b = _per_beat(err_s, n_beats, per_beat); off_b = _per_beat(off_s, n_beats, per_beat)
    spatial = float(math.exp(-math.log(2.0) * float(err_b.mean()) / HALF_MARKS_DEG))
    # a beat is "on time" only if its shape is at least roughly right: timing credit for the wrong move is meaningless
    timing = float(np.mean((np.abs(off_b) <= GREEN_OFF) & (err_b < AMBER_DEG)))
    energy = energy_term(_per_beat(np.asarray(live_energy, np.float32), n_beats, per_beat) if live_energy is not None else [],
                         _per_beat(np.asarray(key_energy, np.float32), n_beats, per_beat) if key_energy is not None else [])
    if partial: energy = 0.0; score = 100.0 * ((W_SPATIAL + W_ENERGY) * spatial + W_TIMING * timing)   # hip energy needs hips
    else: score = 100.0 * (W_SPATIAL * spatial + W_TIMING * timing + W_ENERGY * energy)
    per_group = {}
    for g, idx in GROUPS.items():
        cnt = ch_cnt[:, idx].sum()
        per_group[g] = float(ch_err[:, idx].sum() / cnt) if cnt > 0 else None
    def group_error_in_beat(b, idx):
        seg_err = ch_err[b * per_beat:(b + 1) * per_beat][:, idx]; seg_cnt = ch_cnt[b * per_beat:(b + 1) * per_beat][:, idx]
        return float(seg_err[seg_cnt > 0].mean()) if (seg_cnt > 0).any() else 0.0
    def channel_signed_in_beat(b):
        seg = ch_sgn[b * per_beat:(b + 1) * per_beat]; cnt = ch_cnt[b * per_beat:(b + 1) * per_beat]
        n = cnt.sum(axis=0); out = np.where(n > 0, seg.sum(axis=0) / np.maximum(n, 1), np.nan)
        return out
    rows = []; weak = []; combo = 0; max_combo = 0; channel_signed = []
    for b in range(n_beats):
        e, o = float(err_b[b]), float(off_b[b])
        if e < GREEN_DEG and abs(o) <= GREEN_OFF: v = 'green'; combo += 1; max_combo = max(max_combo, combo)
        elif e < AMBER_DEG and abs(o) <= AMBER_OFF: v = 'amber'; combo = 0
        else: v = 'red'; combo = 0
        if v != 'green': weak.append(b)
        g_err = {g: group_error_in_beat(b, idx) for g, idx in GROUPS.items()}
        cs = channel_signed_in_beat(b); channel_signed.append([None if not np.isfinite(x) else round(float(x), 1) for x in cs])
        wc = int(np.nanargmax(np.abs(cs))) if np.isfinite(cs).any() else None
        rows.append(dict(beat=b, error_deg=round(e, 1), offset_beats=round(o, 2), verdict=v, worst_group=max(g_err, key=g_err.get),
                         worst_channel=wc, signed_deg=(round(float(cs[wc]), 1) if wc is not None else None)))
    display = min(100.0, score * (1.0 + min(0.5, 0.05 * max_combo)))
    return dict(score=round(score, 1), spatial=round(spatial, 3), timing=round(timing, 3), energy=round(energy, 3), scored=True,
                reason='', visible_fraction=round(visible_fraction, 3), per_group={g: (round(v, 1) if v is not None else None) for g, v in per_group.items()},
                per_beat=rows, weak_beats=weak, max_combo=max_combo, display_score=round(display, 1), partial=bool(partial),
                mean_error_deg=round(float(err_b.mean()), 1), mean_offset_beats=round(float(off_b.mean()), 2), dtw_cost=round(cost, 2),
                channel_signed=channel_signed)


def groove_score(live_energy_per_sample, beat_phase_per_sample, live_feats, per_beat=PER_BEAT):
    """Reference-free round: on-beat energy (hip speed peaks near beat boundaries), variety (distinct arm/leg configurations).
    Returns dict(score, on_beat, variety, energy)."""
    e = np.asarray(live_energy_per_sample, np.float32); ph = np.asarray(beat_phase_per_sample, np.float32)
    F = np.asarray(live_feats, np.float32)
    if len(e) < per_beat * 2: return dict(score=0.0, on_beat=0.0, variety=0.0, energy=0.0)
    near = (ph < 0.15) | (ph > 0.85)
    on_beat = float(e[near].mean() / (e.mean() + 1e-6)) if near.any() else 0.0
    on_beat = max(0.0, min(1.0, (on_beat - 0.8) / 0.8))
    q = np.round(F[:, [0, 1, 10, 11, 13]] / 30.0)                   # coarse arm configuration bins (feet are ignored)
    variety = len({tuple(r) for r in q.tolist()}) / max(1.0, len(q) / per_beat)
    variety = max(0.0, min(1.0, variety / 2.0))
    energy = max(0.0, min(1.0, float(e.mean()) / 1.5))
    score = 100.0 * (0.4 * on_beat + 0.3 * variety + 0.3 * energy)
    return dict(score=round(score, 1), on_beat=round(on_beat, 3), variety=round(variety, 3), energy=round(energy, 3))
