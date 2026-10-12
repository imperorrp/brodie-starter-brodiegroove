"""The phrase state machine: show, count-in, attempt, scored, feedback.

Time is injected (seconds, any monotonic clock), so the same engine runs live (wall clock) and offline (video clock).
The music clock starts at `start(t)`: beats are every 60 / (bpm * tempo) seconds; the round-trip lag `rtt_s` shifts the
scoring window so a caller who sees the ghost late is not scored late.

Per frame: on_frame(t, lm_img, lm_world) during an attempt turns landmarks into features and appends them with their
music time; tick(t) moves the state by time and scores at the end of the attempt. ghost_sample(t) gives the key skeleton
to draw (one beat ahead during the attempt), hud(t) gives everything the renderer needs. Events go to on_event(dict).
"""
import math
import numpy as np
from .features import features_from_landmarks, visibility_mask, hip_energy, resample_to_grid, resample_mask, GROUPS
from .scoring import score_phrase, GREEN_DEG, AMBER_DEG
from .reference import phrase_slice

STATES = ('idle', 'show', 'countin', 'attempt', 'scored', 'feedback', 'done')
FEEDBACK_S = 4.0                       # long enough for the result reveal (count-up, grade stamp, title)


def choose_cue(result, count_label='counts'):
    """One cue per phrase (feedback.structured_cue): visibility first, then timing, then the worst joint channel with its side and
    direction on the weakest run of beats, else praise. Keeps text/kind/group/beats for older callers."""
    from .feedback import structured_cue
    return structured_cue(result, count_label)


class PhraseEngine:
    def __init__(self, key, phrase_index=0, tempo=1.0, show_repeats=2, countin_beats=4, rtt_s=0.0, extra_beats=1, on_event=None,
                 skip_show=False, mirror_live=False, freestyle=False, focus=None, demo_only=False):
        """freestyle=True: no ghost, no reference comparison; the attempt is scored by scoring.groove_score (on-beat energy,
        variety, energy) over the same window. The key still supplies the BPM and the phrase length."""
        self.key = key; self.pi = phrase_index; self.tempo = float(tempo); self.show_repeats = int(show_repeats)
        self.countin_beats = int(countin_beats); self.rtt_s = float(rtt_s); self.extra_beats = int(extra_beats)
        self.on_event_cb = on_event; self.skip_show = bool(skip_show or freestyle); self.mirror_live = bool(mirror_live)
        self.mask_hook = None                                                    # optional mask -> mask (live.py: DANCE_DEPTH_GATE)
        self.freestyle = bool(freestyle); self.live_skel = []                    # (t_phrase, (33,2) normalised skeleton) during the attempt
        self.live_3d = []                                                        # (t_phrase, body_frame_signals) during the attempt (depth / facing)
        self.focus = focus                                                       # 'arms': score the arm channels only (the simplified breakdown)
        self.aborted = False; self.step = None
        self.demo_only = bool(demo_only)                                         # the reference only: show, then done (no count-in, no attempt)
        self.bpm = float(key['bpm']) * self.tempo; self.beat_s = 60.0 / self.bpm; self.pb = int(key['per_beat'])
        self.kF, self.kM, self.kE, self.kS = phrase_slice(key, phrase_index); self.n_key = len(self.kF); self.n_beats = self.n_key // self.pb
        self.state = 'idle'; self.t_start = None; self.t_state = None; self.events = []
        self.live = []                                    # (t_phrase, feats, mask, energy) during the attempt
        self.last_lm = None; self.last_t = None; self.prev_lm = None; self.prev_t = None
        self.result = None; self.cue = None; self.live_verdicts = [None] * self.n_beats; self._beat_done = -1
        self.live_channel_err = [None] * self.n_beats                           # signed per-channel error per beat (caller minus key), NaN unseen
        self.phrase_s = self.n_beats * self.beat_s
        self.show_s = 0.0 if skip_show else self.show_repeats * self.phrase_s
        self.countin_s = 0.0 if skip_show else self.countin_beats * self.beat_s
        self.attempt_s = (self.n_beats + self.extra_beats) * self.beat_s

    # ---- lifecycle ----
    def emit(self, type_, **f):
        ev = dict(type=type_, t=self.last_t, state=self.state, phrase=self.pi); ev.update(f); self.events.append(ev)
        if self.on_event_cb:
            try: self.on_event_cb(ev)
            except Exception: pass
        return ev

    def start(self, t):
        self.t_start = t; self.last_t = t; self.live = []; self.result = None; self.cue = None
        self.live_verdicts = [None] * self.n_beats; self._beat_done = -1
        self._enter('attempt' if self.skip_show else 'show', t)
        self.emit('move_shown', key=self.key['name'], bpm=round(self.bpm, 1), tempo=self.tempo, repeats=self.show_repeats, n_beats=self.n_beats)

    def _enter(self, state, t):
        self.state = state; self.t_state = t
        self.emit('phrase_state', to=state)
        if state == 'attempt': self.emit('attempt_started', n_beats=self.n_beats, extra_beats=self.extra_beats)

    # ---- clocks ----
    def t_music(self, t):
        return (t - self.t_start) - self.rtt_s

    def attempt_t(self, t):
        """Seconds since the attempt's first beat (negative before it), lag-compensated."""
        return self.t_music(t) - self.show_s - self.countin_s

    def beat_in_phrase(self, t):
        """Float beat position within the current phase (show loops over the phrase; attempt counts from 0)."""
        tm = self.t_music(t)
        if self.state == 'show': return (tm / self.beat_s) % self.n_beats
        if self.state == 'countin': return (tm - self.show_s) / self.beat_s - self.countin_beats
        if self.state == 'attempt': return self.attempt_t(t) / self.beat_s
        return float(self.n_beats)

    # ---- input ----
    def on_frame(self, t, lm_img, lm_world=None):
        self.last_t = t
        if lm_img is None: self.last_lm = None; return
        if self.mirror_live:
            from .features import mirror_landmarks
            lm_img, lm_world = mirror_landmarks(lm_img, lm_world)
        self.last_lm = np.asarray(lm_img, np.float32)
        if self.state == 'attempt':
            f = features_from_landmarks(lm_img, lm_world); m = visibility_mask(lm_img)
            if self.mask_hook is not None:                       # e.g. live.py's depth gate: joints the camera can't judge this frame
                try: m = self.mask_hook(m)
                except Exception: pass
            e = hip_energy(self.prev_lm, lm_img, (t - self.prev_t) if self.prev_t is not None else 0.0)
            from .features import skeleton2d_normalized, body_frame_signals
            self.live_skel.append((self.attempt_t(t), skeleton2d_normalized(lm_img)))
            if lm_world is not None:
                try:
                    b3 = body_frame_signals(lm_world, lm_img)
                    if b3 is not None: self.live_3d.append((self.attempt_t(t), b3))
                except Exception: pass
            self.on_features(t, f, m, e)
        self.prev_lm = self.last_lm; self.prev_t = t

    def on_features(self, t, feats, mask=None, energy=0.0):
        """Feature-level input (tests and motion-file replays)."""
        self.last_t = t
        if self.state != 'attempt': return
        mask = np.ones(len(feats), bool) if mask is None else np.asarray(mask, bool)
        self.live.append((self.attempt_t(t), np.asarray(feats, np.float32), mask, float(energy)))
        self._update_live_verdicts(t)

    def _update_live_verdicts(self, t):
        """Quick per-beat verdict as each beat completes (beat-indexed, no DTW), for the live timeline. The final verdicts come
        from the scorer at the end of the attempt."""
        b_done = int(math.floor(self.attempt_t(t) / self.beat_s)) - 1
        while self._beat_done < b_done and self._beat_done + 1 < self.n_beats:
            b = self._beat_done + 1; t0, t1 = b * self.beat_s, (b + 1) * self.beat_s
            rows = [(tp, f, m) for tp, f, m, e in self.live if t0 <= tp < t1]
            if len(rows) >= 2:
                grid = t0 + (t1 - t0) * np.arange(self.pb) / self.pb
                F = resample_to_grid([r[0] for r in rows], [r[1] for r in rows], grid); M = resample_mask([r[0] for r in rows], [r[2].astype(np.float32) for r in rows], grid)
                kF = self.kF[b * self.pb:(b + 1) * self.pb]; kM = self.kM[b * self.pb:(b + 1) * self.pb]; both = M & kM
                if self.focus == 'arms':
                    keep = np.zeros(both.shape[1], bool); keep[GROUPS['arms']] = True; both = both & keep[None, :]
                err = float(np.abs(F - kF)[both].mean()) if both.any() else None
                v = None if err is None else ('green' if err < GREEN_DEG else 'amber' if err < AMBER_DEG else 'red')
                cnt = both.sum(axis=0); sgn = np.where(both, F - kF, 0.0).sum(axis=0)
                self.live_channel_err[b] = np.where(cnt > 0, sgn / np.maximum(cnt, 1), np.nan)
            else:
                v = None
            self.live_verdicts[b] = v; self._beat_done = b
            self.emit('beat', beat=b, live_verdict=v, t_beat_end=self.t_start + self.rtt_s + self.show_s + self.countin_s + (b + 1) * self.beat_s)

    # ---- time ----
    def tick(self, t):
        self.last_t = t
        if self.state == 'idle' or self.state == 'done' or self.aborted: return
        tm = self.t_music(t)
        if self.state == 'show' and tm >= self.show_s:
            if self.demo_only: self._enter('done', t); return
            self._enter('countin' if self.countin_s > 0 else 'attempt', t)
        if self.state == 'countin' and tm >= self.show_s + self.countin_s:
            self._enter('attempt', t)
        if self.state == 'attempt' and self.attempt_t(t) >= self.attempt_s:
            self._score(); self._enter('scored', t)
        if self.state == 'scored':
            self._enter('feedback', t)
        if self.state == 'feedback' and t - self.t_state >= FEEDBACK_S:
            self._enter('done', t)

    def _score(self):
        n_live = self.n_key + self.extra_beats * self.pb
        grid = np.arange(n_live) * self.beat_s / self.pb
        if self.live:
            ts = [r[0] for r in self.live]
            F = resample_to_grid(ts, [r[1] for r in self.live], grid); M = resample_mask(ts, [r[2].astype(np.float32) for r in self.live], grid)
            E = resample_to_grid(ts, np.asarray([r[3] for r in self.live], np.float32), grid)[:, 0]
            # samples outside the live window are not evidence: mask them out
            lo, hi = min(ts), max(ts); M[(grid < lo - self.beat_s / self.pb) | (grid > hi + self.beat_s / self.pb)] = False
        else:
            F = np.zeros((n_live, self.kF.shape[1]), np.float32); M = np.zeros((n_live, self.kF.shape[1]), bool); E = np.zeros(n_live, np.float32)
        if self.focus == 'arms' and not self.freestyle:
            M = M.copy(); M[:, GROUPS['legs'] + GROUPS['torso']] = False
        if self.freestyle:
            from .scoring import groove_score
            phase = (grid / self.beat_s) % 1.0
            g = groove_score(E[:self.n_key], phase[:self.n_key], F[:self.n_key], per_beat=self.pb)
            visible = float(M.mean()) if M.size else 0.0
            self.result = dict(score=g['score'] if visible >= 0.6 else 0.0, spatial=g['variety'], timing=g['on_beat'], energy=g['energy'], scored=visible >= 0.6,
                               reason='' if visible >= 0.6 else 'not enough visible joints', visible_fraction=round(visible, 3), per_group={'arms': None, 'legs': None, 'torso': None},
                               per_beat=[dict(beat=b, error_deg=0.0, offset_beats=0.0, verdict='green' if g['score'] >= 60 else 'amber', worst_group='arms') for b in range(self.n_beats)],
                               weak_beats=[], max_combo=self.n_beats if g['score'] >= 60 else 0, display_score=g['score'], mean_error_deg=None, mean_offset_beats=None, groove=g, freestyle=True)
        else:
            self.result = score_phrase(F, M, E, self.kF, self.kM, self.kE, per_beat=self.pb)
        self.result['frames'] = len(self.live); self.result['key'] = self.key['name']; self.result['phrase'] = self.pi; self.result['tempo'] = self.tempo
        if not self.freestyle:
            try: self.result.update(self._extras(F, M, E))
            except Exception as e: self.result['extras_error'] = repr(e)
        if self.freestyle:
            g = self.result.get('groove', {})
            if not self.result['scored']: self.cue = dict(text='Step back so I can see all of you', kind='visibility', group=None, beats=[])
            elif g.get('on_beat', 0) < 0.4: self.cue = dict(text='Hit the beat harder', kind='timing', group=None, beats=[])
            elif g.get('variety', 0) < 0.4: self.cue = dict(text='Change it up, use the arms and the legs', kind='correction', group='arms', beats=[])
            elif g.get('energy', 0) < 0.4: self.cue = dict(text='Bigger. Move the hips', kind='correction', group='legs', beats=[])
            else: self.cue = dict(text='That is a groove', kind='praise', group=None, beats=[])
        else:
            self.cue = choose_cue(self.result)
        self.emit('attempt_ended', frames=len(self.live))
        self.emit('phrase_scored', **{k: v for k, v in self.result.items() if k != 'per_beat'}, per_beat=self.result['per_beat'])
        self.emit('cue_decided', **self.cue)

    def _extras(self, F, M, E):
        """Detail for richer coaching, from data already here: per-count 3D signals (median over the count, shifted by the mean timing
        offset), how big the arm and hip movement was against the reference (range of motion), and the hip-energy level."""
        out = {}
        off = float(self.result.get('mean_offset_beats') or 0.0)
        if self.live_3d:
            per = [dict() for _ in range(self.n_beats)]
            keys = ('arm_fwd_l', 'arm_fwd_r', 'torso_twist', 'facing_yaw')
            for b in range(self.n_beats):
                t0, t1 = (b + off) * self.beat_s, (b + 1 + off) * self.beat_s
                rows = [d for tp, d in self.live_3d if t0 <= tp < t1]
                for k in keys:
                    vals = [r[k] for r in rows if r.get(k) is not None]
                    per[b][k] = float(np.median(vals)) if len(vals) >= 2 else None
            out['body3d'] = per
        n = min(len(F), len(self.kF))
        size = {}
        for name, chans in (('arms', [2, 3, 10, 11]), ('hips', [4, 5])):
            ratios = []
            for c in chans:
                mk = self.kM[:n, c].astype(bool); ml = M[:n, c].astype(bool)
                if mk.sum() < 4 or ml.sum() < 4: continue
                kr = float(np.ptp(self.kF[:n, c][mk])); lr = float(np.ptp(F[:n, c][ml]))
                if kr >= 25.0: ratios.append(lr / kr)
            if ratios: size[name] = round(float(np.median(ratios)), 2)
        ke = float(np.mean(self.kE)) if len(self.kE) else 0.0; le = float(np.mean(E[:n])) if n else 0.0
        if ke > 0.05: size['energy'] = round(le / ke, 2)
        out['size'] = size
        return out

    def abort(self, t):
        """Stop the phrase now (pause or safety stop): no result, state done."""
        if self.state == 'done': return
        self.last_t = t; self.aborted = True; self.result = None; self.cue = None
        self.emit('phrase_aborted', at_state=self.state); self.state = 'done'; self.t_state = t

    # ---- output for the renderer ----
    def ghost_sample(self, t):
        """(33, 2) normalised skeleton of the key to draw now, or None. During the attempt the ghost leads by one beat."""
        if self.freestyle: return None
        if self.state == 'show':
            idx = int(math.floor(self.beat_in_phrase(t) * self.pb)) % self.n_key
        elif self.state == 'countin':
            idx = 0
        elif self.state == 'attempt':
            idx = int(math.floor((self.attempt_t(t) / self.beat_s + 1.0) * self.pb))
            if idx >= self.n_key or idx < 0: return None
        else:
            return None
        return self.kS[idx]

    def verdicts(self):
        if self.result is not None and self.result.get('scored'):
            return [r['verdict'] for r in self.result['per_beat']]
        return list(self.live_verdicts)

    def hud(self, t):
        bip = self.beat_in_phrase(t)
        cur = int(math.floor(bip)) if self.state == 'attempt' else (int(math.floor(bip)) if self.state == 'show' else -1)
        countdown = None
        if self.state == 'countin': countdown = max(1, int(math.ceil(-bip)))
        combo = 0
        for v in self.verdicts():
            if v == 'green': combo += 1
            else: combo = 0 if v is not None else combo
        running = None
        if self.state == 'attempt':
            done = [v for v in self.live_verdicts if v is not None]
            if done: running = round(100.0 * (sum(1.0 if v == 'green' else 0.5 if v == 'amber' else 0.0 for v in done) / len(done)), 0)
        countin_label = None
        if self.state == 'countin': countin_label = self.countin_beats + 1 + int(math.floor(bip + self.countin_beats)) if bip > -self.countin_beats else 5
        if countin_label is not None: countin_label = max(5, min(8, countin_label))
        return dict(state=self.state, key=self.key['name'], title=self.key.get('title'), phrase=self.pi, n_beats=self.n_beats, current_beat=cur, countdown=countdown,
                    verdicts=self.verdicts(), running_score=running, combo=combo, result=self.result, cue=self.cue, bpm=round(self.bpm, 1), tempo=self.tempo,
                    t=t, t_state=self.t_state, beat_pos=bip, countin_label=countin_label, step=self.step, focus=self.focus, freestyle=self.freestyle,
                    beat_s=self.beat_s)
