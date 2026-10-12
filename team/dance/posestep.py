"""Pose by pose: the slow first look at a move, and freeze-frame practice.

At teaching speed (0.5x) one count lasts about a second, still too fast for a first look. Here each count's key pose is held for
`hold_s` (default 3.5 s, DANCE_POSE_HOLD_S) with a short eased move from the previous pose, so a beginner sees one position at a
time:

  mode='watch'   the ghost steps through the counts on its own (WATCH, pose by pose)
  mode='freeze'  the caller matches each pose: held within `match_deg` for `match_hold_s` -> a tick and the next pose early;
                 otherwise it moves on after `max_hold_s` (never stuck). Unscored, with a live hint for the furthest-off limb.

Same interface as engine.PhraseEngine where live.py and render.py touch it: start / tick / on_frame / abort / ghost_sample /
hud / state / step / result / cue / events. There is no music track: no count-in, no attempt window, no ghost lead.
"""
import math, os
import numpy as np
from .features import features_from_landmarks, visibility_mask, GROUPS
from .reference import phrase_slice

HOLD_S = float(os.environ.get('DANCE_POSE_HOLD_S', '3.5') or 3.5)


def _smooth(u):
    u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


class PoseStepEngine:
    def __init__(self, key, phrase_index=0, mode='watch', hold_s=None, ease_s=0.6, match_deg=25.0, match_hold_s=0.5, max_hold_s=4.0,
                 on_event=None, focus=None):
        self.key = key; self.pi = phrase_index; self.mode = mode if mode in ('watch', 'freeze') else 'watch'
        self.hold_s = float(hold_s if hold_s is not None else HOLD_S); self.ease_s = float(min(ease_s, self.hold_s * 0.4))
        self.match_deg = float(match_deg); self.match_hold_s = float(match_hold_s); self.max_hold_s = float(max(max_hold_s, self.hold_s))
        self.on_event_cb = on_event; self.focus = focus; self.step = None
        self.pb = int(key['per_beat']); self.kF, self.kM, self.kE, self.kS = phrase_slice(key, phrase_index)
        self.n_key = len(self.kF); self.n_beats = max(1, self.n_key // self.pb)
        self.idx = [min(self.n_key - 1, b * self.pb) for b in range(self.n_beats)]          # the on-the-count sample of each count
        self.bpm = float(key['bpm']); self.tempo = 0.0; self.beat_s = self.hold_s
        self.state = 'idle'; self.t_start = None; self.t_state = None; self.events = []; self.aborted = False
        self.cur = 0; self.t_cur = None; self.match_since = None; self.best_err = [None] * self.n_beats; self.err_now = None
        self.verdicts_ = [None] * self.n_beats; self.last_t = None; self.last_hint_t = -1e9; self._prof = None
        self.result = None; self.cue = None; self.live = []; self.live_skel = []                # PhraseEngine compatibility (never scored)
        self.live_verdicts = self.verdicts_; self.live_channel_err = [None] * self.n_beats
        self.extra_beats = 0; self.show_s = self.n_beats * self.hold_s; self.countin_s = 0.0; self.attempt_s = 0.0
        self.freestyle = False; self.demo_only = self.mode == 'watch'

    # ---- lifecycle ----
    def emit(self, type_, **f):
        ev = dict(type=type_, t=self.last_t, state=self.state, phrase=self.pi); ev.update(f); self.events.append(ev)
        if self.on_event_cb:
            try: self.on_event_cb(ev)
            except Exception: pass
        return ev

    def start(self, t):
        self.t_start = t; self.last_t = t; self.cur = 0; self.t_cur = t; self.state = 'show'; self.t_state = t
        self.emit('phrase_state', to='show')
        self.emit('move_shown', key=self.key['name'], bpm=round(self.bpm, 1), tempo=0.0, n_beats=self.n_beats, pose_step=self.mode, hold_s=self.hold_s)

    def abort(self, t):
        if self.state == 'done': return
        self.last_t = t; self.aborted = True; self.emit('phrase_aborted', at_state=self.state); self.state = 'done'; self.t_state = t

    def _advance(self, t, why):
        b = self.cur
        if self.mode == 'freeze' and self.verdicts_[b] is None:
            e = self.best_err[b]
            self.verdicts_[b] = 'green' if why == 'matched' else ('amber' if e is not None and e < self.match_deg * 1.4 else 'red' if e is not None else None)
        self.emit('pose_step', count=b, why=why, err=None if self.best_err[b] is None else round(self.best_err[b], 1))
        self.cur += 1; self.t_cur = t; self.match_since = None
        if self.cur >= self.n_beats:
            self.cur = self.n_beats - 1; self.state = 'done'; self.t_state = t; self.emit('phrase_state', to='done')

    # ---- input ----
    def on_frame(self, t, lm_img, lm_world=None):
        self.last_t = t
        if self.mode != 'freeze' or self.state != 'show' or lm_img is None: return
        f = features_from_landmarks(lm_img, lm_world); m = visibility_mask(lm_img)
        i = self.idx[self.cur]; kf = self.kF[i]; both = np.asarray(m, bool) & np.asarray(self.kM[i], bool)
        if self.focus == 'arms':
            keep = np.zeros(len(both), bool); keep[GROUPS['arms']] = True; both = both & keep
        if not both.any(): self.err_now = None; self.match_since = None; return
        d = np.where(both, f - kf, np.nan); err = float(np.nanmean(np.abs(d))); self.err_now = err
        b = self.cur
        if self.best_err[b] is None or err < self.best_err[b]: self.best_err[b] = err
        self.live_channel_err[b] = d
        if err < self.match_deg:
            if self.match_since is None: self.match_since = t
            if t - self.match_since >= self.match_hold_s: self.verdicts_[b] = 'green'; self._advance(t, 'matched')
        else:
            self.match_since = None
            if t - self.t_cur > 1.5 and t - self.last_hint_t > 2.5:             # a hint for the furthest-off limb, depth first
                from .feedback import _pick_channel, channel_cue, depth_issue, ref_profile
                hint = None
                if lm_world is not None:
                    try:
                        from .features import body_frame_signals
                        if self._prof is None: self._prof = ref_profile(self.key, self.pi)
                        sig = body_frame_signals(lm_world, lm_img)
                        for side in ('l', 'r'):
                            hint = hint or (depth_issue(sig, self._prof[min(b, len(self._prof) - 1)], side) if sig and self._prof else None)
                    except Exception:
                        hint = None
                if hint is None:
                    ch = _pick_channel(np.nan_to_num(d, nan=0.0))
                    if ch is not None: hint = channel_cue(ch, float(d[ch]))
                if hint: self.last_hint_t = t; self.emit('pose_hint', count=b, short=hint.get('short') or hint.get('text'), text=hint.get('text'))

    # ---- time ----
    def tick(self, t):
        self.last_t = t
        if self.state != 'show' or self.aborted: return
        limit = self.hold_s if self.mode == 'watch' else self.max_hold_s
        if t - self.t_cur >= limit: self._advance(t, 'held' if self.mode == 'watch' else 'timeout')

    # ---- output ----
    def ghost_sample(self, t):
        if self.state not in ('show', 'done'): return None
        cur = self.kS[self.idx[self.cur]]
        if self.cur == 0 or self.state == 'done' or self.t_cur is None: return cur
        u = (t - self.t_cur) / self.ease_s if self.ease_s > 0 else 1.0
        if u >= 1.0: return cur
        prev = self.kS[self.idx[self.cur - 1]]; k = _smooth(u)
        return (prev + (cur - prev) * k).astype(np.float32)

    def verdicts(self):
        return list(self.verdicts_)

    def hud(self, t):
        hold_p = 0.0 if self.t_cur is None else min(1.0, (t - self.t_cur) / (self.hold_s if self.mode == 'watch' else self.max_hold_s))
        match_p = 0.0 if self.match_since is None else min(1.0, (t - self.match_since) / self.match_hold_s)
        combo = 0
        for v in self.verdicts_:
            if v == 'green': combo += 1
            elif v is not None: combo = 0
        return dict(state='show' if self.state != 'done' else 'done', key=self.key['name'], title=self.key.get('title'), phrase=self.pi, n_beats=self.n_beats,
                    current_beat=self.cur, countdown=None, verdicts=self.verdicts(), running_score=None, combo=combo, result=None, cue=None,
                    bpm=round(self.bpm, 1), tempo=0.15, t=t, t_state=self.t_state, beat_pos=float(self.cur), countin_label=None, step=self.step,
                    focus=self.focus, freestyle=False, beat_s=self.hold_s, pose_step=self.mode, hold_p=hold_p, match_p=match_p,
                    pose_err=None if self.err_now is None else round(self.err_now, 1))
