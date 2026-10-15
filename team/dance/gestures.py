"""Hands-free controls from the pose. The starter's echo guard mutes the caller's mic while Brodie's music plays, so a spoken
"pause" can never land mid-phrase; a held pose can.

  both hands up above the head, held still 1.0 s     -> 'ready'  (also 'resume' when paused)
  arms crossed above the head, still, held 1.2 s     -> 'pause'

The stillness rule (wrists barely moving) keeps arm-heavy moves (waacking, jumping jacks, a stretch) from triggering a pause or
a "yes": a dancer with arms overhead once answered an offer nobody had heard. Image landmarks,
unmirrored (MediaPipe's left wrist is the person's left; facing the camera it sits at the larger x, so crossed means L.x < R.x).

Phone cameras are noisy (live caller, 2026-10-15: the skeleton showed both hands up and no yes ever landed), so a hold is forgiving
where it can be: hands straight up count (wrists at least half a shoulder width apart, not a wide V), a wrist out of the top of a
close portrait frame falls back to its elbow, stillness is measured over 0.25 s rather than frame to frame (landmark jitter), and a
dropout of up to 0.4 s (a missed frame, a wrist flickering out of view) doesn't restart the hold. Moving arms still reset it.
"""
import numpy as np

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 11, 12, 13, 14, 15, 16, 23, 24
VIS = 0.5
HOLD = dict(ready=1.0, pause=1.2)
COOLDOWN_S = 2.0
STILL = 0.6                     # torso lengths per second
STILL_WINDOW_S = 0.25           # stillness is judged against the wrists this long ago
READY_GAP = 0.5                 # shoulder widths between the raised hands (hands straight up are about 1)
GRACE_S = 0.4                   # a dropout this short keeps a hold going (and a fired hold blocked)


def _hand(a, wr, el):
    """A raised hand's (x, y): the wrist, or the elbow when the wrist is out of view (hands up past the top of the frame)."""
    if a[wr, 3] >= VIS: return a[wr, :2]
    if a[el, 3] >= VIS: return a[el, :2]
    return None


def classify(lm):
    """'ready' | 'pause' | None for one frame's landmarks (33, >=4)."""
    if lm is None: return None
    a = np.asarray(lm, np.float32)
    if min(a[NOSE, 3], a[L_SH, 3], a[R_SH, 3]) < VIS: return None
    nose_y = a[NOSE, 1]; sw = abs(a[L_SH, 0] - a[R_SH, 0]) + 1e-6
    if min(a[L_WR, 3], a[R_WR, 3]) >= VIS and a[L_WR, 1] < nose_y and a[R_WR, 1] < nose_y:
        gap = a[L_WR, 0] - a[R_WR, 0]                # positive: uncrossed (left wrist on the image's right)
        if gap < 0 and -gap < 0.8 * sw: return 'pause'
        if gap > READY_GAP * sw: return 'ready'
        return None
    l, r = _hand(a, L_WR, L_EL), _hand(a, R_WR, R_EL)
    if l is None or r is None or not (l[1] < nose_y and r[1] < nose_y): return None
    return 'ready' if l[0] - r[0] > READY_GAP * sw else None


def _torso(a):
    return float(np.linalg.norm(0.5 * (a[L_SH, :2] + a[R_SH, :2]) - 0.5 * (a[L_HIP, :2] + a[R_HIP, :2]))) + 1e-6


class GestureDetector:
    def __init__(self):
        self.kind = None; self.since = None; self.last_fire = -1e9; self.fired = False
        self.hist = []                  # (t, landmarks) over the stillness window
        self.ok_t = None                # last frame that matched the hold in progress
        self.blocked = None             # a hold that already fired and is still held: it can't answer the next question
        self.down_t = None              # when a blocked hold was first seen down

    def reset(self):
        """Start fresh for a new question. A hold still in progress restarts; one that already fired (hands still up after picking
        a card) is blocked until the hands come down, so one gesture never answers two questions."""
        self.blocked = self.kind if self.fired else None; self.down_t = None
        self.kind = None; self.since = None; self.fired = False; self.ok_t = None

    def _moving(self, t, a):
        """Wrists moved faster than STILL against the frame about STILL_WINDOW_S ago (jitter averages out over the window)."""
        self.hist = [(tt, p) for tt, p in self.hist if t - tt <= 2 * STILL_WINDOW_S]
        old = [(tt, p) for tt, p in self.hist if t - tt >= STILL_WINDOW_S]
        self.hist.append((t, a))
        if not old: return False
        tt, p = old[-1]; d = 0.0
        for wr, el in ((L_WR, L_EL), (R_WR, R_EL)):                    # the same point in both frames: the wrist, else its elbow
            j = wr if min(p[wr, 3], a[wr, 3]) >= VIS else el if min(p[el, 3], a[el, 3]) >= VIS else None
            if j is not None: d = max(d, float(np.linalg.norm(a[j, :2] - p[j, :2])))
        return d / _torso(a) / (t - tt) > STILL

    def update(self, t, lm):
        """Feed one frame; returns 'ready' / 'pause' once when a hold completes, else None."""
        raw = k = classify(lm)
        moving = False
        if lm is None: self.hist = []
        else:
            moving = self._moving(t, np.asarray(lm, np.float32))
            if k in ('pause', 'ready') and moving: k = None
        if self.blocked is not None:
            if raw == self.blocked or lm is None: self.down_t = None; return None   # still the same raised arms (or no view of them)
            if self.down_t is None: self.down_t = t
            if t - self.down_t < GRACE_S: return None                           # down for real, not one flickering frame
            self.blocked = None; self.down_t = None
        if k is None and self.kind is not None and not moving and self.ok_t is not None and t - self.ok_t <= GRACE_S:
            return None                                                         # a short dropout: the hold goes on
        if k != self.kind: self.kind = k; self.since = t if k else None; self.fired = False
        if k is not None: self.ok_t = t
        if k is None or self.fired or t - self.last_fire < COOLDOWN_S: return None
        if t - self.since >= HOLD[k]:
            self.fired = True; self.last_fire = t; return k
        return None

    def progress(self, t):
        """(kind, 0..1) of the hold in progress, for the HUD ring; (None, 0) when idle."""
        if self.kind is None or self.since is None or self.fired or t - self.last_fire < COOLDOWN_S: return None, 0.0
        return self.kind, float(min(1.0, (t - self.since) / HOLD[self.kind]))
