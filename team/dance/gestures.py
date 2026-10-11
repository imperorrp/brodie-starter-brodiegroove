"""Hands-free controls from the pose. The starter's echo guard mutes the caller's mic while Brodie's music plays, so a spoken
"pause" can never land mid-phrase; a held pose can.

  both hands up, apart, held 1.0 s                    -> 'ready'  (also 'resume' when paused)
  arms crossed above the head, still, held 1.2 s     -> 'pause'

The stillness rule (wrists barely moving) keeps arm-heavy moves such as waacking from triggering a pause. Image landmarks,
unmirrored (MediaPipe's left wrist is the person's left; facing the camera it sits at the larger x, so crossed means L.x < R.x).
"""
import numpy as np

NOSE, L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 0, 11, 12, 15, 16, 23, 24
VIS = 0.5
HOLD = dict(ready=1.0, pause=1.2)
COOLDOWN_S = 2.0
STILL = 0.6                     # torso lengths per second


def classify(lm):
    """'ready' | 'pause' | None for one frame's landmarks (33, >=4)."""
    if lm is None: return None
    a = np.asarray(lm, np.float32)
    if min(a[NOSE, 3], a[L_SH, 3], a[R_SH, 3], a[L_WR, 3], a[R_WR, 3]) < VIS: return None
    nose_y = a[NOSE, 1]; sw = abs(a[L_SH, 0] - a[R_SH, 0]) + 1e-6
    if not (a[L_WR, 1] < nose_y and a[R_WR, 1] < nose_y): return None
    gap = a[L_WR, 0] - a[R_WR, 0]                    # positive: uncrossed (left wrist on the image's right)
    if gap < 0 and -gap < 0.8 * sw: return 'pause'
    if gap > 1.2 * sw: return 'ready'
    return None


def _torso(a):
    return float(np.linalg.norm(0.5 * (a[L_SH, :2] + a[R_SH, :2]) - 0.5 * (a[L_HIP, :2] + a[R_HIP, :2]))) + 1e-6


class GestureDetector:
    def __init__(self):
        self.kind = None; self.since = None; self.last_fire = -1e9; self.prev = None; self.prev_t = None; self.fired = False

    def reset(self):
        self.kind = None; self.since = None; self.fired = False

    def update(self, t, lm):
        """Feed one frame; returns 'ready' / 'pause' once when a hold completes, else None."""
        k = classify(lm)
        if k == 'pause' and self.prev is not None and self.prev_t is not None and t > self.prev_t:
            a = np.asarray(lm, np.float32); p = self.prev
            speed = max(np.linalg.norm(a[L_WR, :2] - p[L_WR, :2]), np.linalg.norm(a[R_WR, :2] - p[R_WR, :2])) / _torso(a) / (t - self.prev_t)
            if speed > STILL: k = None
        self.prev = None if lm is None else np.asarray(lm, np.float32); self.prev_t = t
        if k != self.kind: self.kind = k; self.since = t if k else None; self.fired = False
        if k is None or self.fired or t - self.last_fire < COOLDOWN_S: return None
        if t - self.since >= HOLD[k]:
            self.fired = True; self.last_fire = t; return k
        return None

    def progress(self, t):
        """(kind, 0..1) of the hold in progress, for the HUD ring; (None, 0) when idle."""
        if self.kind is None or self.since is None or self.fired or t - self.last_fire < COOLDOWN_S: return None, 0.0
        return self.kind, float(min(1.0, (t - self.since) / HOLD[self.kind]))
