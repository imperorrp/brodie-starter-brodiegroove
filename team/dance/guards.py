"""Things that go wrong with one phone in a living room, caught early:

  FrameBudget   the per-frame work (pose + logic + drawing) must fit 1/fps. If the median over the last 5 s runs over 55 ms (15 fps),
                step down: first the lite pose model, then 12 fps. Logged once per step.
  CameraShift   the phone moved or fell: the whole picture jumped. A dancer jumping moves only part of the picture, so this looks at the
                global shift of a tiny grayscale frame (phase correlation; the background dominates). A shift over SHIFT_FRAC of the width
                with a confident peak, or the picture's brightness layout changing all at once, means the camera moved.
  VideoFreeze   the caller's video stopped arriving: under FREEZE_UNJUDGED_S nothing happens; over it the counts are unjudged; over
                FREEZE_ABORT_S the try is stopped (unscored) with a line.
"""
import numpy as np

BUDGET_MS = 55.0
SHIFT_FRAC = 0.08
FREEZE_UNJUDGED_S = 1.5
FREEZE_ABORT_S = 4.0


class FrameBudget:
    def __init__(self, window_s=5.0, budget_ms=BUDGET_MS):
        self.window_s = window_s; self.budget_ms = budget_ms; self.samples = []; self.level = 0; self.since = None

    def add(self, t, ms):
        """One frame's work in ms. Returns None, 'lite' (switch the pose model) or 'fps12' (lower the frame rate), at most once each."""
        self.samples.append((t, float(ms))); self.samples = [(tt, m) for tt, m in self.samples if t - tt <= self.window_s]
        if len(self.samples) < 20 or t - self.samples[0][0] < 0.8 * self.window_s: return None
        med = float(np.median([m for _, m in self.samples]))
        if med <= self.budget_ms: return None
        if self.level == 0: self.level = 1; self.samples = []; return 'lite'
        if self.level == 1: self.level = 2; self.samples = []; return 'fps12'
        return None


class CameraShift:
    def __init__(self, size=(64, 48)):
        self.size = size; self.prev = None; self.prev_t = None

    def update(self, t, bgr):
        """True when the whole picture jumped since the last frame (the phone moved)."""
        import cv2
        if bgr is None: return False
        g = cv2.cvtColor(cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev, self.prev = self.prev, g; pt, self.prev_t = self.prev_t, t
        if prev is None or pt is None or t - pt > 0.5: return False
        (dx, dy), resp = cv2.phaseCorrelate(prev, g)
        moved = resp > 0.3 and np.hypot(dx / self.size[0], dy / self.size[1]) > SHIFT_FRAC
        # a fall or a big tilt: most of the picture changes at once (the correlation peak collapses with a large difference)
        diff = float(np.mean(np.abs(g - prev))) / 255.0
        return bool(moved or (diff > 0.25 and resp < 0.1))


class VideoFreeze:
    @staticmethod
    def status(now, last_frame_t):
        """'ok' / 'unjudged' / 'abort' for the time since the last caller frame."""
        gap = now - last_frame_t if last_frame_t and last_frame_t > 0 else 0.0
        if gap > FREEZE_ABORT_S: return 'abort'
        if gap > FREEZE_UNJUDGED_S: return 'unjudged'
        return 'ok'
