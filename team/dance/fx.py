"""Short-lived HUD animations: judgment pops, tips, combo bursts, confetti, level-up, achievement toasts, grade stamps, K.O.

The session adds effects with the time they start; the renderer asks for the active ones at the frame's time and draws them by
age. Nothing is stateful per frame (confetti positions are analytic from a seed), so dropped frames never desync anything.
"""
import math
import numpy as np


class Fx:
    __slots__ = ('kind', 't0', 'dur', 'data')

    def __init__(self, kind, t0, dur, data):
        self.kind = kind; self.t0 = float(t0); self.dur = float(dur); self.data = data

    def age(self, t): return t - self.t0
    def u(self, t): return max(0.0, min(1.0, (t - self.t0) / self.dur)) if self.dur > 0 else 1.0


DURATIONS = dict(judgment=0.65, tip=1.6, combo=0.9, confetti=2.2, levelup=2.4, achievement=2.6, grade_stamp=0.6, flash=0.25, ko=1.8,
                 mvp=2.2, safety=0.0)


class FxQueue:
    def __init__(self, limit=64):
        self.items = []; self.limit = limit

    def add(self, kind, t, dur=None, **data):
        self.items.append(Fx(kind, t, DURATIONS.get(kind, 1.0) if dur is None else dur, data))
        if len(self.items) > self.limit: self.items = self.items[-self.limit:]
        return self.items[-1]

    def active(self, t, kinds=None):
        self.items = [f for f in self.items if t - f.t0 <= f.dur + 0.05]
        return [f for f in self.items if f.t0 <= t and (kinds is None or f.kind in kinds)]

    def clear(self, kinds=None):
        self.items = [] if kinds is None else [f for f in self.items if f.kind not in kinds]

    def has(self, kind, t):
        return any(f.kind == kind for f in self.active(t))


# ---- easing ----
def ease_out_cubic(u): return 1.0 - (1.0 - u) ** 3


def ease_out_back(u, s=1.70158):
    u = u - 1.0
    return u * u * ((s + 1) * u + s) + 1.0


def pulse(t, hz=2.0): return 0.5 + 0.5 * math.sin(2 * math.pi * hz * t)


def pop_scale(age, grow=0.16):
    """0.6 -> 1.15 -> 1.0 over `grow` seconds, then 1.0."""
    if age >= grow: return 1.0
    u = age / grow
    return 0.6 + 0.55 * math.sin(u * math.pi * 0.5) if u < 0.7 else 1.15 - 0.15 * (u - 0.7) / 0.3


# ---- confetti ----
CONFETTI_COLORS = [(61, 255, 217), (168, 61, 255), (255, 224, 61), (61, 190, 255), (245, 245, 245)]   # BGR: volt, pink, cyan, amber, white


def confetti_particles(seed, n, size, t, origin=None):
    """[(x, y, angle_deg, colour, w, h, alpha)] for confetti of age t seconds on a canvas of `size`."""
    cw, ch = size; rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    ox, oy = origin if origin is not None else (cw * 0.5, ch * 0.42)
    ang = rng.uniform(-math.pi, 0, n); spd = rng.uniform(0.35, 1.0, n) * ch * 0.9
    vx = np.cos(ang) * spd * rng.uniform(0.6, 1.2, n); vy = np.sin(ang) * spd
    g = ch * 1.1
    x = ox + vx * t; y = oy + vy * t + 0.5 * g * t * t
    rot = rng.uniform(0, 360, n) + t * rng.uniform(-540, 540, n)
    col = rng.integers(0, len(CONFETTI_COLORS), n); w = rng.integers(8, 16, n); h = rng.integers(4, 8, n)
    alpha = max(0.0, 1.0 - max(0.0, t - 1.4) / 0.8)
    return [(float(x[i]), float(y[i]), float(rot[i]), CONFETTI_COLORS[col[i]], int(w[i]), int(h[i]), alpha) for i in range(n)
            if -20 <= x[i] <= cw + 20 and -20 <= y[i] <= ch + 20]
