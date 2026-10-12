"""One Euro filter (Casiez et al. 2012): smooths jittery signals (MediaPipe depth, body yaw) while staying responsive to real,
fast movement. Low cutoff when the signal is still, higher when it moves fast.

  f = OneEuro(min_cutoff=1.0, beta=0.05); y = f(x, t)        (x: float or ndarray, t: seconds)
"""
import math
import numpy as np


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff); self.beta = float(beta); self.d_cutoff = float(d_cutoff)
        self.x = None; self.dx = None; self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff); return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x, t):
        x = np.asarray(x, np.float64)
        if self.x is None or self.t is None or t <= self.t:
            self.x = x; self.dx = np.zeros_like(x); self.t = t
            return x if x.ndim else float(x)
        dt = t - self.t; self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt); self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self.x = a * x + (1 - a) * self.x
        return self.x if self.x.ndim else float(self.x)

    def reset(self): self.x = None; self.dx = None; self.t = None
