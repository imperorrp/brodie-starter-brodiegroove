"""Depth from one phone camera: did the body turn, or did a limb move toward the camera? Is the dancer facing away?

A flat picture can't tell "reach forward" from "reach to the side, turned", and MediaPipe's depth (z) is noisy at 2-3 m. Instead of
trusting z alone, a short calibration in the camera check (the dancer standing frontal and still) measures how wide the shoulders
and hips look and how long each arm segment looks *for this person at this distance*. Afterwards:

  how far turned   the shoulder (and hip) width relative to calibration: a turn of yaw shrinks both by cos(yaw), whatever the distance
                   (both widths and the torso length scale together)
  which way        the sign of the shoulders' depth difference, only once it clearly exceeds the calibrated depth noise for a few
                   frames (below ~15 degrees the direction doesn't matter)
  facing away      face landmarks hidden while the shoulders are clearly seen, or a turn that went past 70 degrees and kept going
                   with the face gone; MediaPipe then often swaps left and right, so lr_swapped is set
  limb toward me   a segment that looks much shorter than calibrated while the body is frontal points at or away from the camera
                   (the wrist's depth against the shoulder's says which); a rotation shortens both arms' horizontal parts together
  ambiguous        channels whose segment is strongly foreshortened with unreliable depth: tips and scoring can leave them out

  c = Calib(); c.add(lm_img, lm_world, aspect) per frame -> c.ready
  o = Orient(calib); o.update(lm_img, lm_world, aspect, t) -> dict(yaw, yaw_mag, side, away, lr_swapped, toward={'l','r'}, ambiguous, conf)
"""
import numpy as np

from .filters import OneEuro

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
FACE = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
SEGS = dict(upper_l=(L_SH, L_EL), fore_l=(L_EL, L_WR), upper_r=(R_SH, R_EL), fore_r=(R_EL, R_WR))
ARM_CHANNELS = dict(l=(0, 2, 10), r=(1, 3, 11))            # elbow, shoulder, arm elevation (features.CHANNELS)
CAL_FRAMES = 20
TOWARD_FS = 0.65
AMBIG_FS = 0.55


def _xy(lm, j, aspect): return np.array([lm[j, 0] * aspect, lm[j, 1]], np.float32)


def _torso2d(lm, aspect):
    sh = 0.5 * (_xy(lm, L_SH, aspect) + _xy(lm, R_SH, aspect)); hp = 0.5 * (_xy(lm, L_HIP, aspect) + _xy(lm, R_HIP, aspect))
    return float(np.linalg.norm(sh - hp)) + 1e-6


def _widths(lm, aspect):
    t = _torso2d(lm, aspect)
    return float(np.linalg.norm(_xy(lm, L_SH, aspect) - _xy(lm, R_SH, aspect))) / t, float(np.linalg.norm(_xy(lm, L_HIP, aspect) - _xy(lm, R_HIP, aspect))) / t, t


class Calib:
    """The dancer's frontal proportions at their distance: median over CAL_FRAMES good frames (facing, still, face visible)."""
    def __init__(self):
        self.rows = []; self.ready = False; self.r_sh = self.r_hip = None; self.seg = {}; self.z_noise = 0.03

    def add(self, lm_img, lm_world=None, aspect=1.0):
        if self.ready or lm_img is None: return self.ready
        a = np.asarray(lm_img, np.float32)
        if min(a[[L_SH, R_SH, L_HIP, R_HIP], 3]) < 0.6 or float(np.mean(a[list(FACE), 3])) < 0.6: return False
        if abs(a[L_SH, 1] - a[R_SH, 1]) > 0.03: return False              # shoulders level: not leaning or turned
        rs, rh, t = _widths(a, aspect)
        segs = {k: float(np.linalg.norm(_xy(a, i, aspect) - _xy(a, j, aspect))) / t for k, (i, j) in SEGS.items()}
        z = None if lm_world is None else float(np.asarray(lm_world, np.float32)[L_SH, 2] - np.asarray(lm_world, np.float32)[R_SH, 2])
        self.rows.append((rs, rh, segs, z))
        if len(self.rows) >= CAL_FRAMES:
            R = self.rows; self.r_sh = float(np.median([r[0] for r in R])); self.r_hip = float(np.median([r[1] for r in R]))
            self.seg = {k: float(np.percentile([r[2][k] for r in R], 75)) for k in SEGS}
            zs = [r[3] for r in R if r[3] is not None]
            if len(zs) >= 5: self.z_noise = max(0.01, float(np.std(zs)))
            self.ready = True
        return self.ready


class Orient:
    def __init__(self, calib=None):
        self.calib = calib or Calib(); self.f_mag = OneEuro(min_cutoff=1.5, beta=0.02)
        self.side = 0; self.side_run = 0; self.away_since = None; self.away = False; self.peak_side = 0; self.peak_mag = 0.0
        self.toward_run = dict(l=[], r=[])

    def update(self, lm_img, lm_world=None, aspect=1.0, t=0.0):
        out = dict(yaw=0.0, yaw_mag=0.0, side=0, away=False, lr_swapped=False, toward={}, ambiguous=set(), conf=0.0)
        if lm_img is None: return out
        a = np.asarray(lm_img, np.float32); c = self.calib
        if not c.ready: c.add(a, lm_world, aspect)
        if min(a[[L_SH, R_SH], 3]) < 0.5: return out
        hips = min(a[[L_HIP, R_HIP], 3]) >= 0.5
        if not hips: return out                                     # no hips (seated, close): the torso length is unknown, so no turn estimate
        rs, rh, tor = _widths(a, aspect)
        r0s = c.r_sh or 0.95; r0h = c.r_hip or 0.55                        # typical adult proportions until calibrated
        cs = float(np.clip(rs / r0s, 0.0, 1.0)); ch = float(np.clip(rh / r0h, 0.0, 1.0))
        cosv = 0.6 * cs + 0.4 * ch if hips else cs
        mag = float(np.degrees(np.arccos(np.clip(cosv, 0.0, 1.0))))
        mag = float(self.f_mag(mag, t))
        # direction: + = turned to the dancer's own left (their right shoulder comes toward the camera, so it has the smaller z)
        if lm_world is not None:
            W = np.asarray(lm_world, np.float32); dz = float(W[L_SH, 2] - W[R_SH, 2])
            s = 1 if dz > 2.5 * c.z_noise else (-1 if dz < -2.5 * c.z_noise else 0)
            if s != 0 and s != self.side:
                self.side_run += 1
                if self.side_run >= 3: self.side = s; self.side_run = 0
            else: self.side_run = 0
        face = float(np.mean(a[list(FACE), 3])); sh_vis = float(min(a[[L_SH, R_SH], 3]))
        if mag > 70 and self.side: self.peak_side = self.side; self.peak_mag = max(self.peak_mag, mag)
        hidden = face < 0.35 and sh_vis > 0.6
        if hidden: self.away_since = self.away_since if self.away_since is not None else t
        else: self.away_since = None; self.peak_mag = 0.0 if face > 0.6 else self.peak_mag
        self.away = self.away_since is not None and t - self.away_since >= 0.5
        yaw = (self.side or 1) * mag if mag >= 15 else 0.0
        if self.away:
            s0 = self.peak_side or self.side or 1
            yaw = s0 * (180.0 - mag)
        # limbs: toward / away from the camera, or just foreshortened by the turn
        toward = {}; ambiguous = set()
        for side in ('l', 'r'):
            up = c.seg.get(f'upper_{side}'); fo = c.seg.get(f'fore_{side}')
            if not up or not fo: continue
            i_sh, i_el, i_wr = (L_SH, L_EL, L_WR) if side == 'l' else (R_SH, R_EL, R_WR)
            if min(a[[i_el, i_wr], 3]) < 0.5: continue
            fs_u = float(np.linalg.norm(_xy(a, i_sh, aspect) - _xy(a, i_el, aspect))) / tor / up
            fs_f = float(np.linalg.norm(_xy(a, i_el, aspect) - _xy(a, i_wr, aspect))) / tor / fo
            fs = min(fs_u, fs_f)
            run = self.toward_run[side]
            if fs < TOWARD_FS and mag < 30 and lm_world is not None:
                W = np.asarray(lm_world, np.float32); dzw = float(W[i_wr, 2] - W[i_sh, 2])
                run.append('toward' if dzw < -c.z_noise else ('away' if dzw > c.z_noise else None))
            else: run.append(None)
            del run[:-5]
            if len(run) == 5 and run[0] is not None and all(x == run[0] for x in run): toward[side] = run[0]
            unreliable = (side not in toward) or float(min(a[[i_el, i_wr], 3])) < 0.7
            if fs < AMBIG_FS and unreliable: ambiguous.update(ARM_CHANNELS[side])
        if mag > 50 and not self.away:                                      # side-on: the far arm is hidden behind the body
            far = 'r' if (self.side or 1) > 0 else 'l'
            ambiguous.update(ARM_CHANNELS[far])
        conf = (0.9 if c.ready else 0.5) * (1.0 if sh_vis > 0.7 else 0.6)
        out.update(yaw=float(yaw), yaw_mag=mag, side=self.side, away=self.away, lr_swapped=self.away, toward=toward, ambiguous=ambiguous, conf=conf)
        return out


def facing_phrase(yaw, away=False):
    """What to say in the camera check: 'Turn a little to your left to face me' etc. (yaw: + = turned to their own left, so they
    turn back to their right)."""
    if away: return "Turn around to face me."
    if abs(yaw) < 35: return None
    back = 'right' if yaw > 0 else 'left'
    return f"Turn a little to your {back} to face me."
