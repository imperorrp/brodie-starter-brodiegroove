"""Pose landmarks -> dance features that do not depend on camera distance, body size or (mostly) viewpoint.

MediaPipe Pose gives 33 landmarks per frame: image coordinates (x, y in [0, 1], z, visibility, presence) and world coordinates
(metres, centred on the hips). A frame becomes:
  - 14 angle-like channels in degrees (CHANNELS), computed from world landmarks when present, else from image x, y;
  - a per-channel visibility mask (every joint the channel uses must be visible above a threshold);
  - a hip-energy scalar from the image landmarks (world landmarks are hip-centred, so hip motion is invisible there);
  - a normalised 2D skeleton (hip-centred, torso-length units) used to draw the ghost on another person.
Mirroring swaps left and right landmarks and flips x, so a key recorded facing the camera matches a caller facing the screen.
"""
import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28

CHANNELS = ['elbow_l', 'elbow_r', 'shoulder_l', 'shoulder_r', 'hip_l', 'hip_r', 'knee_l', 'knee_r',
            'torso_lean', 'head_tilt', 'arm_elev_l', 'arm_elev_r', 'leg_spread', 'wrist_dist']
GROUPS = {'arms': [0, 1, 2, 3, 10, 11, 13], 'legs': [4, 5, 6, 7, 12], 'torso': [8, 9]}
CHANNEL_JOINTS = [
    (L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR), (L_HIP, L_SH, L_EL), (R_HIP, R_SH, R_EL),
    (L_SH, L_HIP, L_KN), (R_SH, R_HIP, R_KN), (L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN),
    (L_SH, R_SH, L_HIP, R_HIP), (NOSE, L_SH, R_SH), (L_SH, L_WR, L_HIP, R_HIP), (R_SH, R_WR, L_HIP, R_HIP),
    (L_HIP, R_HIP, L_AN, R_AN), (L_WR, R_WR, L_SH, R_SH)]
WRIST_DIST_SCALE = 30.0            # one shoulder-width of wrist separation counts as 30 "degrees" in the error metric
MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
                (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
SKELETON_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                  (24, 26), (26, 28), (0, 11), (0, 12)]
EPS = 1e-6
N_CHANNELS = len(CHANNELS)


def _angle(a, b, c):
    """Angle at b (degrees) between b->a and b->c; works for 2D or 3D points."""
    v1 = a - b; v2 = c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + EPS))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def _angle_between(v1, v2):
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + EPS))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def features_from_landmarks(lm_img, lm_world=None):
    """(14,) float32 feature vector for one frame. Uses world xyz when given (view-invariant), else image xy."""
    if lm_world is not None:
        P = np.asarray(lm_world, np.float32)[:, :3]
    else:
        P = np.asarray(lm_img, np.float32)[:, :2]
    sh_mid = 0.5 * (P[L_SH] + P[R_SH]); hip_mid = 0.5 * (P[L_HIP] + P[R_HIP])
    torso_down = hip_mid - sh_mid                     # points from shoulders to hips (down the body)
    up = -torso_down                                   # body "up" axis, so lean is measured against the world only via head/torso
    world_up = np.zeros_like(sh_mid); world_up[1] = -1.0        # image and world y both increase downward in MediaPipe
    f = np.zeros(N_CHANNELS, np.float32)
    f[0] = _angle(P[L_SH], P[L_EL], P[L_WR]); f[1] = _angle(P[R_SH], P[R_EL], P[R_WR])
    f[2] = _angle(P[L_HIP], P[L_SH], P[L_EL]); f[3] = _angle(P[R_HIP], P[R_SH], P[R_EL])
    f[4] = _angle(P[L_SH], P[L_HIP], P[L_KN]); f[5] = _angle(P[R_SH], P[R_HIP], P[R_KN])
    f[6] = _angle(P[L_HIP], P[L_KN], P[L_AN]); f[7] = _angle(P[R_HIP], P[R_KN], P[R_AN])
    f[8] = _angle_between(sh_mid - hip_mid, world_up)             # torso lean from vertical
    f[9] = _angle_between(P[NOSE] - sh_mid, up)                   # head tilt relative to the torso axis
    f[10] = _angle_between(P[L_WR] - P[L_SH], torso_down)         # arm elevation: 0 hanging, 180 straight up
    f[11] = _angle_between(P[R_WR] - P[R_SH], torso_down)
    f[12] = _angle(P[L_AN], hip_mid, P[R_AN])                     # leg spread
    sh_w = float(np.linalg.norm(P[L_SH] - P[R_SH])) + EPS
    f[13] = WRIST_DIST_SCALE * float(np.linalg.norm(P[L_WR] - P[R_WR])) / sh_w
    return f


import os as _os
IGNORE_FEET = (_os.environ.get('DANCE_IGNORE_FEET', '1') or '1').strip() not in ('0', 'false', 'no')
FEET_CHANNELS = [i for i, joints in enumerate(CHANNEL_JOINTS) if L_AN in joints or R_AN in joints]   # knee bends (to the ankle), leg spread


def visibility_mask(lm_img, thr=0.5):
    """(14,) bool: every joint a channel uses is visible above thr AND inside the picture (MediaPipe keeps predicting joints
    that left the frame and often still rates them visible; they are guesses, not evidence). With DANCE_IGNORE_FEET (default on)
    the channels that need the ankles are never used: feet were the least reliable part on real calls."""
    a = np.asarray(lm_img, np.float32)
    inside = (a[:, 0] >= -0.02) & (a[:, 0] <= 1.02) & (a[:, 1] >= -0.02) & (a[:, 1] <= 1.02)
    vis = np.where(inside, a[:, 3], 0.0)
    m = np.array([all(vis[j] >= thr for j in joints) for joints in CHANNEL_JOINTS], bool)
    if IGNORE_FEET: m[FEET_CHANNELS] = False
    return m


def mirror_landmarks(lm_img=None, lm_world=None):
    """Swap left and right landmarks and flip x (image: x -> 1 - x; world: x -> -x)."""
    out_i = out_w = None
    if lm_img is not None:
        a = np.array(lm_img, np.float32, copy=True)
        for l, r in MIRROR_PAIRS: a[[l, r]] = a[[r, l]]
        a[:, 0] = 1.0 - a[:, 0]; out_i = a
    if lm_world is not None:
        w = np.array(lm_world, np.float32, copy=True)
        for l, r in MIRROR_PAIRS: w[[l, r]] = w[[r, l]]
        w[:, 0] = -w[:, 0]; out_w = w
    return out_i, out_w


def torso_length(lm_img):
    P = np.asarray(lm_img, np.float32)[:, :2]
    return float(np.linalg.norm(0.5 * (P[L_SH] + P[R_SH]) - 0.5 * (P[L_HIP] + P[R_HIP]))) + EPS


def hip_centre(lm_img):
    P = np.asarray(lm_img, np.float32)[:, :2]
    return 0.5 * (P[L_HIP] + P[R_HIP])


def hip_energy(prev_lm_img, lm_img, dt):
    """Hip-centre speed in torso lengths per second (image landmarks). 0 when either frame is missing."""
    if prev_lm_img is None or lm_img is None or dt <= 0: return 0.0
    d = float(np.linalg.norm(hip_centre(lm_img) - hip_centre(prev_lm_img)))
    return d / torso_length(lm_img) / dt


def skeleton2d_normalized(lm_img):
    """(33, 2) image landmarks centred on the hips and divided by the torso length (so a ghost can be placed on anyone)."""
    P = np.asarray(lm_img, np.float32)[:, :2]
    return (P - hip_centre(lm_img)) / torso_length(lm_img)


def resample_to_grid(times, values, grid_times):
    """Linear interpolation of (n, C) values at grid_times; outside the range holds the end values. Returns (len(grid), C)."""
    times = np.asarray(times, np.float64); V = np.asarray(values, np.float32)
    if V.ndim == 1: V = V[:, None]
    if len(times) == 0: return np.zeros((len(grid_times), V.shape[1]), np.float32)
    if len(times) == 1: return np.repeat(V, len(grid_times), axis=0)
    order = np.argsort(times); times = times[order]; V = V[order]
    out = np.stack([np.interp(grid_times, times, V[:, c]) for c in range(V.shape[1])], axis=1)
    return out.astype(np.float32)


def resample_mask(times, masks, grid_times):
    m = resample_to_grid(times, np.asarray(masks, np.float32), grid_times)
    return m > 0.5


def unwrap_angles(F):
    """Angles here are all in [0, 180] from arccos, so no unwrapping is needed; kept as a hook for signed channels."""
    return np.asarray(F, np.float32)
