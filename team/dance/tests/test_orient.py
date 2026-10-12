"""Depth from one camera (orient.py) on a synthetic 3D body: known turns, a forearm pointing at the camera, a dancer facing away."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from team.dance.orient import Orient, facing_phrase

# a standing body in metres, hip-centred, x to the image's right, y down, z away from the camera (MediaPipe's world frame).
# Facing the camera, the dancer's LEFT side is at +x in the image.
BODY = {0: (0.0, -0.62, -0.05), 2: (0.03, -0.66, -0.06), 5: (-0.03, -0.66, -0.06), 7: (0.07, -0.64, 0.0), 8: (-0.07, -0.64, 0.0),
        11: (0.18, -0.5, 0.0), 12: (-0.18, -0.5, 0.0), 13: (0.22, -0.25, 0.0), 14: (-0.22, -0.25, 0.0), 15: (0.24, -0.02, 0.0), 16: (-0.24, -0.02, 0.0),
        23: (0.1, 0.0, 0.0), 24: (-0.1, 0.0, 0.0), 25: (0.1, 0.45, 0.0), 26: (-0.1, 0.45, 0.0), 27: (0.1, 0.9, 0.0), 28: (-0.1, 0.9, 0.0)}


def pose(turn_left_deg=0.0, edit=None, face_vis=1.0, noise=0.0, rng=None):
    """(lm_img (33,5), lm_world (33,3)) for the body turned turn_left_deg to the dancer's own left (their right shoulder comes
    closer to the camera)."""
    th = np.radians(-turn_left_deg); c, s = np.cos(th), np.sin(th)
    P = np.zeros((33, 3), np.float32); V = np.zeros(33, np.float32)
    pts = dict(BODY)
    if edit: pts.update(edit)
    for j, (x, y, z) in pts.items():
        P[j] = (x * c + z * s, y, -x * s + z * c); V[j] = 1.0
    for j in (0, 2, 5, 7, 8): V[j] = face_vis
    if noise and rng is not None: P[:, 2] += rng.normal(0, noise, 33)
    img = np.zeros((33, 5), np.float32); img[:, 0] = 0.5 + P[:, 0] * 0.5; img[:, 1] = 0.45 + P[:, 1] * 0.5; img[:, 3] = V
    return img, P


def calibrated():
    o = Orient(); rng = np.random.default_rng(0)
    for i in range(25): o.update(*pose(0, noise=0.01, rng=rng), aspect=1.0, t=i / 15)
    assert o.calib.ready
    return o, 25 / 15


def test_turn_size_and_direction():
    for deg in (25, 45, 60, 75, -30, -60):
        o, t = calibrated(); rng = np.random.default_rng(1); r = None
        for i in range(30): r = o.update(*pose(deg, noise=0.01, rng=rng), aspect=1.0, t=t + i / 15)
        assert abs(abs(r['yaw']) - abs(deg)) <= 12, (deg, r['yaw'])
        assert np.sign(r['yaw']) == np.sign(deg), (deg, r['yaw'])


def test_small_turn_is_frontal():
    o, t = calibrated(); r = None
    for i in range(20): r = o.update(*pose(8), aspect=1.0, t=t + i / 15)
    assert r['yaw'] == 0.0 and facing_phrase(r['yaw']) is None


def test_forearm_toward_camera_is_not_a_turn():
    o, t = calibrated(); r = None
    elbow_out = {13: (0.22, -0.25, 0.0), 15: (0.23, -0.27, -0.24)}           # left forearm pointing straight at the camera
    for i in range(10): r = o.update(*pose(0, edit=elbow_out), aspect=1.0, t=t + i / 15)
    assert r['toward'].get('l') == 'toward' and abs(r['yaw']) < 15


def test_facing_away_is_detected_and_left_right_flagged():
    o, t = calibrated(); r = None
    for i in range(15): r = o.update(*pose(0, face_vis=0.05), aspect=1.0, t=t + i / 15)
    assert r['away'] and r['lr_swapped'] and facing_phrase(r['yaw'], r['away']) == 'Turn around to face me.'


def test_side_on_far_arm_is_ambiguous():
    o, t = calibrated(); r = None
    for i in range(30): r = o.update(*pose(70), aspect=1.0, t=t + i / 15)
    assert r['ambiguous'] and facing_phrase(r['yaw']) == 'Turn a little to your right to face me.'
