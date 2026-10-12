"""Left and right, end to end. Reference keys are mirrored at build time so a caller copying the ghost like a mirror matches it, and
every tip says the caller's own side. If any link in that chain flips, Brodie says "left arm" about the right one."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from team.dance.features import features_from_landmarks, mirror_landmarks, CHANNELS
from team.dance import feedback

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16


def body(raise_side=None):
    """Image landmarks of a person facing the camera (their LEFT side at larger x), optionally with one arm straight up."""
    lm = np.zeros((33, 5), np.float32); lm[:, 3] = 1.0
    pts = {0: (0.5, 0.2), 11: (0.56, 0.3), 12: (0.44, 0.3), 13: (0.59, 0.42), 14: (0.41, 0.42), 15: (0.6, 0.54), 16: (0.4, 0.54),
           23: (0.54, 0.55), 24: (0.46, 0.55), 25: (0.54, 0.72), 26: (0.46, 0.72), 27: (0.54, 0.9), 28: (0.46, 0.9)}
    if raise_side == 'left': pts.update({13: (0.57, 0.2), 15: (0.57, 0.1)})
    if raise_side == 'right': pts.update({14: (0.43, 0.2), 16: (0.43, 0.1)})
    for j, p in pts.items(): lm[j, :2] = p
    return lm


def test_mirrored_reference_matches_a_mirror_copy():
    # the reference dancer (facing their camera) raises their RIGHT arm; the key is mirrored at build time
    ref = features_from_landmarks(mirror_landmarks(body('right'))[0])
    copy = features_from_landmarks(body('left'))                 # the caller copies the ghost like a mirror: their LEFT arm
    wrong = features_from_landmarks(body('right'))
    el_l, el_r = CHANNELS.index('arm_elev_l'), CHANNELS.index('arm_elev_r')
    assert abs(copy[el_l] - ref[el_l]) < 5 and abs(copy[el_r] - ref[el_r]) < 5
    assert abs(wrong[el_l] - ref[el_l]) > 60


def test_tip_names_the_callers_own_side():
    ref = features_from_landmarks(mirror_landmarks(body('right'))[0])
    live = features_from_landmarks(body(None))                   # the caller didn't raise anything
    d = live - ref
    tip = feedback.live_tip([d, d])
    assert tip is not None and 'left' in tip['text'].lower() and 'right' not in tip['text'].lower(), tip
