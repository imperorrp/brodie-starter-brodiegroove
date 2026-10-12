"""Several people in one camera: multi-person pose and stable player ids.

A Chert FaceTime call has one caller, but several friends can dance in front of that one phone. The starter's Pose wrapper
(src/rtcoach/pose.py) asks MediaPipe for one person, so with two people in frame it jumps between them. MultiPose asks for up
to four; PlayerTracker (tracker.py) gives each dancer a stable id and colour tag, re-finds them by their clothing colours after
an occlusion or a crossing, ignores small people far in the background (spectators), and in solo mode locks onto one primary
dancer who keeps the scoring even when someone walks in front of or behind them.
No faces, no identities: players are colours (VOLT, PINK, CYAN, AMBER) or a name the dancers say themselves.
"""
import os
import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
COLOURS = ('volt', 'pink', 'cyan', 'amber')
JOIN_S = 1.5
DROP_S = 2.0
MIN_REL_TORSO = 0.4
GATE_TORSO = 1.0


class MultiPose:
    """Same construction as rtcoach.pose.Pose with num_poses=max_poses. __call__ -> [(lm_img (33,5), lm_world (33,3)), ...]."""
    def __init__(self, variant='full', video=True, max_poses=4):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        from team.dance import MODELS_DIR
        self.mp = mp
        opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(MODELS_DIR, f'pose_landmarker_{variant}.task')),
                                            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE, num_poses=int(max_poses),
                                            min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4, min_tracking_confidence=0.4)
        self.lm = vision.PoseLandmarker.create_from_options(opts); self.video = video; self.last_ms = -1

    def __call__(self, rgb, t_ms):
        t_ms = int(t_ms)
        if self.video and t_ms <= self.last_ms: t_ms = self.last_ms + 1          # VIDEO mode needs strictly increasing timestamps
        self.last_ms = t_ms
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        r = self.lm.detect_for_video(img, t_ms) if self.video else self.lm.detect(img)
        out = []
        for i, pl in enumerate(r.pose_landmarks or []):
            a = np.array([[p.x, p.y, p.z, p.visibility or 0, p.presence or 0] for p in pl], np.float32)
            w = np.array([[p.x, p.y, p.z] for p in r.pose_world_landmarks[i]], np.float32) if r.pose_world_landmarks else None
            out.append((a, w))
        return out

    def close(self): self.lm.close()


def hip_centre(lm):
    a = np.asarray(lm, np.float32); return 0.5 * (a[L_HIP, :2] + a[R_HIP, :2])


def torso_len(lm):
    a = np.asarray(lm, np.float32)
    return float(np.linalg.norm(0.5 * (a[L_SH, :2] + a[R_SH, :2]) - 0.5 * (a[L_HIP, :2] + a[R_HIP, :2]))) + 1e-6


def torso_visible(lm, thr=0.5):
    """A person counts once both shoulders are seen inside the picture; hips are optional (a caller at a desk or close to the
    camera shows only the upper body, and must still be tracked)."""
    a = np.asarray(lm, np.float32)
    sh = a[[L_SH, R_SH]]
    return bool(np.all(sh[:, 3] >= thr) and np.all((sh[:, 0] > -0.02) & (sh[:, 0] < 1.02) & (sh[:, 1] > -0.02) & (sh[:, 1] < 1.02)))


def hips_seen(lm, thr=0.5):
    a = np.asarray(lm, np.float32); return bool(np.all(a[[L_HIP, R_HIP], 3] >= thr))


def body_anchor(lm):
    """Hip centre when the hips are seen, else a point one shoulder-width below the shoulders (stable for matching)."""
    a = np.asarray(lm, np.float32)
    if hips_seen(lm): return 0.5 * (a[L_HIP, :2] + a[R_HIP, :2])
    sm = 0.5 * (a[L_SH, :2] + a[R_SH, :2]); return sm + np.array([0.0, abs(a[L_SH, 0] - a[R_SH, 0])], np.float32)


def body_size(lm):
    """Torso length when the hips are seen, else 1.6 x shoulder width (about the same size)."""
    a = np.asarray(lm, np.float32)
    if hips_seen(lm): return torso_len(lm)
    return 1.6 * float(abs(a[L_SH, 0] - a[R_SH, 0])) + 1e-6


def bbox(lm, thr=0.5):
    a = np.asarray(lm, np.float32); v = a[:, 3] >= thr
    if not v.any(): return None
    p = a[v, :2]; return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def iou(b1, b2):
    if b1 is None or b2 is None: return 0.0
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1]); x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    a = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / a if a > 0 else 0.0


# The tracker (stable ids, colour re-identification after an occlusion, the solo primary lock) lives in tracker.py; these names are
# kept here for the code that imports them from multipose.
from .tracker import Player, PlayerTracker  # noqa: E402,F401
