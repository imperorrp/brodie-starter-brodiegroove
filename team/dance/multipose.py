"""Several people in one camera: multi-person pose and stable player ids.

A Chert FaceTime call has one caller, but several friends can dance in front of that one phone. The starter's Pose wrapper
(src/rtcoach/pose.py) asks MediaPipe for one person, so with two people in frame it jumps between them. MultiPose asks for up
to four; PlayerTracker gives each dancer a stable id and colour by position, ignores small people in the background
(spectators), and in solo mode locks onto one primary dancer who keeps the scoring even when someone walks behind them.
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


class Player:
    __slots__ = ('pid', 'lm', 'world', 'first_t', 'last_t', 'centre', 'torso', 'colour', 'name', 'stable')

    def __init__(self, pid, lm, world, t):
        self.pid = pid; self.lm = lm; self.world = world; self.first_t = t; self.last_t = t
        self.centre = body_anchor(lm); self.torso = body_size(lm); self.colour = None; self.name = None; self.stable = False


class PlayerTracker:
    """update(t, poses) with poses = [(lm_img, lm_world), ...] from MultiPose. Keeps stable ids; .primary() for solo."""
    def __init__(self, join_s=JOIN_S, drop_s=DROP_S):
        self.join_s = join_s; self.drop_s = drop_s; self.players = {}; self.next_id = 0; self.primary_id = None
        self.events = []; self._ignored_noted = False; self.squad = False; self.last_cands = []

    def _candidates(self, poses):
        c = [(lm, w) for lm, w in poses if lm is not None and torso_visible(lm)]
        if not c: return []
        biggest = max(body_size(lm) for lm, _ in c)
        keep = [(lm, w) for lm, w in c if body_size(lm) >= MIN_REL_TORSO * biggest]
        if len(keep) < len(c) and not self._ignored_noted: self.events.append(dict(type='extra_person_ignored', reason='background')); self._ignored_noted = True
        return keep

    def update(self, t, poses):
        cands = self._candidates(poses or []); self.last_cands = cands
        # greedy nearest matching, distance in torso lengths
        pairs = []
        for ci, (lm, w) in enumerate(cands):
            c = body_anchor(lm); tl = body_size(lm)
            for pid, p in self.players.items():
                d = float(np.linalg.norm(c - p.centre)) / max(tl, p.torso)
                if d <= GATE_TORSO: pairs.append((d, ci, pid))
        pairs.sort(); used_c = set(); used_p = set()
        for d, ci, pid in pairs:
            if ci in used_c or pid in used_p: continue
            lm, w = cands[ci]; p = self.players[pid]
            p.lm = lm; p.world = w; p.last_t = t; p.centre = body_anchor(lm); p.torso = body_size(lm)
            used_c.add(ci); used_p.add(pid)
        for ci, (lm, w) in enumerate(cands):
            if ci in used_c: continue
            p = Player(self.next_id, lm, w, t); self.players[p.pid] = p; self.next_id += 1
        for pid in list(self.players):
            p = self.players[pid]
            if pid not in used_p and p.last_t < t: p.lm = None if t - p.last_t > 0.3 else p.lm
            if t - p.last_t > self.drop_s:
                if p.stable: self.events.append(dict(type='player_left', pid=pid, colour=p.colour))
                del self.players[pid]
                if pid == self.primary_id: self.primary_id = None
            elif not p.stable and t - p.first_t >= self.join_s and t - p.last_t < 0.3:
                p.stable = True; self.events.append(dict(type='player_joined', pid=pid))
        self._assign_primary()
        return self.stable_players()

    def _assign_primary(self):
        st = [p for p in self.players.values() if p.stable]
        if self.primary_id is not None and self.primary_id in self.players:
            if self.players[self.primary_id].lm is not None or not st: return
            return                                            # primary briefly hidden: keep the lock until it is dropped
        if st:
            self.primary_id = max(st, key=lambda p: p.torso).pid
            self.events.append(dict(type='primary_locked', pid=self.primary_id))

    def stable_players(self):
        return sorted([p for p in self.players.values() if p.stable], key=lambda p: p.centre[0])

    def stable_count(self):
        return sum(1 for p in self.players.values() if p.stable and p.lm is not None)

    def primary(self):
        p = self.players.get(self.primary_id)
        return p if (p is not None and p.lm is not None) else None

    def solo_pose(self):
        """(lm, world) for solo scoring: the locked primary when visible; before anyone is stable, the biggest person this frame;
        None while the locked primary is hidden (never switch to whoever walked in)."""
        p = self.primary()
        if p is not None: return p.lm, p.world
        if self.primary_id is not None and self.primary_id in self.players: return None, None
        if self.last_cands:
            lm, w = max(self.last_cands, key=lambda c: body_size(c[0])); return lm, w
        return None, None

    def assign_colours(self, mirrored=True):
        """Colours in left-to-right order as the dancers see themselves (mirrored display: image x descending)."""
        st = sorted(self.stable_players(), key=lambda p: -p.centre[0] if mirrored else p.centre[0])
        for i, p in enumerate(st): p.colour = COLOURS[i % len(COLOURS)]
        return [(p.pid, p.colour) for p in st]

    def colour_order(self, mirrored=True):
        return sorted(self.stable_players(), key=lambda p: -p.centre[0] if mirrored else p.centre[0])

    def overlapping(self, thr=0.25):
        st = [p for p in self.stable_players() if p.lm is not None]; out = []
        for i in range(len(st)):
            for j in range(i + 1, len(st)):
                if iou(bbox(st[i].lm), bbox(st[j].lm)) > thr: out.append((st[i].colour, st[j].colour))
        return out

    def pop_events(self):
        ev = self.events; self.events = []; return ev
