"""Who is who when several people share one phone camera (squad, class, a friend walking past a solo dancer).

The first tracker matched people by nearest position only: two dancers crossing swapped ids (and colours, and scores), anyone who
stepped out for two seconds came back as a stranger, and in solo mode a bigger person walking in took over the scoring after the
dancer had been hidden for two seconds. This one is OC-SORT-like and runs in well under a millisecond for four people:

  motion      each track predicts where its body anchor will be (constant velocity, smoothed); matching cost uses the prediction
  appearance  a colour histogram (HSV, 16x8) of the torso, learned only while nobody overlaps that person, compared with the
              Bhattacharyya distance; it decides who is who after an occlusion or a crossing
  assignment  Hungarian (scipy.optimize.linear_sum_assignment) on 0.5 motion + 0.35 appearance + 0.15 (1 - box overlap)
  returning   a track missing for over DROP_S leaves the "present" set but is remembered (LOST_KEEP_S: 20 s in squad, 6 s solo); a new
              person whose torso colours match it (distance < REID_MAX) gets the old id back, with its colour tag and name
  crossing    while two people overlap their colour models freeze; when they separate, if swapping the two assignments fits the
              colour models clearly better, the ids are swapped back
  primary     solo: the locked dancer keeps the scoring through up to PRIMARY_HOLD_S of absence (re-found by colour); during an
              activity, nobody else takes over unless they are the only person, near the middle, after that hold
  size        spectators far back are ignored: below MIN_REL_TORSO of the biggest person (solo), or MIN_REL_TORSO_SQUAD in squad
              mode (children and dancers further back still count), and never below MIN_ABS_TORSO of the frame height

No faces, no identities: colours of clothing only, kept in memory for the call.

  PlayerTracker.update(t, poses, frame=None)   poses: [(lm_img (33,5), lm_world)], frame: the RGB image the poses came from
  Same public interface as before: stable_players, stable_count, primary, solo_pose, assign_colours, colour_order, overlapping, pop_events
"""
import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
COLOURS = ('volt', 'pink', 'cyan', 'amber')
JOIN_S = 1.5
DROP_S = 2.0
MIN_REL_TORSO = 0.4
MIN_REL_TORSO_SQUAD = 0.25
MIN_ABS_TORSO = 0.06
GATE = 1.5                       # torso lengths from the predicted anchor
MAX_COST = 0.7
REID_MAX = 0.35
LOST_KEEP_S = dict(squad=20.0, solo=6.0)
PRIMARY_HOLD_S = 8.0
OVERLAP_FREEZE = 0.1             # colour models learn only below this box overlap with anyone else
SWAP_GAIN = 0.15


def _a(lm): return np.asarray(lm, np.float32)


def hips_seen(lm, thr=0.5): return bool(np.all(_a(lm)[[L_HIP, R_HIP], 3] >= thr))


def torso_len(lm):
    a = _a(lm); return float(np.linalg.norm(0.5 * (a[L_SH, :2] + a[R_SH, :2]) - 0.5 * (a[L_HIP, :2] + a[R_HIP, :2]))) + 1e-6


def torso_visible(lm, thr=0.5):
    a = _a(lm); sh = a[[L_SH, R_SH]]
    return bool(np.all(sh[:, 3] >= thr) and np.all((sh[:, 0] > -0.02) & (sh[:, 0] < 1.02) & (sh[:, 1] > -0.02) & (sh[:, 1] < 1.02)))


def body_anchor(lm):
    a = _a(lm)
    if hips_seen(lm): return 0.5 * (a[L_HIP, :2] + a[R_HIP, :2])
    sm = 0.5 * (a[L_SH, :2] + a[R_SH, :2]); return sm + np.array([0.0, abs(a[L_SH, 0] - a[R_SH, 0])], np.float32)


def body_size(lm):
    a = _a(lm)
    if hips_seen(lm): return torso_len(lm)
    return 1.6 * float(abs(a[L_SH, 0] - a[R_SH, 0])) + 1e-6


def bbox(lm, thr=0.5):
    if lm is None: return None
    a = _a(lm); v = a[:, 3] >= thr
    if not v.any(): return None
    p = a[v, :2]; return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def iou(b1, b2):
    if b1 is None or b2 is None: return 0.0
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1]); x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    a = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / a if a > 0 else 0.0


def torso_hist(frame, lm):
    """HSV histogram (16 hue x 8 saturation, normalised) of the torso: the shoulder-hip box shrunk 15%, or a chest box under the
    shoulders when the hips are out of view. None when too small or off the frame."""
    if frame is None or lm is None: return None
    import cv2
    h, w = frame.shape[:2]; a = _a(lm)
    sh = a[[L_SH, R_SH], :2] * (w, h)
    if hips_seen(lm): hp = a[[L_HIP, R_HIP], :2] * (w, h)
    else:
        sw = abs(sh[0, 0] - sh[1, 0]); hp = sh + np.array([0.0, sw], np.float32)
    pts = np.vstack([sh, hp]); x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
    dx, dy = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
    x0, x1 = int(max(0, x0 + dx)), int(min(w, x1 - dx)); y0, y1 = int(max(0, y0 + dy)), int(min(h, y1 - dy))
    if x1 - x0 < 4 or y1 - y0 < 4: return None
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32)
    s = float(hist.sum())
    return hist / s if s > 0 else None


def hist_dist(h1, h2):
    """Bhattacharyya distance in [0, 1]; 0.5 (unknown) when either is missing."""
    if h1 is None or h2 is None: return 0.5
    import cv2
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


class Player:
    __slots__ = ('pid', 'lm', 'world', 'first_t', 'last_t', 'centre', 'vel', 'torso', 'colour', 'name', 'stable', 'hist', 'box')

    def __init__(self, pid, lm, world, t):
        self.pid = pid; self.lm = lm; self.world = world; self.first_t = t; self.last_t = t
        self.centre = body_anchor(lm); self.vel = np.zeros(2, np.float32); self.torso = body_size(lm)
        self.colour = None; self.name = None; self.stable = False; self.hist = None; self.box = bbox(lm)

    def predict(self, t):
        dt = min(1.0, max(0.0, t - self.last_t)); return self.centre + self.vel * dt


class PlayerTracker:
    def __init__(self, join_s=JOIN_S, drop_s=DROP_S):
        self.join_s = join_s; self.drop_s = drop_s; self.players = {}; self.lost = {}; self.next_id = 0; self.primary_id = None
        self.events = []; self._ignored_noted = False; self.squad = False; self.activity = False; self.last_cands = []
        self.overlap_pairs = set(); self.max_seen = 0; self.crowd_noted = False

    # ---- candidates: people big enough to be dancers ----
    def _candidates(self, poses):
        c = [(lm, w) for lm, w in poses if lm is not None and torso_visible(lm)]
        self.max_seen = len(c)
        if not c: return []
        biggest = max(body_size(lm) for lm, _ in c)
        rel = MIN_REL_TORSO_SQUAD if self.squad else MIN_REL_TORSO
        keep = [(lm, w) for lm, w in c if body_size(lm) >= rel * biggest and body_size(lm) >= MIN_ABS_TORSO]
        if len(keep) < len(c) and not self._ignored_noted: self.events.append(dict(type='extra_person_ignored', reason='background')); self._ignored_noted = True
        return keep

    def _cost(self, p, lm, h, t):
        tl = max(body_size(lm), p.torso)
        d = float(np.linalg.norm(body_anchor(lm) - p.predict(t))) / tl
        if d > GATE: return None
        return 0.5 * (d / GATE) + 0.35 * hist_dist(h, p.hist) + 0.15 * (1.0 - iou(bbox(lm), p.box))

    def update(self, t, poses, frame=None):
        cands = self._candidates(poses or []); self.last_cands = cands
        boxes = [bbox(lm) for lm, _ in cands]
        clear = [all(iou(boxes[i], boxes[j]) < OVERLAP_FREEZE for j in range(len(cands)) if j != i) for i in range(len(cands))]
        hists = [torso_hist(frame, lm) for lm, _ in cands]
        pids = [pid for pid, p in self.players.items()]
        matched_c, matched_p = set(), set()
        if pids and cands:
            BIG = 1e3; C = np.full((len(pids), len(cands)), BIG, np.float32)
            for i, pid in enumerate(pids):
                for j, (lm, _) in enumerate(cands):
                    c = self._cost(self.players[pid], lm, hists[j], t)
                    if c is not None: C[i, j] = c
            try:
                from scipy.optimize import linear_sum_assignment
                rows, cols = linear_sum_assignment(C)
            except Exception:                                     # no scipy: greedy on the same costs
                order = sorted(((C[i, j], i, j) for i in range(len(pids)) for j in range(len(cands))))
                rows, cols, ui, uj = [], [], set(), set()
                for c, i, j in order:
                    if i in ui or j in uj: continue
                    rows.append(i); cols.append(j); ui.add(i); uj.add(j)
            for i, j in zip(rows, cols):
                if C[i, j] > MAX_COST: continue
                self._apply(self.players[pids[i]], cands[j], hists[j], clear[j], t); matched_c.add(j); matched_p.add(pids[i])
        for j, (lm, w) in enumerate(cands):                       # unmatched people: someone coming back, or someone new
            if j in matched_c: continue
            back = self._reid(hists[j], t)
            if back is not None:
                p = self.lost.pop(back); self.players[p.pid] = p; p.vel[:] = 0; p.centre = body_anchor(lm)
                self._apply(p, cands[j], hists[j], clear[j], t); p.stable = True
                self.events.append(dict(type='player_back', pid=p.pid, colour=p.colour)); matched_p.add(p.pid)
            else:
                p = Player(self.next_id, lm, w, t); p.hist = hists[j] if clear[j] else None
                self.players[p.pid] = p; self.next_id += 1
        self._fix_swaps(frame)
        for pid in list(self.players):
            p = self.players[pid]
            if pid not in matched_p and p.last_t < t and t - p.last_t > 0.3: p.lm = None
            if t - p.last_t > self.drop_s:
                if p.stable:
                    self.events.append(dict(type='player_left', pid=pid, colour=p.colour)); self.lost[pid] = p
                del self.players[pid]
            elif not p.stable and t - p.first_t >= self.join_s and t - p.last_t < 0.3:
                p.stable = True; self.events.append(dict(type='player_joined', pid=pid))
        keep = LOST_KEEP_S['squad' if self.squad else 'solo']
        for pid in [pid for pid, p in self.lost.items() if t - p.last_t > max(keep, PRIMARY_HOLD_S if pid == self.primary_id else 0)]:
            del self.lost[pid]
            if pid == self.primary_id: self.primary_id = None
        if self.squad and self.max_seen >= 4 and not self.crowd_noted:
            self.crowd_noted = True; self.events.append(dict(type='crowd', n=self.max_seen))
        self._assign_primary(t)
        return self.stable_players()

    def _apply(self, p, cand, h, clear, t):
        lm, w = cand; c = body_anchor(lm); dt = max(1e-3, t - p.last_t)
        if dt < 1.0: p.vel = 0.6 * p.vel + 0.4 * ((c - p.centre) / dt)
        p.lm = lm; p.world = w; p.centre = c; p.torso = body_size(lm); p.box = bbox(lm); p.last_t = t
        if h is not None and clear: p.hist = h if p.hist is None else (0.9 * p.hist + 0.1 * h)

    def _reid(self, h, t):
        if h is None or not self.lost: return None
        best = min(self.lost.values(), key=lambda p: hist_dist(h, p.hist))
        return best.pid if hist_dist(h, best.hist) < REID_MAX else None

    def _fix_swaps(self, frame):
        """Two people who overlapped (their colour models froze) and have now separated: if each one's torso colours now fit the
        *other* track's model clearly better, the ids were swapped during the crossing; swap the poses back."""
        cur = [p for p in self.players.values() if p.lm is not None]
        now_pairs = {(a.pid, b.pid) for i, a in enumerate(cur) for b in cur[i + 1:] if iou(a.box, b.box) > OVERLAP_FREEZE}
        for pa_id, pb_id in self.overlap_pairs - now_pairs:
            a, b = self.players.get(pa_id), self.players.get(pb_id)
            if a is None or b is None or a.lm is None or b.lm is None or a.hist is None or b.hist is None or frame is None: continue
            ha, hb = torso_hist(frame, a.lm), torso_hist(frame, b.lm)
            if ha is None or hb is None: continue
            keep = hist_dist(ha, a.hist) + hist_dist(hb, b.hist); swap = hist_dist(ha, b.hist) + hist_dist(hb, a.hist)
            if keep - swap > 2 * SWAP_GAIN:
                for f in ('lm', 'world', 'centre', 'vel', 'torso', 'box'):
                    va, vb = getattr(a, f), getattr(b, f); setattr(a, f, vb); setattr(b, f, va)
                self.events.append(dict(type='id_swap_fixed', a=a.pid, b=b.pid))
        self.overlap_pairs = now_pairs

    def _assign_primary(self, t):
        st = [p for p in self.players.values() if p.stable and p.lm is not None]
        if self.primary_id is not None:
            if self.primary_id in self.players or self.primary_id in self.lost: return       # present, or hidden: keep the lock
            self.primary_id = None
        if not st: return
        if self.activity:                                          # mid-routine: only an unambiguous single dancer near the middle
            if len(st) != 1 or abs(float(st[0].centre[0]) - 0.5) > 0.2: return
        self.primary_id = max(st, key=lambda p: p.torso).pid
        self.events.append(dict(type='primary_locked', pid=self.primary_id))

    # ---- the public interface (as before) ----
    def stable_players(self):
        return sorted([p for p in self.players.values() if p.stable], key=lambda p: p.centre[0])

    def stable_count(self):
        return sum(1 for p in self.players.values() if p.stable and p.lm is not None)

    def primary(self):
        p = self.players.get(self.primary_id)
        return p if (p is not None and p.lm is not None) else None

    def solo_pose(self):
        p = self.primary()
        if p is not None: return p.lm, p.world
        if self.primary_id is not None and (self.primary_id in self.players or self.primary_id in self.lost): return None, None
        if self.last_cands:
            lm, w = max(self.last_cands, key=lambda c: body_size(c[0])); return lm, w
        return None, None

    def assign_colours(self, mirrored=True, keep=False):
        """Colours left to right as the dancers see themselves (squad start). keep=True: people who have a colour keep it, newcomers
        get free ones (a returning dancer re-found by colour keeps their tag)."""
        st = sorted(self.stable_players(), key=lambda p: -p.centre[0] if mirrored else p.centre[0])
        if not keep:
            for i, p in enumerate(st): p.colour = COLOURS[i % len(COLOURS)]
        else:
            used = {p.colour for p in st if p.colour}
            for p in st:
                if p.colour is None:
                    p.colour = next((c for c in COLOURS if c not in used), COLOURS[len(used) % len(COLOURS)]); used.add(p.colour)
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
