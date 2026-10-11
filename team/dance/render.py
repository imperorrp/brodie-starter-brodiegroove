"""What the caller sees. compose(frame_bgr, lm_img, hud, ghost) -> BGR canvas.

The canvas follows the camera: a phone in portrait gets 720x1280, a laptop webcam or the Chert line (landscape) gets 1280x720,
nothing is cropped away. It is drawn like a rhythm game, not a class:
  - the live skeleton (coloured by the latest beat verdict), the reference ghost on the body and in a corner panel;
  - a top bar (routine, step and count, difficulty, state), "5 6 7 8" count-in, PERFECT / NICE / MISS pops above the head, combo
    and ON FIRE edge glow, the beat timeline, a mid-phrase tip, the hold ring for gestures;
  - the result reveal (score counts up, grade stamp, title, cue, confetti), a recap card between phrases, level-up and achievement
    overlays, paused and safety cards, the squad leaderboard and per-player colours, the battle HP bar;
  - the end scorecard built for screenshots (grade, title, XP, achievements, the challenge code and a QR that opens FaceTime).
The middle of the picture (the dancer's body) stays clear during the attempt: everything else lives on the edges.
Text is real type (text.py, Pillow, cached sprites); translucent plates blend only their own rectangle, so a frame costs a few ms.
Landmarks are normalised to the source frame; the ghost is a (33, 2) skeleton with the hip centre at 0 and the torso length 1.
"""
import math, re
from functools import lru_cache
import numpy as np
import cv2
from .features import SKELETON_EDGES, L_SH, R_SH, L_HIP, R_HIP, IGNORE_FEET as FEET_OFF
FEET_JOINTS = {27, 28, 29, 30, 31, 32}
from . import text as tx
from . import fx as fxm
from . import progression

W, H = 720, 1280                      # the portrait canvas (phones); landscape sources get 1280x720
# BGR palette: volt #D9FF3D, pink #FF3DA8, cyan #3DE0FF, amber #FFB13D, red #FF4D5E, ink #0A0A0B, paper #F4F4F0
VOLT, PINK, CYAN, AMBER, RED = (61, 255, 217), (168, 61, 255), (255, 224, 61), (61, 177, 255), (94, 77, 255)
INK, PAPER, MUTE = (11, 10, 10), (240, 244, 244), (164, 157, 154)
BGR = dict(volt=VOLT, ghost=(255, 200, 120), green=VOLT, amber=AMBER, red=RED, gray=(90, 90, 90), white=(245, 245, 245), black=(0, 0, 0),
           dim=(30, 30, 30), panel=(20, 18, 18), mute=MUTE, pink=PINK, cyan=CYAN, ink=INK, paper=PAPER)
PLAYER = dict(volt=VOLT, pink=PINK, cyan=CYAN, amber=AMBER)
GRADE_COL = dict(S=VOLT, A=CYAN, B=PAPER, C=AMBER, D=RED)
JUDGE = dict(green=('PERFECT', VOLT, PINK), amber=('NICE', CYAN, INK), red=('MISS', RED, INK))
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIS_THR = 0.5
PARTS = dict(head=[0, 2, 5], arms=[13, 14, 15, 16], torso=[11, 12, 23, 24], legs=[25, 26])        # feet are ignored
DIFF_LABEL = dict(beginner='BEGINNER', intermediate='INTERMEDIATE', advanced='PRO')


def canvas_size(src_w, src_h):
    return (1280, 720) if src_w > src_h else (720, 1280)


def sc_of(size):
    return min(size) / 720.0


def to_canvas(bgr, size=None):
    """Fit the source to the canvas of its own orientation: centre-crop only what the aspect ratio forces (nothing for a 16:9
    webcam or a 9:16 phone). Returns (img, transform) with transform = (x0, y0, scale) in source pixels."""
    h, w = bgr.shape[:2]
    cw_, ch_ = size or canvas_size(w, h); target = cw_ / ch_
    if w / h > target + 1e-3:
        cw = int(round(h * target)); x0 = (w - cw) // 2; crop = bgr[:, x0:x0 + cw]; y0 = 0; scale = ch_ / h
    elif w / h < target - 1e-3:
        ch = int(round(w / target)); y0 = (h - ch) // 2; crop = bgr[y0:y0 + ch, :]; x0 = 0; scale = cw_ / w
    else:
        crop = bgr; x0 = y0 = 0; scale = cw_ / w
    img = cv2.resize(crop, (cw_, ch_), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return img, (x0, y0, scale)


to_portrait = to_canvas               # old name


def lm_px(lm_img, src_w, src_h, tr):
    x0, y0, s = tr
    P = np.asarray(lm_img, np.float32)[:, :2].copy()
    P[:, 0] = (P[:, 0] * src_w - x0) * s; P[:, 1] = (P[:, 1] * src_h - y0) * s
    return P


def visibility(lm_img):
    """Per-joint visibility (33,) and per-part booleans. MediaPipe keeps guessing joints that left the picture and often still
    calls them visible, so a joint also has to lie inside the frame to count."""
    if lm_img is None: return np.zeros(33, np.float32), {p: False for p in PARTS}
    a = np.asarray(lm_img, np.float32)
    inside = (a[:, 0] >= -0.02) & (a[:, 0] <= 1.02) & (a[:, 1] >= -0.02) & (a[:, 1] <= 1.02)
    v = np.where(inside, a[:, 3], 0.0).astype(np.float32)
    parts = {p: bool(np.mean(v[idx] > VIS_THR) >= 0.75) for p, idx in PARTS.items()}
    return v, parts


def framing_hint(parts, lm_px_pts=None, size=None):
    """One short line for the caller about what the camera sees, or None when the framing is good."""
    if not parts['torso'] and not parts['arms'] and not parts['head']: return "I can't see you. Step into the picture"
    if not parts['torso'] and parts['head']: return "Step back so I can see your shoulders and hips"
    return None                                                   # head to hips is enough: legs and feet are optional


# ---- drawing primitives ----
def px_of(scale):
    return int(round(scale * 34))


def draw_text(img, text, org, scale=1.0, color=BGR['white'], thick=2, outline=True, font='body', anchor='ls', alpha=1.0):
    """Old Hershey-style call (scale) on real type: px = scale * 34."""
    return tx.put_text(img, text, org, px_of(scale), color, font=font, anchor=anchor, stroke=(2 if px_of(scale) < 40 else 3) if outline else 0, alpha=alpha)


def text_w(text, scale, thick=2, font='body'):
    return tx.text_size(text, px_of(scale), font)[0]


def rounded_rect(img, p0, p1, color, r=14, alpha=1.0):
    """Filled rounded rectangle; with alpha < 1 only the rectangle's own pixels are blended."""
    H_, W_ = img.shape[:2]
    x0, y0 = int(p0[0]), int(p0[1]); x1, y1 = int(p1[0]), int(p1[1])
    if x1 <= x0 or y1 <= y0: return img
    r = max(1, min(int(r), (x1 - x0) // 2, (y1 - y0) // 2))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W_, x1 + 1), min(H_, y1 + 1)
    if cx1 <= cx0 or cy1 <= cy0: return img
    roi = img[cy0:cy1, cx0:cx1]
    over = roi if alpha >= 1.0 else roi.copy()
    ox, oy = x0 - cx0, y0 - cy0; w, h = x1 - x0, y1 - y0
    cv2.rectangle(over, (ox + r, oy), (ox + w - r, oy + h), color, -1); cv2.rectangle(over, (ox, oy + r), (ox + w, oy + h - r), color, -1)
    for cx, cy in ((ox + r, oy + r), (ox + w - r, oy + r), (ox + r, oy + h - r), (ox + w - r, oy + h - r)):
        cv2.circle(over, (cx, cy), r, color, -1, cv2.LINE_AA)
    if alpha < 1.0: cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)
    return img


def plate(img, x0, y0, x1, y1, color=INK, alpha=0.72, r=18):
    return rounded_rect(img, (x0, y0), (x1, y1), color, r=r, alpha=alpha)


def pill(img, text, org, scale=0.8, fg=INK, bg=VOLT, pad=14, thick=2, font='body', alpha=1.0, px=None):
    """A rounded label; org is its text baseline-left (like the old cv2 call). Returns the right edge x."""
    px = px or px_of(scale); tw, th = tx.text_size(text, px, font); x, y = int(org[0]), int(org[1])
    hh = int(px * 0.95) + pad
    rounded_rect(img, (x, y - int(px * 0.78) - pad // 2), (x + tw + 2 * pad, y - int(px * 0.78) - pad // 2 + hh), bg, r=hh // 2, alpha=alpha)
    tx.put_text(img, text, (x + pad, y), px, fg, font=font)
    return x + tw + 2 * pad


def pill_c(img, text, cx, cy, px, fg=INK, bg=VOLT, pad=None, font='display', alpha=1.0):
    """A pill centred on (cx, cy). Returns (x0, y0, x1, y1)."""
    pad = pad or int(px * 0.55); tw, _ = tx.text_size(text, px, font); hh = int(px * 1.5)
    x0 = int(cx - tw / 2 - pad); x1 = int(cx + tw / 2 + pad); y0 = int(cy - hh / 2); y1 = y0 + hh
    rounded_rect(img, (x0, y0), (x1, y1), bg, r=hh // 2, alpha=alpha)
    tx.put_text(img, text, (int(cx), int(cy)), px, fg, font=font, anchor='mm')
    return x0, y0, x1, y1


def _roi_box(pts, pad, size):
    cw, ch = size; P = np.asarray(pts, np.float32)
    x0 = int(max(0, np.floor(P[:, 0].min() - pad))); y0 = int(max(0, np.floor(P[:, 1].min() - pad)))
    x1 = int(min(cw, np.ceil(P[:, 0].max() + pad))); y1 = int(min(ch, np.ceil(P[:, 1].max() + pad)))
    return x0, y0, x1, y1


def draw_skeleton(img, pts, vis=None, color=BGR['volt'], thickness=3, dots=True, alpha=1.0, outline=True):
    if alpha < 1.0:
        x0, y0, x1, y1 = _roi_box(pts, thickness * 3 + 6, (img.shape[1], img.shape[0]))
        if x1 <= x0 or y1 <= y0: return img
        roi = img[y0:y1, x0:x1]; over = roi.copy()
        draw_skeleton(over, np.asarray(pts, np.float32) - np.array([x0, y0], np.float32), vis, color, thickness, dots, 1.0, outline)
        cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)
        return img
    for a, b in SKELETON_EDGES:
        if FEET_OFF and (a in FEET_JOINTS or b in FEET_JOINTS): continue
        if vis is not None and (vis[a] < VIS_THR or vis[b] < VIS_THR): continue
        pa, pb = tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int))
        if outline: cv2.line(img, pa, pb, BGR['black'], thickness + 4, cv2.LINE_AA)
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    if dots:
        for i in ((0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26) if FEET_OFF else (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)):
            if vis is not None and vis[i] < VIS_THR: continue
            cv2.circle(img, tuple(np.round(pts[i]).astype(int)), thickness + 1, color, -1, cv2.LINE_AA)
    return img


def ghost_placement(pts, vis, ghost, size):
    """Where to overlay the ghost on the caller: (anchor_xy, scale_px) or None. Hips visible: hip centre at torso scale.
    Only shoulders: match the shoulder midpoint and shoulder width. Then it must fit the picture; otherwise the panel alone."""
    if pts is None or ghost is None: return None
    g = np.asarray(ghost, np.float32); cw, ch = size
    hips_ok = vis[L_HIP] > VIS_THR and vis[R_HIP] > VIS_THR; sh_ok = vis[L_SH] > VIS_THR and vis[R_SH] > VIS_THR
    if hips_ok and sh_ok:
        anchor = 0.5 * (pts[L_HIP] + pts[R_HIP]); scale = float(np.linalg.norm(0.5 * (pts[L_SH] + pts[R_SH]) - anchor)) + 1e-3
    elif sh_ok:
        g_sh = 0.5 * (g[L_SH] + g[R_SH]); g_sw = float(np.linalg.norm(g[L_SH] - g[R_SH])) + 1e-3
        l_sh = 0.5 * (pts[L_SH] + pts[R_SH]); l_sw = float(np.linalg.norm(pts[L_SH] - pts[R_SH])) + 1e-3
        scale = l_sw / g_sw; anchor = l_sh - g_sh * scale
    else:
        return None
    if scale < 0.04 * ch or scale > 0.6 * ch: return None
    P = anchor + g * scale
    legs_ok = vis[25] > VIS_THR and vis[26] > VIS_THR
    joints = UPPER_JOINTS if not legs_ok else list(range(33))
    inside = np.mean((P[joints, 0] >= -0.05 * cw) & (P[joints, 0] <= 1.05 * cw) & (P[joints, 1] >= -0.05 * ch) & (P[joints, 1] <= 1.05 * ch))
    if inside < 0.85: return None
    return anchor, scale, (not legs_ok)


UPPER_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 11), (0, 12), (11, 23), (12, 24), (23, 24)]
UPPER_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]


def draw_ghost(img, skel_norm, anchor, torso_px, color=BGR['ghost'], alpha=0.6, thickness=9, upper_only=False):
    """The ghost over the caller's body. upper_only: when the caller's legs are out of frame, draw only the ghost's upper body."""
    if skel_norm is None: return img
    pts = anchor + np.asarray(skel_norm, np.float32) * torso_px
    if not upper_only: return draw_skeleton(img, pts, None, color=color, thickness=thickness, dots=True, alpha=alpha)
    sub = pts[UPPER_JOINTS]
    x0, y0, x1, y1 = _roi_box(sub, thickness * 3 + 6, (img.shape[1], img.shape[0]))
    if x1 <= x0 or y1 <= y0: return img
    roi = img[y0:y1, x0:x1]; over = roi.copy(); off = np.array([x0, y0], np.float32)
    for a, b in UPPER_EDGES:
        pa, pb = tuple(np.round(pts[a] - off).astype(int)), tuple(np.round(pts[b] - off).astype(int))
        cv2.line(over, pa, pb, BGR['black'], thickness + 4, cv2.LINE_AA); cv2.line(over, pa, pb, color, thickness, cv2.LINE_AA)
    for i in UPPER_JOINTS: cv2.circle(over, tuple(np.round(pts[i] - off).astype(int)), thickness + 1, color, -1, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)
    return img


def panel_rect(size, small=False):
    cw, ch = size
    sx, sy, sx1, _ = safe_box(size); s0 = sc_of(size)
    if cw > ch:
        pw, ph = (int(cw * 0.17), int(ch * 0.46)) if small else (int(cw * 0.26), int(ch * 0.66)); x1 = sx1; y0 = sy + int(80 * s0)
    else:
        pw, ph = (int(cw * 0.27), int(ch * 0.2)) if small else (int(cw * 0.40), int(ch * 0.33)); x1 = sx1; y0 = sy + int(80 * s0)
    return x1 - pw, y0, x1, y0 + ph


def draw_panel(img, ghost, label, size, sub=None, highlight=False, small=False):
    """The reference dancer in a fixed box, whole body, scaled to fit. Draws nothing without a ghost."""
    if ghost is None: return img
    x0, y0, x1, y1 = panel_rect(size, small); s0 = sc_of(size)
    rounded_rect(img, (x0, y0), (x1, y1), BGR['panel'], r=18, alpha=0.7)
    if highlight: cv2.rectangle(img, (x0, y0), (x1, y1), CYAN, 2, cv2.LINE_AA)
    g = np.asarray(ghost, np.float32)
    gx0, gy0 = g.min(axis=0); gx1, gy1 = g.max(axis=0); gw, gh = max(gx1 - gx0, 0.6), max(gy1 - gy0, 2.4)
    pad = 16 if small else 24; top = 30 if small else 42
    s = min((x1 - x0 - 2 * pad) / gw, (y1 - y0 - top - pad) / gh)
    cx, cy = (x0 + x1) / 2, y0 + top + (y1 - y0 - top - pad) / 2
    anchor = np.array([cx - (gx0 + gx1) / 2 * s, cy - (gy0 + gy1) / 2 * s], np.float32)
    draw_skeleton(img, anchor + g * s, None, color=BGR['ghost'], thickness=max(3, int(s * 0.06)), dots=True)
    tx.put_text(img, label, (x0 + 12, y0 + (22 if small else 30)), int((18 if small else 22) * s0), CYAN if highlight else PAPER, font='display')
    if sub and not small: tx.put_text(img, sub, (x0 + 12, y1 - 10), int(16 * s0), MUTE)
    return img


def edge_glow(img, color, strength):
    """Tinted glow on the four edges (ON FIRE): four stepped bands per edge, each a uint8 blend of its own strip only."""
    if strength <= 0.01: return img
    ch, cw = img.shape[:2]; band = int(min(cw, ch) * 0.07); step = max(1, band // 4)
    for i, f in enumerate((1.0, 0.6, 0.33, 0.14)):
        a = float(min(1.0, strength * f)); o = i * step
        for (x0, y0, x1, y1) in ((o, o, cw - o, o + step), (o, ch - o - step, cw - o, ch - o), (o, o + step, o + step, ch - o - step), (cw - o - step, o + step, cw - o, ch - o - step)):
            if x1 <= x0 or y1 <= y0: continue
            roi = img[y0:y1, x0:x1]; cv2.addWeighted(roi, 1 - a, np.full_like(roi, color), a, 0, dst=roi)
    return img


def grade_ring(img, cx, cy, r, g, scale=1.0, alpha=1.0):
    col = GRADE_COL.get(g, PAPER); rr = max(4, int(r * scale))
    x0, y0, x1, y1 = int(cx - rr - 8), int(cy - rr - 8), int(cx + rr + 8), int(cy + rr + 8)
    H_, W_ = img.shape[:2]
    if x1 <= 0 or y1 <= 0 or x0 >= W_ or y0 >= H_: return img
    cx0, cy0 = max(0, x0), max(0, y0); roi = img[cy0:min(H_, y1), cx0:min(W_, x1)]; over = roi.copy()
    c = (int(cx - cx0), int(cy - cy0))
    cv2.circle(over, c, rr, INK, -1, cv2.LINE_AA); cv2.circle(over, c, rr, col, max(3, rr // 10), cv2.LINE_AA)
    cv2.addWeighted(over, alpha * 0.92, roi, 1 - alpha * 0.92, 0, dst=roi)
    tx.put_text(img, g, (int(cx), int(cy)), int(rr * 1.25), col, font='display', anchor='mm', alpha=alpha)
    return img


def wrap(text, px, max_w, font='body', max_lines=3):
    words = str(text).split(); lines = []; cur = ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if tx.text_size(t, px, font)[0] <= max_w or not cur: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines[:max_lines]


def fit_px(text, px, max_w, font='display', min_px=14):
    while px > min_px and tx.text_size(text, px, font)[0] > max_w: px -= 2
    return px


# ---- HUD pieces ----
def draw_topbar(img, hud, size):
    cw, ch = size; s0 = sc_of(size)
    rounded_rect(img, (0, 0), (cw, int(88 * s0)), INK, r=0, alpha=0.55)
    name = (hud.get('title') or hud['key']).replace('_', ' ')
    name = ' '.join(w[:1].upper() + w[1:] for w in name.split())
    st = hud['state']
    label = {'show': ('WATCH', CYAN), 'countin': ('GET READY', AMBER), 'attempt': ('GO!', VOLT), 'scored': ('RESULT', PINK),
             'feedback': ('RESULT', PINK), 'done': ('RESULT', PINK)}.get(st)
    if hud.get('paused'): label = ('PAUSED', AMBER)
    right = cw - 16
    if label:
        x0, _, _, _ = pill_c(img, label[0], right - tx.text_size(label[0], int(24 * s0), 'display')[0] / 2 - int(14 * s0), int(44 * s0), int(24 * s0), INK, label[1])
        right = x0 - 10
    max_w = right - 20
    tx.put_text(img, name, (20, int(42 * s0)), fit_px(name, int(30 * s0), max_w), PAPER, font='display')
    step = hud.get('step') or {}; bits = []
    if step.get('label'): bits.append(step['label'])
    if st == 'attempt' and hud.get('current_beat', -1) >= 0: bits.append(f"COUNT {min(hud['n_beats'], hud['current_beat'] + 1)}")
    bits.append(f"{hud['bpm']:.0f} BPM")
    x = tx.put_text(img, '  ·  '.join(bits), (20, int(74 * s0)), int(19 * s0), VOLT)
    diff = DIFF_LABEL.get(hud.get('difficulty') or '')
    if diff and x + 140 < right: pill(img, diff, (x + 14, int(73 * s0)), px=int(15 * s0), fg=INK, bg=PAPER, pad=8)
    if hud.get('focus') == 'arms' and x + 300 < right: pill(img, 'ARMS ONLY', (x + 150, int(73 * s0)), px=int(15 * s0), fg=INK, bg=AMBER, pad=8)
    return img


def draw_timeline(img, verdicts, current, n_beats, size, t=0.0, squad=None):
    cw, ch = size; n = max(1, n_beats); pad = 18; h = int(40 * sc_of(size)); gap = 5 if n <= 8 else 3; y0 = ch - h - 22
    cwid = (cw - 2 * pad - gap * (n - 1)) / n; s0 = sc_of(size)
    if squad:
        rh = int(12 * s0); yy = y0 - (rh + 4) * len(squad) - 6
        for row in squad:
            col = PLAYER.get(row['colour'], PAPER); cv2.circle(img, (pad - 8, yy + rh // 2), 5, col, -1, cv2.LINE_AA)
            for b in range(n):
                v = row['verdicts'][b] if b < len(row['verdicts']) else None; x = int(pad + b * (cwid + gap))
                rounded_rect(img, (x, yy), (int(x + cwid), yy + rh), (60, 60, 64) if v is None else BGR[v], r=4, alpha=0.85 if v is None else 1.0)
            yy += rh + 4
    for b in range(n):
        x = int(pad + b * (cwid + gap)); v = verdicts[b] if b < len(verdicts) else None
        color = (64, 62, 62) if v is None else BGR[v]
        rounded_rect(img, (x, y0), (int(x + cwid), y0 + h), color, r=10, alpha=0.85 if v is None else 1.0)
        if b == current:
            grow = int(2 + 3 * fxm.pulse(t, 2.0)); cv2.rectangle(img, (x - grow, y0 - grow), (int(x + cwid) + grow, y0 + h + grow), PAPER, 3, cv2.LINE_AA)
        if cwid >= 26:
            tx.put_text(img, str(b + 1), (int(x + cwid / 2), y0 + h // 2), int(18 * s0), INK if v in ('amber', 'green') else PAPER, font='display', anchor='mm')
    return img


def draw_chips(img, parts, hint, size, only_missing=False):
    """Body-visibility chips above the timeline, and the framing hint on a big amber plate."""
    cw, ch = size; s0 = sc_of(size); y = ch - int(40 * s0) - 22 - 16; x = 18
    if not (only_missing and all(parts.get(p, False) for p in ('head', 'arms', 'torso'))):
        for name in ('head', 'arms', 'torso', 'legs'):
            ok = parts.get(name, False)
            x = pill(img, name.upper(), (x, y), px=int(15 * s0), fg=INK if ok else PAPER, bg=VOLT if ok else (60, 58, 58), pad=9) + 6
    if hint:
        px = int(24 * s0); lines = wrap(hint, px, cw - 80, max_lines=2); lh = int(px * 1.35)
        y1 = y - int(34 * s0); y0 = y1 - lh * len(lines) - 18
        plate(img, 18, y0, cw - 18, y1, AMBER, alpha=0.92, r=16)
        for i, l in enumerate(lines): tx.put_text(img, l, (cw // 2, y0 + 12 + lh * i + int(px * 0.8)), px, INK, font='display', anchor='ms')
    return img


def draw_history(img, history, size, best=None):
    """Past phrase scores: three small chips, newest first, on the left edge above the chips."""
    if not history: return img
    cw, ch = size; s0 = sc_of(size); rows = list(reversed(history[-3:]))
    y = ch - int(40 * s0) - 22 - 16 - int(36 * s0) - 8; x = 18
    tx.put_text(img, 'YOUR SCORES' + (f'   BEST {best:.0f}' if best is not None else ''), (x, y - int(40 * s0)), int(14 * s0), MUTE, font='display')
    for h in rows:
        s = h.get('score'); g = progression.grade(s) if s is not None else '-'
        lab = f"{g} {s:.0f}" if s is not None else '-'
        x = pill(img, lab, (x, y), px=int(17 * s0), fg=INK, bg=GRADE_COL.get(g, (70, 70, 74)), pad=9) + 6
    return img


def _head_anchor(pts, vis, size):
    if pts is None: return size[0] // 2, int(170 * sc_of(size))
    tl = float(np.linalg.norm(0.5 * (pts[L_SH] + pts[R_SH]) - 0.5 * (pts[L_HIP] + pts[R_HIP]))) + 1e-3
    hx = float(pts[0, 0]) if vis[0] > VIS_THR else float(0.5 * (pts[L_SH, 0] + pts[R_SH, 0]))
    hy = float(pts[0, 1]) if vis[0] > VIS_THR else float(0.5 * (pts[L_SH, 1] + pts[R_SH, 1])) - 0.4 * tl
    return int(np.clip(hx, 90, size[0] - 90)), int(max(130 * sc_of(size), hy - 0.55 * tl - 20))


def draw_judgments(img, fxq, t, anchors, size):
    """PERFECT / NICE / MISS above each dancer's head. anchors: {pid or None: (x, y)}."""
    s0 = sc_of(size)
    for f in fxq.active(t, ('judgment',)):
        word, fill, stroke = JUDGE.get(f.data.get('verdict'), (None, None, None))
        if not word: continue
        age = f.age(t); x, y = anchors.get(f.data.get('pid'), anchors.get(None, (size[0] // 2, int(170 * s0))))
        sc = fxm.pop_scale(age) * (0.7 if f.data.get('small') else 1.0); a = 1.0 if age < 0.35 else max(0.0, 1.0 - (age - 0.35) / 0.3)
        col = PLAYER.get(f.data.get('colour'), fill) if f.data.get('small') and f.data.get('verdict') == 'green' else fill
        tx.put_text(img, word, (x, int(y - 40 * min(1.0, age / 0.6))), int(52 * s0), col, font='display', anchor='ms', stroke=4, stroke_color=stroke, alpha=a, scale=sc)
    return img


def draw_combo(img, hud, size, t):
    s0 = sc_of(size); combo = hud.get('combo', 0)
    sx, sy, _, _ = safe_box(size)
    if hud.get('running_score') is not None:
        x = tx.put_text(img, f"{hud['running_score']:.0f}", (sx, sy + int(200 * s0)), int(58 * s0), PAPER, font='display', stroke=3)
        tx.put_text(img, 'SCORE', (x + 8, sy + int(200 * s0)), int(18 * s0), PAPER, font='display', stroke=2)
    if combo >= 2:
        fire = combo >= 4
        bg = tuple(int(a + (b - a) * fxm.pulse(t, 3.0)) for a, b in zip(PINK, VOLT)) if fire else VOLT
        pill(img, (f"ON FIRE x{combo}" if fire else f"x{combo} COMBO"), (sx, sy + int(252 * s0)), px=int(24 * s0), fg=INK, bg=bg, pad=12, font='display')
        if fire: edge_glow(img, PINK, 0.18 + 0.32 * min(combo, 12) / 12.0 * (0.7 + 0.3 * fxm.pulse(t, 2.0)))
    for f in (hud['fx'].active(t, ('combo',)) if hud.get('fx') is not None else ()):
        age = f.age(t); a = max(0.0, 1.0 - age / f.dur)
        tx.put_text(img, f"{f.data.get('n', combo)} COMBO!", (size[0] // 2, int(size[1] * 0.25)), int(46 * s0), VOLT, font='display', anchor='ms', stroke=4,
                    stroke_color=PINK, alpha=a, scale=fxm.pop_scale(age, 0.2))
    return img


def draw_tip(img, fxq, t, size):
    s0 = sc_of(size); cw, ch = size
    for f in fxq.active(t, ('tip',))[-1:]:
        age = f.age(t); a = min(1.0, age / 0.12) * (1.0 if age < f.dur - 0.3 else max(0.0, (f.dur - age) / 0.3))
        y = safe_box(size)[3] - int(110 * s0) - (0 if age > 0.15 else int((0.15 - age) * 200))
        colour = PLAYER.get(f.data.get('colour'), AMBER)
        pill_c(img, f.data.get('short') or f.data.get('text', ''), cw // 2, y, int(26 * s0), INK, colour, alpha=0.95 * a)
    return img


def draw_gesture_ring(img, gesture, size):
    if not gesture or not gesture.get('kind') or gesture.get('p', 0) <= 0.02: return img
    s0 = sc_of(size); cx, cy = size[0] // 2, int(230 * s0); r = int(44 * s0)
    col = AMBER if gesture['kind'] == 'pause' else VOLT
    plate(img, cx - r - 10, cy - r - 10, cx + r + 10, cy + r + 46 * s0, INK, 0.6, r=20)
    cv2.circle(img, (cx, cy), r, (70, 70, 74), 6, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r, r), -90, 0, int(360 * gesture['p']), col, 6, cv2.LINE_AA)
    tx.put_text(img, gesture['kind'].upper(), (cx, cy + r + int(30 * s0)), int(18 * s0), col, font='display', anchor='ms')
    return img


def draw_battle(img, battle, size, t):
    if not battle: return img
    cw, ch = size; s0 = sc_of(size); y0 = int(96 * s0); x0, x1 = 20, cw - 20; hgt = int(22 * s0)
    mx = float(battle.get('ghost_max', 300)); hp = max(0.0, float(battle.get('ghost_hp', mx)))
    plate(img, x0, y0, x1, y0 + hgt, (40, 38, 38), 0.85, r=hgt // 2)
    fw = int((x1 - x0) * hp / mx)
    if fw > hgt: rounded_rect(img, (x0, y0), (x0 + fw, y0 + hgt), PINK, r=hgt // 2)
    tx.put_text(img, f"GHOST HP {hp:.0f}", (x0 + 12, y0 + hgt // 2), int(15 * s0), PAPER, font='display', anchor='lm')
    if battle.get('round'): tx.put_text(img, f"ROUND {battle['round']}/{battle.get('rounds', 3)}", (x1 - 12, y0 + hgt // 2), int(15 * s0), PAPER, font='display', anchor='rm')
    return img


def draw_squad_strip(img, squad, size):
    if not squad: return img
    s0 = sc_of(size); x = 20; y = int(124 * s0)
    for row in sorted(squad, key=lambda r: -(r.get('score') if r.get('score') is not None else (r.get('running') or 0))):
        col = PLAYER.get(row['colour'], PAPER); val = row.get('score') if row.get('score') is not None else row.get('running')
        lab = f"{(row.get('name') or row['colour']).upper()} {'' if val is None else f'{val:.0f}'}" + (f" x{row['combo']}" if row.get('combo', 0) >= 4 else '')
        x = pill(img, lab, (x, y), px=int(17 * s0), fg=INK, bg=col, pad=9, font='display') + 6
    return img


def draw_reveal(img, hud, size, t):
    """After a phrase: practice gets one word; YOUR TURN a grade and title; the challenge the score counting up, then grade and title.
    One cue line at most. No stat lines."""
    cw, ch = size; s0 = sc_of(size); r = hud.get('result') or {}; age = max(0.0, t - (hud.get('t_state') or t))
    if hud.get('fx') is not None and hud['fx'].active(t, ('ko', 'levelup')): return img
    sq = hud.get('squad_result')
    if sq and sq.get('ranking'):
        plate(img, int(cw * 0.08), int(ch * 0.15), int(cw * 0.92), int(ch * 0.66), INK, 0.62, r=28)
        return draw_squad_reveal(img, hud, size, t, sq)
    mid = cw // 2; stage = hud.get('stage') or 'YOUR TURN'
    if hud.get('practice') or stage in ('WITH ME', 'WARM-UP'):
        tx.put_text(img, 'NICE!', (mid, int(ch * 0.40)), int(110 * s0), VOLT, font='display', anchor='ms', stroke=5, stroke_color=INK, scale=fxm.pop_scale(age, 0.2))
        return img
    if not r.get('scored'):
        pill_c(img, 'Step back so I can see you', mid, int(ch * 0.40), int(30 * s0), INK, AMBER)
        return img
    score = float(r.get('score') or 0); g = progression.grade(score); land = cw > ch
    gy = int(ch * (0.32 if land else 0.30)); ty = gy + int(130 * s0)
    if stage == 'CHALLENGE':
        shown = score * fxm.ease_out_cubic(min(1.0, age / 0.8))
        tx.put_text(img, f"{shown:.0f}", (mid, gy + int(55 * s0)), int(140 * s0), PAPER, font='display', anchor='ms', stroke=4)
        gy -= int(150 * s0) if not land else int(120 * s0)
    if age >= (0.85 if stage == 'CHALLENGE' else 0.1):
        u = min(1.0, (age - (0.85 if stage == 'CHALLENGE' else 0.1)) / 0.35)
        grade_ring(img, mid, gy, int(64 * s0), g, scale=1.8 - 0.8 * fxm.ease_out_back(u), alpha=min(1.0, u * 2))
    if age >= 0.9:
        a = min(1.0, (age - 0.9) / 0.25); title = (hud.get('award_title') or progression.title_for(r)).upper()
        tx.put_text(img, title, (mid, ty), fit_px(title, int(40 * s0), cw * 0.84), PINK, font='display', anchor='ms', alpha=a, stroke=3)
    if age >= 1.4 and hud.get('cue') and hud['cue'].get('kind') != 'praise':
        cue = hud['cue']; a = min(1.0, (age - 1.4) / 0.2); t_ = cue.get('short') or cue.get('text', '')
        pill_c(img, t_, mid, ty + int(62 * s0), fit_px(t_, int(26 * s0), cw * 0.8, font='display'), INK, AMBER, alpha=a)
    return img


def draw_squad_reveal(img, hud, size, t, sq):
    cw, ch = size; s0 = sc_of(size); age = t - (hud.get('t_state') or t); mid = cw // 2; y = int(ch * 0.22)
    tx.put_text(img, 'SQUAD RESULTS', (mid, y), int(36 * s0), VOLT, font='display', anchor='ms'); y += int(58 * s0)
    for i, row in enumerate(sq['ranking']):
        if age < 0.25 * i: break
        col = PLAYER.get(row['colour'], PAPER)
        plate(img, int(cw * 0.12), y - int(36 * s0), int(cw * 0.88), y + int(16 * s0), col, 0.9, r=16)
        tx.put_text(img, f"#{row['rank']}  {(row.get('name') or row['colour']).upper()}", (int(cw * 0.15), y), int(24 * s0), INK, font='display')
        tx.put_text(img, f"{row['grade']} {row['score']:.0f}", (int(cw * 0.85), y), int(24 * s0), INK, font='display', anchor='rs')
        y += int(62 * s0)
    if age >= 0.3 * len(sq['ranking']) + 0.2:
        if sq.get('crew_sync') is not None:
            tx.put_text(img, f"CREW SYNC {100 * sq['crew_sync']:.0f}%", (mid, y + int(10 * s0)), int(34 * s0), CYAN, font='display', anchor='ms'); y += int(52 * s0)
        tx.put_text(img, sq['squad_title'].upper(), (mid, y + int(10 * s0)), fit_px(sq['squad_title'].upper(), int(34 * s0), cw * 0.8), PINK, font='display', anchor='ms')
        if hud.get('cue'):
            pill_c(img, hud['cue']['text'], mid, y + int(70 * s0), fit_px(hud['cue']['text'], int(22 * s0), cw * 0.8, 'body'), INK, AMBER, font='body')
    return img


def draw_overlays(img, fxq, t, size):
    """Level-up, achievements, K.O., MVP, flash and confetti (on top of everything)."""
    if fxq is None: return img
    cw, ch = size; s0 = sc_of(size)
    for f in fxq.active(t, ('confetti',)):
        for (x, y, rot, col, w, h, a) in fxm.confetti_particles(f.data.get('seed', 1), 140, size, f.age(t), f.data.get('origin')):
            if a <= 0: continue
            box = cv2.boxPoints(((x, y), (w * s0, h * s0), rot)).astype(np.int32); cv2.fillConvexPoly(img, box, col, cv2.LINE_AA)
    for f in fxq.active(t, ('flash',)):
        a = max(0.0, 1.0 - f.age(t) / f.dur) * 0.55; rounded_rect(img, (0, 0), (cw, ch), PAPER, r=0, alpha=a)
    for f in fxq.active(t, ('ko',)):
        age = f.age(t); a = 1.0 if age < f.dur - 0.4 else max(0.0, (f.dur - age) / 0.4)
        tx.put_text(img, f.data.get('text', 'K.O.!'), (cw // 2, int(ch * 0.42)), int(150 * s0), PINK, font='display', anchor='ms', stroke=6, stroke_color=VOLT,
                    alpha=a, scale=fxm.pop_scale(age, 0.25) * (1.0 + 0.03 * math.sin(age * 40) * max(0.0, 1 - age)))
    for f in fxq.active(t, ('levelup',)):
        age = f.age(t); a = min(1.0, age / 0.2) * (1.0 if age < f.dur - 0.4 else max(0.0, (f.dur - age) / 0.4))
        if age < 0.25: rounded_rect(img, (0, 0), (cw, ch), PAPER, r=0, alpha=0.5 * (1 - age / 0.25))
        y0 = int(ch * 0.30); plate(img, int(cw * 0.06), y0, int(cw * 0.94), y0 + int(250 * s0), INK, 0.88 * a, r=28)
        sc = fxm.pop_scale(age, 0.25)
        tx.put_text(img, 'LEVEL UP', (cw // 2, y0 + int(92 * s0)), int(74 * s0), VOLT, font='display', anchor='ms', stroke=4, stroke_color=PINK, alpha=a, scale=sc)
        tx.put_text(img, str(f.data.get('level_name', '')).upper(), (cw // 2, y0 + int(160 * s0)), int(44 * s0), PAPER, font='display', anchor='ms', alpha=a)
        if f.data.get('unlocks'):
            tx.put_text(img, ' + '.join(f"{u.capitalize()} unlocked" for u in f.data['unlocks']), (cw // 2, y0 + int(214 * s0)), int(24 * s0), CYAN, anchor='ms', alpha=a)
    toasts = fxq.active(t, ('achievement',))
    for i, f in enumerate(toasts[-2:]):
        age = f.age(t); slide = min(1.0, age / 0.25); a = 1.0 if age < f.dur - 0.35 else max(0.0, (f.dur - age) / 0.35)
        y0 = int((100 + 74 * i) * s0) - int((1 - slide) * 60 * s0); x0, x1 = int(cw * 0.12), int(cw * 0.88)
        plate(img, x0, y0, x1, y0 + int(62 * s0), VOLT, 0.95 * a, r=20)
        tx.put_text(img, 'ACHIEVEMENT', (x0 + 18, y0 + int(24 * s0)), int(14 * s0), INK, font='display', alpha=a)
        tx.put_text(img, str(f.data.get('name', '')), (x0 + 18, y0 + int(50 * s0)), int(24 * s0), INK, font='display', alpha=a)
    for f in fxq.active(t, ('mvp',)):
        age = f.age(t); a = min(1.0, age / 0.2) * (1.0 if age < f.dur - 0.4 else max(0.0, (f.dur - age) / 0.4))
        tx.put_text(img, f"MVP: {f.data.get('label', '').upper()}", (cw // 2, int(ch * 0.16)), int(48 * s0), PLAYER.get(f.data.get('colour'), VOLT),
                    font='display', anchor='ms', stroke=4, alpha=a, scale=fxm.pop_scale(age, 0.25))
    return img


def draw_caption(img, caption, size, y_bottom=None):
    if not caption: return img
    cw, ch = size; s0 = sc_of(size); px = int(24 * s0); lines = wrap(caption, px, cw - 90, max_lines=2); lh = int(px * 1.35)
    y1 = y_bottom or ch - int(150 * s0); y0 = y1 - lh * len(lines) - 20
    plate(img, 24, y0, cw - 24, y1, INK, 0.72, r=16)
    for i, l in enumerate(lines): tx.put_text(img, l, (cw // 2, y0 + 10 + lh * i + int(px * 0.85)), px, PAPER, anchor='ms')
    return img


def draw_paused(img, size):
    cw, ch = size; s0 = sc_of(size)
    rounded_rect(img, (0, 0), (cw, ch), INK, r=0, alpha=0.55)
    tx.put_text(img, 'PAUSED', (cw // 2, int(ch * 0.42)), int(96 * s0), AMBER, font='display', anchor='ms', stroke=4)
    tx.put_text(img, "Raise both hands or say 'resume'", (cw // 2, int(ch * 0.42) + int(64 * s0)), int(26 * s0), PAPER, anchor='ms')
    return img


def draw_safety_card(img, size=None):
    ch, cw = img.shape[:2]; size = (cw, ch); s0 = sc_of(size)
    out = cv2.addWeighted(img, 0.25, np.zeros_like(img), 0.75, 0)
    y = int(ch * 0.30)
    tx.put_text(out, 'TAKING A BREAK', (cw // 2, y), int(52 * s0), CYAN, font='display', anchor='ms'); y += int(70 * s0)
    for line in ('Stop if anything hurts or you feel dizzy.', 'Sit down, breathe, drink some water.', "If it doesn't ease up, get help from", 'someone nearby or a doctor.'):
        tx.put_text(out, line, (cw // 2, y), int(26 * s0), PAPER, anchor='ms'); y += int(42 * s0)
    return out


def draw_main(img, hud, size, t=0.0):
    """Centre messages by state: the 5-6-7-8 count-in and the result reveal."""
    cw, ch = size; st = hud['state']; px0, _, _, _ = panel_rect(size); mid = px0 // 2 if st == 'countin' else cw // 2; s0 = sc_of(size)
    if st == 'countin' and hud.get('countin_label'):
        k = int(hud['countin_label']); bp = hud.get('beat_pos') or 0.0; frac = bp - math.floor(bp)
        cy = int(ch * (0.30 if cw < ch else 0.42)); sz = int((150 if cw < ch else 170) * s0)
        tx.put_text(img, str(k), (mid, cy), sz, VOLT, font='display', anchor='ms', stroke=6, stroke_color=PINK, scale=1.25 - 0.25 * fxm.ease_out_cubic(min(1.0, frac * 2)))
        x = mid - int(84 * s0)
        for n in (5, 6, 7, 8):
            tx.put_text(img, str(n), (x, cy + int(56 * s0)), int(30 * s0), VOLT if n == k else PAPER, font='display', anchor='ms', stroke=2); x += int(56 * s0)
    elif st == 'countin' and hud.get('countdown'):
        s = str(hud['countdown']); tx.put_text(img, s, (mid, int(ch * 0.42)), int(150 * s0), VOLT, font='display', anchor='ms', stroke=6)
    elif st in ('scored', 'feedback', 'done') and (hud.get('result') is not None or hud.get('squad_result')):
        draw_reveal(img, hud, size, t)
    return img


# ---- the end scorecard ----
@lru_cache(maxsize=16)
def qr_matrix(data):
    try:
        import segno
    except Exception:
        return None
    q = segno.make(data, error='m', micro=False)
    return np.array([[1 if v else 0 for v in row] for row in q.matrix], np.uint8)


def draw_qr(img, data, x, y, px):
    """A QR code (dark modules on white with a quiet zone) with its top-left at (x, y), px wide. Returns True when drawn."""
    m = qr_matrix(data)
    if m is None: return False
    n = m.shape[0] + 8; cell = max(2, px // n); size = cell * n
    tile = np.full((size, size, 3), 255, np.uint8)
    big = np.kron(m, np.ones((cell, cell), np.uint8)); o = 4 * cell
    tile[o:o + big.shape[0], o:o + big.shape[1]][big == 1] = 0
    H_, W_ = img.shape[:2]
    if x < 0 or y < 0 or x + size > W_ or y + size > H_: return False
    img[y:y + size, x:x + size] = tile
    return True


def _card_bg(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 10), max(1, h // 10)), interpolation=cv2.INTER_AREA)
    blur = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(blur, 0.3, np.full_like(img, INK), 0.7, 0)


def _bar(img, x, y, w, h, frac, col, label, value, s0):
    tx.put_text(img, label, (x, y - 8), int(16 * s0), MUTE, font='display')
    tx.put_text(img, value, (x + w, y - 8), int(16 * s0), PAPER, font='display', anchor='rs')
    rounded_rect(img, (x, y), (x + w, y + h), (48, 46, 46), r=h // 2)
    fw = int(w * max(0.0, min(1.0, frac)))
    if fw > h: rounded_rect(img, (x, y), (x + fw, y + h), col, r=h // 2)


def _draw_classic_card(img, result, key_name, extra_lines=(), code=None, title=None, grade=None, award_title=None, xp=None, achievements=(),
              handle=None, mode_label=None, squad=None, total_line=None, perfects=None, routine=None):
    """End-of-call scorecard over a blurred, darkened frame, built to be screenshotted and shared. Every keyword is optional;
    the old positional call (img, result, key_name, extra_lines, code, title) still works."""
    ch, cw = img.shape[:2]; size = (cw, ch); s0 = sc_of(size); land = cw > ch
    out = _card_bg(img)
    lx0, lx1 = (40, int(cw * 0.52)) if land else (40, cw - 40)
    rx0, rx1 = (int(cw * 0.56), cw - 40) if land else (40, cw - 40)
    mid_l = (lx0 + lx1) // 2
    # header
    tx.put_text(out, 'BRODIE', (lx0, int(66 * s0)), int(30 * s0), VOLT, font='display')
    if mode_label: pill_c(out, mode_label.upper(), (lx1 if not land else cw - 40) - tx.text_size(mode_label.upper(), int(18 * s0), 'display')[0] / 2 - 16, int(56 * s0), int(18 * s0), INK, PINK)
    head = (title or routine or key_name or '').replace('_', ' ')
    head = ' '.join(w[:1].upper() + w[1:] for w in head.split())
    tx.put_text(out, head, (lx0, int(104 * s0)), fit_px(head, int(24 * s0), lx1 - lx0, 'body'), PAPER)
    y = int(130 * s0)
    if squad and squad.get('ranking'):
        y += int(20 * s0)
        tx.put_text(out, 'SQUAD', (mid_l, y + int(30 * s0)), int(34 * s0), VOLT, font='display', anchor='ms'); y += int(60 * s0)
        for row in squad['ranking']:
            col = PLAYER.get(row['colour'], PAPER)
            plate(out, lx0, y, lx1, y + int(64 * s0), col, 0.95, r=18)
            tx.put_text(out, f"#{row['rank']} {(row.get('name') or row['colour']).upper()}", (lx0 + 16, y + int(28 * s0)), int(22 * s0), INK, font='display')
            tx.put_text(out, row.get('title', ''), (lx0 + 16, y + int(52 * s0)), int(16 * s0), INK)
            tx.put_text(out, f"{row['grade']} {row['score']:.0f}", (lx1 - 16, y + int(42 * s0)), int(30 * s0), INK, font='display', anchor='rs')
            y += int(76 * s0)
        if squad.get('crew_sync') is not None:
            _bar(out, lx0, y + int(30 * s0), lx1 - lx0, int(16 * s0), squad['crew_sync'], CYAN, 'CREW SYNC', f"{100 * squad['crew_sync']:.0f}%", s0); y += int(64 * s0)
        st = squad.get('squad_title', '').upper()
        tx.put_text(out, st, (mid_l, y + int(30 * s0)), fit_px(st, int(38 * s0), lx1 - lx0), PINK, font='display', anchor='ms'); y += int(56 * s0)
    elif result is not None and result.get('scored'):
        score = float(result['score']); g = grade or progression.grade(score); k = 0.78 if land else 1.0
        grade_ring(out, mid_l, y + int(110 * s0 * k), int(92 * s0 * k), g); y += int(232 * s0 * k)
        big = total_line or f"{score:.0f}"
        tx.put_text(out, big, (mid_l, y + int(90 * s0 * k)), int((110 if len(big) <= 3 else 80) * s0 * k), PAPER, font='display', anchor='ms', stroke=3); y += int(118 * s0 * k)
        at = (award_title or progression.title_for(result)).upper()
        tx.put_text(out, at, (mid_l, y + int(20 * s0)), fit_px(at, int(42 * s0), lx1 - lx0), PINK, font='display', anchor='ms'); y += int(54 * s0)
        if perfects is not None:
            tx.put_text(out, f"{perfects} PERFECT COUNTS", (mid_l, y + int(30 * s0)), int(20 * s0), PAPER, font='display', anchor='ms'); y += int(48 * s0)
    elif result is not None:
        tx.put_text(out, 'NO SCORE: STEP BACK', (mid_l, y + int(80 * s0)), int(40 * s0), AMBER, font='display', anchor='ms'); y += int(120 * s0)
    else:
        y += int(30 * s0)
    for line in extra_lines or ():
        if y > ch * (0.62 if not land else 0.85): break
        tx.put_text(out, str(line), (mid_l, y + int(26 * s0)), fit_px(str(line), int(24 * s0), lx1 - lx0, 'body'), PAPER, anchor='ms'); y += int(40 * s0)
    # right column (or the bottom half in portrait): XP, achievements, the challenge block
    ry = int(130 * s0) if land else max(y + int(10 * s0), int(ch * 0.66) - (int(110 * s0) if xp else 0) - (int(50 * s0) if achievements else 0))
    if xp:
        tx.put_text(out, f"+{xp.get('gained', 0)} XP", (rx0, ry + int(30 * s0)), int(30 * s0), VOLT, font='display')
        tx.put_text(out, f"{str(xp.get('level_name', '')).upper()}" + (f"   STREAK {xp['streak']}" if xp.get('streak', 0) >= 2 else ''), (rx1, ry + int(30 * s0)), int(18 * s0), PAPER, font='display', anchor='rs')
        lo, nxt, cur = xp.get('level_floor', 0), xp.get('next_at'), xp.get('xp_after', 0)
        frac = 1.0 if not nxt else (cur - lo) / max(1, nxt - lo)
        _bar(out, rx0, ry + int(56 * s0), rx1 - rx0, int(16 * s0), frac, VOLT, '', '' if not nxt else f"next level at {nxt}", s0)
        ry += int(100 * s0)
    if achievements:
        x = rx0
        for a in list(achievements)[:1]:
            nm = a['name'] if isinstance(a, dict) else str(a)
            if x + tx.text_size(nm, int(16 * s0), 'display')[0] + 30 > rx1: break
            x = pill(out, nm, (x, ry + int(30 * s0)), px=int(16 * s0), fg=INK, bg=CYAN, pad=10, font='display') + 8
        ry += int(56 * s0)
    if code:
        by0 = max(ry + int(10 * s0), ch - int(270 * s0)) if not land else max(ry + int(10 * s0), ch - int(300 * s0))
        by1 = ch - int(30 * s0)
        plate(out, rx0 - 10, by0, rx1 + 10, by1, (26, 24, 24), 0.92, r=24)
        qr_px = min(int(200 * s0), by1 - by0 - 30)
        has_qr = bool(handle) and draw_qr(out, f"facetime:{str(handle).replace(' ', '')}", rx1 - qr_px - 6, by0 + (by1 - by0 - qr_px) // 2, qr_px)
        tx_w = (rx1 - qr_px - 26 - rx0) if has_qr else (rx1 - rx0)
        tx.put_text(out, 'BEAT MY SCORE', (rx0 + 10, by0 + int(44 * s0)), int(24 * s0), VOLT, font='display')
        tx.put_text(out, str(code), (rx0 + 10, by0 + int(150 * s0)), fit_px(str(code), int(110 * s0), tx_w - 10), PAPER, font='display')
        how = f"FaceTime {handle}" if handle else 'FaceTime Brodie'
        tx.put_text(out, how, (rx0 + 10, by1 - int(48 * s0)), fit_px(how, int(18 * s0), tx_w - 10, 'body'), MUTE)
        tx.put_text(out, f"say 'battle {code}'", (rx0 + 10, by1 - int(20 * s0)), fit_px(f"say 'battle {code}'", int(20 * s0), tx_w - 10, 'body'), PAPER)
    return out


def draw_photo_card(photo, result, code=None, title=None, award_title=None, xp=None, handle=None, mode_label=None, total_line=None, brag=(), achievements=()):
    """The share card: the dancer's own finish pose (sharp), the kit Brodie sticker beside them, grade, score, title, brag lines,
    the code, the QR and the hashtag. Built for a FaceTime screenshot: everything inside the safe area, code and QR in the middle band."""
    from .avatar import sprite
    ch, cw = photo.shape[:2]; size = (cw, ch); s0 = sc_of(size); x0, y0, x1, y1 = safe_box(size)
    out = photo.copy()
    g = np.linspace(0.0, 1.0, ch, dtype=np.float32)
    shade = (np.clip(0.75 - 1.6 * g, 0, 0.75) + np.clip(2.4 * (g - 0.55), 0, 0.85))[:, None, None]
    out = (out.astype(np.float32) * (1 - shade) + np.array(INK, np.float32) * shade).astype(np.uint8)
    tx.put_text(out, 'BRODIE', (x0, y0 + int(34 * s0)), int(32 * s0), VOLT, font='display')
    if mode_label: pill_c(out, mode_label.upper(), x1 - tx.text_size(mode_label.upper(), int(20 * s0), 'display')[0] // 2 - int(14 * s0), y0 + int(24 * s0), int(20 * s0), INK, PINK)
    y = y0 + int(80 * s0)
    if result is not None and result.get('scored'):
        score = float(result['score']); gr = progression.grade(score)
        grade_ring(out, x0 + int(64 * s0), y + int(64 * s0), int(60 * s0), gr)
        big = total_line or f"{score:.0f}"
        tx.put_text(out, big, (x0 + int(140 * s0), y + int(100 * s0)), int(96 * s0) if len(big) <= 3 else int(64 * s0), PAPER, font='display', stroke=4)
        y += int(150 * s0)
        at = (award_title or progression.title_for(result)).upper()
        tx.put_text(out, at, (x0, y + int(30 * s0)), fit_px(at, int(44 * s0), x1 - x0), PINK, font='display', stroke=3); y += int(56 * s0)
    for b in list(brag)[:2]:
        pill(out, b, (x0, y + int(30 * s0)), px=int(24 * s0), fg=INK, bg=VOLT, pad=12, font='display'); y += int(52 * s0)
    if xp and xp.get('gained'):
        tx.put_text(out, f"+{xp['gained']} XP  ·  {str(xp.get('level_name', '')).upper()}", (x0, y + int(30 * s0)), int(24 * s0), PAPER, font='display', stroke=3); y += int(44 * s0)
    stk = sprite('grin', int(ch * 0.2))                          # the kit Brodie sticker, next to the dancer
    sx = x1 - stk.shape[1]; sy = int(ch * 0.50) - stk.shape[0] // 2
    from .avatar import _blend
    _blend(out, stk, sx, sy)
    if code:
        by1 = y1; by0 = by1 - int(190 * s0)
        plate(out, x0 - 8, by0, x1 + 8, by1, (26, 24, 24), 0.92, r=24)
        qr_px = int(150 * s0)
        has_qr = bool(handle) and draw_qr(out, f"facetime:{str(handle).replace(' ', '')}", x1 - qr_px - 10, by0 + (by1 - by0 - qr_px) // 2, qr_px)
        tw = (x1 - qr_px - 30 - x0) if has_qr else (x1 - x0)
        tx.put_text(out, 'BEAT MY SCORE', (x0 + 12, by0 + int(40 * s0)), int(24 * s0), VOLT, font='display')
        tx.put_text(out, str(code), (x0 + 12, by0 + int(122 * s0)), fit_px(str(code), int(84 * s0), tw - 10), PAPER, font='display')
        tx.put_text(out, f"#BeatMyBrodie {code}", (x0 + 12, by1 - int(20 * s0)), fit_px(f"#BeatMyBrodie {code}", int(22 * s0), tw - 10, 'body'), PAPER)
    return out


def draw_card(img, result, key_name, extra_lines=(), code=None, title=None, grade=None, award_title=None, xp=None, achievements=(),
              handle=None, mode_label=None, squad=None, total_line=None, perfects=None, routine=None, photo=None, brag=()):
    """End-of-call scorecard. With `photo` (the dancer's finish pose, canvas-sized) it is the photo share card; otherwise the
    classic card over a blurred frame. Every keyword is optional; the old positional call still works."""
    if photo is not None and squad is None:
        return draw_photo_card(photo, result, code=code, title=title, award_title=award_title, xp=xp, handle=handle, mode_label=mode_label,
                               total_line=total_line, brag=brag, achievements=achievements)
    return _draw_classic_card(img, result, key_name, extra_lines, code, title, grade, award_title, xp, achievements, handle, mode_label, squad,
                              total_line, perfects, routine)


def mirror_skel(skel):
    """Mirror a normalised (33, 2) skeleton for display: flip x and swap left/right joints."""
    from .features import MIRROR_PAIRS
    s = np.array(skel, np.float32, copy=True); s[:, 0] = -s[:, 0]
    for l, r in MIRROR_PAIRS: s[[l, r]] = s[[r, l]]
    return s


def mirror_view(frame_bgr, lm_img, ghost):
    """Flip the frame, the landmarks and the ghost together for a mirror-like display (scoring stays on the unflipped pose)."""
    from .features import mirror_landmarks
    f = cv2.flip(frame_bgr, 1)
    lm = mirror_landmarks(lm_img)[0] if lm_img is not None else None
    g = mirror_skel(ghost) if ghost is not None else None
    return f, lm, g


# ---- the minimal, per-state layer (2026-10-11 redesign): one coach badge, one state chip, at most one line of text ----
STAGE_CHIP = dict(WATCH=('WATCH', CYAN), **{'WITH ME': ('WITH ME', CYAN), 'YOUR TURN': ('YOUR TURN', VOLT), 'WARM-UP': ('WARM-UP', AMBER)},
                  CHALLENGE=('CHALLENGE', PINK), PREPARE=('SETUP', AMBER), OFFER=('START?', VOLT), CONFIRM=('READY?', VOLT), PAUSED=('PAUSED', AMBER), REVIEW=('NICE', VOLT))
TEMPO_LABEL = ((0.55, 'VERY SLOW'), (0.68, 'SLOW'), (0.83, 'MEDIUM'), (0.95, 'FAST'), (9.0, 'FULL SPEED'))


SAFE = dict(top=0.09, bottom=0.15, side=0.05)                # FaceTime overlays its controls (bottom) and a self-view tile (corners)


def safe_box(size):
    cw, ch = size; land = cw > ch
    top = int(ch * (0.06 if land else SAFE['top'])); bottom = int(ch * (1 - (0.12 if land else SAFE['bottom']))); side = int(cw * SAFE['side'])
    return side, top, cw - side, bottom


def draw_badge(img, size, speaking=False, t=0.0, state='idle', still=False):
    """Brodie (official kit art) small at the top-left of the safe area, with his name. state: idle / listening / speaking /
    thinking / offline (see avatar.py)."""
    from .avatar import avatar
    s0 = sc_of(size); x0, y0, _, _ = safe_box(size); h = int(84 * s0)
    st = 'speaking' if speaking and state == 'idle' else state
    avatar().draw(img, x0 + h // 2 + 4, y0 + h // 2, h, st, t, still=still)
    tx.put_text(img, 'BRODIE', (x0 + h + int(14 * s0), y0 + h // 2), int(24 * s0), PAPER, font='display', anchor='lm', stroke=3)
    return img


def draw_state_chip(img, size, key):
    lab = STAGE_CHIP.get(key)
    if not lab: return img
    s0 = sc_of(size); _, y0, x1, _ = safe_box(size); px = int(26 * s0); w = tx.text_size(lab[0], px, 'display')[0]
    pill_c(img, lab[0], x1 - w // 2 - int(16 * s0), y0 + int(42 * s0), px, INK, lab[1])
    return img


def tempo_label(mult):
    for th, lab in TEMPO_LABEL:
        if mult <= th: return lab
    return 'FULL SPEED'


def draw_progress_row(img, hud, size):
    """Under the badge: the step (as dots when it is a step N/M) and the tempo as a word plus a 4-segment meter."""
    s0 = sc_of(size); sx, sy, _, _ = safe_box(size); x = sx + int(6 * s0); y = sy + int(112 * s0); step = hud.get('step') or {}
    lab = step.get('label') or ''
    m = re.match(r'STEP (\d+)/(\d+)', lab)
    if m:
        k, n = int(m.group(1)), int(m.group(2)); r = int(8 * s0)
        for i in range(n):
            c = (x + r + i * int(24 * s0), y)
            cv2.circle(img, c, r, VOLT if i < k else (90, 88, 88), -1 if i < k else 2, cv2.LINE_AA)
        x += n * int(24 * s0) + int(12 * s0)
    elif lab and lab != (hud.get('stage') or ''):
        x = tx.put_text(img, lab, (x, y + int(9 * s0)), int(22 * s0), PAPER, font='display', stroke=2) + int(14 * s0)
    mult = float(hud.get('tempo') or 1.0); seg = max(1, min(4, sum(1 for th, _ in TEMPO_LABEL[:4] if mult > th)))
    for i in range(4):
        x0 = x + i * int(18 * s0); cv2.rectangle(img, (x0, y - int(8 * s0)), (x0 + int(12 * s0), y + int(8 * s0)), VOLT if i < seg else (90, 88, 88), -1)
    tx.put_text(img, tempo_label(mult), (x + int(80 * s0), y + int(9 * s0)), int(22 * s0), PAPER, font='display', stroke=2)
    return img


def draw_beat_dots(img, verdicts, current, n_beats, size, t=0.0):
    """The count as a row of dots at the bottom of the safe area: grey until danced, then volt / amber / red; the current count is bigger."""
    cw, ch = size; s0 = sc_of(size); n = max(1, n_beats); _, _, _, yb = safe_box(size)
    gap = min(int(60 * s0), int((cw * 0.88) / n)); r = max(5, min(int(15 * s0), int(gap * 0.32)))
    x0 = cw // 2 - gap * (n - 1) // 2; y = yb - int(30 * s0)
    plate(img, x0 - gap // 2 - 6, y - r - int(14 * s0), x0 + gap * (n - 1) + gap // 2 + 6, y + r + int(14 * s0), INK, 0.45, r=r + int(14 * s0))
    for b in range(n):
        v = verdicts[b] if b < len(verdicts) else None; c = (x0 + b * gap, y)
        col = (110, 106, 106) if v is None else BGR[v]
        rr = int(r * (1.35 + 0.15 * fxm.pulse(t, 2.0))) if b == current else r
        cv2.circle(img, c, rr, col, -1, cv2.LINE_AA)
        if b == current: cv2.circle(img, c, rr + 3, PAPER, 2, cv2.LINE_AA)
    return img


def draw_hints(img, size, x0=None, y0=None, w=None):
    """First moments of the call only: two ways to begin, as pills (beside the avatar)."""
    cw, ch = size; s0 = sc_of(size)
    cx = cw // 2 if x0 is None else x0 + w // 2; y = (ch - int(260 * s0)) if y0 is None else y0
    pill_c(img, 'Say "teach me hip hop"', cx, y, fit_px('Say "teach me hip hop"', int(30 * s0), (w or cw) * 0.95, 'body'), INK, VOLT, font='body')
    pill_c(img, 'or ask me anything', cx, y + int(66 * s0), int(26 * s0), PAPER, (40, 38, 38), font='body', alpha=0.85)
    return img


def draw_setup_view(img, setup, size, state, t=0.0):
    """Camera setup: a dashed outline where to stand, three checks (body, distance, light) and one instruction."""
    cw, ch = size; s0 = sc_of(size)
    ok = bool(setup.get('ok_now'))
    col = VOLT if ok else (200, 200, 200)
    bw, bh = int(min(cw * 0.42, ch * 0.32)), int(ch * 0.72); x0, y0 = cw // 2 - bw // 2, int(ch * 0.14)
    head = int(bw * 0.22); cx = cw // 2
    for i in range(0, 360, 12):                                # a dashed outline: head circle, then the body box
        if (i // 12) % 2: continue
        cv2.ellipse(img, (cx, y0 + head), (head, head), 0, i, i + 8, col, 3, cv2.LINE_AA)
    by0 = y0 + 2 * head + int(10 * s0)
    for yy in range(by0, y0 + bh, int(26 * s0)):
        cv2.line(img, (x0, yy), (x0, min(yy + int(14 * s0), y0 + bh)), col, 3, cv2.LINE_AA)
        cv2.line(img, (x0 + bw, yy), (x0 + bw, min(yy + int(14 * s0), y0 + bh)), col, 3, cv2.LINE_AA)
    for xx in range(x0, x0 + bw, int(26 * s0)):
        cv2.line(img, (xx, y0 + bh), (min(xx + int(14 * s0), x0 + bw), y0 + bh), col, 3, cv2.LINE_AA)
    checks = setup.get('checks') or {}
    x = cw // 2 - int(3 * 118 * s0) // 2; y = safe_box(size)[1] + int(132 * s0)
    for name, label in (('body', 'BODY'), ('distance', 'DISTANCE'), ('light', 'LIGHT')):
        good = checks.get(name, False)
        pill_c(img, label, x + int(59 * s0), y, int(18 * s0), INK if good else PAPER, VOLT if good else (60, 58, 58))
        x += int(118 * s0)
    from .setup import SHORT
    line = 'Say "ready" or raise both hands' if state == 'CONFIRM' else (SHORT.get(setup.get('issue')) if setup.get('issue') else ('Hold still...' if ok else None))
    if line:
        pill_c(img, line, cw // 2, ch - int(150 * s0), int(32 * s0), INK, VOLT if (ok or state == 'CONFIRM') else AMBER)
    if ok and state != 'CONFIRM' and setup.get('hold', 0) > 0:
        c = (cw // 2, ch - int(230 * s0)); r = int(28 * s0)
        cv2.circle(img, c, r, (70, 70, 74), 5, cv2.LINE_AA); cv2.ellipse(img, c, (r, r), -90, 0, int(360 * setup['hold']), VOLT, 5, cv2.LINE_AA)
    return img


def compose(frame_bgr, lm_img, hud, ghost=None, show_live=True, size=None, squad_lms=None):
    """The phrase view, by stage: WATCH (the reference, big), WITH ME (ghost on your body), YOUR TURN (dots only), CHALLENGE (score,
    combo, judgments). squad_lms: {pid: (lm_img, colour)} for squad mode (each dancer drawn in their colour)."""
    sh, sw = frame_bgr.shape[:2]; size = size or canvas_size(sw, sh)
    img, tr = to_canvas(frame_bgr, size); t = float(hud.get('t') or 0.0); fxq = hud.get('fx'); s0 = sc_of(size)
    vis, parts = visibility(lm_img)
    pts = lm_px(lm_img, sw, sh, tr) if lm_img is not None else None
    st = hud['state']; squad = hud.get('squad'); stage = hud.get('stage') or ('CHALLENGE' if hud.get('battle') else 'YOUR TURN')
    scoring = stage == 'CHALLENGE'
    if ghost is not None and st in ('show', 'countin', 'attempt') and not squad and (hud.get('ghost_overlay', True) or st == 'show'):
        pl = ghost_placement(pts, vis, ghost, size)
        if pl is not None: img = draw_ghost(img, ghost, pl[0], pl[1], alpha=0.5 if st == 'attempt' else 0.62, upper_only=pl[2])
    anchors = {}
    if squad_lms:
        for pid, (lm, colour) in squad_lms.items():
            if lm is None: continue
            v, _ = visibility(lm); p = lm_px(lm, sw, sh, tr)
            draw_skeleton(img, p, v, color=PLAYER.get(colour, VOLT), thickness=4)
            anchors[pid] = _head_anchor(p, v, size)
            tag = next(((r.get('name') or r['colour']).upper() for r in squad or () if r['pid'] == pid), colour.upper())
            x, y = anchors[pid]; pill_c(img, tag, x, y + int(38 * s0), int(18 * s0), INK, PLAYER.get(colour, VOLT))
    elif pts is not None and show_live:
        col = VOLT
        if st == 'attempt':
            done = [v for v in hud.get('verdicts', []) if v is not None]
            if done: col = BGR[done[-1]]
        draw_skeleton(img, pts, vis, color=col, thickness=3)
        anchors[None] = _head_anchor(pts, vis, size)
    if ghost is not None and st in ('show', 'countin', 'attempt'):
        img = draw_panel(img, ghost, '', size, highlight=(st == 'show'), small=(st == 'attempt'))
    rounded_rect(img, (0, 0), (size[0], safe_box(size)[1] + int(130 * s0)), INK, r=0, alpha=0.35)
    draw_badge(img, size, hud.get('speaking', False), t, state=hud.get('avatar_state', 'idle'), still=(st == 'attempt'))
    draw_state_chip(img, size, 'PAUSED' if hud.get('paused') else stage)
    draw_progress_row(img, hud, size)
    draw_battle(img, hud.get('battle'), size, t)
    if squad: draw_squad_strip(img, squad, size)
    if st == 'attempt' and scoring:
        if not squad: draw_combo(img, hud, size, t)
        if fxq is not None: draw_judgments(img, fxq, t, anchors, size); draw_tip(img, fxq, t, size)
    img = draw_main(img, hud, size, t)
    if st in ('countin', 'attempt', 'show') and not hud.get('demo'):
        img = draw_beat_dots(img, hud['verdicts'] if st == 'attempt' else [None] * hud['n_beats'], hud['current_beat'] if st == 'attempt' else -1, hud['n_beats'], size, t)
    hint = (framing_hint(parts) if lm_img is not None else "I can't see you") if not squad else hud.get('squad_hint')
    if st == 'attempt' and hint and not hint.startswith('Upper body'):
        pill_c(img, hint if len(hint) < 34 else 'Step back so I can see you', size[0] // 2, safe_box(size)[3] - int(110 * s0), int(28 * s0), INK, AMBER)
    draw_gesture_ring(img, hud.get('gesture'), size)
    draw_overlays(img, fxq, t, size)
    if hud.get('paused'): draw_paused(img, size)
    return img


def draw_recap(img, recap, size):
    """Between attempts, while Brodie talks: the grade badge, the title, one tip. Nothing else."""
    r = recap.get('result') or {}; cw, ch = size; s0 = sc_of(size)
    y = ch - int(250 * s0)
    if not r.get('scored'):
        pill_c(img, recap.get('cue') or 'Step back so I can see you', cw // 2, y, int(28 * s0), INK, AMBER); return img
    g = progression.grade(r['score']); grade_ring(img, cw // 2, y - int(70 * s0), int(46 * s0), g)
    title = (recap.get('title') or progression.title_for(r)).upper()
    tx.put_text(img, title, (cw // 2, y + int(10 * s0)), fit_px(title, int(32 * s0), cw * 0.86), PINK, font='display', anchor='ms', stroke=3)
    if recap.get('cue'):
        t_ = recap['cue']; pill_c(img, t_, cw // 2, y + int(62 * s0), fit_px(t_, int(24 * s0), cw * 0.84, 'body'), INK, AMBER, font='body')
    return img


def compose_idle(frame_bgr, lm_img, title, lines=(), size=None, history=None, best=None, recap=None, caption=None, footer=None, fxq=None, t=0.0,
                 gesture=None, paused=False, safety=False, players=None, hints=False, setup=None, speaking=False, state=None, avatar_state='idle',
                 countdown=None):
    """Chat, offer, setup and between-phrase view. In chat Brodie (kit avatar) is big at the bottom-left with his words beside him;
    in setup he moves to the top-left and the setup guide takes over; overlays on top. Everything stays inside the FaceTime safe area."""
    sh, sw = frame_bgr.shape[:2]; size = size or canvas_size(sw, sh)
    img, tr = to_canvas(frame_bgr, size); cw, ch = size; s0 = sc_of(size)
    if safety: return draw_safety_card(img)
    vis, parts = visibility(lm_img)
    x0, y0, x1, y1 = safe_box(size)
    if players:
        for lm, colour in players:
            if lm is not None:
                v, _ = visibility(lm); draw_skeleton(img, lm_px(lm, sw, sh, tr), v, color=PLAYER.get(colour, VOLT), thickness=3)
    elif lm_img is not None and state in ('PREPARE', 'CONFIRM'):
        draw_skeleton(img, lm_px(lm_img, sw, sh, tr), vis, color=VOLT, thickness=3)
    in_setup = setup is not None and state in ('PREPARE', 'CONFIRM')
    big = state in (None, 'EXPLORE', 'OFFER') and not recap and countdown is None
    from .avatar import avatar
    if big:
        h = int(min(cw * 0.36, ch * 0.26)); cx = x0 + h // 2 + 6; cy = y1 - h // 2 - int(10 * s0)
        plate(img, x0 - 6, cy - h // 2 - 14, x1 + 6, cy + h // 2 + 14, INK, 0.55, r=h // 2 + 14)
        avatar().draw(img, cx, cy, h, avatar_state, t)
        tx0 = cx + h // 2 + int(22 * s0); tw = x1 - tx0
        if caption:
            px = int(32 * s0)
            while px > int(20 * s0) and len(wrap(caption, px, tw, max_lines=9)) > 3: px -= 2
            ls = wrap(caption, px, tw, max_lines=3); lh = int(px * 1.3)
            yy = cy - (lh * len(ls)) // 2 + int(px * 0.8)
            for l in ls: tx.put_text(img, l, (tx0, yy), px, PAPER, stroke=2); yy += lh
        elif hints:
            draw_hints(img, size, tx0, cy - int(34 * s0), tw)
        else:
            tx.put_text(img, 'BRODIE', (tx0, cy + int(12 * s0)), int(36 * s0), PAPER, font='display', stroke=3)
        draw_state_chip(img, size, 'OFFER' if state == 'OFFER' else None)
    else:
        rounded_rect(img, (0, 0), (cw, y0 + int(96 * s0)), INK, r=0, alpha=0.35)
        draw_badge(img, size, False, t, state=avatar_state)
        draw_state_chip(img, size, 'PAUSED' if paused else state)
        if in_setup: draw_setup_view(img, setup, size, state, t)
        elif recap: draw_recap(img, recap, size)
        if caption and not in_setup: draw_caption(img, caption, size, y_bottom=y1 - int(4 * s0))
    if countdown is not None:
        tx.put_text(img, str(countdown), (cw // 2, int(ch * 0.42)), int(220 * s0), VOLT, font='display', anchor='ms', stroke=8, stroke_color=PINK)
        pill_c(img, 'FINISH POSE!', cw // 2, int(ch * 0.42) + int(80 * s0), int(34 * s0), INK, VOLT)
    draw_gesture_ring(img, gesture, size)
    draw_overlays(img, fxq, t, size)
    if paused: draw_paused(img, size)
    return img


