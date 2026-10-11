"""Camera setup: what the pose model can actually verify about the caller's framing, one issue at a time.

Verifiable from MediaPipe landmarks and the frame (smoothed over about a second):
  person in view, head to hips inside the frame with a margin (the feet are deliberately ignored: they were the least reliable
  part on real calls), distance (torso length as a share of the frame),
  centred, roughly facing the camera, enough light (frame brightness and landmark confidence).
Not verifiable, so the coach asks instead: clear floor space, obstacles, a safe surface. The exact camera angle is not claimed;
"lower the phone" is offered only as a tip when the feet are cut off.

  evaluate(lm, luma, upper=False) -> {'ok', 'issues': [codes in priority order], 'metrics': {...}}
  SetupTracker.update(t, lm, luma) -> {'ok_stable', 'issue', 'checks': {'body','distance','light'}}
  INSTRUCTIONS[code] -> one short spoken instruction; SHORT[code] -> the on-screen line (a few words)
"""
import numpy as np

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
VIS = 0.5
MARGIN = 0.03
STABLE_S = 1.5

ORDER = ('no_person', 'too_close', 'feet_cut', 'too_far', 'off_center', 'dark', 'not_facing')
INSTRUCTIONS = dict(
    no_person="ask them to step into the picture so you can see them",
    too_close="ask them to take a couple of big steps back from the phone",
    feet_cut="ask them to step back a little, or tilt the phone down, so you can see their feet",
    too_far="ask them to come a step closer to the phone",
    off_center="ask them to take a small step toward the middle of the picture",
    dark="ask them to turn on a light or face a window so you can see them clearly",
    not_facing="ask them to turn and face the screen",
)
REPHRASE = dict(
    no_person="say you still can't see them and ask them to stand where the phone can see them",
    too_close="say they're still a bit close; another step or two back",
    feet_cut="say you still can't see their feet; prop the phone lower, around waist height, or step back",
    too_far="say they're a little far; one step closer",
    off_center="say they're near the edge; a step toward the middle",
    dark="say it's still dark; more light would help you see their moves",
    not_facing="say turn a bit more toward the screen",
)
SHORT = dict(no_person='Step into view', too_close='Step back', feet_cut='Show your feet', too_far='Come closer', off_center='Move to the middle',
             dark='More light', not_facing='Face the screen', lost='Video paused')


def _inside(p, m=MARGIN):
    return (m <= p[0] <= 1 - m) and (m <= p[1] <= 1 - m)


def evaluate(lm, luma=None, upper=False):
    """lm: (33, >=4) image landmarks (x, y, z, visibility) or None. luma: mean frame brightness 0-255 or None."""
    issues = []; m = {}
    if lm is None:
        return dict(ok=False, issues=['no_person'], metrics=m, checks=dict(body=False, distance=False, light=luma is None or luma >= 55))
    a = np.asarray(lm, np.float32); v = a[:, 3]
    seen = lambda j: v[j] >= VIS and _inside(a[j, :2])
    core = [NOSE, L_SH, R_SH] + ([L_EL, R_EL, L_WR, R_WR] if upper else [L_HIP, R_HIP])
    if not all(v[j] >= VIS for j in (L_SH, R_SH)):
        issues.append('no_person')
    head_in = seen(NOSE) or (v[NOSE] < VIS and a[NOSE, 1] > MARGIN)
    hips_in = all(seen(j) for j in (L_HIP, R_HIP))
    top = float(np.min(a[[NOSE, L_SH, R_SH], 1]))
    torso = float(np.linalg.norm(0.5 * (a[L_SH, :2] + a[R_SH, :2]) - 0.5 * (a[L_HIP, :2] + a[R_HIP, :2]))) if hips_in else None
    height = torso                                              # feet are ignored: head to hips is what we need to see
    m.update(top=round(top, 3), height=None if height is None else round(height, 3))
    hip_x = float(np.mean(a[[L_HIP, R_HIP], 0])) if all(v[j] >= VIS for j in (L_HIP, R_HIP)) else float(np.mean(a[[L_SH, R_SH], 0]))
    m['center_x'] = round(hip_x, 3)
    sw = float(abs(a[L_SH, 0] - a[R_SH, 0])); tl = float(np.linalg.norm(0.5 * (a[L_SH, :2] + a[R_SH, :2]) - 0.5 * (a[L_HIP, :2] + a[R_HIP, :2]))) + 1e-6
    m['facing'] = round(sw / tl, 3)
    vis_mean = float(np.mean(v[[NOSE, L_SH, R_SH, L_HIP, R_HIP]]))
    m['visibility'] = round(vis_mean, 3); m['luma'] = None if luma is None else round(float(luma), 1)
    if 'no_person' not in issues:
        if upper:
            if not all(seen(j) for j in core): issues.append('too_close')
        else:
            if not head_in or (top < MARGIN) or not hips_in: issues.append('too_close')     # head to hips must be in frame; feet never matter
            elif height is not None and height > 0.42: issues.append('too_close')
            elif height is not None and height < 0.09: issues.append('too_far')
        if not (0.25 <= hip_x <= 0.75): issues.append('off_center')
    if (luma is not None and luma < 55) or ('no_person' not in issues and vis_mean < 0.45 and not upper): issues.append('dark')
    if 'no_person' not in issues and m['facing'] < 0.3: issues.append('not_facing')
    issues = [c for c in ORDER if c in issues]
    checks = dict(body=not any(c in issues for c in ('no_person', 'too_close', 'feet_cut')), distance=not any(c in issues for c in ('too_far', 'off_center', 'too_close')),
                  light='dark' not in issues)
    return dict(ok=not issues, issues=issues, metrics=m, checks=checks)


# Spoken lines, written by us in second person (Gemini says them word for word; never meta like "ask them to...").
FIRST = dict(
    no_person=["Step into the picture so I can see you.", "I can't see you yet. Stand where your camera can see you."],
    too_close=["Take a step back so I can see you down to your hips.", "You're a bit close. Step back until I can see your hips."],
    feet_cut=["I can't see your feet yet. Step back a little, or tilt your phone down.", "Almost! Your feet are out of frame. Prop the phone lower, or step back."],
    too_far=["Come one step closer to the camera.", "You're a little far away. One step closer."],
    off_center=["Take a small step toward the middle of the picture.", "Shuffle a little toward the middle for me."],
    dark=["It's a bit dark. Can you turn on a light or face a window?", "More light would help me see your moves."],
    not_facing=["Turn to face your camera.", "Face the screen for me."],
)
CHECKIN = dict(
    no_person=["Still can't see you. Take your time, and tell me if you need a minute.", "I'm still waiting to see you. No rush."],
    too_close=["Still can't see your hips. A little further back, or say arms only.", "Almost. Just a bit further back so your hips are in."],
    feet_cut=["Still can't see your feet. If there's no room, say arms only and we'll do the upper body.", "Your feet are still hiding. Lower the phone, step back, or say arms only."],
    too_far=["Still a little far. One more step in.", "Come in just a touch more."],
    off_center=["Nearly. A little more toward the middle.", "Just a bit more to the middle."],
    dark=["Still a little dark for me. Any more light you can add?", "If you can, add a bit more light."],
    not_facing=["Turn a bit more toward the screen.", "Almost facing me. A little more."],
)
PARTS_SEEN = (('head', [NOSE]), ('shoulders', [L_SH, R_SH]), ('hips', [L_HIP, R_HIP]))     # feet are ignored on purpose


def camera_note(lm, issue=None, upper=False):
    """A plain description of what the camera sees, for Gemini's own replies ("can you see me?")."""
    if lm is None: return "[camera now: I can't see anyone in the picture]"
    a = np.asarray(lm, np.float32)
    seen = [n for n, js in PARTS_SEEN if all(a[j, 3] >= VIS and _inside(a[j, :2], 0.0) for j in js)]
    missing = [n for n, _ in PARTS_SEEN if n not in seen]
    msg = f"[camera now: I can see your {', '.join(seen) or 'outline'}"
    if missing: msg += f"; your {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} out of frame"
    if issue: msg += f"; next fix: {SHORT.get(issue, issue).lower()}"
    if upper: msg += "; arms-only mode is on, so legs don't matter"
    return msg + ']'


class SetupTracker:
    """Smooths per-frame verdicts: an issue counts once it holds for most of the last second, and the *current* issue only changes
    after the new one has held HOLD_S (no step-back / step-in flip-flops). 'ok_stable' after STABLE_S of all-clear."""
    HOLD_S = 2.5

    def __init__(self, upper=False):
        self.upper = upper; self.hist = []; self.ok_since = None; self.issue = None; self.cand = None; self.cand_since = None

    def update(self, t, lm, luma=None):
        r = evaluate(lm, luma, self.upper)
        self.hist.append((t, r)); self.hist = [(tt, rr) for tt, rr in self.hist if t - tt <= 1.0]
        n = len(self.hist); counts = {}
        for _, rr in self.hist:
            for c in rr['issues']: counts[c] = counts.get(c, 0) + 1
        now_issue = next((c for c in ORDER if counts.get(c, 0) >= max(1, n // 2)), None)
        if r['ok']:
            if self.ok_since is None: self.ok_since = t
        else:
            self.ok_since = None
        if now_issue != self.issue:
            if now_issue != self.cand: self.cand = now_issue; self.cand_since = t
            elif t - self.cand_since >= (1.0 if now_issue is None else self.HOLD_S if self.issue is not None else 1.0): self.issue = now_issue
        else:
            self.cand = None
        ok_stable = self.ok_since is not None and t - self.ok_since >= STABLE_S
        hold = 0.0 if self.ok_since is None else min(1.0, (t - self.ok_since) / STABLE_S)
        return dict(ok_stable=ok_stable, ok_now=r['ok'], issue=self.issue, checks=r['checks'], hold=hold, metrics=r['metrics'])
