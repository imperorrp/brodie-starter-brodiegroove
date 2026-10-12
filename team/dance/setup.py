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


TOO_CLOSE_ON, TOO_CLOSE_OFF = 0.42, 0.38      # hysteresis: an issue starts at one threshold and clears only past the other,
TOO_FAR_ON, TOO_FAR_OFF = 0.09, 0.11          # so a caller standing right on a line doesn't flip between "step back" and "come closer"
CENTER_ON, CENTER_OFF = 0.25, 0.28


def evaluate(lm, luma=None, upper=False, aspect=None, prev=None):
    """lm: (33, >=4) image landmarks (x, y, z, visibility) or None. luma: mean frame brightness 0-255 or None. aspect: frame
    width / height (landmark x and y are normalised separately, so a 16:9 webcam squeezed the shoulder width by about half and a
    caller turned 40 degrees read as 'not facing'). prev: the issue currently being worked on (its clear threshold is stricter)."""
    issues = []; m = {}
    close_t = TOO_CLOSE_OFF if prev == 'too_close' else TOO_CLOSE_ON
    far_t = TOO_FAR_OFF if prev == 'too_far' else TOO_FAR_ON
    edge = CENTER_OFF if prev == 'off_center' else CENTER_ON
    if lm is None:
        return dict(ok=False, issues=['no_person'], metrics=m, checks=dict(body=False, distance=False, light=luma is None or luma >= 55))
    a = np.asarray(lm, np.float32); v = a[:, 3]
    seen = lambda j: v[j] >= VIS and _inside(a[j, :2])
    core = [NOSE, L_SH, R_SH] + ([] if upper else [L_HIP, R_HIP])        # arms only: head and shoulders (arms come into view as they move)
    if not all(v[j] >= VIS for j in (L_SH, R_SH)):
        issues.append('no_person')
    head_in = seen(NOSE) or (v[NOSE] < VIS and a[NOSE, 1] > MARGIN)
    hips_in = all(seen(j) for j in (L_HIP, R_HIP))
    top = float(np.min(a[[NOSE, L_SH, R_SH], 1]))
    torso = float(np.linalg.norm(0.5 * (a[L_SH, :2] + a[R_SH, :2]) - 0.5 * (a[L_HIP, :2] + a[R_HIP, :2]))) if hips_in else None
    height = torso                                              # feet are ignored: head to hips is what we need to see
    m.update(top=round(top, 3), height=None if height is None else round(height, 3), hips_in=hips_in)
    hip_x = float(np.mean(a[[L_HIP, R_HIP], 0])) if all(v[j] >= VIS for j in (L_HIP, R_HIP)) else float(np.mean(a[[L_SH, R_SH], 0]))
    m['center_x'] = round(hip_x, 3)
    ax = float(aspect) if aspect else 1.0
    sw = float(abs(a[L_SH, 0] - a[R_SH, 0])) * ax
    d = 0.5 * (a[L_SH, :2] + a[R_SH, :2]) - 0.5 * (a[L_HIP, :2] + a[R_HIP, :2]); tl = float(np.hypot(d[0] * ax, d[1])) + 1e-6
    m['facing'] = round(sw / tl, 3)
    vis_mean = float(np.mean(v[[NOSE, L_SH, R_SH, L_HIP, R_HIP]]))
    m['visibility'] = round(vis_mean, 3); m['luma'] = None if luma is None else round(float(luma), 1)
    if 'no_person' not in issues:
        if upper:
            if not all(seen(j) for j in core): issues.append('too_close')
        else:
            if not head_in or (top < MARGIN) or not hips_in: issues.append('too_close')     # head to hips must be in frame; feet never matter
            elif height is not None and height > close_t: issues.append('too_close')
            elif height is not None and height < far_t: issues.append('too_far')
        if not (edge <= hip_x <= 1.0 - edge): issues.append('off_center')
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
FIRST_UPPER = dict(
    too_close=["Lean back a little so I can see your head and shoulders.", "Sit back a bit so your shoulders are in the picture."],
)
CHECKIN_UPPER = dict(
    too_close=["Still a bit close. Lean back from the camera a little.", "Almost. Sit back so both shoulders are in."],
)


def lines_for(issue, upper, checkin=False):
    """The spoken lines for an issue; arms-only mode has its own (never "down to your hips" or "say arms only" again)."""
    if upper:
        pool = (CHECKIN_UPPER if checkin else FIRST_UPPER).get(issue)
        if pool: return pool
    return (CHECKIN if checkin else FIRST)[issue]


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
    """Smooths per-frame verdicts. 'ok_stable' once at least 65% of the frames in the last STABLE_S were fine (a single bad frame
    used to reset it, so a reading that failed on a fifth of the frames never became stable and the setup went silent). The issue is
    the most frequent one over the last HOLD_S, and the *current* issue only changes after the new one has led for a second (no
    step-back / step-in flip-flops). 'top_raw' is the most frequent problem even when none is frequent enough to be the issue."""
    HOLD_S = 2.5
    OK_SHARE = 0.65

    def __init__(self, upper=False):
        self.upper = upper; self.hist = []; self.issue = None; self.cand = None; self.cand_since = None; self.t_first = None

    def update(self, t, lm, luma=None, aspect=None):
        r = evaluate(lm, luma, self.upper, aspect, prev=self.issue)
        if self.t_first is None: self.t_first = t
        self.hist.append((t, r)); self.hist = [(tt, rr) for tt, rr in self.hist if t - tt <= self.HOLD_S]
        recent = [rr for tt, rr in self.hist if t - tt <= STABLE_S]
        ok_share = sum(1 for rr in recent if rr['ok']) / max(1, len(recent))
        covered = min(1.0, (t - self.t_first) / STABLE_S) if len(recent) > 1 else 0.0
        ok_stable = covered >= 0.8 and ok_share >= self.OK_SHARE
        counts = {}
        for _, rr in self.hist:
            for c in rr['issues'][:1]: counts[c] = counts.get(c, 0) + 1
        n = len(self.hist); bad = sum(counts.values())
        top_raw = max(counts, key=lambda c: (counts[c], -ORDER.index(c))) if counts else None
        now_issue = None if ok_stable or bad < 0.4 * n else top_raw
        if now_issue != self.issue:
            if now_issue != self.cand: self.cand = now_issue; self.cand_since = t
            elif t - self.cand_since >= 1.0: self.issue = now_issue
        else:
            self.cand = None
        if ok_stable: self.issue = None
        hold = min(1.0, covered * ok_share / self.OK_SHARE) if ok_share >= 0.5 else 0.0
        return dict(ok_stable=ok_stable, ok_now=r['ok'], issue=self.issue, checks=r['checks'], hold=hold, metrics=r['metrics'],
                    top_raw=top_raw, bad_share=round(bad / max(1, n), 2))
