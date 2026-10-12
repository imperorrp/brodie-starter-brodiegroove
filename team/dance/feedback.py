"""Feedback: turn scoring results into one specific, directional cue ("Lift your left arm higher on counts 5 to 7").

Channels are the 14 angle features of features.py. Signed error = caller minus reference, per channel, from scoring.score_phrase
(`channel_signed`, per beat) or from the engine's live per-beat errors. "Left" and "right" are the caller's own anatomical sides
(MediaPipe labels landmarks by the person's side; the display is a mirror, so their left arm is on the screen's left).

  structured_cue(result)   -> {kind, text, short, group, beats, body_part, direction, error_type, confidence}
  live_tip(rows)           -> {text, short, channel} or None   (mid-phrase, from the last two red beats)
  describe_phrase(key, i)  -> plain-language outline of a phrase, for "explain"
"""
import numpy as np
from .features import CHANNELS, GROUPS

MIN_DEG = 12.0                         # smaller signed errors are not worth a cue
SIDE = {'l': 'left', 'r': 'right'}
CH_GROUP = {i: g for g, idx in GROUPS.items() for i in idx}
CH_GROUP[9] = 'torso'

# channel index -> (body part, text when caller < reference, text when caller > reference, short <, short >)
def _table():
    t = {}
    for i, s in ((0, 'l'), (1, 'r')):
        t[i] = (f'{SIDE[s]} arm', f'Straighten your {SIDE[s]} arm', f'Bend your {SIDE[s]} elbow', f'{SIDE[s].upper()} ARM STRAIGHT', f'{SIDE[s].upper()} ELBOW BEND')
    for i, s in ((2, 'l'), (3, 'r'), (10, 'l'), (11, 'r')):
        t[i] = (f'{SIDE[s]} arm', f'Lift your {SIDE[s]} arm higher', f'Bring your {SIDE[s]} arm down', f'{SIDE[s].upper()} ARM HIGHER', f'{SIDE[s].upper()} ARM LOWER')
    for i, s in ((4, 'l'), (5, 'r')):
        t[i] = ('hips', 'Stand taller', 'Sink into your hips', 'STAND TALLER', 'SINK LOWER')
    for i, s in ((6, 'l'), (7, 'r')):
        t[i] = (f'{SIDE[s]} leg', f'Straighten your {SIDE[s]} leg', f'Bend your {SIDE[s]} knee more', f'{SIDE[s].upper()} LEG STRAIGHT', f'BEND {SIDE[s].upper()} KNEE')
    t[8] = ('torso', 'Lean into it more', 'Stay more upright', 'LEAN IN', 'STAY UPRIGHT')
    t[9] = ('head', 'Watch your head position', 'Watch your head position', 'HEAD UP', 'HEAD UP')
    t[12] = ('feet', 'Wider stance', 'Feet closer together', 'WIDER STANCE', 'FEET IN')
    t[13] = ('hands', 'Hands wider apart', 'Hands closer together', 'HANDS WIDER', 'HANDS CLOSER')
    return t


PHRASES = _table()
assert len(PHRASES) == len(CHANNELS)


def channel_cue(ch, signed):
    part, lo, hi, slo, shi = PHRASES[int(ch)]
    low = signed < 0
    direction = {0: 'straighter', 1: 'straighter', 2: 'higher', 3: 'higher', 10: 'higher', 11: 'higher', 4: 'taller', 5: 'taller',
                 6: 'straighter', 7: 'straighter', 8: 'lean', 12: 'wider', 13: 'wider'}.get(int(ch), 'adjust')
    if not low:
        direction = {'straighter': 'bend', 'higher': 'lower', 'taller': 'lower', 'lean': 'upright', 'wider': 'closer'}.get(direction, direction)
    return dict(body_part=part, text=lo if low else hi, short=slo if low else shi, direction=direction)


def _pick_channel(signed):
    """Index of the channel with the largest |signed| error (head only as a last resort), or None below MIN_DEG."""
    a = np.asarray(signed, np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    w = np.abs(a).copy(); w[9] *= 0.5; w[13] *= 0.8
    i = int(np.argmax(w))
    return i if abs(a[i]) >= MIN_DEG else None


def _runs(beats):
    if not beats: return []
    runs = []; run = [beats[0]]
    for b in beats[1:]:
        if b == run[-1] + 1: run.append(b)
        else: runs.append(run); run = [b]
    runs.append(run)
    return runs


def _span(run, label='counts'):
    a, b = run[0] + 1, run[-1] + 1
    return f'{label[:-1] if a == b else label} {a}' if a == b else f'{label} {a} to {b}'


def structured_cue(result, count_label='counts'):
    """One cue per phrase: visibility first, then timing, then the worst channel on the longest weak run, else praise."""
    r = result or {}
    base = dict(body_part=None, direction=None, confidence=0.0)
    if not r.get('scored'):
        return dict(base, text='Step back so I can see all of you', short='STEP BACK', kind='visibility', group=None, beats=[], error_type='visibility', confidence=1.0)
    weak = list(r.get('weak_beats') or [])
    longest = max(_runs(weak), key=len) if weak else []
    if (r.get('timing') if r.get('timing') is not None else 1.0) < 0.6:
        t = timing_cue(r)                                                  # only counts whose shape was right (no false "you're early")
        if t: return dict(base, **{k: v for k, v in t.items() if k != 'sev'}, group=None, confidence=round(float(t['sev']), 2))
    if r.get('score', 0) >= 85 or not weak:
        return dict(base, text='Clean, that is locked in', short='LOCKED IN', kind='praise', group=None, beats=[], error_type=None, confidence=1.0)
    cs = r.get('channel_signed')
    if cs:
        rows = [cs[b] for b in longest if b < len(cs)]
        if rows:
            arr = np.array([[np.nan if v is None else v for v in row] for row in rows], np.float64)
            with np.errstate(all='ignore'): mean = np.nanmean(arr, axis=0)
            ch = _pick_channel(mean)
            if ch is not None:
                c = channel_cue(ch, mean[ch]); span = _span(longest, count_label)
                conf = float(min(1.0, abs(mean[ch]) / 40.0))
                return dict(c, text=f"{c['text']} on {span}", kind='correction', group=CH_GROUP.get(ch), beats=longest, error_type='shape', confidence=round(conf, 2), channel=CHANNELS[ch])
    groups = {g: v for g, v in (r.get('per_group') or {}).items() if v is not None}
    if not groups: return dict(base, text='Nice. Again', short='AGAIN', kind='praise', group=None, beats=[], error_type=None)
    worst = max(groups, key=groups.get)
    return dict(base, text=f'{worst.capitalize()}, {_span(longest, count_label)}', short=worst.upper(), kind='correction', group=worst, beats=longest,
                error_type='shape', body_part=worst, confidence=0.5)


def live_tip(rows):
    """Mid-phrase tip from the signed per-channel errors of the last two red beats (each an array of 14, NaN where unseen)."""
    if not rows or len(rows) < 2: return None
    arr = np.array(rows[-2:], np.float64)
    with np.errstate(all='ignore'): mean = np.nanmean(arr, axis=0)
    if not np.isfinite(mean).any(): return None
    ch = _pick_channel(mean)
    if ch is None or abs(mean[ch]) < 15.0: return None
    c = channel_cue(ch, mean[ch])
    return dict(text=c['text'], short=c['short'], channel=CHANNELS[ch], body_part=c['body_part'], direction=c['direction'])


_MOVES = {0: ('arm straightens', 'elbow bends'), 1: ('arm straightens', 'elbow bends'), 2: ('arm lifts', 'arm comes down'),
          3: ('arm lifts', 'arm comes down'), 10: ('arm goes up', 'arm comes down'), 11: ('arm goes up', 'arm comes down'),
          4: ('hips straighten', 'hips sink'), 5: ('hips straighten', 'hips sink'), 6: ('leg straightens', 'knee bends'),
          7: ('leg straightens', 'knee bends'), 8: ('you lean', 'you straighten up'), 12: ('feet step wider', 'feet come together'),
          13: ('hands open out', 'hands come together')}
_SIDED = {0: 'left', 1: 'right', 2: 'left', 3: 'right', 10: 'left', 11: 'right', 6: 'left', 7: 'right'}
_PAIR = {0: 1, 2: 3, 10: 11, 6: 7}
_BOTH = {0: ('both arms straighten', 'both elbows bend'), 2: ('both arms lift', 'both arms come down'),
         10: ('both arms go up', 'both arms come down'), 6: ('both legs straighten', 'both knees bend')}


def describe_phrase(key, i, per=2):
    """'counts 1 to 2: both arms go up; counts 3 to 4: left knee bends; ...' from the reference phrase itself."""
    from .reference import phrase_slice
    F, M, _, _ = phrase_slice(key, i); pb = int(key['per_beat']); n_beats = len(F) // pb; out = []
    for b0 in range(0, n_beats, per):
        a, b = b0 * pb, min(len(F) - 1, (b0 + per) * pb - 1)
        if b <= a: continue
        span = f'counts {b0 + 1} to {min(n_beats, b0 + per)}'
        d = np.where(M[a] & M[b], F[b] - F[a], 0.0).astype(np.float64); d[9] = 0.0; d[13] *= 0.7
        from .features import IGNORE_FEET, FEET_CHANNELS
        if IGNORE_FEET: d[FEET_CHANNELS] = 0.0                                 # never describe the feet
        ch = int(np.argmax(np.abs(d)))
        if abs(d[ch]) < 15: out.append(f'{span}: hold and groove'); continue
        up = d[ch] > 0; first = min(ch, _PAIR.get(ch, ch)); pair = _PAIR.get(ch) or next((k for k, v in _PAIR.items() if v == ch), None)
        if pair is not None and abs(d[pair]) >= 0.6 * abs(d[ch]) and np.sign(d[pair]) == np.sign(d[ch]) and first in _BOTH:
            what = _BOTH[first][0 if up else 1]
        else:
            verb = _MOVES.get(ch, ('moves', 'moves'))[0 if up else 1]
            what = f'{_SIDED[ch]} {verb}' if ch in _SIDED else verb
        out.append(f'{span}: {what}')
    return '; '.join(out)


# ---- richer coaching: depth / facing, timing per count, size, up to three ranked corrections, progress ----
DEPTH_DEADBAND = 35.0                   # degrees out of the frontal plane before a forward/back cue (MediaPipe z is noisy)
FACING_DEADBAND = 30.0


def ref_profile(key, pi):
    """Per count of a reference phrase, from its 2D skeleton (what the caller sees as the ghost): how flat each arm is in the picture
    plane (projected length / the arm's longest projection in the phrase), where it points, and how turned the shoulders look."""
    from .reference import phrase_slice
    from .features import arm_planarity, arm_direction_2d, L_SH, R_SH
    _, _, _, S = phrase_slice(key, pi); pb = int(key['per_beat']); n = max(1, len(S) // pb)
    if not len(S): return []
    lmax = {s: max(1e-3, max(arm_planarity(x, s) for x in S)) for s in ('l', 'r')}
    sw = [float(np.linalg.norm(x[L_SH] - x[R_SH])) for x in S]; swmax = max(1e-3, max(sw))
    out = []
    for b in range(n):
        x = S[min(len(S) - 1, b * pb)]
        out.append(dict(flat_l=arm_planarity(x, 'l') / lmax['l'], flat_r=arm_planarity(x, 'r') / lmax['r'],
                        dir_l=arm_direction_2d(x, 'l'), dir_r=arm_direction_2d(x, 'r'), facing=sw[min(len(sw) - 1, b * pb)] / swmax))
    return out


def depth_issue(sig, prof, side):
    """A forward/back issue for one arm on one count, or None. Only when the reference is clearly in the plane (flat) or clearly
    foreshortened, and the live arm clearly disagrees by more than the deadband."""
    a = (sig or {}).get(f'arm_fwd_{side}')
    if a is None: return None
    flat = prof.get(f'flat_{side}', 1.0); d = prof.get(f'dir_{side}', 'out'); name = SIDE[side]
    if flat >= 0.85 and abs(a) > DEPTH_DEADBAND:
        where = {'up': 'straight up', 'down': 'down by your side', 'across': 'across your body'}.get(d, 'out to the side')
        fwd = a > 0
        return dict(kind='depth', body_part=f'{name} arm', side=side, sev=abs(a) / 90.0,
                    text=f"Your {name} arm is reaching {'forward' if fwd else 'back'}; take it {where}",
                    short=f"{name.upper()} ARM {'OUT' if d == 'out' else d.upper()}", direction='side')
    if flat <= 0.55 and abs(a) < 15.0:
        return dict(kind='depth', body_part=f'{name} arm', side=side, sev=0.5,
                    text=f"Bring your {name} arm forward, out in front of you", short=f"{name.upper()} ARM FORWARD", direction='forward')
    return None


def _runs_of(flags):
    """[(start, end)] runs of consecutive True."""
    runs = []; start = None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None: start = i
        if not f and start is not None: runs.append((start, i - 1)); start = None
    return runs


def depth_cues(result, key, pi):
    """Depth and facing cues from the per-count 3D signals (engine result['body3d']) against the reference profile. A cue covers a run
    of at least two counts (jitter on a single count never makes one)."""
    b3 = (result or {}).get('body3d')
    if not b3: return []
    try: prof = ref_profile(key, pi)
    except Exception: return []
    n = min(len(b3), len(prof)); out = []
    for side in ('l', 'r'):
        issues = [(depth_issue(b3[b], prof[b], side) if b3[b] else None) for b in range(n)]
        for s0, s1 in _runs_of([i is not None for i in issues]):
            if s1 - s0 + 1 < 2 and n > 2: continue
            best = max((issues[b] for b in range(s0, s1 + 1)), key=lambda i: i['sev'])
            beats = list(range(s0, s1 + 1))
            out.append(dict(best, beats=beats, text=f"{best['text']} on {_span(beats)}", error_type='depth'))
    flags = []
    for b in range(n):
        y = (b3[b] or {}).get('facing_yaw'); f = prof[b].get('facing', 1.0)
        flags.append(None if y is None else ('front' if f >= 0.9 and abs(y) > FACING_DEADBAND else 'turn' if f <= 0.65 and abs(y) < 15.0 else None))
    for want in ('front', 'turn'):
        for s0, s1 in _runs_of([x == want for x in flags]):
            if s1 - s0 + 1 < 2 and n > 2: continue
            beats = list(range(s0, s1 + 1)); y = float(np.median([b3[b].get('facing_yaw') or 0.0 for b in beats]))
            if want == 'front':
                txt = f"You're turned to your {'left' if y > 0 else 'right'}; face the front on {_span(beats)}"; short = 'FACE FRONT'
            else:
                txt = f"Turn your body to the side on {_span(beats)}, like the ghost"; short = 'TURN TO THE SIDE'
            out.append(dict(kind='depth', body_part='body', text=txt, short=short, beats=beats, sev=0.7, error_type='facing', direction=want))
    return out


def timing_cue(result):
    """Late / early on the counts where the shape was roughly right (an offset on a wrong shape is a shape problem, not timing)."""
    from .scoring import AMBER_DEG
    rows = [r for r in (result or {}).get('per_beat') or [] if r.get('offset_beats') is not None and (r.get('error_deg') if r.get('error_deg') is not None else 99) < AMBER_DEG]
    late = [r['beat'] for r in rows if r['offset_beats'] > 0.25]; early = [r['beat'] for r in rows if r['offset_beats'] < -0.25]
    if len(late) < 2 and len(early) < 2: return None
    is_late = len(late) >= len(early); beats = late if is_late else early
    run = max(_runs(beats), key=len)
    amount = float(np.median([abs(r['offset_beats']) for r in rows if r['beat'] in beats]))
    how = 'half a count' if amount < 0.7 else 'a full count'
    text = (f"You're {how} late on {_span(run)}: hit it on the click" if is_late else f"You're rushing {_span(run)}, about {how} early: wait for the click")
    return dict(kind='timing', body_part=None, text=text, short='LATE: ON THE CLICK' if is_late else 'EARLY: WAIT FOR IT', beats=run, sev=min(1.0, amount),
                error_type='late' if is_late else 'early', direction='earlier' if is_late else 'later')


def size_cue(result):
    """Too small (marking it) or too big, from the range of motion and hip energy against the reference."""
    sz = (result or {}).get('size') or {}
    a = sz.get('arms'); e = sz.get('energy'); h = sz.get('hips')
    if a is not None and a < 0.6:
        frac = 'half' if a > 0.35 else 'a third'
        return dict(kind='size', body_part='arms', text=f"Your arm moves are about {frac} the size of mine: go bigger, full reach", short='BIGGER ARMS', beats=[], sev=1.0 - a)
    if e is not None and e < 0.5:
        return dict(kind='size', body_part='legs', text="More bounce: bend your knees and move from the hips", short='MORE BOUNCE', beats=[], sev=1.0 - e)
    if h is not None and h < 0.5:
        return dict(kind='size', body_part='hips', text="Sink lower into your hips on the moves", short='SINK LOWER', beats=[], sev=1.0 - h)
    if a is not None and a > 1.8:
        return dict(kind='size', body_part='arms', text="Smaller and sharper with the arms: control it", short='SHARPER ARMS', beats=[], sev=0.4)
    return None


def layered_cues(result, key=None, pi=0, arms_only=False, limit=3):
    """Up to `limit` corrections for one try, most important first, one per body part: the shape cue (structured_cue), depth and facing,
    timing on the counts where the shape was right, and size. Each has kind, body_part, text, short, beats. Not scored -> the
    visibility cue alone; nothing to fix -> the praise cue alone."""
    base = structured_cue(result)
    if base.get('kind') == 'visibility': return [base]
    cands = []
    shape = base if base.get('kind') == 'correction' else structured_cue(dict(result, timing=1.0))   # the shape cue even when timing led
    if shape.get('kind') == 'correction': cands.append(dict(shape, sev=0.6 + float(shape.get('confidence') or 0.4)))
    if key is not None:
        try: cands += depth_cues(result, key, pi)
        except Exception: pass
    t = timing_cue(result)
    if t: cands.append(dict(t, sev=0.5 + t['sev'] * 0.5))
    z = size_cue(result)
    if z: cands.append(dict(z, sev=0.4 + z['sev'] * 0.4))
    if arms_only:
        cands = [c for c in cands if c.get('kind') == 'timing' or 'arm' in str(c.get('body_part') or '') or c.get('body_part') in ('arms', 'hands')]
    out = []; parts = set()
    for c in sorted(cands, key=lambda c: -c.get('sev', 0)):
        bp = c.get('body_part') or c.get('kind')
        if bp in parts: continue
        parts.add(bp); out.append(c)
        if len(out) >= limit: break
    return out or [base]


def progress_note(prev_result, result, prev_cue):
    """A short note on whether the last correction landed ("Your left arm is way better."), or None. Compares the cued channel's mean
    error on its counts between the two tries."""
    if not prev_result or not result or not prev_cue or prev_cue.get('kind') != 'correction': return None
    ch_name = prev_cue.get('channel'); beats = prev_cue.get('beats') or []
    if not ch_name or ch_name not in CHANNELS or not beats: return None
    ch = CHANNELS.index(ch_name)

    def err(r):
        cs = r.get('channel_signed') or []
        vals = [abs(cs[b][ch]) for b in beats if b < len(cs) and cs[b] is not None and cs[b][ch] is not None]
        return float(np.mean(vals)) if vals else None
    a, b = err(prev_result), err(result)
    if a is None or b is None or a < 12: return None
    part = prev_cue.get('body_part') or 'that'
    if b <= 0.6 * a: return f"Your {part} is way better."
    if b >= 0.95 * a: return None
    return f"Your {part} is getting there."
