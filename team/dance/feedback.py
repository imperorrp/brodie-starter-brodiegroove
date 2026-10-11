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
    if (r.get('timing') if r.get('timing') is not None else 1.0) < 0.6 and r.get('mean_offset_beats') is not None:
        late = r['mean_offset_beats'] > 0; span = f' on {_span(longest, count_label)}' if longest else ''
        text = (f"You're a little late{span}, hit it on the click" if late else f"You're early{span}, wait for the click")
        return dict(base, text=text, short='LATE: ON THE CLICK' if late else 'EARLY: WAIT FOR IT', kind='timing', group=None, beats=longest or weak,
                    error_type='late' if late else 'early', direction='earlier' if late else 'later', confidence=min(1.0, abs(r['mean_offset_beats']) * 2))
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
