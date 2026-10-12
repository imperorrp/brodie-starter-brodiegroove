"""Coaching cues from the reference itself: what each count looks like and how it should feel, in words a beginner can follow.

The routines are stitched AIST++ phrases with no authored text, so everything here is read off the stored reference arrays
(reference.phrase_slice: 14 angle channels and the hip-centred 2D skeleton, per_beat samples per count):
  arms      where the hands are: up in a V, straight up, out wide, at your chest, down, one arm up ...
  level     knees bent vs standing tall (knee angles against the phrase's own standing height)
  bounce    the knee angle dipping and rising within a count
  weight    which foot the hips sit over (only when the stance is wide enough to tell)
  facing    shoulders turned side-on (the projected shoulder width against the phrase's widest)
  dyn       hit (a burst of angular speed) vs flow (steady speed)
Sides are the dancer's own: the keys are mirrored at build time so the reference's left is what the caller does with their left
(the same convention as feedback.py). Depth is not guessed here: a facing cue only says "turn side-on", never which way.

  count_cues(key, pi) -> [dict(n, arms, level, bounce, weight, facing, dyn)]
  move_card(key, pi) -> dict(name, setup, counts, feel, genre, tags)
  intro_line(card, label) (<= 16 words), count_line(cue) (<= 8 words), card_note(card) (<= 60 words)
"""
import numpy as np

from .reference import phrase_slice
from . import style_vocab as sv

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
KNEE_L, KNEE_R = 6, 7
ARM_CH = (0, 1, 2, 3, 10, 11)
SIDE = dict(l='left', r='right')


def _arm(S, side):
    """Where one arm is, from the 2D skeleton (y down, torso units, hip-centred): 'up' / 'out' / 'chest' / 'down' / 'across'."""
    sh, wr = (L_SH, L_WR) if side == 'l' else (R_SH, R_WR)
    other_sh = R_SH if side == 'l' else L_SH
    d = S[wr] - S[sh]
    if S[wr][1] < S[NOSE][1] - 0.05: return 'up'
    outward = np.sign(S[sh][0] - S[other_sh][0])                # +1 if this side is on the +x side
    lateral = d[0] * outward
    if abs(d[1]) < 0.35 and lateral > 0.55: return 'out'
    if lateral < -0.15: return 'across'
    if d[1] < 0.75 and abs(S[wr][0]) < 0.45: return 'chest'
    return 'down'


def _arms_phrase(S):
    l, r = _arm(S, 'l'), _arm(S, 'r')
    wide = abs(S[L_WR][0] - S[R_WR][0]) > 1.0
    if l == r == 'up': return 'arms up in a V' if wide else 'arms straight up'
    if l == r == 'out': return 'arms out wide'
    if l == r == 'chest': return 'hands at your chest'
    if l == r == 'down': return 'arms down'
    if l == r == 'across': return 'arms crossed in front'
    # one arm does the move: name that arm (priority: up, out, across, chest), the other just hangs
    rank = dict(up=4, out=3, across=2, chest=1, down=0)
    s, a = ('l', l) if rank[l] >= rank[r] else ('r', r)
    other = r if s == 'l' else l
    if a == 'up' and other == 'out': return f"{SIDE[s]} arm up, {SIDE['r' if s == 'l' else 'l']} arm out"
    return {'up': f"{SIDE[s]} arm up", 'out': f"{SIDE[s]} arm out", 'across': f"{SIDE[s]} arm across",
            'chest': f"{SIDE[s]} hand to your chest"}.get(a, 'arms down')


def count_cues(key, pi):
    F, M, E, S = phrase_slice(key, pi)
    pb = int(key['per_beat']); n = max(1, len(S) // pb)
    if not len(S): return []
    knee = np.nanmean(np.where(M[:, [KNEE_L, KNEE_R]], F[:, [KNEE_L, KNEE_R]], np.nan), axis=1) if M.size else np.full(len(F), np.nan)
    stand = float(np.nanpercentile(knee, 90)) if np.isfinite(knee).any() else 175.0
    sw = np.abs(S[:, L_SH, 0] - S[:, R_SH, 0]); swmax = max(1e-3, float(np.percentile(sw, 95)))
    speed = np.abs(np.diff(F[:, list(ARM_CH)], axis=0)).mean(axis=1) if len(F) > 1 else np.zeros(1)
    speed = np.concatenate([speed[:1], speed]); mean_sp = float(np.mean(speed)) + 1e-3
    out = []
    for b in range(n):
        a0, a1 = b * pb, min(len(S), (b + 1) * pb)
        Sk = S[min(len(S) - 1, a0 + pb // 2)]                    # the pose in the middle of the count
        kw = knee[a0:a1]; kw = kw[np.isfinite(kw)]
        k_mid = float(np.median(kw)) if len(kw) else stand
        level = 'low' if stand - k_mid > 28 else None
        bounce = bool(len(kw) >= 3 and (kw.max() - kw.min()) > 14 and np.argmin(kw) not in (0, len(kw) - 1))
        ax_l, ax_r = float(Sk[L_AN][0]), float(Sk[R_AN][0])
        weight = None
        if abs(ax_l - ax_r) > 0.7:
            if abs(ax_l) < 0.35 * abs(ax_r): weight = 'left'
            elif abs(ax_r) < 0.35 * abs(ax_l): weight = 'right'
        facing = 'side' if sw[min(len(sw) - 1, a0 + pb // 2)] / swmax < 0.5 else 'front'
        sp = float(np.max(speed[a0:a1])) / mean_sp if a1 > a0 else 1.0
        dyn = 'hit' if sp > 2.2 else ('flow' if sp < 1.3 else None)
        out.append(dict(n=b + 1, arms=_arms_phrase(Sk), level=level, bounce=bounce, weight=weight, facing=facing, dyn=dyn))
    return out


def _name(counts):
    """A short, stable nickname from what dominates the move ('Bounce Reach', 'Side Step', 'Open Hit')."""
    n = max(1, len(counts))
    bounce = sum(c['bounce'] for c in counts) / n; low = sum(c['level'] == 'low' for c in counts) / n
    side = sum(c['facing'] == 'side' for c in counts) / n
    shifts = sum(1 for a, b in zip(counts, counts[1:]) if a['weight'] and b['weight'] and a['weight'] != b['weight'])
    hits = sum(c['dyn'] == 'hit' for c in counts) / n
    arms = [c['arms'] for c in counts]
    up = sum('up' in a for a in arms) / n; wide = sum('wide' in a or ' out' in a for a in arms) / n; chest = sum('chest' in a for a in arms) / n
    first = 'Bounce' if bounce >= 0.4 else ('Low' if low >= 0.4 else ('Side' if side >= 0.3 else ('Step' if shifts >= 2 else 'Groove')))
    second = 'Reach' if up >= 0.3 else ('Open' if wide >= 0.3 else ('Chest' if chest >= 0.3 else ('Hit' if hits >= 0.3 else 'Step')))
    if first == 'Step' and second == 'Step': second = 'Touch'
    return f"{first} {second}"


def _tags(counts):
    n = max(1, len(counts))
    score = dict(bouncy=sum(c['bounce'] for c in counts) / n, sharp=sum(c['dyn'] == 'hit' for c in counts) / n,
                 smooth=sum(c['dyn'] == 'flow' for c in counts) / n, big=sum(('wide' in c['arms'] or 'up' in c['arms']) for c in counts) / n)
    return [t for t, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def move_card(key, pi):
    """Everything Brodie can say about one phrase (or half): name, how to start, one cue per count, how it should feel."""
    counts = count_cues(key, pi)
    p = key['phrases'][pi]; moves = key.get('moves') or []
    src_i = p.get('of', pi) if isinstance(p.get('of'), int) else pi
    src = (moves[src_i] if src_i < len(moves) else {}).get('src') or key.get('source') or key.get('name', '')
    genre = sv.genre_of_src(src, fallback=sv.GENRE_OF.get(str(key.get('style', '')).lower(), 'hiphop') if key.get('style') else 'hiphop')
    first = counts[0] if counts else dict(weight=None, facing='front', level=None)
    stance = 'feet hip-width' if not first.get('weight') else f"weight on your {first['weight']} foot"
    setup = f"Start {'side-on' if first.get('facing') == 'side' else 'facing me'}, {stance}" + (', knees bent' if first.get('level') == 'low' else '') + '.'
    tags = _tags(counts)
    return dict(name=_name(counts), setup=setup, counts=counts, feel=sv.feel_line(genre, tags), genre=genre, tags=tags)


def count_line(c, prev=None):
    """One count, at most 8 words: what changes from the last count ("Three: arms up in a V, sink low")."""
    parts = []
    if prev is None or c['arms'] != prev['arms']: parts.append(c['arms'])
    if c['level'] == 'low' and (prev is None or prev['level'] != 'low'): parts.append('sink low')
    elif c['level'] is None and prev is not None and prev['level'] == 'low': parts.append('stand up')
    if c['weight'] and (prev is None or c['weight'] != prev['weight']): parts.append(f"weight {c['weight']}")
    if c['facing'] == 'side' and (prev is None or prev['facing'] != 'side'): parts.append('turn side-on')
    elif c['facing'] == 'front' and prev is not None and prev['facing'] == 'side': parts.append('face me')
    if not parts: parts.append('hit it' if c['dyn'] == 'hit' else ('bounce' if c['bounce'] else 'hold it'))
    words = []
    for p in parts:
        if len(words) + len(p.split()) > 7: break
        words += p.split() + [',']
    text = ' '.join(words).rstrip(' ,').replace(' ,', ',')
    return text[:1].upper() + text[1:]


def count_lines(card):
    out = []; prev = None
    for c in card['counts']:
        out.append(count_line(c, prev)); prev = c
    return out


def intro_line(card, label):
    """'Move one, counts 1 to 4: Bounce Reach. Start facing me, feet hip-width.' (at most 16 words)"""
    line = f"{label}: {card['name']}. {card['setup']}"
    w = line.split()
    return ' '.join(w[:16]) if len(w) > 16 else line


def card_note(card, label=''):
    """The background note for Gemini (facts only, at most 60 words), so it can answer "which foot?" in its own words."""
    lines = count_lines(card)
    body = '; '.join(f"{c['n']}: {t}" for c, t in zip(card['counts'], lines))
    note = f"Move {label} '{card['name']}'. {card['setup']} Counts: {body}. Feel: {card['feel']}"
    w = note.split()
    return ' '.join(w[:60])
