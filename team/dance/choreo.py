"""Choreography generation: routines stitched from library phrases by style and ability.

Every library phrase gets a motion-complexity number (how fast the joint angles change, plus hip energy). The phrases are split into
tertiles: beginner, intermediate, advanced (the keys' own `difficulty` field is a constant 2, so this replaces it). A routine is
N distinct phrases from a style pool at one BPM, filtered by tier and ordered easiest first, concatenated into a new key with the
same layout as any other (so the engine, scorer and renderer need nothing new).

Generated keys are named gen_<style>_<level>_<seed> (or gen_daily_<yyyymmdd>) and are deterministic: from_name() rebuilds the
same routine on any worker, so a friend's challenge code can load it without storing the key anywhere.

  generate(style, level, n_phrases=None, seed=None) -> key
  from_name(name) -> key
  with_combined(key, upto) -> (key copy with an extra phrase spanning sections 0..upto, its phrase index)
  daily(date_str=None) -> key
  tier_of(key_name, phrase_index) -> 'beginner' | 'intermediate' | 'advanced'
"""
import datetime, random
import numpy as np
from .reference import list_keys, load_key, key_from_arrays, phrase_slice

STYLE_POOLS = {
    'hiphop': ('aist_lh_', 'aist_mh_'),
    'pop': ('aist_po_', 'aist_lo_'),
    'house': ('aist_ho_',),
    'waack': ('aist_wa_',),
    'krump': ('aist_kr_',),
    'jazz': ('aist_js_',),
    'lock': ('aist_lo_',),
    'fitness': ('aist_',),                     # filtered to the gentlest third below, plus house
}
STYLE_TITLES = dict(hiphop='Hip-Hop', pop='Pop Choreo', house='House', waack='Waacking', krump='Krump', jazz='Street Jazz', lock='Locking',
                    fitness='Dance Cardio', daily='Daily Challenge')
LEVEL_TITLES = dict(beginner='Starter', intermediate='Level Up', advanced='Pro')
LEVEL_TEMPO = dict(beginner=0.8, intermediate=0.95, advanced=1.05)
LEVEL_PHRASES = dict(beginner=3, intermediate=3, advanced=4)
LEVEL_INDEX = dict(beginner=1, intermediate=2, advanced=3)
DAILY_ROTATION = ('hiphop', 'pop', 'house', 'fitness', 'waack', 'jazz', 'krump')    # Monday first
# live.py style codes -> pool names
STYLE_CODE = dict(lh='hiphop', mh='hiphop', po='pop', lo='lock', ho='house', wa='waack', kr='krump', js='jazz', fitness='fitness',
                  any='hiphop', squats='fitness')

_INDEX = None
_KEYS = {}


def _key(name):
    if name not in _KEYS: _KEYS[name] = load_key(name)
    return _KEYS[name]


def phrase_complexity(key, i):
    """Mean absolute change of the visible joint angles per sample (degrees), plus a weighted mean hip energy."""
    F, M, E, _ = phrase_slice(key, i)
    if len(F) < 2: return 0.0
    dF = np.abs(np.diff(F, axis=0)); both = M[1:] & M[:-1]
    vel = float(dF[both].mean()) if both.any() else 0.0
    return vel + 4.0 * float(np.mean(E)) if len(E) else vel


def index():
    """[{key, phrase, bpm, per_beat, complexity, tier}] over the AIST library phrases (the stitchable pool). Cached."""
    global _INDEX
    if _INDEX is not None: return _INDEX
    rows = []
    for k in list_keys():
        name = k.get('name', '')
        if 'error' in k or not name.startswith('aist_'): continue
        key = _key(name)
        for i in range(len(key['phrases'])):
            if key['phrases'][i]['beats'] < 8: continue
            rows.append(dict(key=name, phrase=i, bpm=float(key['bpm']), per_beat=int(key['per_beat']), complexity=phrase_complexity(key, i)))
    if rows:
        c = np.array([r['complexity'] for r in rows]); lo, hi = np.quantile(c, [1 / 3, 2 / 3])
        for r in rows: r['tier'] = 'beginner' if r['complexity'] <= lo else 'intermediate' if r['complexity'] <= hi else 'advanced'
    _INDEX = rows
    return rows


def tier_of(key_name, phrase_index):
    for r in index():
        if r['key'] == key_name and r['phrase'] == phrase_index: return r['tier']
    return None


def pool(style):
    style = STYLE_CODE.get(style, style)
    rows = index()
    if style == 'fitness':
        easy = [r for r in rows if r['tier'] == 'beginner']
        house = [r for r in rows if r['key'].startswith('aist_ho_') and r['tier'] != 'advanced']
        seen = set(); out = []
        for r in easy + house:
            if (r['key'], r['phrase']) not in seen: seen.add((r['key'], r['phrase'])); out.append(r)
        return out
    prefixes = STYLE_POOLS.get(style, STYLE_POOLS['hiphop'])
    return [r for r in rows if r['key'].startswith(prefixes)]


def _pick(rows, level, n, rng):
    order = ['beginner', 'intermediate', 'advanced']; li = order.index(level)
    pref = {'beginner': [0, 1, 2], 'intermediate': [1, 0, 2], 'advanced': [2, 1, 0]}[level]
    chosen = []
    for t in pref:                                     # fill from the preferred tier, then the neighbours
        cands = [r for r in rows if r['tier'] == order[t] and r not in chosen]
        rng.shuffle(cands)
        chosen += cands[:max(0, n - len(chosen))]
        if len(chosen) >= n: break
    chosen = chosen[:n]
    chosen.sort(key=lambda r: r['complexity'])         # easiest first: the routine builds up
    return chosen, li


def generate(style='hiphop', level='beginner', n_phrases=None, seed=None, name=None):
    """A new routine key. Deterministic for (style, level, n_phrases, seed)."""
    style = STYLE_CODE.get(style, style)
    if level not in LEVEL_TITLES: level = 'beginner'
    seed = int(seed if seed is not None else random.randint(1000, 9999))
    rows = pool(style)
    if not rows: raise RuntimeError(f'no library phrases for style {style!r}')
    bpms = {}
    for r in rows: bpms.setdefault((r['bpm'], r['per_beat']), []).append(r)
    rows = max(bpms.values(), key=len)                 # one BPM so the phrases butt together on the grid
    n = int(n_phrases or LEVEL_PHRASES[level]); n = max(1, min(n, len(rows)))
    rng = random.Random(f'{style}:{level}:{n}:{seed}')
    chosen, _ = _pick(rows, level, n, rng)
    parts = [phrase_slice(_key(r['key']), r['phrase']) for r in chosen]
    F = np.concatenate([p[0] for p in parts]); M = np.concatenate([p[1] for p in parts])
    E = np.concatenate([p[2] for p in parts]); S = np.concatenate([p[3] for p in parts])
    bpm = chosen[0]['bpm']; pb = chosen[0]['per_beat']
    name = name or f'gen_{style}_{level}_{seed}'
    key = key_from_arrays(name, bpm, F, M=M, E=E, S=S, per_beat=pb, counts_per_phrase=8, source='generated from the library', save=False)
    key.update(style=style, difficulty=LEVEL_INDEX[level], level=level, generated=True, seed=seed,
               title=(f"{STYLE_TITLES.get(style, style)} {LEVEL_TITLES[level]}" if style == 'fitness' else f"{STYLE_TITLES.get(style, style)} {LEVEL_TITLES[level]} Mix"),
               moves=[dict(src=r['key'], phrase=r['phrase'], tier=r['tier']) for r in chosen], default_tempo=LEVEL_TEMPO[level] if style != 'fitness' else 0.85)
    return key


def daily(date_str=None):
    """Today's challenge: everyone gets the same beginner routine; the style rotates by weekday."""
    d = datetime.date.today() if date_str is None else datetime.datetime.strptime(str(date_str), '%Y%m%d').date()
    style = DAILY_ROTATION[d.weekday()]; ymd = d.strftime('%Y%m%d')
    key = generate(style, 'beginner', seed=int(ymd), name=f'gen_daily_{ymd}')
    key['title'] = f"Daily: {STYLE_TITLES.get(style, style)}"; key['daily'] = ymd
    return key


def from_name(name):
    """Rebuild a generated key from its name (gen_<style>_<level>_<seed> or gen_daily_<yyyymmdd>)."""
    parts = str(name).split('_')
    if len(parts) == 3 and parts[1] == 'daily': return daily(parts[2])
    if len(parts) != 4 or parts[0] != 'gen': raise ValueError(f'not a generated key name: {name!r}')
    return generate(parts[1], parts[2], seed=int(parts[3]))


def load_any(name):
    """load_key for library keys, from_name for generated ones."""
    return from_name(name) if str(name).startswith('gen_') else load_key(name)


def with_combined(key, upto):
    """A shallow copy of `key` with one more phrase entry spanning sections 0..upto (a combine run). Returns (key, index)."""
    k = dict(key); ph = list(key['phrases']); upto = max(0, min(int(upto), len(ph) - 1))
    beats = sum(p['beats'] for p in ph[:upto + 1])
    ph.append(dict(start_beat=ph[0]['start_beat'], beats=beats, combined=upto + 1)); k['phrases'] = ph
    return k, len(ph) - 1


def with_half(key, section, half):
    """A shallow copy of `key` with one more phrase entry: half (0 or 1) of 8-count section `section`, i.e. 4 counts. Returns (key, index)."""
    k = dict(key); ph = list(key['phrases']); p = ph[max(0, min(int(section), len(ph) - 1))]
    h = p['beats'] // 2
    ph.append(dict(start_beat=p['start_beat'] + (h if half else 0), beats=(p['beats'] - h) if half else h, half=int(half))); k['phrases'] = ph
    return k, len(ph) - 1


def with_counts(key, section, first, last):
    """A shallow copy of `key` with one more phrase entry: counts first..last (0-based, inclusive) of section `section`, at least two
    counts (looping the weak bar). Returns (key, index)."""
    k = dict(key); ph = list(key['phrases']); p = ph[max(0, min(int(section), len(ph) - 1))]
    a = max(0, min(int(first), p['beats'] - 2)); b = max(a + 1, min(int(last), p['beats'] - 1))
    ph.append(dict(start_beat=p['start_beat'] + a, beats=b - a + 1, loop=(a, b))); k['phrases'] = ph
    return k, len(ph) - 1


def routine_seconds(key, tempo=1.0):
    return sum(p['beats'] for p in key['phrases'] if not p.get('combined') and 'half' not in p) * 60.0 / (float(key['bpm']) * float(tempo))
