"""Progression: grades, playful titles, XP, levels, unlocks and achievements. Pure functions, no I/O.

The backend (team/backend/db.py) stores a profile per first name and calls these to award a session; the live session uses them
to draw the result reveal, the level-up moment and the scorecard.

  grade(score)                       -> 'S' | 'A' | 'B' | 'C' | 'D'
  title_for(result, ctx)             -> a short fun title for one result ("Human Metronome")
  level_for(xp)                      -> (index, name, floor_xp, next_xp or None)
  unlocked(level_index)              -> set of difficulty names and features the level opens
  xp_for_session(rounds, mode, difficulty, extras) -> (xp, breakdown)
  new_achievements(profile, session) -> [{id, name, desc}] earned this session and not before
"""
import os

GRADES = (('S', 92.0), ('A', 80.0), ('B', 65.0), ('C', 50.0))
LEVELS = [(0, 'Rookie'), (100, 'Groover'), (250, 'Mover'), (450, 'Shaker'), (700, 'Headliner'), (1000, 'Trendsetter'),
          (1400, 'Icon'), (1900, 'Legend'), (2500, 'Main Character')]
DIFFICULTIES = ('beginner', 'intermediate', 'advanced')
DIFF_MULT = dict(beginner=1.0, intermediate=1.3, advanced=1.6)
# level index -> what it opens (cumulative)
UNLOCKS = {0: {'beginner', 'learn', 'battle', 'daily'}, 1: {'intermediate'}, 2: {'advanced'}}
DIFF_UNLOCK_LEVEL = dict(beginner=0, intermediate=1, advanced=2)


def grade(score):
    s = float(score or 0.0)
    for g, lo in GRADES:
        if s >= lo: return g
    return 'D'


def level_for(xp):
    xp = max(0, int(xp or 0)); idx = 0
    for i, (lo, _) in enumerate(LEVELS):
        if xp >= lo: idx = i
    nxt = LEVELS[idx + 1][0] if idx + 1 < len(LEVELS) else None
    return idx, LEVELS[idx][1], LEVELS[idx][0], nxt


def unlock_all():
    return (os.environ.get('DANCE_UNLOCK_ALL', '0') or '0').strip().lower() in ('1', 'true', 'yes', 'on')


def unlocked(level_index):
    if unlock_all(): return set().union(*UNLOCKS.values())
    out = set()
    for lv, feats in UNLOCKS.items():
        if level_index >= lv: out |= feats
    return out


def difficulty_unlocked(difficulty, level_index):
    return difficulty in unlocked(level_index)


def xp_to_unlock(difficulty, xp):
    """XP still needed to open a difficulty (0 when already open)."""
    if unlock_all(): return 0
    lv = DIFF_UNLOCK_LEVEL.get(difficulty, 0)
    return max(0, LEVELS[lv][0] - int(xp or 0))


def title_for(result, ctx=None):
    """One title for one result. ctx: {glow_up: bool, duel_win: bool, freestyle: bool}."""
    ctx = ctx or {}; r = result or {}
    if not r.get('scored', True) and not r.get('freestyle'): return 'Camera Shy (step back)'
    s = float(r.get('score') or 0); timing = float(r.get('timing') or 0); spatial = float(r.get('spatial') or 0)
    energy = float(r.get('energy') or 0); combo = int(r.get('max_combo') or 0)
    if r.get('freestyle') or ctx.get('freestyle'):
        if s >= 70: return 'Freestyle Gremlin'
    if s >= 95: return 'Main Character'
    if timing >= 0.9 and s >= 70: return 'Human Metronome'
    if combo >= 8: return 'Combo Menace'
    if ctx.get('glow_up'): return 'Glow-Up Arc'
    if ctx.get('duel_win'): return 'Duel Boss'
    if spatial >= 0.85: return 'Clean Lines Club'
    if energy >= 0.85 and s >= 50: return 'Hype Engine'
    pg = {g: v for g, v in (r.get('per_group') or {}).items() if v is not None}
    if pg and min(pg, key=pg.get) == 'arms' and s >= 60: return 'Arms Day Champion'
    if s < 40: return 'Certified Off-Beat (for now)'
    return dict(S='Main Character', A='Smooth Operator', B='Rising Mover', C='Warming Up', D='Brave First Try')[grade(s)]


def squad_title(crew_sync, crew_mean, spread):
    if crew_sync is not None and crew_sync >= 0.85: return 'Synchronized Menaces'
    if crew_mean >= 80: return 'Main Character Crew'
    if spread > 30: return 'One Carry, Rest Vibes'
    return 'Crew in Training'


def xp_for_session(rounds, mode='learn', difficulty='beginner', extras=None):
    """rounds: [{score, scored, max_combo?, glow_up?}] for every attempt. extras: {first_today, duel_win, battle_done,
    combo_run}. Returns (xp, [(label, amount), ...])."""
    extras = extras or {}; mult = DIFF_MULT.get(difficulty, 1.0); out = []
    scored = [r for r in rounds or [] if r.get('scored') and r.get('score') is not None]
    base = sum(int(round(float(r['score']) / 4.0 * mult)) for r in scored)
    if base: out.append(('dancing', base))
    if scored: out.append(('first phrase', 10))
    best = max((float(r['score']) for r in scored), default=0.0)
    if best >= 92: out.append(('S rank', 25))
    elif best >= 80: out.append(('A rank', 15))
    if any(int(r.get('max_combo') or 0) >= 8 for r in scored): out.append(('combo 8+', 10))
    if any(r.get('glow_up') for r in scored): out.append(('glow-up', 10))
    if extras.get('battle_done'): out.append(('battle', 20))
    if extras.get('duel_win'): out.append(('duel win', 30))
    if extras.get('combo_run'): out.append(('full routine', 15))
    if extras.get('first_today') and scored: out.append(('first today', 20))
    return sum(a for _, a in out), out


ACHIEVEMENTS = [
    ('first_dance', 'First Steps', 'Finished your first phrase'),
    ('s_rank', 'S-Rank', 'Scored 92 or more'),
    ('perfect_8', 'Eight Clean', 'Eight perfect counts in a row'),
    ('glow_up', 'Glow-Up', 'Improved a phrase by 15+ points'),
    ('duel_win', 'Duel Winner', "Beat a friend's challenge"),
    ('style_explorer', 'Style Hopper', 'Danced three different styles'),
    ('streak_3', '3-Day Streak', 'Danced three days in a row'),
    ('battle_done', 'Ghost Fighter', 'Finished a Ghost Battle'),
    ('combo_run', 'Full Routine', 'Passed a full combined run'),
    ('teacher', 'Sensei', 'Created a class'),
    ('daily', 'Daily Grinder', 'Played the daily challenge'),
    ('squad', 'Crew Up', 'Danced in a squad'),
]
ACH_BY_ID = {a[0]: dict(id=a[0], name=a[1], desc=a[2]) for a in ACHIEVEMENTS}


def session_achievements(session):
    """Which achievement ids a session (plus the profile after it) qualifies for. session: {rounds, styles (all-time set), streak,
    duel_win, battle_done, combo_run, teacher, daily, squad}."""
    rounds = session.get('rounds') or []; scored = [r for r in rounds if r.get('scored') and r.get('score') is not None]
    got = set()
    if scored: got.add('first_dance')
    if any(float(r['score']) >= 92 for r in scored): got.add('s_rank')
    if any(int(r.get('max_combo') or 0) >= 8 for r in scored): got.add('perfect_8')
    if any(r.get('glow_up') for r in scored): got.add('glow_up')
    if len(set(session.get('styles') or ())) >= 3: got.add('style_explorer')
    if int(session.get('streak') or 0) >= 3: got.add('streak_3')
    for flag in ('duel_win', 'battle_done', 'combo_run', 'teacher', 'daily', 'squad'):
        if session.get(flag): got.add(flag)
    return got


def new_achievements(before_ids, session):
    """Achievements earned this session that the profile did not have. Returns a list of {id, name, desc} in catalogue order."""
    have = set(before_ids or ()); got = session_achievements(session) - have
    return [ACH_BY_ID[a[0]] for a in ACHIEVEMENTS if a[0] in got]
