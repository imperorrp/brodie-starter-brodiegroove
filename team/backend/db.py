"""SQLite storage for the team backend: raw events, dance results with four-digit codes, telephone chains, classes, player
profiles (XP, level, streak, achievements, preferences, keyed by a spoken first name), and share/challenge tracking."""
import json, os, sqlite3, threading

_HERE = os.path.dirname(os.path.abspath(__file__))
# TEAM_DB overrides. On Vercel the package directory is read-only, so the default moves to /tmp (ephemeral: resets on a
# cold start and is not shared between instances; a hosted database is the real fix for a public deployment).
DB_PATH = os.environ.get('TEAM_DB') or (os.path.join('/tmp', 'team.db') if os.environ.get('VERCEL') else os.path.join(_HERE, 'team.db'))
_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False); c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, type TEXT, t_wall REAL, body TEXT)')
        c.execute('CREATE INDEX IF NOT EXISTS events_t ON events (t_wall)')
        c.execute('CREATE TABLE IF NOT EXISTS dance_results (id INTEGER PRIMARY KEY, code TEXT UNIQUE, name TEXT, key TEXT, mode TEXT, total REAL, rounds TEXT, created REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS dance_chains (id INTEGER PRIMARY KEY, code TEXT UNIQUE, key TEXT, links TEXT, created REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS dance_classes (id INTEGER PRIMARY KEY, code TEXT UNIQUE, teacher TEXT, key TEXT, students TEXT, created REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS dance_profiles (name_key TEXT PRIMARY KEY, display_name TEXT, xp INTEGER, level INTEGER, streak INTEGER, '
                  'last_day TEXT, styles TEXT, achievements TEXT, prefs TEXT, created REAL, updated REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS dance_shares (id INTEGER PRIMARY KEY, code TEXT, kind TEXT, t REAL)')
        # a battle winner's dance as body points (never video), only with their yes: friends with the code battle it (30 days)
        c.execute('CREATE TABLE IF NOT EXISTS dance_ghosts (code TEXT PRIMARY KEY, name TEXT, src_key TEXT, bpm REAL, per_beat INTEGER, '
                  'data TEXT, total REAL, score100 REAL, difficulty TEXT, created REAL)')
        # the one-time token that lets the caller who earned a code save their ghost under it (returned only to them by post_result)
        c.execute('CREATE TABLE IF NOT EXISTS dance_ghost_tokens (code TEXT PRIMARY KEY, token TEXT, created REAL)')
        cols = {r[1] for r in c.execute('PRAGMA table_info(dance_results)').fetchall()}
        if 'score100' not in cols: c.execute('ALTER TABLE dance_results ADD COLUMN score100 REAL')
        if 'challenge' not in cols: c.execute('ALTER TABLE dance_results ADD COLUMN challenge TEXT')


# ---- dance: results with four-digit codes (async duels), telephone chains ----
import random, time as _time


CODE_TABLES = ('dance_results', 'dance_chains', 'dance_classes')


CODE_DAYS = 30


def _free_old_codes(c):
    """Codes older than CODE_DAYS that nobody has taken on lately go back in the pool. Everything keyed by such a code (its ghost,
    the ghost's save token, its share log) is deleted in the same transaction, so a reissued code never inherits someone else's
    ghost or history. A code with a live ghost is never freed."""
    cut = _time.time() - CODE_DAYS * 86400
    old = [r[0] for r in c.execute('SELECT code FROM dance_results WHERE code IS NOT NULL AND created < ? AND code NOT IN '
                                   '(SELECT code FROM dance_shares WHERE kind = ? AND t > ?) AND code NOT IN (SELECT code FROM dance_ghosts WHERE created >= ?)',
                                   (cut, 'challenge_start', cut, cut)).fetchall()]
    for code in old:
        for t in ('dance_ghosts', 'dance_ghost_tokens', 'dance_shares'): c.execute(f'DELETE FROM {t} WHERE code = ?', (code,))
        c.execute('UPDATE dance_results SET code = NULL WHERE code = ?', (code,))


def _new_code(c, table):
    """A four-digit code free in every code table: a spoken code then means exactly one thing (a battle, a telephone chain or a
    class). `table` is kept for the callers' readability."""
    for _ in range(200):
        code = f'{random.randint(1000, 9999)}'
        if all(c.execute(f'SELECT 1 FROM {t} WHERE code = ?', (code,)).fetchone() is None
               for t in CODE_TABLES + ('dance_ghosts', 'dance_ghost_tokens', 'dance_shares')): return code   # never a code something still points at
    raise RuntimeError('no free code')


def resolve_code(code):
    """What a spoken four-digit code is: {'code', 'kinds': [...]} with each match ('battle' with the result's name and score,
    'telephone' with the number of links, 'class' with the teacher). Old codes made before codes were unique can match twice."""
    code = str(code); kinds = []
    r = get_result(code)
    if r: kinds.append(dict(kind='battle', name=r.get('name'), score100=r.get('score100') if r.get('score100') is not None else r.get('total'), key=r.get('key'), mode=r.get('mode'),
                            has_ghost=get_ghost(code, with_data=False) is not None))
    ch = get_chain(code)
    if ch: kinds.append(dict(kind='telephone', links=len(ch.get('links') or []), key=ch.get('key')))
    cl = get_class(code, with_key=False)
    if cl: kinds.append(dict(kind='class', teacher=cl.get('teacher'), n_students=len(cl.get('students') or [])))
    return dict(code=code, kinds=kinds)


def add_result(name, key, mode, total, rounds, score100=None, challenge=None, issue_code=True):
    """score100: the result on a 0-100 scale for like-for-like duels across modes (learn: best phrase, battle: total / 3).
    challenge: the code this result challenged (it then appears on that code's crew board). issue_code=False (a lost battle: the
    site promises codes to winners) stores the result with no code (NULL; SQLite allows many) and returns None."""
    if score100 is None: score100 = float(total) / 3.0 if mode in ('battle', 'squad_battle') else float(total)
    with _lock, _conn() as c:
        _free_old_codes(c)
        code = _new_code(c, 'dance_results') if issue_code else None
        c.execute('INSERT INTO dance_results (code, name, key, mode, total, rounds, created, score100, challenge) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                  (code, name or 'anonymous', key, mode, float(total), json.dumps(rounds), _time.time(), float(score100), str(challenge) if challenge else None))
        if code: c.execute('INSERT INTO dance_shares (code, kind, t) VALUES (?, ?, ?)', (code, 'issued', _time.time()))
        if challenge: c.execute('INSERT INTO dance_shares (code, kind, t) VALUES (?, ?, ?)', (str(challenge), 'challenge_complete', _time.time()))
    return code


def get_result(code):
    with _lock, _conn() as c:
        r = c.execute('SELECT * FROM dance_results WHERE code = ?', (str(code),)).fetchone()
    if r is None: return None
    d = dict(r); d['rounds'] = json.loads(d['rounds'] or '[]'); return d


def history(name, limit=10):
    """A caller's past results, newest first (first-name match, case-insensitive)."""
    with _lock, _conn() as c:
        rows = c.execute('SELECT code, name, key, mode, total, rounds, created FROM dance_results WHERE lower(name) = lower(?) ORDER BY created DESC LIMIT ?', (name or '', int(limit))).fetchall()
    out = []
    for r in rows:
        d = dict(r); d['rounds'] = json.loads(d['rounds'] or '[]'); out.append(d)
    return out


def leaderboard(key=None, limit=20, mode=None, since=None):
    """Results by score, best first. mode / since (epoch seconds) filter it: the daily board is mode='daily' since local midnight."""
    q = 'SELECT code, name, key, mode, total, score100, created FROM dance_results WHERE 1=1'; args = []
    if key: q += ' AND key = ?'; args.append(key)
    if mode: q += ' AND mode = ?'; args.append(mode)
    if since is not None: q += ' AND created >= ?'; args.append(float(since))
    q += ' ORDER BY COALESCE(score100, total) DESC LIMIT ?'; args.append(int(limit))
    with _lock, _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(r) for r in rows]


def daily_played(name, key):
    """True when this (named) caller already had today's daily shot on this routine."""
    if not name: return False
    with _lock, _conn() as c:
        r = c.execute("SELECT 1 FROM dance_results WHERE lower(name) = lower(?) AND key = ? AND mode = 'daily' LIMIT 1", (name, key)).fetchone()
    return r is not None


def start_chain(key, link):
    with _lock, _conn() as c:
        code = _new_code(c, 'dance_chains')
        c.execute('INSERT INTO dance_chains (code, key, links, created) VALUES (?, ?, ?, ?)', (code, key, json.dumps([link]), _time.time()))
    return code


def add_link(code, link):
    with _lock, _conn() as c:
        r = c.execute('SELECT links FROM dance_chains WHERE code = ?', (str(code),)).fetchone()
        if r is None: return None
        links = json.loads(r['links'] or '[]'); links.append(link)
        c.execute('UPDATE dance_chains SET links = ? WHERE code = ?', (json.dumps(links), str(code)))
    return len(links)


def get_chain(code):
    with _lock, _conn() as c:
        r = c.execute('SELECT * FROM dance_chains WHERE code = ?', (str(code),)).fetchone()
    if r is None: return None
    d = dict(r); d['links'] = json.loads(d['links'] or '[]'); return d


# ---- class mode: a teacher's recorded phrases under a code, students scored against them, a class report ----
VERDICT_VALUE = dict(green=1.0, amber=0.5, red=0.0)


def start_class(teacher, key):
    """key: a full key dict (the teacher's own phrases, skeleton included) so students can be served from any worker."""
    with _lock, _conn() as c:
        code = _new_code(c, 'dance_classes')
        c.execute('INSERT INTO dance_classes (code, teacher, key, students, created) VALUES (?, ?, ?, ?, ?)',
                  (code, teacher or 'teacher', json.dumps(key), json.dumps([]), _time.time()))
    return code


def get_class(code, with_key=True):
    with _lock, _conn() as c:
        r = c.execute('SELECT * FROM dance_classes WHERE code = ?', (str(code),)).fetchone()
    if r is None: return None
    d = dict(r); d['students'] = json.loads(d['students'] or '[]')
    key = json.loads(d['key'] or 'null')
    d['n_phrases'] = len((key or {}).get('phrases', [])); d['key_name'] = (key or {}).get('name'); d['bpm'] = (key or {}).get('bpm')
    if with_key: d['key'] = key
    else: d.pop('key', None)
    return d


def add_student(code, student):
    """student: {name, phrases: [{phrase, score, scored, verdicts: ['green',...], per_group: {...}, weak_beats: [...]}]}.
    A returning student (same name) replaces their earlier row. Returns the class summary for the spoken line."""
    with _lock, _conn() as c:
        r = c.execute('SELECT students FROM dance_classes WHERE code = ?', (str(code),)).fetchone()
        if r is None: return None
        students = json.loads(r['students'] or '[]')
        name = (student.get('name') or 'anonymous')
        students = [s for s in students if (s.get('name') or '').lower() != name.lower()]
        student = dict(student); student['name'] = name; student['total'] = _student_total(student); student['at'] = _time.time()
        students.append(student)
        c.execute('UPDATE dance_classes SET students = ? WHERE code = ?', (json.dumps(students), str(code)))
    ranked = sorted(students, key=lambda s: -float(s.get('total') or 0))
    rank = next(i + 1 for i, s in enumerate(ranked) if s['name'] == name)
    mean = sum(float(s.get('total') or 0) for s in students) / len(students)
    return dict(code=str(code), name=name, total=student['total'], rank=rank, n_students=len(students), class_mean=round(mean, 1))


def _student_total(student):
    scores = [float(p.get('score') or 0) for p in student.get('phrases', []) if p.get('scored', True)]
    return round(sum(scores) / len(scores), 1) if scores else 0.0


def class_report(code):
    """The teacher's view: students x phrases heatmap, per-beat class health per phrase, the weakest phrase and joint group, one
    suggestion. Verdict colours count green 1, amber 0.5, red 0."""
    cls = get_class(code, with_key=False)
    if cls is None: return None
    n_ph = cls['n_phrases'] or max([len(s.get('phrases', [])) for s in cls['students']] or [0])
    heat = []
    for s in cls['students']:
        row = [None] * n_ph
        for p in s.get('phrases', []):
            i = int(p.get('phrase', 0))
            if 0 <= i < n_ph: row[i] = (round(float(p['score']), 0) if p.get('scored', True) and p.get('score') is not None else None)
        heat.append(dict(name=s['name'], scores=row, total=s.get('total')))
    phrase_mean = []; beat_health = []; group_err = []
    for i in range(n_ph):
        vals = [r['scores'][i] for r in heat if r['scores'][i] is not None]
        phrase_mean.append(round(sum(vals) / len(vals), 1) if vals else None)
        beats = {}; groups = {}
        for s in cls['students']:
            for p in s.get('phrases', []):
                if int(p.get('phrase', 0)) != i or not p.get('scored', True): continue
                for b, v in enumerate(p.get('verdicts') or []): beats.setdefault(b, []).append(VERDICT_VALUE.get(v, 0.0))
                for g, e in (p.get('per_group') or {}).items():
                    if e is not None: groups.setdefault(g, []).append(float(e))
        beat_health.append([round(sum(beats[b]) / len(beats[b]), 2) for b in sorted(beats)])
        group_err.append({g: round(sum(v) / len(v), 1) for g, v in groups.items()})
    weakest_phrase = None; weakest_beat = None; weakest_group = None
    if any(m is not None for m in phrase_mean):
        weakest_phrase = min((i for i in range(n_ph) if phrase_mean[i] is not None), key=lambda i: phrase_mean[i])
        bh = beat_health[weakest_phrase]
        if bh: weakest_beat = min(range(len(bh)), key=lambda b: bh[b])
        ge = group_err[weakest_phrase]
        if ge: weakest_group = max(ge, key=ge.get)
    n = len(cls['students'])
    if n == 0: suggestion = 'No students yet. Share the code.'
    elif weakest_phrase is None: suggestion = 'Students called but nothing scored: ask them to prop the phone lower and step back.'
    else:
        parts = [f"Phrase {weakest_phrase + 1} is where the class drops (mean {phrase_mean[weakest_phrase]:.0f})"]
        if weakest_beat is not None: parts.append(f"count {weakest_beat + 1} is the weak count")
        if weakest_group: parts.append(f"the {weakest_group} are the joint group most off")
        parts.append('slow it to 85 percent and loop that phrase' if phrase_mean[weakest_phrase] < 60 else 'one more pass at full speed should do it')
        suggestion = '; '.join(parts) + '.'
    not_attempted = [r['name'] for r in heat if all(v is None for v in r['scores'])]
    return dict(code=str(code), teacher=cls['teacher'], key_name=cls['key_name'], n_phrases=n_ph, n_students=n, heatmap=heat, phrase_mean=phrase_mean,
                beat_health=beat_health, group_error=group_err, weakest_phrase=weakest_phrase, weakest_beat=weakest_beat, weakest_group=weakest_group,
                not_attempted=not_attempted, suggestion=suggestion)


def add_event(ev):
    with _lock, _conn() as c:
        c.execute('INSERT INTO events (type, t_wall, body) VALUES (?, ?, ?)', (ev.get('type'), ev.get('t_wall'), json.dumps(ev)))


def recent_events(limit=50):
    with _lock, _conn() as c:
        rows = c.execute('SELECT body FROM events ORDER BY id DESC LIMIT ?', (int(limit),)).fetchall()
    return [json.loads(r['body']) for r in rows]


# ---- profiles: XP, level, streak, achievements, preferences (keyed by a spoken first name, case-insensitive) ----
def _progression():
    import importlib.util, sys as _sys
    if 'team_progression' in _sys.modules: return _sys.modules['team_progression']
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location('team_progression', os.path.join(os.path.dirname(here), 'dance', 'progression.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _sys.modules['team_progression'] = m
    return m


def _profile_row(r):
    if r is None: return None
    d = dict(r)
    for k, default in (('styles', '[]'), ('achievements', '[]'), ('prefs', '{}')): d[k] = json.loads(d.get(k) or default)
    pg = _progression(); lv = pg.level_for(d['xp'])
    d.update(level=lv[0], level_name=lv[1], level_floor=lv[2], next_at=lv[3], unlocked=sorted(pg.unlocked(lv[0])), name=d['display_name'])
    return d


def get_profile(name):
    if not name: return None
    with _lock, _conn() as c:
        r = c.execute('SELECT * FROM dance_profiles WHERE name_key = ?', (name.strip().lower(),)).fetchone()
    return _profile_row(r)


def _ensure_profile(c, key, display, now, prefs=None):
    c.execute('INSERT OR IGNORE INTO dance_profiles (name_key, display_name, xp, level, streak, last_day, styles, achievements, prefs, created, updated) '
              'VALUES (?, ?, 0, 0, 0, NULL, ?, ?, ?, ?, ?)', (key, display, '[]', '[]', json.dumps(prefs or {}), now, now))


def award(name, xp, style=None, achievements=(), day=None, session=None):
    """Add XP for a session. day: 'YYYY-MM-DD' (default today). session: the progression session summary (rounds, flags) used to
    compute new achievements with the profile's all-time styles and streak. Returns the before/after summary."""
    import datetime
    pg = _progression(); key = name.strip().lower(); day = day or datetime.date.today().isoformat(); now = _time.time()
    with _lock, _conn() as c:
        _ensure_profile(c, key, name.strip(), now)
        d = dict(c.execute('SELECT * FROM dance_profiles WHERE name_key = ?', (key,)).fetchone())
        styles = set(json.loads(d['styles'] or '[]')); have = list(json.loads(d['achievements'] or '[]'))
        last = d['last_day']; streak = int(d['streak'] or 0); first_today = last != day
        if last == day: pass
        elif last and (datetime.date.fromisoformat(day) - datetime.date.fromisoformat(last)).days == 1: streak += 1
        else: streak = 1
        if style: styles.add(style)
        sess = dict(session or {}); sess.update(styles=styles, streak=streak)
        new = [a['id'] for a in pg.new_achievements(have, sess)] + [a for a in (achievements or ()) if a not in have]
        new = list(dict.fromkeys(new))
        xp_before = int(d['xp'] or 0); xp_after = xp_before + max(0, int(xp))
        lb = pg.level_for(xp_before); la = pg.level_for(xp_after)
        c.execute('UPDATE dance_profiles SET xp = ?, level = ?, streak = ?, last_day = ?, styles = ?, achievements = ?, updated = ? WHERE name_key = ?',
                  (xp_after, la[0], streak, day, json.dumps(sorted(styles)), json.dumps(have + new), now, key))
    newly = sorted(pg.unlocked(la[0]) - pg.unlocked(lb[0]))
    return dict(name=name.strip(), xp_before=xp_before, xp_after=xp_after, gained=xp_after - xp_before, level_before=lb[0], level_after=la[0],
                level_name=la[1], level_floor=la[2], next_at=la[3], leveled_up=la[0] > lb[0], streak=streak, first_today=first_today,
                new_achievements=[pg.ACH_BY_ID[a] for a in new if a in pg.ACH_BY_ID], unlocked=sorted(pg.unlocked(la[0])), newly_unlocked=newly)


def set_prefs(name, prefs):
    key = name.strip().lower(); now = _time.time()
    with _lock, _conn() as c:
        _ensure_profile(c, key, name.strip(), now)
        p = json.loads(c.execute('SELECT prefs FROM dance_profiles WHERE name_key = ?', (key,)).fetchone()['prefs'] or '{}')
        p.update({k: v for k, v in (prefs or {}).items() if v})
        c.execute('UPDATE dance_profiles SET prefs = ?, updated = ? WHERE name_key = ?', (json.dumps(p), now, key))
    return p


def delete_user(name):
    """'Forget me': delete the profile and every result under that name; anonymise the name inside chains and classes."""
    key = (name or '').strip().lower(); out = dict(profile=0, results=0, chains=0, classes=0)
    if not key: return out
    with _lock, _conn() as c:
        out['profile'] = c.execute('DELETE FROM dance_profiles WHERE name_key = ?', (key,)).rowcount
        out['ghosts'] = c.execute('DELETE FROM dance_ghosts WHERE lower(name) = ?', (key,)).rowcount
        out['results'] = c.execute('DELETE FROM dance_results WHERE lower(name) = ?', (key,)).rowcount
        for table, col, label in (('dance_chains', 'links', 'chains'), ('dance_classes', 'students', 'classes')):
            for r in c.execute(f'SELECT code, {col} FROM {table}').fetchall():
                items = json.loads(r[col] or '[]'); hit = False
                for it in items:
                    if isinstance(it, dict) and (it.get('name') or '').lower() == key: it['name'] = 'anonymous'; hit = True
                if hit: c.execute(f'UPDATE {table} SET {col} = ? WHERE code = ?', (json.dumps(items), r['code'])); out[label] += 1
        c.execute('UPDATE dance_classes SET teacher = ? WHERE lower(teacher) = ?', ('anonymous', key))
    return out


# ---- shares and challenges: how codes spread ----
def log_share(code, kind):
    with _lock, _conn() as c:
        c.execute('INSERT INTO dance_shares (code, kind, t) VALUES (?, ?, ?)', (str(code), kind, _time.time()))


def share_count(code, kind='challenge_start'):
    with _lock, _conn() as c:
        return c.execute('SELECT COUNT(*) FROM dance_shares WHERE code = ? AND kind = ?', (str(code), kind)).fetchone()[0]


def funnel():
    with _lock, _conn() as c:
        cnt = {k: c.execute('SELECT COUNT(*) FROM dance_shares WHERE kind = ?', (k,)).fetchone()[0] for k in ('issued', 'challenge_start', 'challenge_complete')}
        codes_with_starts = c.execute("SELECT COUNT(DISTINCT code) FROM dance_shares WHERE kind = 'challenge_start'").fetchone()[0]
    issued = cnt['issued']
    return dict(codes_issued=issued, challenge_starts=cnt['challenge_start'], challenge_completes=cnt['challenge_complete'],
                codes_with_starts=codes_with_starts, start_rate=round(codes_with_starts / issued, 3) if issued else 0.0)


def crew_board(code, limit=50):
    """The original result plus every result that challenged it, best first (score100)."""
    orig = get_result(code)
    if orig is None: return None
    with _lock, _conn() as c:
        rows = c.execute('SELECT code, name, key, mode, total, score100, created FROM dance_results WHERE challenge = ? ORDER BY score100 DESC LIMIT ?',
                         (str(code), int(limit))).fetchall()
    own = orig.get('score100') if orig.get('score100') is not None else orig['total']
    entries = [dict(code=orig['code'], name=orig['name'], score100=round(float(own), 1), owner=True)]
    entries += [dict(code=r['code'], name=r['name'], score100=round(float(r['score100'] or 0), 1), owner=False) for r in rows]
    entries.sort(key=lambda e: -e['score100'])
    for i, e in enumerate(entries): e['rank'] = i + 1
    return dict(code=str(code), key=orig['key'], mode=orig['mode'], owner=orig['name'], entries=entries, n=len(entries))


# ---- metrics over session summaries ----
def metrics(limit=2000):
    import statistics
    with _lock, _conn() as c:
        rows = c.execute("SELECT body FROM events WHERE type = 'session_summary' ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    s = [json.loads(r['body']) for r in rows]
    def med(k):
        v = [x[k] for x in s if x.get(k) is not None]; return round(statistics.median(v), 3) if v else None
    calls = sum(int(x.get('tool_calls') or 0) for x in s); errs = sum(int(x.get('tool_errors') or 0) for x in s)
    return dict(sessions=len(s), completion_rate=round(sum(1 for x in s if x.get('completed')) / len(s), 3) if s else None,
                median_tracking_rate=med('tracking_rate'), median_fullbody_rate=med('fullbody_rate'), median_cue_latency_ms=med('median_cue_latency_ms'),
                p90_cue_latency_ms=med('p90_cue_latency_ms'), tool_calls=calls, tool_error_rate=round(errs / calls, 3) if calls else None,
                mean_score=med('mean_score'), median_time_to_start_s=med('time_to_start_s'), funnel=funnel())


def post_result(b):
    """Store a result dict {name, key, mode, total, rounds, score100?, challenge?}; with a valid challenge code, compare like for like
    on score100 and rank on that code's crew board. Shared by the HTTP API and the in-process backend."""
    ch = b.get('challenge'); opp = get_result(ch) if ch else None
    code = add_result(b.get('name'), b.get('key', ''), b.get('mode', 'learn'), float(b.get('total', 0)), b.get('rounds', []),
                      score100=b.get('score100'), challenge=ch if opp is not None else None, issue_code=b.get('issue_code', True) is not False)
    if code: s100 = get_result(code).get('score100')
    else: s100 = b.get('score100') if b.get('score100') is not None else (float(b.get('total', 0)) / 3.0 if b.get('mode') in ('battle', 'squad_battle') else float(b.get('total', 0)))
    out = dict(code=code, total=float(b.get('total', 0)), score100=s100, name=b.get('name') or 'anonymous')
    if code and b.get('mode') == 'battle':                       # only this caller can attach a ghost to the code
        import secrets
        tok = secrets.token_urlsafe(16)
        with _lock, _conn() as c: c.execute('INSERT OR REPLACE INTO dance_ghost_tokens (code, token, created) VALUES (?, ?, ?)', (code, tok, _time.time()))
        out['ghost_token'] = tok
    if ch:
        if opp is not None:
            theirs = opp.get('score100') if opp.get('score100') is not None else opp['total']
            out['versus'] = dict(code=ch, name=opp['name'], total=opp['total'], score100=theirs, key=opp['key'])
            out['winner'] = (out['name'] if out['score100'] > theirs else opp['name'] if theirs > out['score100'] else 'tie')
            board = crew_board(ch) if code else None
            if board: out['board'] = dict(rank=next((e['rank'] for e in board['entries'] if e['code'] == code), None), n=board['n'], owner=board['owner'])
        else:
            out['versus'] = None
    return out


# ---- ghosts: a battle winner's dance (body points only), for friends to battle ----
GHOST_DAYS = 30


GHOST_MAX_BYTES = 400_000


def valid_ghost(data, bpm, per_beat):
    """A ghost is 1-2 rounds of (n, 14) angles, (n, 14) visibility, (n,) energy and (n, 33, 2) skeleton, n = 16..64 samples (4 per
    count), finite numbers, a sane tempo. Anything else is refused (it is loaded and drawn on someone else's call)."""
    import math
    try:
        if int(per_beat) != 4 or not (40.0 <= float(bpm) <= 220.0): return False
        if not isinstance(data, list) or not 1 <= len(data) <= 2: return False
        n0 = None
        for r in data:
            F, M, E, S = r['F'], r['M'], r['E'], r['S']; n = len(F)
            if not 16 <= n <= 64 or n % 4 or (n0 is not None and n != n0): return False
            n0 = n
            if len(M) != n or len(E) != n or len(S) != n: return False
            if any(len(row) != 14 for row in F) or any(len(row) != 14 for row in M): return False
            if any(len(p) != 33 or any(len(xy) != 2 for xy in p) for p in S): return False
            for v in [x for row in F for x in row] + list(E) + [c for p in S for xy in p for c in xy]:
                if not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 1e4: return False
        return len(json.dumps(data)) <= GHOST_MAX_BYTES
    except Exception:
        return False


def save_ghost(code, token, src_key, bpm, per_beat, data, total, score100, difficulty=None):
    """Attach a battle winner's ghost to their code. Needs the code's one-time token (only the winner's call has it); never overwrites
    an existing ghost; the name is the result's own. data: team.dance.ghost.pack(rounds), validated. Ghosts expire after GHOST_DAYS."""
    code = str(code); now = _time.time()
    if not valid_ghost(data, bpm, per_beat): return dict(ok=False, error='invalid ghost')
    res = get_result(code)
    if res is None: return dict(ok=False, error='no such code')
    with _lock, _conn() as c:
        row = c.execute('SELECT token FROM dance_ghost_tokens WHERE code = ?', (code,)).fetchone()
        if row is None or not token or row['token'] != str(token): return dict(ok=False, error='not allowed')
        c.execute('DELETE FROM dance_ghosts WHERE created < ?', (now - GHOST_DAYS * 86400,))
        if c.execute('SELECT 1 FROM dance_ghosts WHERE code = ?', (code,)).fetchone(): return dict(ok=False, error='ghost already saved')
        c.execute('INSERT INTO dance_ghosts (code, name, src_key, bpm, per_beat, data, total, score100, difficulty, created) '
                  'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (code, res.get('name') or 'anonymous', src_key, float(bpm), int(per_beat), json.dumps(data),
                                                            float(total), float(score100), difficulty, now))
        c.execute('DELETE FROM dance_ghost_tokens WHERE code = ?', (code,))
    return dict(ok=True, code=code)


def get_ghost(code, with_data=True):
    with _lock, _conn() as c:
        r = c.execute('SELECT * FROM dance_ghosts WHERE code = ? AND created >= ?', (str(code), _time.time() - GHOST_DAYS * 86400)).fetchone()
    if r is None: return None
    d = dict(r); d.pop('name', None)                           # body points only: the name comes with the result, not the ghost
    if with_data: d['data'] = json.loads(d['data'] or '[]')
    else: d.pop('data', None)
    return d
