"""Team backend: receives Brodie events from team/skill/handlers.py and keeps state in SQLite.

Run (repo root, venv active):  python -m uvicorn app:app --app-dir team/backend --port 8787
Endpoints:
  GET  /health              liveness
  POST /events              one event (the dance session posts its session_summary here)
  GET  /events/recent       the last N events, newest first
  GET  /sessions/last       reps per exercise for the most recent call (events grouped by a gap of SESSION_GAP_S)
  Dance: POST /dance/result (duels on score100, crew boards), GET /dance/result/{code}, /dance/leaderboard, /dance/history,
         /dance/profile/{name} (GET; POST .../award, .../prefs; DELETE = forget me), POST /dance/share/{code}/start,
         GET /dance/funnel, /dance/board/{code}[/page], chains, classes, GET /dance (board page), GET /metrics
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import db

SESSION_GAP_S = 120          # events more than this far apart belong to different calls
app = FastAPI(title='team backend')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])   # team/web pages call it from anywhere
db.init()


@app.get('/health')
def health():
    return {'ok': True}


@app.post('/events')
async def ingest(request: Request):
    ev = await request.json()
    db.add_event(ev)
    return {'ok': True}


@app.get('/events/recent')
def recent(limit: int = 50):
    return db.recent_events(limit)


# ---- dance: codes for async duels and telephone chains, a leaderboard, a results page ----
@app.post('/dance/result')
async def dance_result(request: Request):
    """{name, key, mode, total, rounds, challenge?} -> {code, versus?}. If `challenge` is a code, the stored result is returned as
    `versus` with a winner, so the caller hears the duel verdict on the same call."""
    b = await request.json()
    return result_with_versus(b)


def result_with_versus(b):
    return db.post_result(b)


# ---- profiles, deletion, shares, crew boards, metrics ----
@app.get('/dance/profile/{name}')
def dance_profile(name: str):
    return db.get_profile(name) or {'error': 'no such profile', 'name': name}


@app.post('/dance/profile/{name}/award')
async def dance_award(name: str, request: Request):
    b = await request.json()
    return db.award(name, int(b.get('xp', 0)), style=b.get('style'), achievements=b.get('achievements') or (), day=b.get('day'), session=b.get('session'))


@app.post('/dance/profile/{name}/prefs')
async def dance_prefs(name: str, request: Request):
    return db.set_prefs(name, await request.json())


@app.delete('/dance/profile/{name}')
def dance_forget(name: str):
    return dict(deleted=db.delete_user(name), name=name)


@app.post('/dance/share/{code}/start')
def dance_share_start(code: str):
    db.log_share(code, 'challenge_start')
    return dict(code=code, starts=db.share_count(code))


@app.get('/dance/share/{code}')
def dance_share_get(code: str):
    return dict(code=code, starts=db.share_count(code), completes=db.share_count(code, 'challenge_complete'))


@app.get('/dance/funnel')
def dance_funnel():
    return db.funnel()


@app.get('/dance/board/{code}')
def dance_board(code: str):
    return db.crew_board(code) or {'error': 'no such code'}


@app.get('/dance/board/{code}/page', response_class=HTMLResponse)
def dance_board_page(code: str):
    b = db.crew_board(code)
    if not b: return HTMLResponse(_page('No such code', '<p>That code does not exist.</p>'), status_code=404)
    rows = ''.join(f"<tr class='{'me' if e['owner'] else ''}'><td>{e['rank']}</td><td>{_esc(e['name'])}{' (started it)' if e['owner'] else ''}</td>"
                   f"<td><b>{e['score100']:.0f}</b></td><td><span class='g g{_grade(e['score100'])}'>{_grade(e['score100'])}</span></td></tr>" for e in b['entries'])
    body = (f"<h1>{_esc(b['owner'])}'s challenge <code>{code}</code></h1><p class='sub'>{b['n']} dancer(s) on {_esc(str(b['key']).replace('_', ' '))}. "
            f"FaceTime Brodie and say <b>battle {code}</b> to get on this board.</p><table><tr><th>#</th><th>dancer</th><th>score</th><th>grade</th></tr>{rows}</table>")
    return _page(f'Board {code}', body)


@app.get('/metrics')
def metrics():
    return db.metrics()


@app.get('/dance/result/{code}')
def dance_get_result(code: str):
    r = db.get_result(code)
    return r or {'error': 'no such code'}


@app.post('/dance/ghost/{code}')
async def dance_save_ghost(code: str, request: Request):
    """A battle winner's dance as body points (they said yes to saving it): {token, src_key, bpm, per_beat, data, total, score100}.
    The token is the one /dance/result returned to the winner's call; the shape is validated; an existing ghost is never replaced."""
    b = await request.json()
    return db.save_ghost(code, b.get('token'), b.get('src_key'), b.get('bpm', 100), b.get('per_beat', 4), b.get('data') or [], b.get('total', 0),
                         b.get('score100', 0), b.get('difficulty'))


@app.get('/dance/ghost/{code}')
def dance_get_ghost(code: str):
    return db.get_ghost(code) or {'error': 'no such ghost'}


@app.get('/dance/leaderboard')
def dance_leaderboard(key: str = None, limit: int = 20, mode: str = None, since: float = None):
    return db.leaderboard(key, limit, mode=mode, since=since)


@app.get('/dance/code/{code}')
def dance_resolve_code(code: str):
    """What a spoken code is: a battle result, a telephone chain or a class."""
    return db.resolve_code(code)


@app.get('/dance/daily_played')
def dance_daily_played(name: str, key: str):
    return dict(played=db.daily_played(name, key))


@app.get('/dance/history')
def dance_history(name: str, limit: int = 10):
    return db.history(name, limit)


@app.post('/dance/chain')
async def dance_chain_start(request: Request):
    b = await request.json()
    return dict(code=db.start_chain(b.get('key', ''), b.get('link', {})), link_no=1)


@app.post('/dance/chain/{code}/link')
async def dance_chain_link(code: str, request: Request):
    b = await request.json(); n = db.add_link(code, b.get('link', {}))
    return dict(code=code, link_no=n) if n else {'error': 'no such chain'}


@app.get('/dance/chain/{code}')
def dance_chain_get(code: str):
    return db.get_chain(code) or {'error': 'no such chain'}


@app.get('/dance/chain/{code}/page', response_class=HTMLResponse)
def dance_chain_page(code: str):
    """The telephone drift: one row per link with fidelity to the previous dancer and to the original."""
    ch = db.get_chain(code)
    if not ch: return HTMLResponse('<p>no such chain</p>', status_code=404)
    def num(v): return '' if v is None else f'{float(v):.0f}'
    rows = ''.join(f"<tr><td>{i + 1}</td><td>{l.get('name', '?')}</td><td>{num(l.get('fidelity_prev'))}</td><td>{num(l.get('fidelity_origin'))}</td></tr>"
                   for i, l in enumerate(ch['links']))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Chain {code}</title>
<style>body{{font-family:system-ui;background:#0b0b0b;color:#f2f2f2;max-width:720px;margin:40px auto;padding:0 16px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #333;text-align:left}}code{{color:#D9FF3D}}</style></head>
<body><h1>Dance telephone <code>{code}</code></h1><p>{len(ch['links'])} links. Fidelity is 0 to 100 against the previous dancer and against the original move.</p>
<table><tr><th>link</th><th>name</th><th>vs previous</th><th>vs original</th></tr>{rows}</table>
<p>To add a link: FaceTime the number, say "telephone" and the code <code>{code}</code>.</p></body></html>"""


@app.post('/dance/class')
async def dance_class_start(request: Request):
    """{teacher, key} -> {code}. The key is the teacher's recorded phrases (full key dict, skeleton included)."""
    b = await request.json()
    return dict(code=db.start_class(b.get('teacher'), b.get('key') or {}))


@app.get('/dance/class/{code}')
def dance_class_get(code: str, key: int = 1):
    return db.get_class(code, with_key=bool(key)) or {'error': 'no such class'}


@app.post('/dance/class/{code}/student')
async def dance_class_student(code: str, request: Request):
    """{name, phrases: [{phrase, score, scored, verdicts, per_group, weak_beats}]} -> {rank, n_students, class_mean, total}."""
    b = await request.json()
    return db.add_student(code, b) or {'error': 'no such class'}


@app.get('/dance/class/{code}/report')
def dance_class_report(code: str):
    return db.class_report(code) or {'error': 'no such class'}


@app.get('/dance/class/{code}/page', response_class=HTMLResponse)
def dance_class_page(code: str):
    """The teacher's heatmap: students by phrase (colour by score), class health per count under each phrase, the suggestion."""
    rep = db.class_report(code)
    if not rep: return HTMLResponse('<p>no such class</p>', status_code=404)
    def cell(v):
        if v is None: return "<td class='none'>-</td>"
        cls = 'g' if v >= 80 else 'a' if v >= 50 else 'r'
        return f"<td class='{cls}'>{v:.0f}</td>"
    def strip(bh):
        return ''.join(f"<i style='opacity:{0.25 + 0.75 * h:.2f}' title='count {b + 1}: {h:.2f}'></i>" for b, h in enumerate(bh))
    head = ''.join(f"<th>phrase {i + 1}<div class='strip'>{strip(bh)}</div><small>mean {'' if m is None else f'{m:.0f}'}</small></th>"
                   for i, (bh, m) in enumerate(zip(rep['beat_health'], rep['phrase_mean'])))
    rows = ''.join(f"<tr><td>{r['name']}</td>{''.join(cell(v) for v in r['scores'])}<td><b>{'' if r['total'] is None else f'{float(r['total']):.0f}'}</b></td></tr>" for r in rep['heatmap'])
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Class {code}</title>
<style>body{{font-family:system-ui;background:#0b0b0b;color:#f2f2f2;max-width:760px;margin:40px auto;padding:0 16px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #333;text-align:left;vertical-align:top}}
code{{color:#D9FF3D}}td.g{{background:#1f5a2a}}td.a{{background:#6b4a00}}td.r{{background:#5a1f1f}}td.none{{color:#666}}.strip{{display:flex;gap:2px;margin:4px 0}}.strip i{{display:block;width:14px;height:8px;background:#D9FF3D;border-radius:2px}}small{{color:#9a9da4}}</style></head>
<body><h1>Class <code>{code}</code> by {rep['teacher']}</h1><p>{rep['n_students']} student(s), {rep['n_phrases']} phrase(s). Cells are scores out of 100; the strip under each phrase is class health per count (bright = green).</p>
<table><tr><th>student</th>{head}<th>total</th></tr>{rows or '<tr><td colspan=9>nobody yet</td></tr>'}</table>
<p><b>Brodie's suggestion:</b> {rep['suggestion']}</p>{('<p>Not attempted yet: ' + ', '.join(rep['not_attempted']) + '</p>') if rep['not_attempted'] else ''}
<p>Students: FaceTime the number, say "class" and the code <code>{code}</code>. Teacher: call again, say your name and "report".</p></body></html>"""


def _esc(s):
    import html
    return html.escape(str(s if s is not None else ''))


def _grade(score):
    s = float(score or 0)
    return 'S' if s >= 92 else 'A' if s >= 80 else 'B' if s >= 65 else 'C' if s >= 50 else 'D'


def _page(title, body):
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{_esc(title)}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Space+Grotesk:wght@500;700&display=swap');
:root{{--ink:#0A0A0B;--paper:#F4F4F0;--volt:#D9FF3D;--pink:#FF3DA8;--cyan:#3DE0FF;--mute:#9a9da4;--line:#26262b}}
*{{box-sizing:border-box}}body{{font-family:'Space Grotesk',system-ui,sans-serif;background:var(--ink);color:var(--paper);max-width:760px;margin:0 auto;padding:32px 18px 60px;font-size:17px;line-height:1.45}}
h1{{font-family:'Archivo Black',system-ui,sans-serif;font-size:clamp(28px,6vw,44px);line-height:1.05;margin:0 0 8px;letter-spacing:-.5px}}
.sub{{color:var(--mute);margin:0 0 22px}}code{{color:var(--volt);font-family:'Archivo Black',monospace}}
table{{width:100%;border-collapse:collapse}}td,th{{padding:12px 8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--mute);font-weight:500;font-size:14px;text-transform:uppercase;letter-spacing:.06em}}
tr.me td{{color:var(--volt)}}.g{{display:inline-block;min-width:30px;text-align:center;padding:2px 8px;border-radius:999px;font-family:'Archivo Black';color:var(--ink)}}
.gS{{background:var(--volt)}}.gA{{background:var(--cyan)}}.gB{{background:#c9c9c4}}.gC{{background:#ffb13d}}.gD{{background:#ff4d5e}}
a{{color:var(--cyan)}}.pill{{display:inline-block;background:var(--volt);color:var(--ink);font-family:'Archivo Black';padding:4px 12px;border-radius:999px;font-size:14px}}
</style></head><body>{body}</body></html>"""


@app.get('/dance', response_class=HTMLResponse)
def dance_page(key: str = None):
    rows = db.leaderboard(key, 50)
    daily = bool(key and key.startswith('gen_daily_'))
    trs = ''.join(f"<tr><td>{i + 1}</td><td>{_esc(r['name'])}</td><td>{_esc(str(r['key']).replace('_', ' '))}</td><td>{_esc(r['mode'])}</td>"
                  f"<td><b>{r['total']:.0f}</b></td><td><a href='/dance/board/{r['code']}/page'><code>{r['code']}</code></a></td></tr>" for i, r in enumerate(rows))
    head = "<span class='pill'>DAILY CHALLENGE</span><h1>Today's Daily</h1>" if daily else "<h1>Brodie dance board</h1>"
    body = (f"{head}<p class='sub'>FaceTime Brodie and say a four-digit code to battle the person who owns it. Tap a code to see its crew board.</p>"
            f"<table><tr><th>#</th><th>name</th><th>routine</th><th>mode</th><th>total</th><th>code</th></tr>{trs or '<tr><td colspan=6>nobody yet</td></tr>'}</table>")
    return _page('Brodie dance', body)


@app.get('/sessions/last')
def last_session():
    evs = db.recent_events(5000)                    # newest first
    if not evs: return {'events': 0}
    call = [evs[0]]
    for ev in evs[1:]:
        if (call[-1].get('t_wall') or 0) - (ev.get('t_wall') or 0) > SESSION_GAP_S: break
        call.append(ev)
    reps = {}
    for ev in call:
        if ev.get('type') == 'rep':
            reps[ev.get('exercise')] = max(reps.get(ev.get('exercise'), 0), ev.get('n') or 0)
    faults = sorted({ev.get('label') for ev in call if ev.get('type') == 'fault'})
    return {'events': len(call), 'started': call[-1].get('t_wall'), 'ended': call[0].get('t_wall'), 'reps': reps, 'faults': faults}
