"""The team backend's dance endpoints (codes, duels, chains, leaderboard) with FastAPI's test client on a temporary SQLite file."""
import importlib.util, os, sys, tempfile
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); BACKEND = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'backend')


@pytest.fixture(scope='module')
def client():
    os.environ['TEAM_DB'] = os.path.join(tempfile.gettempdir(), 'team_test_dance.db')
    if os.path.exists(os.environ['TEAM_DB']): os.remove(os.environ['TEAM_DB'])
    sys.path.insert(0, BACKEND)
    for m in ('db', 'app'):
        sys.modules.pop(m, None)
    spec = importlib.util.spec_from_file_location('app', os.path.join(BACKEND, 'app.py')); app_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(app_mod)
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app)


def test_result_code_and_duel(client):
    a = client.post('/dance/result', json=dict(name='Maya', key='wave', mode='battle', total=210, rounds=[{'score': 70}] * 3)).json()
    assert len(a['code']) == 4 and a['name'] == 'Maya'
    b = client.post('/dance/result', json=dict(name='Dev', key='wave', mode='battle', total=180, rounds=[], challenge=a['code'])).json()
    assert b['versus']['name'] == 'Maya' and b['winner'] == 'Maya'
    assert client.get(f"/dance/result/{a['code']}").json()['total'] == 210
    assert client.get('/dance/result/0000').json().get('error')
    board = client.get('/dance/leaderboard?key=wave').json()
    assert board[0]['name'] == 'Maya' and board[1]['name'] == 'Dev'
    assert 'Maya' in client.get('/dance').text


def test_chain(client):
    c = client.post('/dance/chain', json=dict(key='wave', link=dict(name='A', score=100))).json()
    assert len(c['code']) == 4 and c['link_no'] == 1
    n = client.post(f"/dance/chain/{c['code']}/link", json=dict(link=dict(name='B', fidelity_prev=72))).json()
    assert n['link_no'] == 2
    chain = client.get(f"/dance/chain/{c['code']}").json()
    assert [l['name'] for l in chain['links']] == ['A', 'B']
    assert client.post('/dance/chain/0000/link', json=dict(link={})).json().get('error')


def test_class(client):
    key = dict(name='class_maya', bpm=100.0, per_beat=4, phrases=[dict(start_beat=0, beats=8), dict(start_beat=8, beats=8)], features=[], mask=[], energy=[], skeleton=[])
    c = client.post('/dance/class', json=dict(teacher='Maya', key=key)).json()
    assert len(c['code']) == 4
    got = client.get(f"/dance/class/{c['code']}").json()
    assert got['teacher'] == 'Maya' and got['n_phrases'] == 2 and got['key']['name'] == 'class_maya'
    assert 'key' not in client.get(f"/dance/class/{c['code']}?key=0").json()
    rep0 = client.get(f"/dance/class/{c['code']}/report").json()
    assert rep0['n_students'] == 0 and 'Share the code' in rep0['suggestion']
    v_good = ['green'] * 8; v_bad = ['green', 'green', 'red', 'red', 'amber', 'green', 'green', 'red']
    a = client.post(f"/dance/class/{c['code']}/student", json=dict(name='Dev', phrases=[dict(phrase=0, score=85, scored=True, verdicts=v_good, per_group=dict(arms=8, legs=10, torso=4)),
                                                                                       dict(phrase=1, score=40, scored=True, verdicts=v_bad, per_group=dict(arms=30, legs=12, torso=6))])).json()
    assert a['rank'] == 1 and a['n_students'] == 1 and a['total'] == 62.5
    b = client.post(f"/dance/class/{c['code']}/student", json=dict(name='Ana', phrases=[dict(phrase=0, score=90, scored=True, verdicts=v_good, per_group=dict(arms=6, legs=8, torso=3)),
                                                                                       dict(phrase=1, score=55, scored=True, verdicts=v_bad, per_group=dict(arms=25, legs=10, torso=5))])).json()
    assert b['rank'] == 1 and b['n_students'] == 2 and b['class_mean'] == 67.5
    # a returning student replaces their row rather than adding one
    again = client.post(f"/dance/class/{c['code']}/student", json=dict(name='dev', phrases=[dict(phrase=0, score=95, scored=True, verdicts=v_good, per_group={})])).json()
    assert again['n_students'] == 2 and again['rank'] == 1
    rep = client.get(f"/dance/class/{c['code']}/report").json()
    assert rep['n_students'] == 2 and rep['n_phrases'] == 2
    assert rep['weakest_phrase'] == 1 and rep['weakest_group'] == 'arms' and rep['weakest_beat'] in (2, 3, 7)
    assert [r['name'] for r in rep['heatmap']] == ['Ana', 'dev'] and rep['heatmap'][1]['scores'] == [95, None]
    assert 'Phrase 2' in rep['suggestion'] and 'arms' in rep['suggestion']
    page = client.get(f"/dance/class/{c['code']}/page").text
    assert 'Ana' in page and 'phrase 2' in page and 'suggestion' in page
    assert client.get('/dance/class/0000').json().get('error') and client.post('/dance/class/0000/student', json=dict(name='x', phrases=[])).json().get('error')


def test_profiles_award_streak_and_prefs(client):
    a = client.post('/dance/profile/Zoe/award', json=dict(xp=120, style='hiphop', day='2026-10-10', session=dict(rounds=[dict(scored=True, score=93, max_combo=8)]))).json()
    assert a['leveled_up'] and a['level_after'] == 1 and a['streak'] == 1 and {x['id'] for x in a['new_achievements']} >= {'first_dance', 's_rank', 'perfect_8'}
    assert 'intermediate' in a['newly_unlocked']
    b = client.post('/dance/profile/zoe/award', json=dict(xp=10, style='pop', day='2026-10-11', session=dict(rounds=[dict(scored=True, score=60)]))).json()
    assert b['streak'] == 2 and b['xp_after'] == 130 and not b['new_achievements']
    client.post('/dance/profile/Zoe/prefs', json=dict(persona='chill', difficulty='intermediate'))
    p = client.get('/dance/profile/ZOE').json()
    assert p['xp'] == 130 and p['level_name'] == 'Groover' and p['prefs']['persona'] == 'chill' and set(p['styles']) == {'hiphop', 'pop'}
    assert client.get('/dance/profile/Nobody').json().get('error')


def test_duel_like_for_like_board_and_funnel(client):
    f0 = client.get('/dance/funnel').json()
    a = client.post('/dance/result', json=dict(name='Ana', key='gen_pop_beginner_7', mode='learn', total=74, rounds=[])).json()
    client.post(f"/dance/share/{a['code']}/start")
    b = client.post('/dance/result', json=dict(name='Ben', key='gen_pop_beginner_7', mode='battle', total=240, rounds=[], challenge=a['code'])).json()
    assert b['score100'] == 80 and b['winner'] == 'Ben' and b['board']['rank'] == 1 and b['board']['n'] == 2 and b['board']['owner'] == 'Ana'
    c = client.post('/dance/result', json=dict(name='Cal', key='gen_pop_beginner_7', mode='learn', total=50, rounds=[], challenge=a['code'])).json()
    board = client.get(f"/dance/board/{a['code']}").json()
    assert [e['name'] for e in board['entries']] == ['Ben', 'Ana', 'Cal'] and c['board']['rank'] == 3
    assert 'Ben' in client.get(f"/dance/board/{a['code']}/page").text
    f1 = client.get('/dance/funnel').json()
    assert f1['codes_issued'] - f0['codes_issued'] == 3 and f1['challenge_starts'] - f0['challenge_starts'] == 1 and f1['challenge_completes'] - f0['challenge_completes'] == 2


def test_forget_me(client):
    client.post('/dance/profile/Gus/award', json=dict(xp=50))
    client.post('/dance/result', json=dict(name='Gus', key='k', mode='learn', total=60, rounds=[]))
    ch = client.post('/dance/chain', json=dict(key='wave', link=dict(name='Gus', score=100))).json()
    out = client.delete('/dance/profile/gus').json()['deleted']
    assert out['profile'] == 1 and out['results'] >= 1 and out['chains'] >= 1
    assert client.get('/dance/profile/Gus').json().get('error') and client.get('/dance/history?name=Gus').json() == []
    assert client.get(f"/dance/chain/{ch['code']}").json()['links'][0]['name'] == 'anonymous'


def test_result_page_and_preview_card(client, monkeypatch):
    import io
    from PIL import Image
    monkeypatch.setenv('DANCE_PUBLIC_URL', 'https://b.example.app'); monkeypatch.setenv('DANCE_CALL_HANDLE', 'brodie@example.com')
    a = client.post('/dance/result', json=dict(name='Maya', key='k', mode='learn', total=94, rounds=[])).json()
    v0 = client.get('/dance/funnel').json()['page_views']
    page = client.get(f"/r/{a['code']}")
    assert page.status_code == 200
    t = page.text
    assert f"og:image' content='https://b.example.app/r/{a['code']}/og.png'" in t and "twitter:card' content='summary_large_image'" in t
    assert 'Maya scored 94 (S)' in t and "href='facetime:brodie@example.com'" in t and f"battle {a['code']}" in t
    assert client.get('/dance/funnel').json()['page_views'] == v0 + 1
    png = client.get(f"/r/{a['code']}/og.png")
    assert png.headers['content-type'] == 'image/png' and Image.open(io.BytesIO(png.content)).size == (1200, 630)
    assert client.get('/r/0000').status_code == 404 and client.get('/r/abcd/og.png').status_code == 404
    assert client.get('/dance/funnel').json()['page_views'] == v0 + 1          # unknown codes and the image fetch don't count


def test_metrics(client):
    for done in (True, False):
        client.post('/events', json=dict(type='session_summary', t_wall=1.0, completed=done, tracking_rate=0.95, median_cue_latency_ms=400.0, tool_calls=4, tool_errors=1))
    m = client.get('/metrics').json()
    assert m['sessions'] >= 2 and m['median_tracking_rate'] == 0.95 and m['tool_error_rate'] == 0.25 and 'funnel' in m


def test_health_status_tracks_database(client, monkeypatch):
    import db
    monkeypatch.setattr(db, 'ping', lambda: True)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == dict(ok=True, db='sqlite', db_ok=True)
    monkeypatch.setattr(db, 'ping', lambda: False)
    response = client.get('/health')
    assert response.status_code == 503
    assert response.json() == dict(ok=False, db='sqlite', db_ok=False)
