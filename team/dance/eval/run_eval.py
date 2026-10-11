"""End-to-end eval: scripted sessions through the simulator (real pose, real engine, in-process backend), in parallel, with a
check per scenario and the numbers the criteria ask for.

  python -m team.dance.eval.run_eval [--only learn,safety] [--jobs 4] [--clip data/tmp/mac_handoff/segments/run3_jacks.mp4]

Reports: sessions completed without a crash (target >= 95%), tracking rate on full-body clips (>= 90%), median mid-phrase cue
latency (<= 1 s), time to first dance (<= 30 s), tool/backend errors, and per-scenario checks (adaptation, pause/resume, safety stop,
forget me, squad, walk-in lock, challenge duel and crew board). Writes data/tmp/eval_report.json and data/tmp/cue_review.csv
(every cue with its time and context, for the human relevance review).
"""
import argparse, csv, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = sys.executable
OUT = os.path.join(ROOT, 'data', 'tmp', 'eval')
DB = os.path.join(OUT, 'eval_team.db')
FAST = dict(DANCE_MAX_PHRASES='1', DANCE_MAX_S='400')


def scenarios(clip):
    base = ['--simulate', clip, '--pose', 'lite']
    flow = ['--say', '16:yes', '--say', '30:yes', '--say', '38:yes', '--say', '46:ready', '--say', '54:ready']      # offer yes, space yes, ready
    return [
        dict(name='long_chat', args=base + ['--mode', 'learn', '--lobby', '--say', "2:Hi I'm Maya", '--say', '6:where do I start?', '--say', "10:what's a battle?",
                                            '--say', "14:I'm not ready", '--say', "18:ok let's go", '--say', '22:maybe dance later', '--say', '26:I like K-pop'],
             env=dict(DANCE_MAX_S='70'), gestures=['8:ready:2'], checks=['completed', 'no_start', 'no_offer', 'proactive']),
        dict(name='proactive', args=base + ['--mode', 'learn', '--lobby', '--say', '19:yes', '--say', '30:yes', '--say', '38:yes', '--say', '46:ready']
             + sum((['--say', f'{t}:ready'] for t in range(58, 200, 9)), []),
             env=dict(DANCE_MAX_PHRASES='1', DANCE_MAX_S='300'), checks=['completed', 'proactive', 'confirmed_before_dance']),
        dict(name='question_no_offer', args=base + ['--mode', 'learn', '--lobby', '--say', '3:Today I want to dance some hip hop. Is that possible?',
                                                    '--say', '9:what styles do you have?'], env=dict(DANCE_MAX_S='40'), checks=['completed', 'no_offer', 'no_start']),
        dict(name='offer_needs_yes', args=base + ['--mode', 'learn', '--lobby', '--say', '3:teach me hip hop', '--say', '12:hmm what is that like?',
                                                  '--say', '20:not yet'], env=dict(DANCE_MAX_S='60'), checks=['completed', 'offered', 'no_start', 'no_setup']),
        dict(name='learn_lobby', args=base + ['--mode', 'learn', '--lobby', '--say', "2:I'm Maya, teach me hip hop easy", '--say', '5:chill coach'] + flow
             + sum((['--say', f"{t}:{'next' if (t // 8) % 3 == 2 else 'ready'}"] for t in range(62, 560, 8)), []),
             env=dict(DANCE_MAX_PHRASES='1', DANCE_MAX_S='600'), checks=['completed', 'persona_chill', 'offered', 'confirmed_before_dance', 'ladder', 'award', 'finish_pose']),
        dict(name='patient_silence', args=base + ['--mode', 'learn', '--lobby', '--say', '2:teach me hip hop', '--say', '12:yes', '--say', '24:yes', '--say', '32:ready'],
             env=dict(DANCE_MAX_S='160'), checks=['completed', 'demo_first', 'waits_after_demo']),
        dict(name='patient_answers', args=base + ['--mode', 'learn', '--lobby', '--say', '2:teach me hip hop', '--say', '12:yes', '--say', '24:yes']
             + sum((['--say', f'{t}:ready'] for t in range(32, 300, 9)), []),
             env=dict(DANCE_MAX_PHRASES='1', DANCE_MAX_S='300'), checks=['completed', 'demo_first', 'teaching_order', 'slow_start', 'speed_by_consent']),
        dict(name='setup_wait', args=base + ['--mode', 'learn', '--lobby', '--crop-bottom', '0.62', '--say', '3:teach me pop', '--say', '14:yes'],
             env=dict(DANCE_MAX_S='90'), checks=['completed', 'setup_blocked', 'no_repeat_lines']),
        dict(name='setup_listens', args=base + ['--mode', 'learn', '--lobby', '--crop-bottom', '0.35', '--say', '3:teach me pop', '--say', '14:yes',
                                                '--say', "30:I can't step back any further", '--say', '44:yes', '--say', '52:ready'],
             env=dict(DANCE_MAX_PHRASES='1', DANCE_MAX_S='300'), checks=['completed', 'upper_body_used']),
        dict(name='desk_tracking', args=base + ['--mode', 'battle', '--crop-bottom', '0.45'], env=FAST, checks=['completed', 'tracked']),
        dict(name='battle', args=base + ['--mode', 'battle', '--name', 'Dev'], env=FAST, checks=['completed', 'battle_three_rounds', 'code_posted', 'award']),
        dict(name='safety', args=base + ['--mode', 'learn', '--say', '18:ow my knee hurts'], env=FAST, checks=['completed', 'safety_stop']),
        dict(name='pause_resume', args=base + ['--mode', 'battle', '--gesture', '16:pause:2', '--gesture', '24:ready:2'], env=FAST,
             checks=['completed', 'paused_resumed']),
        dict(name='forget', args=base + ['--mode', 'learn', '--lobby', '--say', "2:I'm Gus", '--say', '4:forget me', '--say', '7:forget me'],
             env=dict(DANCE_MAX_S='40'), checks=['completed', 'forget', 'no_start']),
        dict(name='walk_in', args=base + ['--mode', 'battle', '--background'], env=FAST, checks=['completed', 'primary_locked_once']),
        dict(name='squad', args=base + ['--mode', 'learn', '--players', '2', '--lobby', '--say', "3:let's do a squad battle, hip hop"] + flow,
             env=dict(DANCE_MAX_PHRASES='1', DANCE_MAX_S='300'), checks=['completed', 'squad_two', 'squad_rounds']),
        dict(name='daily', args=base + ['--mode', 'daily', '--name', 'Ana'], env=FAST, checks=['completed', 'daily_key']),
    ]


def run_one(sc, clip, stage_env=None):
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, f"{sc['name']}.json")
    env = dict(os.environ, BACKEND_URL='fake', TEAM_DB=DB, DANCE_SESSION_LOG=log, PYTHONUTF8='1', **sc.get('env', {}), **(stage_env or {}))
    t = time.time()
    gargs = sum((['--gesture', g] for g in sc.get('gestures', [])), [])
    p = subprocess.run([PY, '-m', 'team.dance.live'] + sc['args'] + gargs + ['--out', os.path.join(OUT, f"{sc['name']}.mp4")], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
    res = dict(name=sc['name'], rc=p.returncode, wall_s=round(time.time() - t, 1), crashed=p.returncode != 0 or 'Traceback' in p.stdout + p.stderr)
    if res['crashed']: res['tail'] = (p.stdout + p.stderr)[-1500:]
    try:
        d = json.load(open(log, encoding='utf-8')); res['log'] = d
        tpath = os.path.join(ROOT, 'data', 'tmp', 'sessions', f"{d['summary']['session']}.jsonl")
        res['rows'] = [json.loads(l) for l in open(tpath, encoding='utf-8')] if os.path.exists(tpath) else []
    except Exception as e:
        res['log'] = None; res['rows'] = []; res['load_error'] = repr(e)
    return res


def check(res, name):
    d = res.get('log') or {}; s = d.get('summary') or {}; rows = res.get('rows') or []; types = [r['type'] for r in rows]
    asks = ' | '.join(e.get('ask', '') for e in d.get('log', []) if 'ask' in e)
    if name == 'completed': return bool(s.get('completed')) and not res['crashed']
    if name == 'started_30s': return s.get('time_to_start_s') is not None and s['time_to_start_s'] <= 30
    if name == 'persona_chill': return s.get('persona') == 'chill'
    if name == 'award': return 'award' in types
    if name == 'adapt_or_pass': return 'adapt' in types or any((r.get('score') or 0) >= 75 for r in rows if r['type'] == 'phrase_scored')
    if name == 'cues': return 'cue' in types or 'phrase_scored' in types
    if name == 'battle_three_rounds': return sum(1 for r in d.get('rounds', []) if r.get('round')) == 3
    if name == 'code_posted': return any(e.get('event') == 'result_posted' and e.get('code') for e in d.get('log', []))
    if name == 'safety_stop': return 'safety_stop' in types and "Let's stop here" in asks
    if name == 'paused_resumed': return 'paused' in types and 'resumed' in types
    if name == 'rerun_after_pause': return types.count('move_shown') >= 2
    if name == 'forget': return 'forget_me' in types
    if name == 'primary_locked_once': return types.count('primary_locked') == 1
    if name == 'squad_two': return any(r['type'] == 'squad_mode_started' and r.get('n', 0) >= 2 for r in rows)
    if name == 'squad_rounds': return any(r.get('squad') and len((r['squad'].get('ranking') or [])) >= 2 for r in d.get('rounds', []))
    if name == 'daily_key': return str(d.get('key', '')).startswith('gen_daily_')
    if name == 'no_offer': return not any(r['type'] == 'offer' and r.get('source') != 'brodie' for r in rows)    # Brodie's own suggestions are fine
    if name == 'proactive': return any(r['type'] == 'offer' and r.get('source') == 'brodie' for r in rows)
    if name == 'offered': return 'offer' in types
    if name == 'no_setup': return not any(r['type'] == 'state' and r.get('to') == 'PREPARE' for r in rows)
    if name == 'finish_pose': return 'finish_pose' in types
    if name == 'upper_body_used': return 'upper_body' in types and any(r['type'] == 'state' and r.get('to') == 'CONFIRM' for r in rows)
    if name == 'tracked': return (s.get('tracking_rate') or 0) >= 0.8
    if name == 'no_repeat_lines':
        said = [e.get('ask', '') for e in d.get('log', []) if 'ask' in e and 'note' not in e and 'result' not in e]
        return all(a != b for a, b in zip(said, said[1:])) and len(said) <= 12
    if name == 'demo_first':
        st = [r.get('why') for r in rows if r['type'] == 'state' and r.get('to') == 'ACTIVITY']
        return bool(st) and st[0] == 'WATCH'
    if name == 'waits_after_demo':
        st = [r.get('why') for r in rows if r['type'] == 'state' and r.get('to') == 'ACTIVITY']
        return 'WATCH' in st and 'YOUR TURN' not in st and any(r['type'] == 'speech' and r.get('key') == 'gotit' for r in rows)
    if name == 'teaching_order':
        st = [r.get('why') for r in rows if r['type'] == 'state' and r.get('to') == 'ACTIVITY']
        try: return st.index('WATCH') < st.index('WITH ME') < st.index('YOUR TURN') and any(r['type'] == 'speech' and r.get('key') == 'choice' for r in rows)
        except ValueError: return False
    if name == 'slow_start':
        starts = [e for e in d.get('log', []) if e.get('event') == 'start']
        return bool(starts) and starts[0].get('tempo') is not None and starts[0]['tempo'] <= 0.5 + 1e-6
    if name == 'speed_by_consent':
        ok = True; seen_choice = False
        for r in rows:
            if r['type'] == 'speech' and r.get('key') == 'choice': seen_choice = True
            if r['type'] == 'adapt' and r.get('action') == 'faster' and not seen_choice: ok = False
        tempos = [e.get('tempo') for e in d.get('log', []) if e.get('event') == 'start' and e.get('tempo')]
        jumps = all(b - a <= 0.16 for a, b in zip(tempos, tempos[1:]))
        return ok and jumps and (max(tempos) if tempos else 0) < 1.0
    if name == 'no_start': return 'attempt_started' not in types and 'activity_requested' not in types
    if name == 'setup_blocked': return 'setup_issue' in types and 'attempt_started' not in types
    if name == 'confirmed_before_dance':
        ti = [r['t'] for r in rows if r['type'] == 'confirmed']; ta = [r['t'] for r in rows if r['type'] == 'attempt_started']
        return bool(ti) and bool(ta) and min(ta) > min(ti) and 'space_confirmed' in types
    if name == 'ladder':
        st = [r.get('why') for r in rows if r['type'] == 'state' and r.get('to') == 'ACTIVITY']
        return 'WATCH' in st and 'WITH ME' in st and 'YOUR TURN' in st
    if name == 'duel': return any(e.get('event') == 'result_posted' and e.get('versus') for e in d.get('log', []))
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--clip', default=os.path.join('data', 'tmp', 'mac_handoff', 'segments', 'run3_jacks.mp4'))
    ap.add_argument('--only', default=''); ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--no-duel', action='store_true')
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(DB) and not a.only: os.remove(DB)
    scs = scenarios(a.clip)
    if a.only: scs = [s for s in scs if s['name'] in a.only.split(',')]
    with ThreadPoolExecutor(max_workers=a.jobs) as ex: results = list(ex.map(lambda s: run_one(s, a.clip), scs))
    # the duel: a second caller says the battle's code
    if not a.no_duel and (not a.only or 'battle' in a.only.split(',')):
        code = next((e.get('code') for r in results if r['name'] == 'battle' and r.get('log') for e in r['log'].get('log', []) if e.get('event') == 'result_posted'), None)
        if code:
            sc = dict(name='duel', args=['--simulate', a.clip, '--pose', 'lite', '--mode', 'battle', '--name', 'Eve', '--code', code], env=FAST, checks=['completed', 'duel'])
            results.append(run_one(sc, a.clip)); scs.append(sc)
    report = []; cue_rows = []
    for sc, res in zip(scs, results):
        checks = {c: check(res, c) for c in sc['checks']}
        s = (res.get('log') or {}).get('summary') or {}
        report.append(dict(name=sc['name'], ok=all(checks.values()), checks=checks, crashed=res['crashed'], wall_s=res['wall_s'], tracking_rate=s.get('tracking_rate'),
                           median_cue_latency_ms=s.get('median_cue_latency_ms'), time_to_start_s=s.get('time_to_start_s'), phrases=s.get('phrases'),
                           tool_errors=s.get('tool_errors'), tail=res.get('tail')))
        for r in res.get('rows') or []:
            if r['type'] == 'cue': cue_rows.append(dict(session=sc['name'], t=r['t'], kind=r.get('kind'), text=r.get('text'), latency_ms=r.get('latency_ms')))
        for e in ((res.get('log') or {}).get('log') or []):
            if 'ask' in e and ('cue' in e['ask'].lower() or 'Sentence two' in e['ask']): cue_rows.append(dict(session=sc['name'], t=e['t'], kind='spoken', text=e['ask'][:160], latency_ms=''))
    n = len(report); done = sum(1 for r in report if not r['crashed'] and r['checks'].get('completed'))
    tr = [r['tracking_rate'] for r in report if r['tracking_rate'] is not None]; lat = [r['median_cue_latency_ms'] for r in report if r['median_cue_latency_ms'] is not None]
    agg = dict(sessions=n, completed=done, completion_rate=round(done / n, 3) if n else None, median_tracking_rate=float(np.median(tr)) if tr else None,
               median_cue_latency_ms=float(np.median(lat)) if lat else None, scenarios_passing=sum(1 for r in report if r['ok']))
    json.dump(dict(aggregate=agg, scenarios=report), open(os.path.join(ROOT, 'data', 'tmp', 'eval_report.json'), 'w', encoding='utf-8'), indent=1)
    with open(os.path.join(ROOT, 'data', 'tmp', 'cue_review.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['session', 't', 'kind', 'text', 'latency_ms', 'relevant (y/n)']); w.writeheader()
        for r in cue_rows: w.writerow(r)
    print(f"{'scenario':<16} {'ok':<4} {'track':>6} {'cue ms':>7} {'start s':>8} {'wall s':>7}  failed checks")
    for r in report:
        bad = [c for c, v in r['checks'].items() if not v]
        print(f"{r['name']:<16} {'yes' if r['ok'] else 'NO':<4} {str(r['tracking_rate']):>6} {str(r['median_cue_latency_ms']):>7} {str(r['time_to_start_s']):>8} {r['wall_s']:>7}  {', '.join(bad)}")
        if r['crashed'] and r.get('tail'): print('   ', r['tail'][-600:].replace('\n', '\n    '))
    print('aggregate:', json.dumps(agg))
    return 0 if agg['scenarios_passing'] == n else 1


if __name__ == '__main__':
    sys.exit(main())
