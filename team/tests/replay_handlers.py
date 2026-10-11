"""Feed an events file through our handlers with fake say_now / ask_voice, no call needed (docs/BUILD_ON_BRODIE.md section 3).

  python team/tests/replay_handlers.py data/tmp/run3.events.jsonl [--handlers team/skill/handlers.py] [--forward]

Accepts JSONL (replay_local --events) or a JSON list (tests/fixtures). Forwarding to the backend is off unless --forward.
"""
import argparse, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_events(path):
    with open(path, encoding='utf-8') as f:
        text = f.read().strip()
    if text.startswith('['): return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('events')
    ap.add_argument('--handlers', default=os.path.join(HERE, '..', 'skill', 'handlers.py'))
    ap.add_argument('--forward', action='store_true', help='also send events to BACKEND_URL (start the backend first)')
    a = ap.parse_args()
    if not a.forward: os.environ['BACKEND_URL'] = 'off'
    spec = importlib.util.spec_from_file_location('team_handlers', a.handlers)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    calls = {'say_now': 0, 'ask_voice': 0}

    def say_now(text, intent='other'):
        calls['say_now'] += 1; print(f'          -> say_now {text!r}'); return {'status': 'ok'}

    def ask_voice(prompt):
        calls['ask_voice'] += 1; print(f'          -> ask_voice {prompt[:70]!r}...'); return 'queued'

    events = load_events(a.events)
    for ev in events:
        if ev['type'] == 'session':
            print(f"{ev['t']:7.1f}  session {ev['state']:6} {ev.get('exercise') or '-':14} target {ev.get('target')} count {ev.get('count')}")
        elif ev['type'] == 'rep':
            print(f"{ev['t']:7.1f}  rep     {ev.get('exercise')} {ev.get('n')}")
        elif ev['type'] == 'cue':
            print(f"{ev['t']:7.1f}  cue     {ev['intent']:10} {ev['text']!r}")
        h.on_event(ev, say_now, ask_voice)
    print(f'\n{len(events)} events, say_now x{calls["say_now"]}, ask_voice x{calls["ask_voice"]}')
    if a.forward:
        import time; time.sleep(1.0)        # let the sender thread drain


if __name__ == '__main__':
    sys.exit(main())
