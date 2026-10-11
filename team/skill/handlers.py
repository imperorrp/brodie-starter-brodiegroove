"""Our Brodie skill: on_event(event, say_now, ask_voice), loaded by the worker with --handlers team/skill/handlers.py.

Same contract as starter/handlers.py (read its docstring and docs/STARTER_EVENTS.md):
  say_now(text)    a pre-rendered bank string, ~0.2 s, never over a count
  ask_voice(text)  a Gemini line, ~1 s; only spoken between sets (rest, done, wait, a gesture)
Runs on the worker's event loop: return fast, never block.

On top of the default coaching, every event is forwarded to our backend (BACKEND_URL, default http://127.0.0.1:8787) from a
background thread, so a slow or missing backend never stalls the call. BACKEND_URL=off turns forwarding off.
Note: events here carry no room or call id (the worker's JSONL log adds those); the backend groups them by time for now.
"""
import json, os, queue, threading, urllib.request

BACKEND_URL = os.environ.get('BACKEND_URL', 'http://127.0.0.1:8787').rstrip('/')
SKIP_TYPES = {'status'}                      # every 5 s; not worth sending
_q = queue.Queue(maxsize=2000)


def _sender():
    while True:
        ev = _q.get()
        try:
            req = urllib.request.Request(BACKEND_URL + '/events', data=json.dumps(ev).encode(),
                                         headers={'content-type': 'application/json'}, method='POST')
            urllib.request.urlopen(req, timeout=2).read()
        except Exception:
            pass                             # backend down: drop it, the call matters more


if BACKEND_URL.lower() != 'off':
    threading.Thread(target=_sender, daemon=True, name='team-backend-sender').start()


def forward(event):
    if BACKEND_URL.lower() == 'off' or event.get('type') in SKIP_TYPES: return
    try: _q.put_nowait(event)
    except queue.Full: pass


def on_event(event, say_now, ask_voice):
    forward(event)
    t = event['type']
    if t == 'session' and event.get('state') == 'rest':
        ask_voice('The set just ended: in ONE short sentence, say how it went and name what comes next.')
    elif t == 'session' and event.get('state') == 'done':
        ask_voice('The whole session is over: close it warmly in one short sentence.')
