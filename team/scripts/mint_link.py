"""Mint a room token for the phone page and print the links. Used by chrome.ps1, link_local.ps1 and chrome_test.py.

Why not facetime/mint_token.py: when the worker runs under a dispatch name (AGENT_NAME in .env.local, needed for the Chert
line), a room only gets that worker if the token carries an explicit agent dispatch; automatic dispatch never reaches it. This
minter adds the dispatch whenever AGENT_NAME is set (or --agent is given), so one worker serves the line and the Chrome tests.

Room-name prefixes the session understands (team/dance/live.py):
  demo-   one routine, no chat/offer/ready gates (DANCE_LOBBY=0 for that call only)
  desk-   arms-only camera check and scoring (a laptop webcam at a desk)
  demo-desk-...  both.

  .venv\\Scripts\\python team\\scripts\\mint_link.py [--prefix demo-desk] [--facing user|environment] [--ip 10.0.0.5] [--port 8443]
                                                     [--identity chrome-laptop] [--agent NAME] [--json]
Prints: room, the page URL (carries the token: do not post it), the LiveKit Meet fallback; writes team/.go_url for /go.
"""
import argparse, datetime, json, os, socket, sys, time
from urllib.parse import urlencode

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def load_env():
    p = os.path.join(ROOT, '.env.local')
    if not os.path.exists(p): return
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1); k = k.strip(); v = v.strip().strip('"').strip("'")
        if k not in os.environ: os.environ[k] = v


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return '127.0.0.1'


def mint(room, identity='chrome-laptop', agent=None, ttl_s=3600):
    from livekit import api
    t = (api.AccessToken(os.environ['LIVEKIT_API_KEY'], os.environ['LIVEKIT_API_SECRET']).with_identity(identity).with_name(identity)
         .with_ttl(datetime.timedelta(seconds=int(ttl_s)))
         .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True, can_publish_data=True)))
    if agent:
        t = t.with_room_config(api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=agent)]))
    return t.to_jwt()


def make(prefix='dance', facing='user', ip=None, port=8443, identity='chrome-laptop', agent=None, host=None):
    load_env()
    agent = agent if agent is not None else (os.environ.get('AGENT_NAME') or None)
    room = f"{prefix.rstrip('-')}-{time.strftime('%m%d-%H%M%S')}"
    token = mint(room, identity, agent)
    host = host or ip or lan_ip()
    page = f"https://{host}:{port}/phone.html?" + urlencode(dict(url=os.environ['LIVEKIT_URL'], token=token, facing=facing))
    meet = 'https://meet.livekit.io/custom?' + urlencode(dict(liveKitUrl=os.environ['LIVEKIT_URL'], token=token))
    return dict(room=room, agent=agent, page=page, meet=meet, short=f"https://{host}:{port}/go", token=token)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--prefix', default='dance'); ap.add_argument('--facing', default='user', choices=['user', 'environment'])
    ap.add_argument('--ip', default=None); ap.add_argument('--port', type=int, default=8443); ap.add_argument('--identity', default='chrome-laptop')
    ap.add_argument('--agent', default=None, help='dispatch name (default: AGENT_NAME from .env.local; empty string = automatic dispatch)')
    ap.add_argument('--localhost', action='store_true', help='link to https://localhost (this laptop) instead of the LAN address')
    ap.add_argument('--json', action='store_true'); ap.add_argument('--no-go', action='store_true', help='do not update team/.go_url')
    a = ap.parse_args(argv)
    r = make(a.prefix, a.facing, a.ip, a.port, a.identity, a.agent, host='localhost' if a.localhost else None)
    if not a.no_go:
        with open(os.path.join(ROOT, 'team', '.go_url'), 'w', encoding='ascii') as f: f.write(r['page'])
    if a.json: print(json.dumps(r)); return 0
    print(f"room: {r['room']}   dispatch: {r['agent'] or 'automatic'}   (links carry a one-hour token for this room only; do not post them)")
    print(f"short link: {r['short']}")
    print(f"page: {r['page']}")
    print(f"LiveKit Meet fallback: {r['meet']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
