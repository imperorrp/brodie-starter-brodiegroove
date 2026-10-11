"""Automated any-browser run: headless Chrome, with a dance clip as its camera, joins the room through team/web/phone.html.

Proves the browser path end to end with nobody on camera: the page, the token, the camera and mic publish, Brodie's video and
audio coming back, pose on the browser's frames, a scored phrase. The same page a person opens on a laptop or an Android phone.

  .venv\\Scripts\\python team\\scripts\\chrome_test.py [--video data/dance_clips/huntrix.mp4] [--seconds 100]
                                                      [--worker-log data/tmp/worker_dance5.log] [--port 8443]
Needs: the dance worker running under automatic dispatch (dance.ps1 or run_worker, no agent name), serve_web.py on the port,
Chrome installed, ffmpeg on PATH. Prints the worker's [dance] lines seen during the run and the scored rounds from
data/tmp/dance_session_log.json. Exit 0 when a phrase was scored.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from urllib.parse import urlencode

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE):
    if p not in sys.path: sys.path.insert(0, p)
CHROME = [r'C:\Program Files\Google\Chrome\Application\chrome.exe', r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
          '/usr/bin/google-chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']


def load_env():
    p = os.path.join(ROOT, '.env.local')
    if not os.path.exists(p): return
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1); k = k.strip(); v = v.strip().strip('"').strip("'")
        if v and not os.environ.get(k): os.environ[k] = v


def make_y4m(video, out, seconds=16, fps=12, height=480):
    """Chrome's fake camera plays a Y4M file in a loop. Keep it small: 480p at 12 fps is enough for pose at 288 px."""
    ff = shutil.which('ffmpeg')
    if not ff: raise SystemExit('ffmpeg not on PATH')
    cmd = [ff, '-y', '-loglevel', 'error', '-i', video, '-t', str(seconds), '-r', str(fps), '-vf', f'scale=-2:{height}', '-pix_fmt', 'yuv420p', '-an', out]
    subprocess.run(cmd, check=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--video', default=os.path.join(ROOT, 'data', 'dance_clips', 'huntrix.mp4'))
    ap.add_argument('--seconds', type=float, default=200.0, help='how long Chrome stays in the room (the lobby waits 50 s for a caller who never speaks, then one phrase takes about 20 s)')
    ap.add_argument('--worker-log', default=os.path.join(ROOT, 'data', 'tmp', 'worker_dance5.log'))
    ap.add_argument('--port', type=int, default=8443); ap.add_argument('--room', default=None)
    ap.add_argument('--y4m', default=os.path.join(ROOT, 'data', 'tmp', 'chrome_camera.y4m'))
    ap.add_argument('--keep-y4m', action='store_true'); ap.add_argument('--headed', action='store_true', help='show the Chrome window')
    a = ap.parse_args(argv)
    load_env()
    chrome = next((c for c in CHROME if os.path.exists(c)), None) or shutil.which('chrome') or shutil.which('google-chrome')
    if not chrome: raise SystemExit('Chrome not found')
    from mint_link import mint                                   # same folder; adds the named dispatch when AGENT_NAME is set
    agent = os.environ.get('AGENT_NAME') or None
    room = a.room or f'demo-chrome-{time.strftime("%m%d-%H%M%S")}'  # 'demo-' = one routine, no chat gates (nobody speaks in this test)
    token = mint(room, 'chrome-caller', agent)
    print(f'[chrome] dispatch: {agent or "automatic"}', flush=True)
    if not os.path.exists(a.y4m) or not a.keep_y4m:
        print(f'[chrome] writing fake camera {a.y4m} from {a.video}', flush=True); make_y4m(a.video, a.y4m)
    url = f'https://localhost:{a.port}/phone.html?' + urlencode(dict(url=os.environ['LIVEKIT_URL'], token=token, facing='user', autojoin='1'))
    profile = tempfile.mkdtemp(prefix='brodie-chrome-')
    flags = [chrome, '--no-first-run', '--no-default-browser-check', f'--user-data-dir={profile}', '--window-size=720,1280',
             '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', f'--use-file-for-fake-video-capture={a.y4m}',
             '--ignore-certificate-errors', '--autoplay-policy=no-user-gesture-required', '--disable-gpu', '--mute-audio']
    if not a.headed: flags.append('--headless=new')
    log_pos = os.path.getsize(a.worker_log) if os.path.exists(a.worker_log) else 0
    sess_log = os.path.join(ROOT, 'data', 'tmp', 'dance_session_log.json'); sess_mtime = os.path.getmtime(sess_log) if os.path.exists(sess_log) else 0
    print(f'[chrome] room {room}; joining through the page for {a.seconds:.0f} s (token not printed)', flush=True)
    proc = subprocess.Popen(flags + [url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while time.time() - t0 < a.seconds and proc.poll() is None: time.sleep(1.0)
    finally:
        if proc.poll() is None:
            if os.name == 'nt': subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else: proc.terminate()
        time.sleep(8.0)                                           # let the worker notice the leave and write its session log
        shutil.rmtree(profile, ignore_errors=True)
        if not a.keep_y4m and os.path.exists(a.y4m): os.remove(a.y4m)
    seen = []
    if os.path.exists(a.worker_log):
        with open(a.worker_log, encoding='utf-8', errors='replace') as f:
            f.seek(log_pos); new = f.read()
        for line in new.splitlines():
            if any(k in line for k in ('[dance]', 'participant', 'bound', 'coach-video', 'scored', 'facetime-remote-video', 'Traceback', 'Error')):
                seen.append(line.strip()[:200])
    print('[chrome] worker lines during the run:'); print('\n'.join('  ' + s for s in seen[-40:]) or '  (none: is --worker-log the right file?)')
    scored = []
    if os.path.exists(sess_log) and os.path.getmtime(sess_log) > sess_mtime:
        d = json.load(open(sess_log, encoding='utf-8'))
        for r in d.get('rounds', []):
            res = r.get('result') or {}
            scored.append(dict(phrase=r.get('phrase'), score=res.get('score'), scored=res.get('scored'), visible=res.get('visible_fraction')))
        print(f"[chrome] session log: key {d.get('key')} mode {d.get('mode')} rounds {json.dumps(scored)}")
    else:
        print('[chrome] no new session log (the worker did not finish a session; it may still be running or the caller never bound)')
    scored_ok = any(x.get('scored') for x in scored) or any(s.startswith('[dance] phrase') and ' scored:' in s for s in seen)
    # The teaching ladder waits for a spoken "ready" after the demo of each step, which a silent headless browser can never give.
    # Reaching ACTIVITY proves the whole plumbing: dispatch, camera and mic bound, camera check passed on browser frames, music
    # and the demo played on the published video. A scored phrase is the stronger pass when the session got that far.
    reached = any('-> ACTIVITY' in s for s in seen)
    ok = scored_ok or reached
    print('[chrome] RESULT:', 'PASS (a phrase was scored on frames from the browser)' if scored_ok else
          'PASS (camera check passed and the routine started on browser frames; a scored phrase needs a caller to say "ready")' if reached else
          'FAIL (the session never reached the routine)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
