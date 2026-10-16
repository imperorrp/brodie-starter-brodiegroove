"""Render the animated promo pieces: the sticker/GIF pack, the incoming-call loop, the count timeline, the missed-calls gag and
the stack diagram. Each piece is a page in team/promo/anim/ whose CSS animations loop over `secs`.

Run:  python -m team.promo.animate [stickers call timeline missed stack] [--fps N] [--only perfect,miss] [--jobs 2]
Headless Edge/Chrome is driven over the DevTools protocol: every frame is window.seek(t) then a screenshot, so the timing is
exact whatever the machine's speed. ffmpeg then writes, per piece/variant/size:
  <name>.mp4         H.264, the loop repeated to at least `min_mp4` seconds (posts, Reels, X)
  <name>.gif         palette GIF, smaller (comment replies, Slack, X)
  <name>.png         the still at `still` seconds (static posts)
  stickers also get  <name>.webp (512 px, transparent, animated: IG/WhatsApp-style stickers) and <name>_green.mp4 (green
                     screen, for CapCut/TikTok overlays); their GIF keeps 1-bit transparency.
Output: data/tmp/promo/anim/<piece>/... + data/tmp/promo/anim/sheet.jpg (all the stills).
Brodie is kit art only: export_art() writes the kit poses and avatar.py's moods to data/tmp/promo/anim/art/ and runs the
kit's check_cream.py on them. The call line comes from config.json (call_line); while it is empty the pieces show the handle.
"""
import os, sys, json, time, shutil, base64, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from pathlib import Path

from team.promo.render import browser, load_config, write_fonts_css, write_config_js, contact_sheet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ANIM = HERE / 'anim'
OUT = ROOT / 'data' / 'tmp' / 'promo' / 'anim'
KIT = ROOT / 'team' / 'dance' / 'assets' / 'brodie'
SIZES = {'1x1': (720, 720), '4x5': (1080, 1350), '9x16': (1080, 1920), '16x9': (1600, 900)}
GIF_W = {'1x1': 480, '4x5': 540, '9x16': 540, '16x9': 800}

# piece -> page, variants, sizes, loop length, the still's time, alpha, the MP4's minimum length
PIECES = {
    'stickers': dict(page='stickers.html', variants=('perfect', 'miss', 'combo', 'srank', 'lockin', 'calling', 'chefskiss'),
                     sizes=('1x1',), secs=2.0, still=1.0, alpha=True, min_mp4=0),
    'call': dict(page='call.html', variants=('',), sizes=('9x16', '4x5'), secs=4.0, still=1.3, min_mp4=8),
    'timeline': dict(page='timeline.html', variants=('',), sizes=('16x9', '9x16'), secs=7.2, still=6.4, min_mp4=14),
    'missed': dict(page='missed.html', variants=('1', '2', '3'), sizes=('9x16', '4x5'), secs=8.0, still=7.9, min_mp4=0),
    'stack': dict(page='stack.html', variants=('',), sizes=('16x9',), secs=8.0, still=7.0, min_mp4=16),
}


# ---------- kit art (Brodie never comes from anywhere else) ----------

ART_FILES = ('brodie-flex.png', 'brodie-sit-wave.png', 'brodie-pfp-square.png')
MOOD_SPRITES = ('grin', 'closed', 'sleepy', 'focus_grin', 'focus_closed', 'encourage_grin', 'encourage_closed', 'pumped_grin')


def export_art():
    """the kit PNGs plus avatar.py's mood edits (the same sprites the call shows), trimmed, then cream-checked."""
    from team.dance import avatar
    dst = OUT / 'art'; dst.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in ART_FILES:
        shutil.copyfile(KIT / f, dst / f); paths.append(dst / f)
    for name, im in avatar._get().items():
        if name in MOOD_SPRITES:
            p = dst / f'mood_{name}.png'; im.crop(im.getbbox()).save(p); paths.append(p)
    chk = subprocess.run([sys.executable, str(KIT / 'tools' / 'check_cream.py'), *map(str, paths)],
                         cwd=KIT / 'tools', capture_output=True, text=True)
    if chk.returncode:
        sys.exit('cream check failed:\n' + chk.stdout + chk.stderr)
    return paths


# ---------- a minimal DevTools client ----------

class Tab:
    def __init__(self, ws_url):
        from websockets.sync.client import connect
        self.ws = connect(ws_url, max_size=None, open_timeout=20)
        self.n = 0

    def call(self, method, **params):
        self.n += 1; i = self.n
        self.ws.send(json.dumps({'id': i, 'method': method, 'params': params}))
        while True:
            m = json.loads(self.ws.recv(timeout=60))
            if m.get('id') == i:
                if 'error' in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get('result', {})

    def js(self, expr, wait=False):
        r = self.call('Runtime.evaluate', expression=expr, awaitPromise=wait, returnByValue=True)
        if 'exceptionDetails' in r:
            raise RuntimeError(f'{expr}: {r["exceptionDetails"].get("text")} {r.get("result", {}).get("description", "")}')
        return r.get('result', {}).get('value')


def launch(exe):
    import httpx
    prof = tempfile.mkdtemp(prefix='promo_anim_')
    p = subprocess.Popen([exe, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run', '--mute-audio',
                          '--allow-file-access-from-files', '--force-device-scale-factor=1', '--remote-allow-origins=*',
                          f'--user-data-dir={prof}', '--remote-debugging-port=0', 'about:blank'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    port_file, t0 = Path(prof) / 'DevToolsActivePort', time.time()
    def read_port():
        try:                               # Edge may still hold the file open while writing it (PermissionError on Windows)
            return port_file.read_text().split()[0]
        except (OSError, IndexError):
            return None
    while not (port := read_port()):
        if time.time() - t0 > 30:
            p.kill(); raise RuntimeError('browser never opened its DevTools port')
        time.sleep(.1)
    page = next(t for t in httpx.get(f'http://127.0.0.1:{port}/json/list', timeout=10).json() if t['type'] == 'page')
    return p, prof, Tab(page['webSocketDebuggerUrl'])


def close(p, prof, tab):
    """Browser.close over DevTools: on Windows the PID we started is only Edge's launcher, so killing it leaves the real
    browser (and its renderers) running; dozens of runs leaked ~80 of them and starved the laptop of memory."""
    try:
        import httpx
        from websockets.sync.client import connect
        port = (Path(prof) / 'DevToolsActivePort').read_text().split()[0]
        with connect(httpx.get(f'http://127.0.0.1:{port}/json/version', timeout=5).json()['webSocketDebuggerUrl'], open_timeout=5) as ws:
            ws.send(json.dumps({'id': 1, 'method': 'Browser.close'}))
    except Exception:
        pass
    try:
        tab.ws.close()
    except Exception:
        pass
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], capture_output=True) if os.name == 'nt' else p.kill()
    for _ in range(20):                                   # the profile stays locked until the browser has really exited
        shutil.rmtree(prof, ignore_errors=True)
        if not Path(prof).exists():
            break
        time.sleep(.25)


# ---------- frames and encoding ----------

def frames(tab, page, size, v, secs, fps, alpha, dst):
    w, h = SIZES[size]
    tab.call('Emulation.setDeviceMetricsOverride', width=w, height=h, deviceScaleFactor=1, mobile=False)
    tab.call('Emulation.setDefaultBackgroundColorOverride',
             color={'r': 0, 'g': 0, 'b': 0, 'a': 0} if alpha else {'r': 10, 'g': 10, 'b': 11, 'a': 1})
    def load(**extra):
        tab.js('window.ANIM_READY = false')
        q = urlencode({'size': size, 'v': v, **({'alpha': 1} if alpha else {}), **extra})
        tab.call('Page.navigate', url=(ANIM / page).as_uri() + '?' + q)
        t0 = time.time()
        while not tab.js('window.ANIM_READY === true'):
            if time.time() - t0 > 30:
                raise RuntimeError(f'{page}?{q} never got ready')
            time.sleep(.1)
    dst.mkdir(parents=True, exist_ok=True)
    if alpha:
        load()
    else:                                  # the static background once, then frames over it (4x faster per frame)
        load(plate=1)
        tab.js('window.seek(0)')
        plate = dst / 'plate.png'                    # not matched by ffmpeg's %04d.png
        plate.write_bytes(base64.b64decode(tab.call('Page.captureScreenshot', format='png')['data']))
        load(flat=plate.as_uri())
    for i in range(round(secs * fps)):
        tab.js(f'window.seek({i / fps:.4f})')
        tab.js('new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))', wait=True)
        png = base64.b64decode(tab.call('Page.captureScreenshot', format='png')['data'])
        (dst / f'{i:04d}.png').write_bytes(png)


def ff(*args):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError('ffmpeg failed: ' + r.stderr[-800:])


def encode(src, base, fps, secs, size, alpha, min_mp4):
    seq = ['-framerate', fps, '-i', src / '%04d.png']
    n = round(secs * fps)
    gw = GIF_W[size]
    if alpha:
        ff(*seq, '-vf', f'fps=20,scale={gw}:-1:flags=lanczos,split[a][b];[a]palettegen=reserve_transparent=1:stats_mode=diff[p];'
                        '[b][p]paletteuse=dither=bayer:bayer_scale=4:alpha_threshold=96', '-loop', 0, base.with_suffix('.gif'))
        ff(*seq, '-vf', 'fps=20,scale=512:512:flags=lanczos', '-c:v', 'libwebp_anim', '-q:v', 65, '-pix_fmt', 'yuva420p', '-loop', 0,
           base.with_suffix('.webp'))
        reps = 3                                         # ~6 s of green screen is enough to drop into an edit
        ff('-f', 'lavfi', '-i', f'color=c=0x00FF00:s={SIZES[size][0]}x{SIZES[size][1]}:r={fps}', *seq,
           '-filter_complex', f'[1:v]loop=loop={reps - 1}:size={n}:start=0[fg];[0:v][fg]overlay=shortest=1,format=yuv420p',
           '-c:v', 'libx264', '-crf', 18, '-preset', 'slow', '-movflags', '+faststart', base.with_name(base.name + '_green.mp4'))
        return
    reps = max(1, -(-min_mp4 // secs)) if min_mp4 else 1
    ff(*seq, '-vf', f'loop=loop={int(reps) - 1}:size={n}:start=0,format=yuv420p', '-c:v', 'libx264', '-crf', 17, '-preset', 'slow',
       '-movflags', '+faststart', base.with_suffix('.mp4'))
    ff(*seq, '-vf', f'fps=15,scale={gw}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff:max_colors=200[p];'
                    '[b][p]paletteuse=dither=sierra2_4a', '-loop', 0, base.with_suffix('.gif'))


def job(piece, v, size, fps, exe):
    spec = PIECES[piece]
    name = '_'.join(x for x in (piece, v, size) if x)
    base = OUT / piece / name
    base.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f'anim_{name}_'))
    p, prof, tab = launch(exe)
    try:
        frames(tab, spec['page'], size, v, spec['secs'], fps, spec.get('alpha', False), tmp)
    finally:
        close(p, prof, tab)
    still = min(round(spec['still'] * fps), round(spec['secs'] * fps) - 1)
    shutil.copyfile(tmp / f'{still:04d}.png', base.with_suffix('.png'))
    encode(tmp, base, fps, spec['secs'], size, spec.get('alpha', False), spec.get('min_mp4', 0))
    shutil.rmtree(tmp, ignore_errors=True)
    return base


def main(argv):
    fps = int(argv[argv.index('--fps') + 1]) if '--fps' in argv else 30
    only = set(argv[argv.index('--only') + 1].split(',')) if '--only' in argv else None
    jobs_n = int(argv[argv.index('--jobs') + 1]) if '--jobs' in argv else 2   # more browsers starve a 16 GB laptop
    skip = {argv[i + 1] for i, a in enumerate(argv) if a in ('--fps', '--only', '--jobs')}
    names = [a for a in argv if not a.startswith('--') and a not in skip] or list(PIECES)
    cfg = load_config()
    write_fonts_css(); write_config_js(cfg)
    print(f'kit art: {len(export_art())} files, cream check passed')
    exe = browser()
    jobs = [(pc, v, s) for pc in names for v in PIECES[pc]['variants'] for s in PIECES[pc]['sizes'] if not only or v in only]
    t0 = time.time()
    with ThreadPoolExecutor(jobs_n) as ex:
        done = list(ex.map(lambda j: job(*j, fps, exe), jobs))
    for b in done:
        print(b.relative_to(ROOT), ' '.join(sorted(x.name[len(b.name):] for x in b.parent.glob(b.name + '*'))))
    stills = sorted(OUT.glob('*/*.png'))
    if stills:
        print(contact_sheet(stills, OUT / 'sheet.jpg').relative_to(ROOT))
    print(f'{len(done)} renders in {time.time() - t0:.0f} s')
    if not (cfg.get('call_line') or '').strip():
        print('note: call_line is empty in config.json, so the pieces show the handle and hashtag instead of the line.')


if __name__ == '__main__':
    main(sys.argv[1:])
