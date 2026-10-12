"""Render the launch posters: every day x size x slide in team/promo/days/ to PNG, from config.json.

Run:  python -m team.promo.render [d0 d1 ...] [--clean] [--guides]
  no day args   -> all days (and the reel end cards)
  --clean       -> no CTA band (add your own in Canva/CapCut); files get a _clean suffix
  --guides      -> 9:16 renders show the TikTok/Reels UI safe zones in red; files get a _guides suffix
Fill call_line (what people read) and call_url (what the QR opens) in config.json, re-run, and every poster, end card
and captions_filled.md picks them up. Output: data/tmp/promo/<day>/<day>_<size>[_<slide>].png + data/tmp/promo/sheet.jpg
Needs Edge or Chrome (headless screenshot); nothing to pip install beyond segno (QR) and Pillow.
"""
import os, sys, json, time, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / 'data' / 'tmp' / 'promo'
SIZES = {'4x5': (1080, 1350), '9x16': (1080, 1920), '16x9': (1600, 900)}

# day -> (html file, number of slides, sizes). Carousels render one PNG per slide.
DAYS = {
    'd0': ('d0_teaser.html', 1, ('4x5', '9x16', '16x9')),
    'd1': ('d1_launch.html', 1, ('4x5', '9x16', '16x9')),
    'd2': ('d2_how.html', 5, ('4x5',)),
    'd2s': ('d2_how.html', 1, ('9x16', '16x9')),
    'd3': ('d3_scores.html', 1, ('4x5', '9x16', '16x9')),
    'd4': ('d4_pick.html', 1, ('4x5', '9x16', '16x9')),
    'd5': ('d5_squad.html', 1, ('4x5', '9x16', '16x9')),
    'd6': ('d6_tiers.html', 2, ('4x5', '9x16')),
    'd6x': ('d6_tiers.html', 1, ('16x9',)),
    'd7': ('d7_recap.html', 1, ('4x5', '9x16', '16x9')),
    'daily': ('daily.html', 7, ('9x16',)),
    'end': ('endcard.html', (0, 1, 2, 3, 4, 5, 6, 7), ('9x16',)),     # slide = day: the reels' closing card
}
SHEET_ORDER = ['d0', 'd1', 'd2', 'd2s', 'd3', 'd4', 'd5', 'd6', 'd6x', 'd7', 'daily', 'end']


def browser():
    for p in [os.environ.get('PROMO_BROWSER', ''),
              r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
              r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
              r'C:\Program Files\Google\Chrome\Application\chrome.exe',
              '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
              shutil.which('google-chrome') or '', shutil.which('chromium') or '', shutil.which('msedge') or '']:
        if p and os.path.exists(p):
            return p
    sys.exit('No Edge/Chrome found: set PROMO_BROWSER to a Chromium browser executable.')


def load_config():
    return json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


FONTS = [('Archivo Black', 'ArchivoBlack-Regular.ttf', ''), ('Space Grotesk', 'SpaceGrotesk.ttf', 'font-weight: 300 700;'),
         ('Plus Jakarta Sans', 'PlusJakartaSans.ttf', 'font-weight: 200 800;')]


def write_fonts_css():
    """the brand TTFs (team/web/groove/assets) as data URIs: headless Chromium won't load fonts from file:// URLs."""
    import base64
    dst = HERE / 'fonts.css'
    if dst.exists():
        return
    css = ''
    for fam, f, extra in FONTS:
        b64 = base64.b64encode((ROOT / 'team' / 'web' / 'groove' / 'assets' / f).read_bytes()).decode()
        css += f"@font-face {{ font-family: '{fam}'; src: url(data:font/ttf;base64,{b64}) format('truetype'); {extra} }}\n"
    dst.write_text(css, encoding='utf-8')


def write_config_js(cfg):
    """config.js is what the HTML reads (file:// pages can't fetch JSON). The QR is inlined as an SVG data URI."""
    promo = dict(cfg)
    url = (cfg.get('call_url') or '').strip()
    if url:
        import segno
        promo['qr'] = segno.make(url, error='m').svg_data_uri(dark='#0A0A0B', light='#F4F4F0', border=1, scale=8)
    (HERE / 'config.js').write_text('window.PROMO = ' + json.dumps(promo, ensure_ascii=False, indent=1) + ';\n', encoding='utf-8')


def shoot(html, out, size, params=None, exe=None):
    """one headless screenshot of days/<html>?size=..&<params> at the exact pixel size."""
    w, h = SIZES[size]
    q = urlencode(dict(size=size, **(params or {})))
    url = (HERE / 'days' / html).as_uri() + '?' + q
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    prof = tempfile.mkdtemp(prefix='promo_edge_')        # one profile per shot so parallel browsers don't collide
    cmd = [exe or browser(), '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run', '--mute-audio',
           '--allow-file-access-from-files', '--force-device-scale-factor=1', f'--user-data-dir={prof}',
           f'--window-size={w},{h}', '--virtual-time-budget=4000', f'--screenshot={out}', url]
    if (params or {}).get('alpha'):                      # transparent PNG (reel overlays)
        cmd.insert(1, '--default-background-color=00000000')
    for attempt in range(2):                             # headless Edge sometimes writes the PNG and then never exits
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while p.poll() is None and time.time() - t0 < 45:
            if out.exists() and out.stat().st_size > 0 and time.time() - out.stat().st_mtime > 1.0:
                break
            time.sleep(.25)
        if p.poll() is None:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], capture_output=True) if os.name == 'nt' else p.kill()
        if out.exists() and out.stat().st_size > 0:
            break
    shutil.rmtree(prof, ignore_errors=True)              # Edge's helpers can hold cache locks a moment longer: leave those
    if not out.exists():
        raise RuntimeError(f'screenshot failed: {url}')
    from PIL import Image
    im = Image.open(out)
    if im.size != (w, h):                    # some builds include window chrome in --window-size: pad/crop to exact
        fixed = Image.new('RGB', (w, h), (10, 10, 11)); fixed.paste(im.convert('RGB'), (0, 0)); fixed.save(out)
    return out


def jobs_for(day, clean=False, guides=False):
    html, slides, sizes = DAYS[day]
    folder = OUT / day.rstrip('sx') if day not in ('daily', 'end') else OUT / day
    ids = range(1, slides + 1) if isinstance(slides, int) else slides
    for size in sizes:
        for s in ids:
            params, suffix = {}, ''
            if len(ids) > 1:
                params['slide'] = s; suffix += f'_{s}'
            if clean:
                params['clean'] = 1; suffix += '_clean'
            if guides:
                params['guides'] = 1; suffix += '_guides'
            yield html, folder / f'{Path(html).stem}_{size}{suffix}.png', size, params


def fill_captions(cfg):
    src = HERE / 'CAPTIONS.md'
    if not src.exists():
        return None
    line = (cfg.get('call_line') or '').strip()
    rep = {'{{CALL_LINE}}': line or '{{CALL_LINE}}', '{{CALL_URL}}': (cfg.get('call_url') or '').strip() or '{{CALL_URL}}',
           '{{HANDLE}}': cfg.get('handle', ''), '{{HASHTAG}}': cfg.get('hashtag', ''), '{{LAUNCH}}': cfg.get('launch', '')}
    txt = src.read_text(encoding='utf-8')
    for k, v in rep.items():
        txt = txt.replace(k, v)
    dst = OUT / 'captions_filled.md'; OUT.mkdir(parents=True, exist_ok=True)
    dst.write_text(txt, encoding='utf-8')
    return dst


def contact_sheet(paths, dst):
    from PIL import Image, ImageDraw
    th = 360
    thumbs = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        thumbs.append((p, im.resize((int(im.width * th / im.height), th), Image.LANCZOS)))
    rows, row, x, maxw = [], [], 0, 2400
    for p, t in thumbs:
        if row and x + t.width > maxw:
            rows.append(row); row, x = [], 0
        row.append((p, t)); x += t.width + 16
    if row:
        rows.append(row)
    sheet = Image.new('RGB', (maxw + 16, len(rows) * (th + 40) + 16), (30, 31, 36))
    d = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        x = 16
        for p, t in row:
            y = 16 + r * (th + 40)
            sheet.paste(t, (x, y)); d.text((x, y + th + 6), Path(p).stem, fill=(200, 200, 200))
            x += t.width + 16
    sheet.save(dst, quality=88)
    return dst


def main(argv):
    clean, guides = '--clean' in argv, '--guides' in argv
    days = [a for a in argv if not a.startswith('--')]
    picked = [d for d in SHEET_ORDER if not days or d in days or d.rstrip('sx') in days]
    cfg = load_config()
    write_fonts_css()
    write_config_js(cfg)
    exe = browser()
    jobs = [j for d in picked for j in jobs_for(d, clean, guides)]
    with ThreadPoolExecutor(6) as ex:
        done = list(ex.map(lambda j: shoot(*j, exe=exe), jobs))
    for p in done:
        print(p.relative_to(ROOT))
    cap = fill_captions(cfg)
    if cap:
        print(cap.relative_to(ROOT))
    if done:
        print(contact_sheet(done, OUT / 'sheet.jpg').relative_to(ROOT))
    if not (cfg.get('call_line') or '').strip():
        print('note: call_line is empty, so posters show the "line opens" slot. Fill it in config.json and re-run.')
    if any(not str(v).strip() for v in cfg.get('recap', {}).values()):
        print('note: recap numbers in config.json are empty, so Day 7 shows "-". Fill in the real numbers before posting it.')


if __name__ == '__main__':
    main(sys.argv[1:])
