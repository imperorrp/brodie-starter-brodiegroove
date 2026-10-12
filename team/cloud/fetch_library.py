"""Fetch the private part of the reference library at boot: the AIST++-derived keys (research-licensed, git-ignored, never
redistributed) and any recorded class keys, as a zip made by pack_library.py and hosted somewhere private (a Supabase Storage
bucket with a long-lived signed URL is the simplest). DANCE_LIBRARY_URL unset: nothing happens (the Commons keys in git still
work; generated routines, the daily key and every style pool need the AIST keys, so set it).

    DANCE_LIBRARY_URL=https://... python team/cloud/fetch_library.py

Only top-level *.json entries are taken, by basename, into team/dance/library/; files already there are left alone.
"""
import io, os, sys, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARY = os.path.join(ROOT, 'team', 'dance', 'library')


def main():
    url = (os.environ.get('DANCE_LIBRARY_URL') or '').strip()
    if not url:
        n = len([f for f in os.listdir(LIBRARY) if f.startswith('aist_')]) if os.path.isdir(LIBRARY) else 0
        print(f'[library] DANCE_LIBRARY_URL unset; {n} AIST keys on disk', flush=True); return 0 if n else 1
    try:
        with urllib.request.urlopen(url, timeout=60) as r: blob = r.read()
    except Exception as e:
        print(f'[library] download failed: {e!r}', flush=True); return 1
    os.makedirs(LIBRARY, exist_ok=True); new = 0; kept = 0
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for info in z.infolist():
                name = os.path.basename(info.filename)
                if info.is_dir() or not name.endswith('.json') or name.startswith('.'): continue
                dest = os.path.join(LIBRARY, name)
                if os.path.exists(dest): kept += 1; continue
                with z.open(info) as src, open(dest, 'wb') as out: out.write(src.read())
                new += 1
    except zipfile.BadZipFile:
        print('[library] the URL did not return a zip (a signed URL that expired returns JSON)', flush=True); return 1
    print(f'[library] {new} keys added, {kept} already present', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
