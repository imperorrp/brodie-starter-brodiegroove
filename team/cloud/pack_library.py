"""Zip the git-ignored keys (aist_*.json, class_*.json) from team/dance/library/ for the cloud box to fetch (fetch_library.py).

    .venv\\Scripts\\python team\\cloud\\pack_library.py            -> data/tmp/library.zip

Upload the zip somewhere private (Supabase > Storage > a private bucket > upload > "Get URL" > signed, a long expiry) and set
DANCE_LIBRARY_URL on the box to that URL. The AIST++ terms allow our own use, not redistribution: never a public bucket.
"""
import glob, os, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARY = os.path.join(ROOT, 'team', 'dance', 'library')
OUT = os.path.join(ROOT, 'data', 'tmp', 'library.zip')


def main(out=OUT):
    files = sorted(glob.glob(os.path.join(LIBRARY, 'aist_*.json')) + glob.glob(os.path.join(LIBRARY, 'class_*.json')))
    if not files: print('no aist_*.json keys in team/dance/library (run python -m team.dance.ingest_aist first)'); return 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, os.path.basename(f))
    print(f'{len(files)} keys -> {out} ({os.path.getsize(out) // 1024} KB)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else OUT))
