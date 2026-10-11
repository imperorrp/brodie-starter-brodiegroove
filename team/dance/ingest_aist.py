"""Batch-ingest AIST++ 3D keypoints into keys.

AIST++ file names: g<GENRE>_s<TYPE>_cAll_d<dancer>_m<GENRE><music>_ch<nn>.pkl. TYPE sBM = basic motion (short, loopable moves:
our 8-count content), sFM = advanced free motion. Music index 0..5 maps to 80, 90, 100, 110, 120, 130 BPM (AIST++ paper, from
memory; verify against a clip's audio if a key feels off). Genres: BR break, PO pop, LO lock, MH middle hip-hop, LH LA-style
hip-hop, HO house, WA waack, KR krump, JS street jazz, JB ballet jazz. Keypoints are COCO-17 at 60 fps in cm, y up.

  .venv\Scripts\python -m team.dance.ingest_aist [--genres LH MH HO PO LO WA] [--per-genre 3] [--music 3] [--mirror]
Keys are named aist_<genre>_<bpm>_ch<nn> and carry the dataset licence note. Licence: AIST Dance Video DB terms of use
(research; cite Tsuchida et al. 2019 and Li et al. 2021); demo use only, never redistribute the data.
"""
import argparse, glob, os, re, sys
import numpy as np
from . import ROOT
from .ingest import load_aist_keypoints, key_from_coco_sequence

AIST_DIR = os.path.join(ROOT, 'data', 'aist', 'keypoints3d')
GENRE_NAMES = dict(BR='break', PO='pop', LO='lock', MH='middle hip-hop', LH='LA hip-hop', HO='house', WA='waack', KR='krump', JS='street jazz', JB='ballet jazz')
NAME_RE = re.compile(r'g(?P<genre>[A-Z]{2})_s(?P<type>[A-Z]{2})_c(?P<cam>\w+)_d(?P<dancer>\d+)_m(?P<mgenre>[A-Z]{2})(?P<music>\d)_ch(?P<ch>\d+)')
LICENCE = 'AIST Dance Video DB terms of use (research); AIST++ annotations (Li et al. 2021); demo use only, not redistributable'


def bpm_of(music_idx):
    return 80.0 + 10.0 * int(music_idx)


def list_sequences(genres, type_='BM', music=None):
    out = []
    for p in sorted(glob.glob(os.path.join(AIST_DIR, 'g*_s*_cAll_*.pkl'))):
        m = NAME_RE.search(os.path.basename(p))
        if not m or m['type'] != type_ or m['genre'] not in genres: continue
        if music is not None and int(m['music']) != int(music): continue
        out.append((p, m))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--genres', nargs='+', default=['LH', 'MH', 'HO', 'PO', 'LO', 'WA'])
    ap.add_argument('--per-genre', type=int, default=3); ap.add_argument('--music', type=int, default=3, help='music index 0..5 (3 = 110 bpm); -1 = any')
    ap.add_argument('--type', default='BM', choices=['BM', 'FM']); ap.add_argument('--mirror', action='store_true')
    ap.add_argument('--counts', type=int, default=8); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    if not os.path.isdir(AIST_DIR): raise SystemExit(f'{AIST_DIR} not found: download keypoints3d.zip from the AIST++ release and extract into data/aist/')
    seqs = list_sequences(a.genres, a.type, None if a.music < 0 else a.music)
    per = {}
    for p, m in seqs: per.setdefault(m['genre'], []).append((p, m))
    made = []
    for g in a.genres:
        for p, m in per.get(g, [])[:a.per_genre]:
            bpm = bpm_of(m['music']); name = f"aist_{g.lower()}_{int(bpm)}_ch{m['ch']}"
            if a.dry_run: print(f'{name}: {os.path.basename(p)} ({GENRE_NAMES.get(g, g)}, {bpm:.0f} bpm)'); continue
            seq, fps = load_aist_keypoints(p)
            try:
                key = key_from_coco_sequence(seq, fps, name, bpm=bpm, is_3d=True, counts_per_phrase=a.counts, mirror=a.mirror,
                                             source=f'AIST++ {os.path.basename(p)}', licence=LICENCE, style=GENRE_NAMES.get(g, g), difficulty=2 if a.type == 'BM' else 4)
                made.append(name); print(f"{name}: {key['n_beats']} beats at {key['bpm']:.0f} bpm, {len(key['phrases'])} phrases, {len(seq)} frames ({GENRE_NAMES.get(g, g)})", flush=True)
            except Exception as e:
                print(f'{name}: FAILED {e!r}', flush=True)
    print(f'{len(made)} keys written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
