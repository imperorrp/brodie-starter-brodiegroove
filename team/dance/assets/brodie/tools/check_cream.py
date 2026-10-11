#!/usr/bin/env python3
"""Fail any Brodie image whose body fill has drifted off canon cream.

Image models drift Brodie toward pink/peach (a* creeps up) or toward white.
This measures the median body fill and compares it to #EFE7DA in CIELAB.

Usage: check_cream.py img.png|https://... [more ...]     exit 1 if any fails
"""
import sys
import numpy as np
from PIL import Image
from cream import CREAM, rgb_to_lab, hue_chroma

CANON = rgb_to_lab(CREAM)          # L 91.9, a 0.5, b 7.3
MAX_A = 1.8                        # above this reads pink/peach next to canon
MIN_A = -1.5                       # below this reads green/grey
B_RANGE = (5.0, 11.0)              # under 5 reads white, over 11 reads yellow/tan
MAX_DE = 3.0                       # chromatic (a*b*) distance budget; L is left to lighting


def body_mask(lab, alpha):
    hue, chroma = hue_chroma(lab)
    return ((alpha > 250) & (lab[..., 0] > 84) & (chroma > 3.5) & (chroma < 16)
            & ((hue < 120) | (hue > 330)))


def _open(path):
    path = str(path)
    if path.startswith(("http://", "https://")):
        import io, urllib.request
        return Image.open(io.BytesIO(urllib.request.urlopen(path, timeout=60).read()))
    return Image.open(path)


def measure(path):
    im = np.asarray(_open(path).convert("RGBA"))
    lab = rgb_to_lab(im[..., :3])
    m = body_mask(lab, im[..., 3])
    if m.sum() < 500:
        return None, 0.0
    return np.median(lab[m], axis=0), m.mean()


def verdict(fill):
    L, a, b = fill
    de = float(np.linalg.norm(fill[1:] - CANON[1:]))
    problems = []
    if a > MAX_A: problems.append(f"too pink (a*={a:.1f} > {MAX_A})")
    if a < MIN_A: problems.append(f"too green/grey (a*={a:.1f})")
    if b < B_RANGE[0]: problems.append(f"too white (b*={b:.1f})")
    if b > B_RANGE[1]: problems.append(f"too yellow/tan (b*={b:.1f})")
    if de > MAX_DE and not problems: problems.append(f"off canon (dE={de:.1f})")
    return de, problems


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        fill, cover = measure(p)
        if fill is None:
            print(f"?    {p}: no cream body found"); bad += 1; continue
        de, problems = verdict(fill)
        rgb = "#%02X%02X%02X" % tuple(int(v) for v in __import__("cream").lab_to_rgb(fill))
        tag = "PASS" if not problems else "FAIL"
        print(f"{tag} {p}: fill {rgb} L{fill[0]:.1f} a{fill[1]:+.1f} b{fill[2]:+.1f} dE {de:.1f}"
              + ("" if not problems else "  <- " + "; ".join(problems)))
        bad += bool(problems)
    sys.exit(1 if bad else 0)
