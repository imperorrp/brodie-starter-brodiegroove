"""Download the MediaPipe pose landmarker models into models/ when they are missing (git-ignored; Apache 2.0; Google's public
bucket). The build runs this (nixpacks.toml) and start.sh runs it again, so a box that lost its disk heals itself.

    python team/cloud/fetch_models.py [full lite heavy]      default: full lite (DANCE_POSE picks one; the worker defaults to full)
"""
import os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, 'models')
URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task'
MIN_BYTES = 1_000_000


def main(variants=('full', 'lite')):
    os.makedirs(MODELS, exist_ok=True); ok = True
    for v in variants:
        p = os.path.join(MODELS, f'pose_landmarker_{v}.task')
        if os.path.exists(p) and os.path.getsize(p) > MIN_BYTES:
            print(f'[models] {os.path.basename(p)} present ({os.path.getsize(p) // 1_000_000} MB)'); continue
        try:
            urllib.request.urlretrieve(URL.format(v=v), p + '.part'); os.replace(p + '.part', p)
            print(f'[models] downloaded {os.path.basename(p)} ({os.path.getsize(p) // 1_000_000} MB)', flush=True)
        except Exception as e:
            print(f'[models] {v}: download failed: {e!r}', flush=True); ok = False
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(tuple(sys.argv[1:]) or ('full', 'lite')))
