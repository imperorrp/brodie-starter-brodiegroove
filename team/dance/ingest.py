"""Keys from motion files and keypoint datasets (no video): COCO-17 keypoint sequences in 2D or 3D, such as AIST++'s
`keypoints2d` / `keypoints3d`, PopDanceSet or any pose-estimator export. The 17 COCO joints cover every joint our features use,
so a sequence maps onto the MediaPipe layout and goes through the same feature code as live frames.

  .venv/Scripts/python -m team.dance.ingest --npy seq.npy --fps 60 --bpm 100 --name house_basic [--3d] [--start 2 --end 14]
      [--source "AIST++ gBR_sBM_c01_d04_mBR0_ch01" --licence "AIST Dance DB terms, research; demo only"]
  .venv/Scripts/python -m team.dance.ingest --aist keypoints3d/gBR_sBM_cAll_d04_mBR0_ch01.pkl --bpm 80 --name break_basic

COCO order: 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_sho, 6 r_sho, 7 l_elb, 8 r_elb, 9 l_wri, 10 r_wri, 11 l_hip, 12 r_hip,
13 l_knee, 14 r_knee, 15 l_ank, 16 r_ank. AIST++ pickles hold 'keypoints3d' (N, 17, 3) and 'keypoints3d_optim'.
"""
import argparse, os, pickle, sys
import numpy as np
from . import LIBRARY_DIR
from .features import features_from_landmarks, visibility_mask, mirror_landmarks, hip_energy, skeleton2d_normalized, resample_to_grid, resample_mask
from .reference import key_from_arrays, save_key
from .scoring import PER_BEAT

COCO_TO_MP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}


def coco_to_mediapipe(kp, conf=None, is_3d=False):
    """(17, 2|3) COCO joints -> (33, 5) image-style landmarks (x, y, z, vis, pres) and, for 3D input, (33, 3) world landmarks.
    Unmapped MediaPipe joints (fingers, feet, mouth) are filled with the nearest mapped joint so no NaN reaches the features."""
    kp = np.asarray(kp, np.float32)
    img = np.zeros((33, 5), np.float32); world = np.zeros((33, 3), np.float32)
    if conf is None: conf = np.ones(17, np.float32)
    for c, m in COCO_TO_MP.items():
        img[m, 0] = kp[c, 0]; img[m, 1] = kp[c, 1]; img[m, 2] = kp[c, 2] if kp.shape[1] > 2 else 0.0
        img[m, 3] = img[m, 4] = float(conf[c])
        if is_3d: world[m] = kp[c, :3]
    # fill the rest from anchors: hands from wrists, feet from ankles, face from nose
    for src, dsts in ((15, (17, 19, 21)), (16, (18, 20, 22)), (27, (29, 31)), (28, (30, 32)), (0, (1, 3, 4, 6, 9, 10))):
        for d in dsts:
            img[d] = img[src]; world[d] = world[src]
    if is_3d:
        hip = 0.5 * (world[23] + world[24]); world = world - hip                 # hip-centred like MediaPipe's world landmarks
        # image coordinates for the ghost: project x, y and normalise into roughly [0, 1]
        xy = world[:, :2].copy(); span = float(np.abs(xy).max()) + 1e-6
        img[:, 0] = 0.5 + 0.4 * xy[:, 0] / span; img[:, 1] = 0.5 - 0.4 * xy[:, 1] / span if _y_up(world) else 0.5 + 0.4 * xy[:, 1] / span
        world[:, 1] = -world[:, 1] if _y_up(world) else world[:, 1]             # our features expect y increasing downward
    return img, (world if is_3d else None)


def _y_up(world):
    """Heuristic: in a y-up frame the nose has a larger y than the ankles."""
    return float(world[0, 1]) > float(0.5 * (world[27, 1] + world[28, 1]))


def key_from_coco_sequence(seq, fps, name, bpm=None, beat_times=None, conf=None, is_3d=False, start=0.0, end=None, counts_per_phrase=8,
                           per_beat=PER_BEAT, mirror=False, source='', licence='', style='', difficulty=2, save=True):
    """seq: (N, 17, 2|3) keypoints at `fps`; conf: optional (N, 17). Either bpm (uniform grid from `start`) or beat_times."""
    seq = np.asarray(seq, np.float32); N = len(seq); times = np.arange(N) / float(fps)
    if end is None: end = float(times[-1])
    if beat_times is None:
        if not bpm: raise ValueError('give bpm or beat_times')
        n = int((end - start) * bpm / 60.0) + 1; beat_times = start + np.arange(n) * 60.0 / bpm
    beat_times = np.asarray(beat_times, np.float64); beat_times = beat_times[(beat_times >= start) & (beat_times <= end + 1e-6)]
    feats = []; masks = []; energies = []; skels = []; ok_t = []; prev = None; prev_t = None
    for i in range(N):
        a, w = coco_to_mediapipe(seq[i], None if conf is None else conf[i], is_3d=is_3d)
        if mirror: a, w = mirror_landmarks(a, w)
        feats.append(features_from_landmarks(a, w)); masks.append(visibility_mask(a).astype(np.float32))
        energies.append(hip_energy(prev, a, (times[i] - prev_t) if prev_t is not None else 0.0)); skels.append(skeleton2d_normalized(a))
        ok_t.append(times[i]); prev = a; prev_t = times[i]
    grid = []
    for k in range(len(beat_times) - 1):
        grid.extend(beat_times[k] + (beat_times[k + 1] - beat_times[k]) * np.arange(per_beat) / per_beat)
    grid = np.asarray(grid, np.float64)
    if len(grid) < per_beat * 2: raise ValueError('fewer than two beats in range')
    F = resample_to_grid(ok_t, feats, grid); M = resample_mask(ok_t, masks, grid)
    E = resample_to_grid(ok_t, np.asarray(energies, np.float32), grid)[:, 0]
    S = resample_to_grid(ok_t, np.asarray(skels, np.float32).reshape(len(skels), -1), grid).reshape(len(grid), 33, 2)
    bpm_est = float(bpm) if bpm else 60.0 / float(np.median(np.diff(beat_times)))
    key = key_from_arrays(name, bpm_est, F, M=M, E=E, S=S, per_beat=per_beat, counts_per_phrase=counts_per_phrase, source=source, licence=licence, save=False)
    key.update(style=style, difficulty=int(difficulty), mirrored=bool(mirror), beat_times=[round(float(b), 4) for b in beat_times], fps=float(fps), frames=int(N), pose_frames=int(N))
    if save: save_key(key)
    return key


def load_aist_keypoints(path, optim=True):
    """AIST++ keypoints3d pickle -> (N, 17, 3) at 60 fps (the dataset's rate)."""
    with open(path, 'rb') as f: d = pickle.load(f)
    k = d.get('keypoints3d_optim' if optim and 'keypoints3d_optim' in d else 'keypoints3d')
    return np.asarray(k, np.float32), 60.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument('--npy'); g.add_argument('--aist')
    ap.add_argument('--fps', type=float, default=60.0); ap.add_argument('--bpm', type=float, required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--3d', dest='is_3d', action='store_true'); ap.add_argument('--start', type=float, default=0.0); ap.add_argument('--end', type=float)
    ap.add_argument('--counts', type=int, default=8); ap.add_argument('--mirror', action='store_true'); ap.add_argument('--source', default=''); ap.add_argument('--licence', default='')
    ap.add_argument('--style', default=''); ap.add_argument('--difficulty', type=int, default=2)
    a = ap.parse_args(argv)
    if a.aist: seq, fps = load_aist_keypoints(a.aist); is_3d = True
    else: seq = np.load(a.npy); fps = a.fps; is_3d = a.is_3d or seq.shape[-1] == 3
    key = key_from_coco_sequence(seq, fps, a.name, bpm=a.bpm, is_3d=is_3d, start=a.start, end=a.end, counts_per_phrase=a.counts, mirror=a.mirror,
                                 source=a.source or (a.aist or a.npy), licence=a.licence, style=a.style, difficulty=a.difficulty)
    print(f"key {key['name']}: {key['n_beats']} beats at {key['bpm']:.1f} bpm, {len(key['phrases'])} phrases -> {os.path.join(LIBRARY_DIR, key['name'] + '.json')}")


if __name__ == '__main__':
    sys.exit(main())
