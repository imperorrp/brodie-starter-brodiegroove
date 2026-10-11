import numpy as np
from team.dance.features import FEET_CHANNELS
from team.dance.ingest import coco_to_mediapipe, key_from_coco_sequence
from team.dance.features import features_from_landmarks, visibility_mask
from team.dance.reference import phrase_slice
from team.dance.scoring import score_phrase


def _walking_coco(n=240, fps=30.0, is_3d=False):
    """A synthetic COCO-17 sequence: an upright figure whose arms swing and knees bend on a 100 bpm beat."""
    t = np.arange(n) / fps; beat = 2 * np.pi * t * 100 / 60
    base = {0: (0.5, 0.15), 5: (0.42, 0.3), 6: (0.58, 0.3), 7: (0.38, 0.45), 8: (0.62, 0.45), 9: (0.36, 0.6), 10: (0.64, 0.6),
            11: (0.45, 0.55), 12: (0.55, 0.55), 13: (0.45, 0.75), 14: (0.55, 0.75), 15: (0.45, 0.95), 16: (0.55, 0.95),
            1: (0.48, 0.13), 2: (0.52, 0.13), 3: (0.46, 0.14), 4: (0.54, 0.14)}
    seq = np.zeros((n, 17, 3 if is_3d else 2), np.float32)
    for j, (x, y) in base.items():
        seq[:, j, 0] = x; seq[:, j, 1] = y
    seq[:, 9, 1] -= 0.25 * (1 + np.sin(beat)) / 2; seq[:, 10, 1] -= 0.25 * (1 + np.sin(beat + np.pi)) / 2     # wrists swing up alternately
    seq[:, 13, 0] -= 0.05 * (1 + np.sin(beat)) / 2; seq[:, 14, 0] += 0.05 * (1 + np.sin(beat)) / 2          # knees out on the beat
    if is_3d:
        seq[:, :, 1] = 1.0 - seq[:, :, 1]                           # y up, like a mocap frame
    return seq, fps


def test_coco_mapping_feeds_features():
    seq, fps = _walking_coco()
    img, world = coco_to_mediapipe(seq[0])
    assert img.shape == (33, 5) and world is None and np.delete(visibility_mask(img), FEET_CHANNELS).all()
    f = features_from_landmarks(img); assert f.shape == (14,) and np.isfinite(f).all()


def test_key_from_coco_2d_and_3d_score_themselves_high():
    for is_3d in (False, True):
        seq, fps = _walking_coco(is_3d=is_3d)
        key = key_from_coco_sequence(seq, fps, 'synthetic_coco', bpm=100, is_3d=is_3d, save=False)
        kF, kM, kE, kS = phrase_slice(key, 0); pb = key['per_beat']
        assert kF.shape[0] == 8 * pb and kS.shape == (8 * pb, 33, 2) and key['bpm'] == 100.0
        live = np.concatenate([kF, np.repeat(kF[-1:], pb, axis=0)])
        r = score_phrase(live, None, np.concatenate([kE, np.repeat(kE[-1:], pb)]), kF, kM, kE, pb)
        assert r['scored'] and r['score'] >= 95, (is_3d, r['score'])
