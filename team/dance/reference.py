"""Build a reference key from a video (or load one).

A key is one move: 8 or 16 counts at a BPM, as feature samples on a beat grid (PER_BEAT samples per beat), with a per-sample
visibility mask, a hip-energy sample, and a normalised 2D skeleton per sample for the ghost. Keys are JSON files in
team/dance/library/<name>.json with provenance (source, licence) so the README can list them.

Video -> key: MediaPipe pose offline on every frame (rtcoach.pose.Pose, the same wrapper the live pipeline uses), a beat grid
from the audio (librosa beat tracking) or from a given BPM, optional mirroring (a dancer facing the camera is mirrored so the
caller who faces the screen matches like a mirror), resampling to the grid, cutting into phrases.
"""
import json, os, subprocess, tempfile
import numpy as np
import cv2
from . import LIBRARY_DIR, MODELS_DIR
from .features import (features_from_landmarks, visibility_mask, mirror_landmarks, hip_energy, skeleton2d_normalized,
                       resample_to_grid, resample_mask, N_CHANNELS)
from .scoring import PER_BEAT

KEY_VERSION = 1


def extract_landmarks(video_path, variant='full', short_side=480, max_frames=None, progress=None):
    """Run MediaPipe pose over a video. Returns dict(times, img, world, fps, size): times in seconds, img a list of (33,5) or None,
    world a list of (33,3) or None. Frames are resized to `short_side` for speed (landmarks are normalised, so no effect on
    features)."""
    from rtcoach.pose import Pose
    model = os.path.join(MODELS_DIR, f'pose_landmarker_{variant}.task')
    if not os.path.exists(model): raise FileNotFoundError(f'pose model missing: {model} (README step 1)')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f'cannot open {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose = Pose(variant, video=True); times = []; img = []; world = []; i = 0; last_ms = -1
    try:
        while True:
            ok, bgr = cap.read()
            if not ok: break
            if max_frames and i >= max_frames: break
            s = min(bgr.shape[:2])
            if s > short_side:
                f = short_side / s; bgr = cv2.resize(bgr, (max(1, round(bgr.shape[1] * f)), max(1, round(bgr.shape[0] * f))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = i / fps; ms = int(round(t * 1000))
            if ms <= last_ms: ms = last_ms + 1
            last_ms = ms
            a, wld = pose(rgb, ms)
            times.append(t); img.append(a); world.append(wld); i += 1
            if progress and i % 100 == 0: progress(i)
    finally:
        cap.release(); pose.close()
    return dict(times=np.array(times), img=img, world=world, fps=fps, size=(w, h), n=i)


def extract_audio(video_path, sr=22050):
    """ffmpeg -> mono wav at sr in a temp file; returns the path or None when the video has no audio."""
    out = os.path.join(tempfile.gettempdir(), 'dance_audio_%d.wav' % (abs(hash(video_path)) % 10 ** 8))
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', video_path, '-vn', '-ac', '1', '-ar', str(sr), out]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return out if os.path.exists(out) and os.path.getsize(out) > 1000 else None


def beat_grid(video_path=None, bpm=None, start=0.0, end=None, duration=None):
    """Beat times in seconds. With bpm: a uniform grid from `start`. Without: librosa beat tracking on the video's audio
    (falls back to 100 bpm from `start` when there is no audio or no librosa). Returns (beat_times, bpm_estimate)."""
    if bpm:
        stop = end if end is not None else (duration if duration is not None else start + 60.0)
        n = int((stop - start) * bpm / 60.0) + 1
        return start + np.arange(n) * 60.0 / bpm, float(bpm)
    wav = extract_audio(video_path) if video_path else None
    if wav is not None:
        try:
            import librosa
            y, sr = librosa.load(wav, sr=None, mono=True)
            tempo, frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
            times = librosa.frames_to_time(frames, sr=sr)
            tempo = float(np.atleast_1d(tempo)[0])
            times = times[(times >= start) & ((times <= end) if end is not None else True)]
            if len(times) >= 4: return np.asarray(times, np.float64), tempo
        except Exception as e:
            print(f'[reference] librosa beat tracking failed ({e!r}); using a fixed grid', flush=True)
    stop = end if end is not None else (duration if duration is not None else start + 60.0)
    n = int((stop - start) * 100.0 / 60.0) + 1
    return start + np.arange(n) * 0.6, 100.0


def build_key(video_path, name, bpm=None, start=0.0, end=None, counts_per_phrase=8, per_beat=PER_BEAT, mirror=True,
              variant='full', source='', licence='', style='', difficulty=2, save=True, landmarks=None, progress=None):
    """Video -> key dict (and JSON in the library). `landmarks` can be a precomputed extract_landmarks() result."""
    L = landmarks or extract_landmarks(video_path, variant=variant, progress=progress)
    times = L['times']; duration = float(times[-1]) if len(times) else 0.0
    beats, bpm_est = beat_grid(video_path, bpm=bpm, start=start, end=end, duration=duration)
    beats = beats[beats <= duration + 1e-6]
    if len(beats) < 2: raise RuntimeError('fewer than two beats inside the clip; give --bpm or a longer clip')
    feats = []; masks = []; energies = []; skels = []; ok_t = []; prev = None; prev_t = None
    for t, a, w in zip(times, L['img'], L['world']):
        if a is None: continue
        if mirror: a, w = mirror_landmarks(a, w)
        feats.append(features_from_landmarks(a, w)); masks.append(visibility_mask(a).astype(np.float32))
        energies.append(hip_energy(prev, a, (t - prev_t) if prev_t is not None else 0.0)); skels.append(skeleton2d_normalized(a))
        ok_t.append(t); prev = a; prev_t = t
    if len(ok_t) < 4: raise RuntimeError('no pose found in the clip')
    # the sample grid: per_beat samples per beat interval, from each beat to the next
    grid = []
    for k in range(len(beats) - 1):
        grid.extend(beats[k] + (beats[k + 1] - beats[k]) * np.arange(per_beat) / per_beat)
    grid = np.asarray(grid, np.float64)
    F = resample_to_grid(ok_t, feats, grid); M = resample_mask(ok_t, masks, grid)
    E = resample_to_grid(ok_t, np.asarray(energies, np.float32), grid)[:, 0]
    S = resample_to_grid(ok_t, np.asarray(skels, np.float32).reshape(len(skels), -1), grid).reshape(len(grid), 33, 2)
    n_beats = len(beats) - 1
    phrases = [dict(start_beat=s, beats=min(counts_per_phrase, n_beats - s)) for s in range(0, n_beats, counts_per_phrase)]
    phrases = [p for p in phrases if p['beats'] >= max(2, counts_per_phrase // 2)]
    key = dict(version=KEY_VERSION, name=name, bpm=round(float(bpm_est), 2), per_beat=per_beat, n_beats=n_beats, counts_per_phrase=counts_per_phrase,
               beat_times=[round(float(b), 4) for b in beats], phrases=phrases, mirrored=bool(mirror), style=style, difficulty=int(difficulty),
               source=source or os.path.basename(str(video_path)), licence=licence, video=os.path.abspath(video_path) if video_path else '',
               features=F.tolist(), mask=M.astype(int).tolist(), energy=E.tolist(), skeleton=S.tolist(),
               pose_frames=len(ok_t), frames=int(L['n']), fps=L['fps'])
    if save: save_key(key)
    return key


def key_from_arrays(name, bpm, F, M=None, E=None, S=None, per_beat=PER_BEAT, counts_per_phrase=8, source='synthetic', licence='', save=False):
    """A key straight from sample arrays (tests, motion-file ingesters)."""
    F = np.asarray(F, np.float32); n = len(F); n_beats = n // per_beat
    M = np.ones_like(F, bool) if M is None else np.asarray(M, bool)
    E = np.zeros(n, np.float32) if E is None else np.asarray(E, np.float32)
    S = np.zeros((n, 33, 2), np.float32) if S is None else np.asarray(S, np.float32)
    beats = np.arange(n_beats + 1) * 60.0 / bpm
    phrases = [dict(start_beat=s, beats=min(counts_per_phrase, n_beats - s)) for s in range(0, n_beats, counts_per_phrase)]
    key = dict(version=KEY_VERSION, name=name, bpm=float(bpm), per_beat=per_beat, n_beats=n_beats, counts_per_phrase=counts_per_phrase,
               beat_times=beats.tolist(), phrases=phrases, mirrored=False, style='', difficulty=2, source=source, licence=licence, video='',
               features=F.tolist(), mask=M.astype(int).tolist(), energy=E.tolist(), skeleton=S.tolist(), pose_frames=n, frames=n, fps=0.0)
    if save: save_key(key)
    return key


def phrase_slice(key, phrase_index=0):
    """(features, mask, energy, skeleton) arrays for one phrase of the key."""
    p = key['phrases'][phrase_index]; pb = key['per_beat']; a = p['start_beat'] * pb; b = a + p['beats'] * pb
    return (np.asarray(key['features'], np.float32)[a:b], np.asarray(key['mask'], bool)[a:b],
            np.asarray(key['energy'], np.float32)[a:b], np.asarray(key['skeleton'], np.float32)[a:b])


def key_path(name):
    return os.path.join(LIBRARY_DIR, f'{name}.json')


def save_key(key):
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    with open(key_path(key['name']), 'w', encoding='utf-8') as f: json.dump(key, f)
    return key_path(key['name'])


def load_key(name):
    p = name if name.endswith('.json') else key_path(name)
    with open(p, encoding='utf-8') as f: return json.load(f)


def list_keys():
    if not os.path.isdir(LIBRARY_DIR): return []
    out = []
    for fn in sorted(os.listdir(LIBRARY_DIR)):
        if fn.endswith('.json'):
            try:
                k = load_key(os.path.join(LIBRARY_DIR, fn))
                out.append(dict(name=k['name'], bpm=k['bpm'], n_beats=k['n_beats'], phrases=len(k['phrases']), style=k.get('style', ''),
                                difficulty=k.get('difficulty', 2), source=k.get('source', ''), licence=k.get('licence', '')))
            except Exception as e:
                out.append(dict(name=fn, error=repr(e)))
    return out
