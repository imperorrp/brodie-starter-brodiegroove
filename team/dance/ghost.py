"""Your ghost: what you actually danced in a battle, kept as body points so a friend with your code battles *you*, not the
reference routine. Only landmark-derived arrays are kept (angles, visibility, hip energy, the hip-centred skeleton), never video,
and only when the winner says yes to saving it.

  live_to_arrays(engine) -> dict(F, M, E, S) on the beat grid, or None when too little was seen
  ghost_key(code, owner, rounds, bpm, per_beat) -> a key like any routine (one phrase per recorded round), titled "<owner>'s ghost"
  pack(rounds) / unpack(data): compact JSON for the backend (about 40 KB for two 8-count rounds)
"""
import numpy as np

from .features import resample_to_grid, resample_mask
from .reference import key_from_arrays

MIN_VISIBLE = 0.6


def live_to_arrays(eng, per_beat=4):
    """The engine's attempt (features, mask, energy and skeleton samples, timed from the attempt start) resampled to per_beat
    samples per count. None when fewer than two samples per count arrived or under 60% of the joints were seen."""
    n = int(getattr(eng, 'n_beats', 0) or 0); beat_s = float(getattr(eng, 'beat_s', 0) or 0)
    live = list(getattr(eng, 'live', []) or []); skel = list(getattr(eng, 'live_skel', []) or [])
    if n <= 0 or beat_s <= 0 or len(live) < n * 2 or len(skel) < n * 2: return None
    grid = np.arange(n * per_beat) * beat_s / per_beat
    ts = [r[0] for r in live]
    F = resample_to_grid(ts, [r[1] for r in live], grid)
    M = resample_mask(ts, [np.asarray(r[2], np.float32) for r in live], grid)
    E = resample_to_grid(ts, np.asarray([r[3] for r in live], np.float32), grid)[:, 0]
    st = [r[0] for r in skel]
    S = resample_to_grid(st, np.asarray([r[1] for r in skel], np.float32).reshape(len(skel), -1), grid).reshape(len(grid), 33, 2)
    if float(np.mean(M)) < MIN_VISIBLE: return None
    return dict(F=F, M=M, E=E, S=S)


def ghost_key(code, owner, rounds, bpm, per_beat=4):
    """rounds: [dict(F, M, E, S)] of equal length; one phrase per round."""
    F = np.concatenate([r['F'] for r in rounds]); M = np.concatenate([np.asarray(r['M'], bool) for r in rounds])
    E = np.concatenate([r['E'] for r in rounds]); S = np.concatenate([r['S'] for r in rounds])
    n_beats = len(rounds[0]['F']) // per_beat
    key = key_from_arrays(f'ghost_{code}', bpm, F, M=M, E=E, S=S, per_beat=per_beat, counts_per_phrase=n_beats, source='ghost', save=False)
    key['title'] = f"{owner or 'your friend'}'s ghost"; key['ghost_of'] = owner; key['generated'] = True
    return key


def valid(data, bpm, per_beat):
    """The backend's shape and size check, run again before a fetched ghost is unpacked and drawn (team/backend/db.valid_ghost)."""
    import importlib.util, os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'db.py')
    spec = importlib.util.spec_from_file_location('team_db_validate', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.valid_ghost(data, bpm, per_beat)


def pack(rounds):
    return [dict(F=np.round(r['F'], 1).tolist(), M=np.asarray(r['M'], bool).astype(int).tolist(), E=np.round(r['E'], 3).tolist(),
                 S=np.round(r['S'], 3).tolist()) for r in rounds]


def unpack(data):
    return [dict(F=np.asarray(r['F'], np.float32), M=np.asarray(r['M'], bool), E=np.asarray(r['E'], np.float32), S=np.asarray(r['S'], np.float32))
            for r in data]
