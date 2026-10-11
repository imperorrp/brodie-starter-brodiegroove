"""Per-session telemetry: tool latency and errors, engine events, tracking quality, cue latency, and a summary row.

One JSONL file per session under data/tmp/sessions/. The caller's name is never written in clear (sha1 prefix). The summary is
also POSTed to the backend's /events as type 'session_summary' so GET /metrics can aggregate across sessions.
"""
import hashlib, json, os, time
from contextlib import contextmanager
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESS_DIR = os.path.join(ROOT, 'data', 'tmp', 'sessions')


def name_hash(name):
    return hashlib.sha1(str(name).strip().lower().encode()).hexdigest()[:8] if name else None


class Telemetry:
    def __init__(self, session_id=None, now=time.monotonic, write=True):
        self.id = session_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid() % 10000:04d}"
        self.now = now; self.t0 = now(); self.rows = []; self.write = write; self.path = None; self._f = None
        self.frames = 0; self.frames_pose = 0; self.frames_full = 0; self.frames_upper = 0
        self.cue_lat = []; self.tool_calls = 0; self.tool_errors = 0; self.tool_ms = []; self.scores = []; self.errors = []
        self.mode = None; self.caller = None; self.closed = False; self.extra = {}
        if write:
            try:
                os.makedirs(SESS_DIR, exist_ok=True); self.path = os.path.join(SESS_DIR, f'{self.id}.jsonl')
                self._f = open(self.path, 'a', encoding='utf-8')
            except Exception as e:
                print(f'[telemetry] no session file ({e!r})', flush=True); self._f = None

    def _t(self):
        return round(self.now() - self.t0, 3)

    def event(self, type_, **f):
        row = dict(t=self._t(), type=type_); row.update({k: v for k, v in f.items() if k not in ('t', 'type')})
        self.rows.append(row)
        if self._f is not None:
            try: self._f.write(json.dumps(row, default=_jsonable) + '\n'); self._f.flush()
            except Exception: pass
        return row

    # ---- counters ----
    def frame(self, pose_ok, full_body=False, upper=False):
        self.frames += 1
        if pose_ok: self.frames_pose += 1
        if full_body: self.frames_full += 1
        if upper: self.frames_upper += 1

    def cue(self, t_detect, t_shown, kind, text=''):
        ms = max(0.0, (t_shown - t_detect) * 1000.0); self.cue_lat.append(ms)
        self.event('cue', kind=kind, text=text, latency_ms=round(ms, 1))

    def score(self, result, **f):
        r = result or {}
        if r.get('scored'): self.scores.append(float(r.get('score') or 0))
        self.event('phrase_scored', score=r.get('score'), scored=r.get('scored'), spatial=r.get('spatial'), timing=r.get('timing'),
                   energy=r.get('energy'), visible=r.get('visible_fraction'), **f)

    @contextmanager
    def tool(self, name, **args):
        t = time.perf_counter(); ok = True; err = None
        try:
            yield
        except Exception as e:
            ok = False; err = repr(e); raise
        finally:
            self.tool_result(name, (time.perf_counter() - t) * 1000.0, ok, err, **args)

    def tool_result(self, name, ms, ok=True, error=None, **args):
        self.tool_calls += 1; self.tool_ms.append(ms)
        if not ok: self.tool_errors += 1
        self.event('tool', name=name, ms=round(ms, 1), ok=bool(ok), error=error, **{k: v for k, v in args.items() if k not in ('name', 'ms', 'ok', 'error')})

    def error(self, where, exc):
        self.errors.append(f'{where}: {exc!r}'); self.event('error', where=where, error=repr(exc))

    # ---- summary ----
    def summary(self, completed=True):
        lat = np.asarray(self.cue_lat, np.float64)
        return dict(session=self.id, caller=name_hash(self.caller), mode=self.mode, duration_s=self._t(), phrases=len(self.scores),
                    mean_score=round(float(np.mean(self.scores)), 1) if self.scores else None, scores=[round(s, 1) for s in self.scores],
                    tracking_rate=round(self.frames_pose / self.frames, 3) if self.frames else None,
                    fullbody_rate=round(self.frames_full / self.frames, 3) if self.frames else None,
                    median_cue_latency_ms=round(float(np.median(lat)), 1) if lat.size else None,
                    p90_cue_latency_ms=round(float(np.percentile(lat, 90)), 1) if lat.size else None,
                    tool_calls=self.tool_calls, tool_errors=self.tool_errors,
                    median_tool_ms=round(float(np.median(self.tool_ms)), 1) if self.tool_ms else None,
                    completed=bool(completed and not self.errors), errors=self.errors[:5], **self.extra)

    def close(self, completed=True, post=None):
        """Write the summary row; post(summary) is an optional callable (the backend client) for /events."""
        if self.closed: return None
        self.closed = True; s = self.summary(completed)
        self.event('session_summary', **s)
        if self._f is not None:
            try: self._f.close()
            except Exception: pass
        if post is not None:
            try: post(dict(type='session_summary', t_wall=round(time.time(), 3), **s))
            except Exception as e: print(f'[telemetry] summary post failed: {e!r}', flush=True)
        return s


def _jsonable(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
