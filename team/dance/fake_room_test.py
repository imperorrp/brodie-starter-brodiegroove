"""Run the REAL worker job (facetime.worker.CoachJob) in dance mode against a fake room, with a video file standing in for the
caller's camera track: no LiveKit server, no keys, no phone. This exercises the actual plumbing the live call uses: the handlers
module loaded by path, the consumer registration, job.latest frames as rtc.VideoFrame objects, frame conversion, the video
track publish call, VideoSource.capture_frame, the cue player with a real rtc.AudioSource, and the session's learn flow.

  .venv/Scripts/python -m team.dance.fake_room_test [--video data/tmp/mac_handoff/segments/live_squats.mp4] [--key test_squats]
      [--seconds 70] [--out data/tmp/dance_fake_room.mp4]

Pass/fail is printed at the end; the composed track is written to --out for inspection.
"""
import argparse, asyncio, json, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, 'src')):
    if p not in sys.path: sys.path.insert(0, p)

from livekit import rtc                                            # noqa: E402
from facetime import worker                                       # noqa: E402
from facetime.perception import Config                            # noqa: E402

PASS = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True); PASS.append(bool(cond)); return cond


class FakeLocal:
    identity = 'brodie-coach'
    def __init__(self): self.published = []
    async def publish_track(self, track, opts):
        self.published.append(getattr(track, 'name', '?')); return None
    async def publish_data(self, *a, **k): return None


class FakeRoom:
    def __init__(self):
        self.handlers = {}; self.name = 'dance-fake-room'; self.local_participant = FakeLocal(); self.remote_participants = {}
        self.connection_state = rtc.ConnectionState.CONN_CONNECTED
    def on(self, name, fn=None):
        if fn is None: return lambda f: self.on(name, f)
        self.handlers.setdefault(name, []).append(fn); return fn
    def emit(self, name, *args):
        for fn in self.handlers.get(name, []): fn(*args)


class FakeJobInfo:
    id = 'job-dance-fake'; metadata = json.dumps({'chert_call': 'fake'})


class FakeCtx:
    def __init__(self, room):
        self.room = room; self.job = FakeJobInfo(); self.callbacks = []; self.shutdowns = []; self.connected = False
    def add_shutdown_callback(self, fn): self.callbacks.append(fn)
    async def connect(self, auto_subscribe=None): await asyncio.sleep(0.05); self.connected = True
    def shutdown(self, reason='user requested'): self.shutdowns.append(reason)


class P:
    def __init__(self, identity): self.identity = identity; self.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD


class VideoFileStream:
    """Stands in for rtc.VideoStream: yields rtc.VideoFrame objects from an mp4 at its own frame rate, looping."""
    def __init__(self, path, fps=None):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise SystemExit(f'cannot open {path}')
        self.fps = fps or float(cap.get(cv2.CAP_PROP_FPS) or 10.0); self.frames = []
        while True:
            ok, bgr = cap.read()
            if not ok: break
            self.frames.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        cap.release(); self.closed = False; self.n = 0
    def __aiter__(self): return self
    async def __anext__(self):
        if self.closed: raise StopAsyncIteration
        await asyncio.sleep(1.0 / self.fps); self.n += 1
        if self.closed: raise StopAsyncIteration
        rgb = self.frames[self.n % len(self.frames)]; h, w = rgb.shape[:2]
        frame = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGB24, rgb.tobytes())
        class Ev: pass
        ev = Ev(); ev.frame = frame; ev.timestamp_us = int(time.time() * 1e6); return ev
    async def aclose(self): self.closed = True


class FakeTrack:
    def __init__(self, kind): self.kind = kind


class FakePub:
    def __init__(self, name, w, h): self.name = name; self.width = w; self.height = h; self.simulcasted = False


async def main(a):
    os.environ['DANCE_KEY'] = a.key; os.environ['DANCE_MODE'] = 'learn'; os.environ['DANCE_MAX_PHRASES'] = '1'
    os.environ['DANCE_DEBUG_MP4'] = a.out; os.environ['DANCE_RTT'] = '0.0'; os.environ['DANCE_FPS'] = str(a.fps)
    os.environ['DANCE_POSE'] = a.pose
    args = worker.load_args(); args.perception = 'off'; args.voice = 'off'; args.vision = False; args.floor = None
    args.handlers = os.path.join(HERE, 'live.py'); args.log_caller = True; args.leave_grace_s = 0.5
    args.event_log = os.path.join(ROOT, 'data', 'tmp', 'dance_fake_room_events.jsonl'); args.cue_log = os.path.join(ROOT, 'data', 'tmp', 'dance_fake_room_cues.jsonl')
    worker.resolve_mode(args); cfg = Config.from_args(args)
    from facetime.audio_out import CueBank
    bank = CueBank()
    ctx = FakeCtx(FakeRoom()); job = worker.CoachJob(ctx, args, cfg, None, bank, worker.parse_metadata(ctx.job.metadata))
    stream = VideoFileStream(a.video); job.video_frames = lambda track: stream
    print(f'[fake-room] video {a.video}: {len(stream.frames)} frames at {stream.fps:g} fps; key {a.key}; running {a.seconds} s', flush=True)
    t = asyncio.create_task(job.run()); await asyncio.sleep(0.5)
    check(job.voice is not None and job.voice.handlers is not None, 'handlers module loaded by path (voice layer with handlers, no model)')
    check(len(worker.CONSUMERS) >= 1 and any(fn.__name__ == 'dance_consumer' for fn in worker.CONSUMERS), f'dance consumer registered: {[fn.__name__ for fn in worker.CONSUMERS]}')
    w, h = stream.frames[0].shape[1], stream.frames[0].shape[0]
    job.on_track_subscribed(FakeTrack(rtc.TrackKind.KIND_VIDEO), FakePub('facetime-remote-video', w, h), P('chert-bridge-dance1'))
    await asyncio.sleep(3.0)
    check(job.bound == 'chert-bridge-dance1' and job.seq >= 10, f'bound on the video track, {job.seq} frames received')
    check('coach-video' in ctx.room.local_participant.published, f'video track published: {ctx.room.local_participant.published}')
    sess = None
    for _ in range(int(a.seconds)):
        await asyncio.sleep(1.0)
        import team.dance.live as live_mod
        from importlib import import_module
        sess = sess or getattr(sys.modules.get('starter_handlers'), 'STATE', {}).get('session')
        if sess is not None and getattr(sess, 'rounds', None) and len(sess.rounds) >= 1 and sess.engine is None and sess.phrase >= 1: break
    hmod = sys.modules.get('starter_handlers')
    st = getattr(hmod, 'STATE', {}) if hmod else {}
    cues = [json.loads(l) for l in open(args.cue_log)] if os.path.exists(args.cue_log) else []
    played = [c for c in cues if c.get('status') == 'played']
    check(any(c.get('text') == '[dance:track]' for c in cues), f'metronome track went through the cue player ({len(played)} clips played, {len(cues)} records)')
    job.on_participant_disconnected(P('chert-bridge-dance1'))
    try: await asyncio.wait_for(t, 15); exc = None
    except Exception as e: exc = e; t.cancel()
    check(exc is None and job.closed, f'job ended cleanly: {job.end_reason!r}, shutdowns {ctx.shutdowns}')
    log_path = os.path.join(ROOT, 'data', 'tmp', 'dance_session_log.json')
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    rounds = log.get('rounds', [])
    check(len(rounds) >= 1 and rounds[0].get('result') is not None, f'session scored {len(rounds)} phrase(s): ' + ', '.join(f"{r['result'].get('score')}" for r in rounds if r.get('result')))
    import glob                                      # live runs add a time suffix (one file per call): <out>_HHMMSS.mp4
    base, ext = os.path.splitext(a.out); outs = sorted(glob.glob(f'{base}_*{ext or ".mp4"}') + ([a.out] if os.path.exists(a.out) else []), key=os.path.getmtime)
    out = outs[-1] if outs else a.out
    check(os.path.exists(out) and os.path.getsize(out) > 100_000, f'composed track written: {out} ({os.path.getsize(out) if os.path.exists(out) else 0} bytes)')
    ok = all(PASS); print('fake-room test', 'PASSED' if ok else 'FAILED', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--video', default=os.path.join(ROOT, 'data', 'tmp', 'mac_handoff', 'segments', 'live_squats.mp4'))
    ap.add_argument('--key', default='test_squats'); ap.add_argument('--seconds', type=float, default=75); ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'tmp', 'dance_fake_room.mp4'))
    ap.add_argument('--fps', type=float, default=12); ap.add_argument('--pose', default='full')
    sys.exit(asyncio.run(main(ap.parse_args())))
