"""Tier (b): headless stand-in for the Chert bridge, for a LiveKit loopback test on one machine.

Joins the room as BRIDGE_IDENTITY (default "chert-bridge"), publishes an mp4 as a video track named
"facetime-remote-video" through rtc.VideoSource at wall-clock rate (frame k goes out at t0 + k/fps; --loop
repeats), subscribes to the agent's audio track and writes it to --out (24 kHz mono wav), logging the
wall-clock time of every audio onset (RMS above --onset-dbfs after at least --min-silence s of silence).
At the end prints the onsets and the audio seconds received; --onsets writes them as JSON for
facetime/latency_report.py.

Frames are streamed from the clip one at a time (cv2.VideoCapture; memory stays at one frame whatever the clip
length or the output size), so the bridge can stand in for a phone: --scale WxH resizes each frame on the fly
(cv2.resize) and --fps is converted against the clip's native rate (a source frame is repeated, nearest, when
--fps is higher and skipped when lower, so the clip keeps its duration). The --stamp pattern is drawn after the
resize, so it is published at the output size and the agent reads it at whatever size it receives. E.g.
--scale 720x1280 --fps 30 --no-simulcast turns the 288x512 10 fps test clip into roughly what a phone sends
through the Chert bridge; --max-bitrate raises the encoder cap for the larger picture.

Run:  .venv/bin/python -m facetime.fake_bridge --room rtcoach-test-abc --video data/tmp/0013_squats.mp4 \
        --out data/tmp/bridge_rx.wav [--fps 10] [--scale 720x1280] [--loop] [--tail 3] [--wait-agent 30] [--onsets data/tmp/onsets.json]
"""
import argparse, asyncio, json, math, os, signal, time, wave
import numpy as np
import cv2
from livekit import rtc
from facetime import DATA, BRIDGE_IDENTITY, AGENT_IDENTITY, VIDEO_TRACK_NAME, SAMPLE_RATE
from facetime.mint_token import mint

def probe(path):
    """(width, height, fps, frame count) of the clip's video stream as cv2 decodes it (rotation metadata applied)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open video {path!r}')
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(cap.get(cv2.CAP_PROP_FPS) or 0.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    finally:
        cap.release()

def parse_scale(s):
    """--scale WxH -> (W, H); 'native' or '' -> None."""
    if not s or s.lower() == 'native':
        return None
    try:
        w, h = (int(v) for v in s.lower().split('x'))
        if w <= 0 or h <= 0: raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f'--scale wants WxH, e.g. 720x1280, got {s!r}')
    return w, h

class Frames:
    """Streams the clip for the publisher, one decoded frame in memory at a time.

    Iterating yields (H, W, 3) RGB uint8 frames at `fps`, each resized to `scale` (W, H) when given. Frame-rate
    conversion is nearest: output frame k shows the source frame on screen at k/fps, i.e. source index
    floor(k * src_fps / fps), so a source frame is repeated when fps > src_fps and skipped when fps < src_fps and the
    clip keeps its duration. `loop` restarts the clip at its end (the iterator never stops). `frames`, an (N, H, W, 3)
    RGB array such as loopback_inproc preloads with rtcoach.video.read_frames, replaces the decoder; the clip is still
    probed for its native rate. `n` is the output frame count of one pass, `duration` the clip length in seconds."""
    def __init__(self, path, fps, scale=None, loop=False, frames=None):
        self.path, self.loop, self.frames = path, loop, frames
        w, h, src_fps, n = probe(path)
        if frames is not None:
            n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
        self.src_fps = src_fps if src_fps > 0 else float(fps)        # unreadable rate: no conversion
        self.fps = float(fps) if fps else self.src_fps
        self.src_n = n; self.src_size = (w, h)
        self.size = tuple(scale) if scale else (w, h)
        self.n = int(math.ceil(round(n * self.fps / self.src_fps, 6))) if n else 0
        self.duration = n / self.src_fps if n else 0.0

    @property
    def width(self): return self.size[0]
    @property
    def height(self): return self.size[1]

    def describe(self):
        conv = '' if self.fps == self.src_fps else f', {"repeating" if self.fps > self.src_fps else "skipping"} frames {self.src_fps:g} -> {self.fps:g} fps'
        return (f'{self.src_n} frames {self.src_size[0]}x{self.src_size[1]} at {self.src_fps:g} fps ({self.duration:.1f} s) -> '
                f'publishing {self.width}x{self.height} at {self.fps:g} fps ({self.n} frames per pass, streamed{conv})'
                f'{" looping" if self.loop else ""}')

    def _decode(self):
        """One pass over the source frames, RGB."""
        if self.frames is not None:
            yield from self.frames; return
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f'cannot open video {self.path!r}')
        try:
            while True:
                ok, bgr = cap.read()
                if not ok: break
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def __iter__(self):
        resize = self.size != self.src_size
        while True:
            k = 0                                                     # output frames yielded so far in this pass
            for i, rgb in enumerate(self._decode()):
                k_end = int(math.ceil(round((i + 1) * self.fps / self.src_fps, 6)))   # output frames up to the end of source frame i
                if k_end <= k: continue                               # fps below the source rate: this frame is skipped
                if resize: rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_LINEAR)
                for _ in range(k_end - k): yield rgb
                k = k_end
            if not self.loop: return

class Bridge:
    def __init__(self, a, loop):
        self.a = a; self.loop = loop; self.room = rtc.Room(loop=loop)
        self.audio = []; self.onsets = []; self.n_samples = 0; self.silent_s = 1e9; self.loud_s = 0.0
        self.thr = 32767 * 10 ** (a.onset_dbfs / 20); self.off = self.thr / 3
        self.audio_task = None; self.stopping = asyncio.Event(); self.t_sub = None; self.frames = None

    def on_track_subscribed(self, track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO and self.audio_task is None:
            print(f'[bridge] subscribed to audio {pub.name!r} from {participant.identity!r}', flush=True)
            self.audio_task = asyncio.create_task(self.record(track))

    async def record(self, track):
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1, frame_size_ms=20)
        self.t_sub = time.time()
        try:
            async for ev in stream:
                x = np.frombuffer(ev.frame.data.cast('B'), np.int16); t = time.time()
                self.audio.append(x.copy()); dur = len(x) / SAMPLE_RATE
                rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2))) if len(x) else 0.0
                if rms > self.thr:
                    if self.silent_s >= self.a.min_silence:
                        self.onsets.append(dict(t=t, at_s=round(self.n_samples / SAMPLE_RATE, 3), rms=round(rms)))
                        print(f'[bridge] audio onset #{len(self.onsets)} at {time.strftime("%H:%M:%S")}.{int(t * 1000) % 1000:03d} (rx +{self.n_samples / SAMPLE_RATE:.2f} s, rms {rms:.0f})', flush=True)
                    self.silent_s = 0.0; self.loud_s += dur
                elif rms < self.off:
                    self.silent_s += dur
                self.n_samples += len(x)
        finally:
            await stream.aclose()

    async def wait_for_agent(self):
        if not self.a.wait_agent: return
        t0 = time.time()
        while AGENT_IDENTITY not in self.room.remote_participants:
            if time.time() - t0 > self.a.wait_agent:
                print(f'[bridge] agent {AGENT_IDENTITY} not in room after {self.a.wait_agent} s, sending anyway', flush=True); return
            await asyncio.sleep(0.1)
        print(f'[bridge] agent present after {time.time() - t0:.1f} s', flush=True)

    def make_frames(self, frames=None):
        a = self.a
        self.frames = Frames(a.video, a.fps, scale=getattr(a, 'scale', None), loop=a.loop, frames=frames)
        return self.frames

    async def send(self, source, fr):
        """Push frames at wall-clock rate: frame k at t0 + k/fps. `fr` is a Frames iterator or, as loopback_inproc passes it,
        the preloaded (N, H, W, 3) array (then wrapped, so --scale and the fps conversion apply too). Returns (t0, t_end, frames sent)."""
        a = self.a
        if not isinstance(fr, Frames):
            fr = self.make_frames(fr); print(f'[bridge] {a.video}: {fr.describe()}', flush=True)
        W, H = fr.width, fr.height
        stamper = None
        if getattr(a, 'stamp', False):                           # opt-in: send time drawn into each frame (facetime/frame_stamp.py)
            from facetime.frame_stamp import Stamper
            stamper = Stamper(getattr(a, 'stamp_t0', None)); self.stamp_t0 = stamper.t0
            print(f'[bridge] stamping frames with ms since t0={stamper.t0:.3f} (wall clock), drawn at {W}x{H}', flush=True)
        t0 = time.time(); k = 0; late = 0
        print(f'[bridge] sending from t0={t0:.3f}', flush=True)
        for img in fr:                                           # the stamp goes on after the resize, so it is published at the output size
            if self.stopping.is_set(): break
            due = t0 + k / a.fps; now = time.time()
            if now < due: await asyncio.sleep(due - now)
            elif now - due > 0.5 / a.fps: late += 1
            if stamper: img = stamper(img, time.time())
            vf = rtc.VideoFrame(W, H, rtc.VideoBufferType.RGB24, img.tobytes()).convert(rtc.VideoBufferType.I420)
            source.capture_frame(vf)
            k += 1
        t_end = time.time()
        print(f'[bridge] sent {k} frames in {t_end - t0:.1f} s ({k / max(t_end - t0, 1e-6):.2f} fps, {late} late); waiting {a.tail} s for trailing audio', flush=True)
        return t0, t_end, k

    def write_outputs(self, t0, t_end):
        a = self.a; fr = self.frames
        pcm = np.concatenate(self.audio) if self.audio else np.zeros(0, np.int16)
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        w = wave.open(a.out, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(pcm.tobytes()); w.close()
        print(f'[bridge] audio received {len(pcm) / SAMPLE_RATE:.1f} s (frames since subscribe), {self.loud_s:.1f} s above {a.onset_dbfs} dBFS; wrote {a.out}')
        print(f'[bridge] {len(self.onsets)} audio onsets (wall clock):')
        for j, o in enumerate(self.onsets):
            print(f'   #{j + 1}  {time.strftime("%H:%M:%S", time.localtime(o["t"]))}.{int(o["t"] * 1000) % 1000:03d}  t={o["t"]:.3f}  clip+{o["t"] - t0:6.2f} s  rx+{o["at_s"]:.2f} s  rms {o["rms"]}')
        if a.onsets:
            json.dump(dict(room=a.room, video=a.video, fps=a.fps, t0=t0, t_end=t_end, t_sub=self.t_sub, onsets=self.onsets,
                           audio_s=len(pcm) / SAMPLE_RATE, loud_s=self.loud_s, thr_dbfs=a.onset_dbfs,
                           stamp_t0=getattr(self, 'stamp_t0', None),
                           size=f'{fr.width}x{fr.height}' if fr else None, src_size=f'{fr.src_size[0]}x{fr.src_size[1]}' if fr else None,
                           src_fps=fr.src_fps if fr else None, simulcast=a.simulcast), open(a.onsets, 'w'), indent=1)
            print('[bridge] wrote', a.onsets)

    async def run(self):
        a = self.a
        fr = self.make_frames(); W, H = fr.width, fr.height
        print(f'[bridge] {a.video}: {fr.describe()}', flush=True)
        self.room.on('track_subscribed', self.on_track_subscribed)
        self.room.on('participant_connected', lambda p: print(f'[bridge] participant joined: {p.identity}', flush=True))
        await self.room.connect(os.environ['LIVEKIT_URL'], mint(BRIDGE_IDENTITY, a.room), rtc.RoomOptions(auto_subscribe=True))
        print(f'[bridge] connected to {a.room} as {BRIDGE_IDENTITY}; participants {list(self.room.remote_participants)}', flush=True)
        source = rtc.VideoSource(W, H)
        track = rtc.LocalVideoTrack.create_video_track(VIDEO_TRACK_NAME, source)
        opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA, simulcast=a.simulcast,
                                       video_encoding=rtc.VideoEncoding(max_framerate=a.fps, max_bitrate=a.max_bitrate))
        await self.room.local_participant.publish_track(track, opts)
        print(f'[bridge] published video track {VIDEO_TRACK_NAME} {W}x{H} at {a.fps:g} fps, max {a.max_bitrate / 1e6:.1f} Mbps '
              f'(simulcast {a.simulcast}: the SDK adds half and quarter layers; the agent asks for the lowest)', flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try: self.loop.add_signal_handler(sig, self.stopping.set)
            except NotImplementedError: pass                      # Windows: asyncio has no signal handlers; Ctrl-C raises KeyboardInterrupt instead
        await self.wait_for_agent()
        t0, t_end, k = await self.send(source, fr)
        await asyncio.sleep(a.tail)
        try: await self.room.local_participant.unpublish_track(track.sid)
        except Exception as e: print('[bridge] unpublish:', e)
        if self.audio_task: self.audio_task.cancel()
        await self.room.disconnect()
        self.write_outputs(t0, t_end)

def build_parser(room_required=True):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--room', required=room_required, default='inproc'); ap.add_argument('--video', required=True)
    ap.add_argument('--out', default=os.path.join(DATA, 'tmp', 'bridge_rx.wav')); ap.add_argument('--onsets', default=None)
    ap.add_argument('--fps', type=float, default=10.0, help='publish rate; converted from the clip rate by repeating (nearest) or skipping frames')
    ap.add_argument('--scale', type=parse_scale, default=None, metavar='WxH', help='resize each frame on the fly to this size, e.g. 720x1280 or 1080x1920 (default: the clip size)')
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--tail', type=float, default=3.0); ap.add_argument('--wait-agent', type=float, default=30.0)
    ap.add_argument('--onset-dbfs', type=float, default=-35.0, help='onset threshold, dBFS RMS per 20 ms frame')
    ap.add_argument('--min-silence', type=float, default=0.15)
    ap.add_argument('--stamp', action='store_true', help='draw the send time into each frame (facetime/frame_stamp.py; measurement runs only)')
    ap.add_argument('--stamp-t0', type=float, default=None, help='epoch (wall clock) the stamp counts from; give the same value to the agent')
    ap.add_argument('--simulcast', action=argparse.BooleanOptionalAction, default=True, help='publish simulcast layers (the agent asks the SFU for the lowest); --no-simulcast sends one full-size layer')
    ap.add_argument('--max-bitrate', type=int, default=2_500_000, help='encoder cap in bit/s (2.5 Mbps; a 1080x1920 30 fps picture wants about 4 Mbps)')
    return ap

def main():
    a = build_parser().parse_args()
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    loop.run_until_complete(Bridge(a, loop).run())

if __name__ == '__main__':
    main()
