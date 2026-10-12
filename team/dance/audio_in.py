"""The caller's audio on its way to Gemini: background-voice cancellation (Krisp BVC) and automatic gain control, without editing
the starter. facetime/voice.py builds RoomOptions(audio_input=True) inside GeminiModel.start; wrap_start() swaps the RoomOptions
name in livekit.agents.voice.room_io for the duration of that call, so audio_input becomes AudioInputOptions(noise_cancellation=...,
auto_gain_control=...).

  DANCE_NC   auto (default) | bvc | bvc_tel | off
             auto: BVC only when the LiveKit server is LiveKit Cloud (*.livekit.cloud: the Krisp models need it) and the plugin
             imports; otherwise off, with the reason logged. bvc_tel: the telephony model (narrow-band audio).
  DANCE_AGC  1 (default) | 0: gain control for callers standing 2-3 m from the phone

Squad and class calls: BVC keeps the nearest voice, so friends further back may be cut; every question there also offers "or
raise both hands", and the gestures count per dancer.
"""
import os
from urllib.parse import urlparse

STATUS = dict(nc='off', reason='not configured', agc=False)


def choose_nc(env=None, url=None):
    """('bvc' | 'bvc_tel' | None, reason)."""
    env = (env if env is not None else os.environ.get('DANCE_NC', 'auto') or 'auto').strip().lower()
    if env in ('off', '0', 'no', 'none'): return None, 'DANCE_NC=off'
    try:
        from livekit.plugins import noise_cancellation  # noqa: F401
    except Exception as e:
        return None, f'plugin not installed ({type(e).__name__})'
    if env in ('bvc', 'bvc_tel'): return env, 'forced'
    host = urlparse(url if url is not None else (os.environ.get('LIVEKIT_URL') or '')).hostname or ''
    if host.endswith('.livekit.cloud'): return 'bvc', 'LiveKit Cloud'
    return None, f'server {host or "unknown"} is not LiveKit Cloud'


def make_nc(kind):
    if kind is None: return None
    from livekit.plugins import noise_cancellation as nc
    return nc.BVCTelephony() if kind == 'bvc_tel' else nc.BVC()


def audio_input_options(kind=None, agc=None):
    from livekit.agents.voice.room_io import AudioInputOptions
    agc = (os.environ.get('DANCE_AGC', '1') or '1').strip() != '0' if agc is None else agc
    kw = {}
    nc = make_nc(kind)
    if nc is not None: kw['noise_cancellation'] = nc
    if agc: kw['auto_gain_control'] = True
    return AudioInputOptions(**kw) if kw else True


def wrap_start(orig_start):
    """GeminiModel.start, with the caller's audio input upgraded (falls back to the original options on any error)."""
    async def start(self, *a, **kw):
        import livekit.agents.voice.room_io as rio
        Orig = rio.RoomOptions
        kind, why = choose_nc()
        try:
            opts = audio_input_options(kind)
        except Exception as e:
            print(f'[dance] audio input options failed ({e!r}); plain input', flush=True); opts = True; kind = None; why = repr(e)
        STATUS.update(nc=kind or 'off', reason=why, agc=bool(opts is not True and getattr(opts, 'auto_gain_control', False) is True))
        print(f"[dance] caller audio: noise cancellation {kind or 'off'} ({why}), gain control {'on' if STATUS['agc'] else 'off'}", flush=True)

        def Patched(*pa, **pk):
            if pk.get('audio_input') is True and opts is not True: pk['audio_input'] = opts
            return Orig(*pa, **pk)
        rio.RoomOptions = Patched
        try:
            return await orig_start(self, *a, **kw)
        finally:
            rio.RoomOptions = Orig
    start.__wrapped_audio_in__ = True
    return start
