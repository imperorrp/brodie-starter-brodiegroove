"""Generation helpers for the pitch film: Gemini images (keyframes, model sheets) and Veo video, both cached by a hash of
the request so a rerun never pays twice. Everything goes through facetime.cues_tts.gemini_client (the team relay).

Spend is logged to data/tmp/film/spend.jsonl; veo() refuses to go past VEO_BUDGET_S generated seconds.
"""
import os, io, json, time, hashlib
from team.video.voice import load_env, ROOT

FILM = os.path.join(ROOT, 'data', 'tmp', 'film')
CACHE = os.path.join(FILM, 'cache')
SPEND = os.path.join(FILM, 'spend.jsonl')
IMAGE_MODEL = 'gemini-3-pro-image'
VEO_MODEL = 'veo-3.1-fast-generate-preview'
VEO_BUDGET_S = 160
_client = None


def client():
    global _client
    if _client is None:
        load_env()
        from facetime.cues_tts import gemini_client
        _client = gemini_client()
    return _client


def _key(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode('utf-8'))
    return h.hexdigest()[:16]


def _log(**kw):
    os.makedirs(FILM, exist_ok=True)
    with open(SPEND, 'a') as f:
        f.write(json.dumps(dict(ts=time.time(), **kw)) + '\n')


def veo_seconds_used():
    if not os.path.exists(SPEND):
        return 0
    return sum(json.loads(l).get('veo_s', 0) for l in open(SPEND))


def image(prompt, refs=(), aspect='9:16', name=None, take=0):
    """one generated image (PNG path). refs: image paths attached before the prompt. take: bump to force a new variant."""
    from google.genai import types
    ref_bytes = [open(r, 'rb').read() for r in refs]
    k = _key(IMAGE_MODEL, prompt, aspect, take, *ref_bytes)
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f'{name or "img"}_{k}.png')
    if os.path.exists(dst):
        return dst
    parts = [types.Part.from_bytes(data=b, mime_type='image/png') for b in ref_bytes] + [prompt]
    cfg = types.GenerateContentConfig(response_modalities=['IMAGE', 'TEXT'], image_config=types.ImageConfig(aspect_ratio=aspect))
    last = None
    for attempt in range(4):
        try:
            r = client().models.generate_content(model=IMAGE_MODEL, contents=parts, config=cfg)
            for p in r.candidates[0].content.parts:
                if getattr(p, 'inline_data', None) and p.inline_data.data:
                    from PIL import Image
                    Image.open(io.BytesIO(p.inline_data.data)).save(dst)
                    _log(kind='image', name=name, file=dst)
                    return dst
            last = RuntimeError(f'no image in response: {r.text!r}')
        except Exception as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise last


def music(prompt, model='lyria-3-pro-preview', name=None, take=0):
    """one Lyria track (MP3 path). pro returns a full song (~3 min), clip ~30 s."""
    k = _key(model, prompt, take)
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f'{name or "music"}_{k}.mp3')
    if os.path.exists(dst):
        return dst
    r = client().models.generate_content(model=model, contents=prompt)
    for p in r.candidates[0].content.parts:
        if getattr(p, 'inline_data', None) and p.inline_data.data:
            open(dst, 'wb').write(p.inline_data.data)
            _log(kind='music', name=name, file=dst)
            return dst
    raise RuntimeError('no audio from Lyria')


def veo(prompt, first_frame=None, seconds=8, aspect='9:16', name=None, take=0, last_frame=None, negative=None):
    """one Veo clip (MP4 path), image-to-video when first_frame is given."""
    from google.genai import types
    fb = open(first_frame, 'rb').read() if first_frame else b''
    lb = open(last_frame, 'rb').read() if last_frame else b''
    k = _key(VEO_MODEL, prompt, seconds, aspect, take, fb, lb, negative or '')
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f'{name or "veo"}_{k}.mp4')
    if os.path.exists(dst):
        return dst
    used = veo_seconds_used()
    if used + seconds > VEO_BUDGET_S:
        raise SystemExit(f'Veo budget: {used}s used, {seconds}s more would pass {VEO_BUDGET_S}s')
    cfg = types.GenerateVideosConfig(aspect_ratio=aspect, duration_seconds=seconds, number_of_videos=1,
                                     negative_prompt=negative,
                                     last_frame=types.Image(image_bytes=lb, mime_type='image/png') if lb else None)
    kw = dict(model=VEO_MODEL, prompt=prompt, config=cfg)
    if fb:
        kw['image'] = types.Image(image_bytes=fb, mime_type='image/png')
    t0 = time.time()
    op = client().models.generate_videos(**kw)
    while not op.done:
        time.sleep(8)
        op = client().operations.get(op)
    if op.error:
        _log(kind='veo_error', name=name, error=str(op.error))
        raise RuntimeError(f'veo failed: {op.error}')
    vids = (op.response or op.result).generated_videos
    if not vids:
        raise RuntimeError(f'veo returned no video (filtered?): {op.response}')
    v = vids[0].video
    data = v.video_bytes or client().files.download(file=v)
    open(dst, 'wb').write(data)
    _log(kind='veo', name=name, veo_s=seconds, secs=round(time.time() - t0), file=dst)
    print(f'veo {name}: {seconds}s clip in {time.time() - t0:.0f}s, budget used {used + seconds}/{VEO_BUDGET_S}s', flush=True)
    return dst
