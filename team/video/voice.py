"""Voice lines for the pitch video: Brodie in the product voice (Gemini TTS, Puck) plus a few user lines in a second voice.

Run:  python -m team.video.voice [--force] [key ...]   (keys re-render just those lines, e.g. when the model ad-libs)
Writes data/tmp/pitch/vo/<key>.wav (24 kHz mono) and lines.json (key -> text, voice, duration_s). Cached: existing clips are kept.
"""
import os, sys, json, wave
from facetime.cues_tts import gemini_client, gemini_pick_model, gemini_render_one, trim

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'tmp', 'pitch', 'vo')
RATE = 24000
BRODIE = ('Puck', '(hyped, warm, playful dance coach on a video call, big smile, punchy)')
USER = ('Kore', '(excited young adult on a video call, casual, natural)')
USER_CALM = ('Kore', '(young adult mid-dance, casual, short and plain, no breath or laugh before speaking)')   # "excited" adds a gasp
LINES = {
    'u_teach':   (USER,   'Brodie, teach me hip hop!'),
    'b_ready':   (BRODIE, "Let's go! Ready to dance?"),
    'b_teach':   (BRODIE, 'Counts one to two, both arms go up. Follow my ghost.'),
    'b_tip':     (BRODIE, 'Lift your left arm higher on five to seven.'),
    'u_slower':  (USER_CALM, 'Brodie, slower.'),
    'b_slower':  (BRODIE, 'Got you. Slowing it down.'),
    'b_combo':   (BRODIE, 'Combo menace!'),
    'b_screen':  (BRODIE, 'Human Metronome! Screenshot this.'),
    'b_cta':     (BRODIE, 'Your first eight count is one call away.'),
}


def load_env():
    p = os.path.join(ROOT, '.env.local')
    if not os.path.exists(p):
        return
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def transcribe(client, path):
    """what the clip actually says (the TTS model sometimes ad-libs or adds a gasp)."""
    from google.genai import types
    for _ in range(3):
        r = client.models.generate_content(model='gemini-3.8-flash', contents=[
            types.Part.from_bytes(data=open(path, 'rb').read(), mime_type='audio/wav'),
            'Transcribe verbatim, including gasps, laughs or extra words in brackets. Output only the transcript.'])
        if r.text:
            return r.text.strip()
    return '?'


def matches(text, heard):
    """same words, ignoring case, punctuation and the Brodie/Brody spelling; bracketed sounds don't count."""
    import re
    words = 'zero one two three four five six seven eight nine ten'.split()
    def norm(s):
        s = re.sub(r'\[[^\]]*\]', '', s.lower()).replace('brody', 'brodie').replace('-', ' ')
        s = re.sub(r'\d+', lambda m: ' '.join(words[int(d)] for d in m.group()), s)   # "5 6" == "five six"
        return re.sub(r'[^a-z ]', '', s).split()
    return norm(text) == norm(heard)


def render(lines, out, redo=(), force=False):
    """lines: key -> ((voice, direction), text). Renders missing (or redo) clips, prints each transcript to check."""
    load_env(); os.makedirs(out, exist_ok=True)
    client = gemini_client(); model = gemini_pick_model(client, 'gemini-3.8-flash-tts')
    meta = {}
    for key, ((voice, direction), text) in lines.items():
        dst = os.path.join(out, key + '.wav')
        fresh = force or key in redo or not os.path.exists(dst)
        heard = ''
        for attempt in range(4 if fresh else 0):     # retake until the words match (it ad-libs, or reads the direction)
            prompt = f'{direction} {text}' if direction else text
            pcm, usage = gemini_render_one(client, model, voice, prompt)
            pcm = trim(pcm, RATE, -40.0, 0.08)
            with wave.open(dst, 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm)
            heard = transcribe(client, dst)
            if matches(text, heard):
                break
            print(f'{key}: retake, heard "{heard}"')
        with wave.open(dst) as w:
            meta[key] = dict(text=text, voice=voice, duration_s=round(w.getnframes() / w.getframerate(), 3))
        print(f'{key}: {meta[key]["duration_s"]}s  "{text}"' + (f'  heard: {heard}' if heard else ''))
    json.dump(meta, open(os.path.join(out, 'lines.json'), 'w'), indent=1)
    return meta


def main():
    redo = [a for a in sys.argv[1:] if not a.startswith('--')]      # re-render just these keys
    render(LINES, OUT, redo, '--force' in sys.argv)


if __name__ == '__main__':
    main()
