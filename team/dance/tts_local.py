"""Spoken cue lines without any API: Windows' built-in speech (System.Speech via PowerShell) rendered to 24 kHz mono int16 PCM
files under team/dance/voice/. Good enough for testing and for a demo fallback; re-render the same lines with the starter's
Gemini TTS (facetime/cues_tts.py) when the relay token is available so the voice matches Brodie's bank.

  .venv/Scripts/python -m team.dance.tts_local            # renders LINES below (skips files that exist)
  .venv/Scripts/python -m team.dance.tts_local --force    # re-render
Other platforms: `say` on macOS is wired too; Linux falls back to espeak if present.
"""
import os, subprocess, sys, tempfile, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, 'voice')
SR = 24000
LINES = {
    'watch': 'Watch.',
    'your_turn': 'Your turn.',
    'countin': 'Five, six, seven, eight.',
    'again': 'Again.',
    'slower': 'Slower this time.',
    'nice': 'Nice.',
    'last_one': 'Last one.',
    'round_two': 'Round two.',
    'round_three': 'Round three. Ten percent faster.',
    'step_back': 'Step back so I can see all of you.',
    'hello': "Hi, I'm Brodie. Step back so I can see your whole body.",
    'intro': "I'll show you the move twice. Then it's your turn.",
}


def _resample(x, sr_in, sr_out=SR):
    if sr_in == sr_out: return x
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x.astype(np.float32)).astype(np.float32)


def _read_wav(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth(); n = w.getnframes(); raw = w.readframes(n)
    if sw == 2: x = np.frombuffer(raw, np.int16).astype(np.float32)
    elif sw == 1: x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) * 256
    else: raise ValueError(f'unsupported sample width {sw}')
    if ch > 1: x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def synth_to_wav(text, path, rate=0):
    if sys.platform.startswith('win'):
        ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.Rate = {int(rate)}; $s.SetOutputToWaveFile('{path}'); $s.Speak('{text.replace(chr(39), chr(39) * 2)}'); $s.Dispose()")
        subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', ps], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif sys.platform == 'darwin':
        aiff = path + '.aiff'; subprocess.run(['say', '-o', aiff, text], check=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', aiff, '-ac', '1', '-ar', str(SR), path], check=True); os.remove(aiff)
    else:
        subprocess.run(['espeak-ng', '-w', path, text], check=True)


def render(name, text, force=False, trim_dbfs=-45.0):
    """Render one line to VOICE_DIR/<name>.pcm (24 kHz mono int16, trimmed, level-normalised). Returns the path."""
    os.makedirs(VOICE_DIR, exist_ok=True)
    out = os.path.join(VOICE_DIR, f'{name}.pcm')
    if os.path.exists(out) and not force: return out
    tmp = os.path.join(tempfile.gettempdir(), f'dance_tts_{name}.wav')
    synth_to_wav(text, tmp)
    x, sr = _read_wav(tmp); x = _resample(x, sr)
    thr = 32767 * 10 ** (trim_dbfs / 20); loud = np.flatnonzero(np.abs(x) > thr)
    if len(loud): x = x[max(0, loud[0] - int(0.02 * SR)):min(len(x), loud[-1] + int(0.12 * SR))]
    peak = float(np.max(np.abs(x))) or 1.0; x = x * (0.7 * 32767 / peak)
    np.clip(x, -32767, 32767).astype(np.int16).tofile(out)
    return out


def load(name):
    p = os.path.join(VOICE_DIR, f'{name}.pcm')
    return np.fromfile(p, np.int16) if os.path.exists(p) else None


def render_all(force=False):
    done = {}
    for name, text in LINES.items():
        try: done[name] = render(name, text, force=force)
        except Exception as e: print(f'[tts] {name}: {e!r}', flush=True)
    return done


if __name__ == '__main__':
    force = '--force' in sys.argv
    d = render_all(force=force)
    for k, p in d.items():
        x = load(k); print(f'{k:12} {len(x) / SR:5.2f} s  {p}')
