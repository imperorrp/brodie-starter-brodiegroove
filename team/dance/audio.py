"""Synthetic audio for the dance engine: clicks, a count-in and a metronome track at an exact BPM, as 24 kHz mono int16 PCM
that the starter's CuePlayer can play through the one audio track. Registered into the CueBank at runtime (its cache is a dict
keyed by the cue string), so no files and no licences are involved. Real music clips can be registered the same way later.
"""
import numpy as np

SR = 24000
FRAME = 480
TRACK = '[dance:track]'
COUNT_IN = '[dance:countin]'
CLICK = '[dance:click]'
STOP = '[dance:stop]'


def _tone(freq, ms, amp=0.5, decay=True):
    n = int(SR * ms / 1000.0); t = np.arange(n) / SR
    env = np.exp(-t * 18.0) if decay else np.ones(n)
    return (amp * 32767 * np.sin(2 * np.pi * freq * t) * env).astype(np.int16)


def _pad(pcm):
    pad = (-len(pcm)) % FRAME
    return np.concatenate([pcm, np.zeros(pad, np.int16)]) if pad else pcm


def click(freq=1000.0, ms=45, amp=0.5):
    return _pad(_tone(freq, ms, amp))


def metronome(bpm, n_beats, accent_every=8, countin_beats=0, lead_silence_s=0.0, tail_s=0.3, hat=True):
    """Clicks on every beat: a high click for the count-in beats, an accented low click on beat 1 of every `accent_every`,
    a plain click elsewhere, and a soft offbeat hat. Total beats = countin_beats + n_beats."""
    beat_s = 60.0 / bpm; total = countin_beats + n_beats
    n = int(SR * (lead_silence_s + total * beat_s + tail_s)); out = np.zeros(n, np.float32)
    for b in range(total):
        t0 = lead_silence_s + b * beat_s; i = int(t0 * SR)
        if b < countin_beats: tone = _tone(1400.0, 60, 0.6)
        elif (b - countin_beats) % accent_every == 0: tone = _tone(700.0, 80, 0.7)
        else: tone = _tone(1000.0, 45, 0.45)
        out[i:i + len(tone)] += tone[:max(0, min(len(tone), n - i))]
        if hat and b >= countin_beats:
            j = int((t0 + beat_s / 2) * SR); h = (0.12 * 32767 * np.random.default_rng(b).normal(size=int(SR * 0.02)) * np.exp(-np.arange(int(SR * 0.02)) / (SR * 0.004))).astype(np.float32)
            out[j:j + len(h)] += h[:max(0, min(len(h), n - j))]
    return _pad(np.clip(out, -32767, 32767).astype(np.int16))


def register_clip(bank, text, pcm):
    """Put a clip into a CueBank at runtime so CuePlayer.say(text) plays it. Returns the duration in seconds."""
    pcm = _pad(np.asarray(pcm, np.int16)); dur = len(pcm) / SR
    bank.cache[text] = pcm
    bank.manifest[text] = dict(file='synthetic', duration_s=round(dur, 3))
    return dur


def mix_at(base, clip, offset_s, gain=1.0):
    """Add `clip` into `base` (both int16) starting at offset_s, clipping to int16; extends base if needed."""
    base = np.asarray(base, np.int32); clip = (np.asarray(clip, np.float32) * gain).astype(np.int32)
    i = max(0, int(offset_s * SR)); end = i + len(clip)
    if end > len(base): base = np.concatenate([base, np.zeros(end - len(base), np.int32)])
    base[i:end] += clip
    return np.clip(base, -32767, 32767).astype(np.int16)


def register_phrase_track(bank, bpm, show_beats, countin_beats, attempt_beats, text=TRACK, voice=None):
    """One track for a whole phrase run: show (plain clicks), count-in (high clicks), attempt (accented clicks), with optional
    spoken lines mixed in: voice = {'watch': pcm, 'your_turn': pcm}. "Watch" lands at the start, "your turn" ends just before
    the count-in. Returns (duration_s, offsets) with the second offsets of the count-in start and the attempt start."""
    beat_s = 60.0 / bpm
    parts = [metronome(bpm, show_beats, tail_s=0.0, hat=True), metronome(bpm, 0, countin_beats=countin_beats, tail_s=0.0, hat=False),
             metronome(bpm, attempt_beats, tail_s=0.4, hat=True)]
    # trim each part to its exact beat length so the parts butt together on the grid
    exact = [int(SR * show_beats * beat_s), int(SR * countin_beats * beat_s), None]
    fixed = []
    for p, e in zip(parts, exact):
        if e is not None:
            p = p[:e] if len(p) >= e else np.concatenate([p, np.zeros(e - len(p), np.int16)])
        fixed.append(p)
    pcm = np.concatenate(fixed)
    voice = voice or {}
    if voice.get('watch') is not None and show_beats > 0:
        pcm = mix_at(pcm, voice['watch'], 0.05, gain=0.9)
    if voice.get('your_turn') is not None:
        at = show_beats * beat_s - len(voice['your_turn']) / SR - 0.15
        if at > 0: pcm = mix_at(pcm, voice['your_turn'], at, gain=0.9)
    dur = register_clip(bank, text, pcm)
    return dur, dict(countin=show_beats * beat_s, attempt=(show_beats + countin_beats) * beat_s)


class FakeBank:
    """A stand-in CueBank for tests and simulation (same attributes the registrar touches)."""
    def __init__(self): self.cache = {}; self.manifest = {}
    def get(self, text): return self.cache.get(text)
    def duration(self, text): m = self.manifest.get(text); return m['duration_s'] if m else None


# ---- sound effects (synthesized; played after the phrase track or to cut it) ----
SFX = dict(levelup='[dance:sfx:levelup]', stamp='[dance:sfx:stamp]', ko='[dance:sfx:ko]', combo='[dance:sfx:combo]',
           pause='[dance:sfx:pause]', achievement='[dance:sfx:achievement]', mvp='[dance:sfx:mvp]')


def _note(freq, ms, amp=0.4, decay=8.0, harm=0.3):
    n = int(SR * ms / 1000.0); t = np.arange(n) / SR; env = np.exp(-t * decay) * np.minimum(1.0, t / 0.004)
    w = np.sin(2 * np.pi * freq * t) + harm * np.sin(4 * np.pi * freq * t)
    return (amp * 32767 * w * env / (1 + harm)).astype(np.float32)


def _seq(notes, gap_ms=0):
    parts = []
    for f, ms, amp in notes:
        parts.append(_note(f, ms, amp)); parts.append(np.zeros(int(SR * gap_ms / 1000.0), np.float32))
    return _pad(np.clip(np.concatenate(parts), -32767, 32767).astype(np.int16))


def sfx_levelup():                                   # C-E-G-C up, bright
    return _seq([(523.3, 110, 0.35), (659.3, 110, 0.35), (784.0, 110, 0.38), (1046.5, 380, 0.42)])


def sfx_achievement():
    return _seq([(880.0, 90, 0.3), (1318.5, 260, 0.35)])


def sfx_combo():
    return _seq([(660.0, 70, 0.3), (880.0, 70, 0.3), (1320.0, 160, 0.32)])


def sfx_stamp():                                     # low thump plus a short noise burst
    n = int(SR * 0.22); t = np.arange(n) / SR
    thump = 0.6 * 32767 * np.sin(2 * np.pi * (90 + 60 * np.exp(-t * 30)) * t) * np.exp(-t * 14)
    noise = 0.18 * 32767 * np.random.default_rng(7).normal(size=n) * np.exp(-t * 60)
    return _pad(np.clip(thump + noise, -32767, 32767).astype(np.int16))


def sfx_ko():
    a = sfx_stamp().astype(np.float32); b = _seq([(392.0, 120, 0.35), (523.3, 120, 0.35), (784.0, 420, 0.4)]).astype(np.float32)
    out = np.zeros(len(a) + len(b), np.float32); out[:len(a)] += a; out[len(a) // 2:len(a) // 2 + len(b)] += b
    return _pad(np.clip(out, -32767, 32767).astype(np.int16))


def sfx_pause():                                     # soft two notes down; also cuts the music when played urgently
    return _seq([(784.0, 120, 0.25), (523.3, 220, 0.25)])


def register_sfx(bank):
    """Register every sound effect in the bank; returns {name: cue text}."""
    gens = dict(levelup=sfx_levelup, stamp=sfx_stamp, ko=sfx_ko, combo=sfx_combo, pause=sfx_pause, achievement=sfx_achievement, mvp=sfx_levelup)
    for name, fn in gens.items(): register_clip(bank, SFX[name], fn())
    return dict(SFX)
