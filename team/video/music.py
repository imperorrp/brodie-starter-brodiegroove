"""Soundtrack for the pitch video: the landing page's Web Audio groove (team/web/groove/index.html, kick / snare / hats / A-minor
bass, same 16-step pattern) ported to numpy, arranged on timeline.py's beats, with SFX and the voice lines ducked on top.

Run:  python -m team.video.music
Writes data/tmp/pitch/music.wav (music + SFX) and data/tmp/pitch/mix.wav (everything, ducked under the voice). 48 kHz stereo.
"""
import os, json, wave
import numpy as np
from scipy.signal import butter, sosfilt, resample_poly
from team.video.timeline import PITCH, BEAT, DURATION, VOICE, SFX, t

SR = 48000
N = int(round(DURATION * SR))
rng = np.random.default_rng(7)
NOISE = rng.uniform(-1, 1, SR * 6)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def env(dur, peak=1.0, floor=0.0008):
    """WebAudio exponentialRampToValueAtTime(floor, t + dur) from peak."""
    x = tt(dur)
    return peak * (floor / peak) ** (x / dur)


def filt(x, kind, freq, order=2):
    if kind == 'bandpass':
        sos = butter(order, [freq / 1.5, min(freq * 1.5, SR / 2 - 1)], 'bandpass', fs=SR, output='sos')
    else:
        sos = butter(order, freq, kind, fs=SR, output='sos')
    return sosfilt(sos, x)


def osc(freq, dur, shape='sine'):
    """freq: number or per-sample array."""
    n = int(dur * SR)
    f = np.broadcast_to(np.asarray(freq, float), (n,)) if np.ndim(freq) == 0 else np.asarray(freq)[:n]
    ph = np.cumsum(f) / SR
    if shape == 'sine':
        return np.sin(2 * np.pi * ph)
    if shape == 'square':
        return np.sign(np.sin(2 * np.pi * ph))
    if shape == 'saw':
        return 2 * (ph % 1.0) - 1
    if shape == 'triangle':
        return 2 * np.abs(2 * (ph % 1.0) - 1) - 1
    raise ValueError(shape)


def sweep(f0, f1, dur, ramp):
    """exponential frequency ramp f0 -> f1 over `ramp` seconds, then hold."""
    x = tt(dur)
    return np.where(x < ramp, f0 * (f1 / f0) ** (np.minimum(x, ramp) / ramp), f1)


def noise(dur, kind, freq, peak):
    n = int(dur * SR)
    x = filt(NOISE[:n], kind, freq)
    return x * env(dur, peak) * 2.2


# ---- the site's instruments ----
def kick(peak=1.0):
    return osc(sweep(150, 42, 0.35, 0.14), 0.35) * env(0.35, 1.0)[: int(0.35 * SR)] * peak


def snare(peak=1.0):
    n = noise(0.2, 'bandpass', 1900, 0.55)
    o = osc(190, 0.1, 'triangle') * env(0.1, 0.35)
    out = n.copy(); out[: len(o)] += o
    return out * peak


def hat(on_beat, peak=1.0):
    return noise(0.06, 'highpass', 7500, 0.08 if on_beat else 0.16) * peak


def bass(hz, dur):
    s = osc(hz, dur, 'saw')
    bright, dark = filt(s, 'lowpass', 900), filt(s, 'lowpass', 160)
    k = np.exp(-tt(dur) / (dur / 2.5))
    g = np.minimum(1, tt(dur) / 0.01) * env(dur, 0.32)
    return (bright * k + dark * (1 - k)) * g


def clap(peak=1.0):
    out = np.zeros(int(0.25 * SR))
    for i, d in enumerate((0.0, 0.011, 0.022)):
        b = noise(0.12 if i == 2 else 0.02, 'bandpass', 1200, 0.5)
        a = int(d * SR); out[a:a + len(b)] += b
    return out * peak


def pluck(hz, dur=0.5, peak=0.3):
    """the ringtone voice: a soft marimba-ish sine with a fast-decaying overtone."""
    x = osc(hz, dur) + 0.35 * osc(hz * 4, dur) * np.exp(-tt(dur) / 0.03)
    return x * np.exp(-tt(dur) / 0.16) * np.minimum(1, tt(dur) / 0.003) * peak


BASS = [55, 0, 0, 55, 0, 65.4, 0, 73.4, 55, 0, 0, 55, 0, 82.4, 73.4, 65.4]
CHORDS = [(220.0, 261.6, 329.6), (174.6, 220.0, 261.6), (196.0, 261.6, 329.6), (196.0, 246.9, 293.7)]   # Am F C/G G, one per bar


class Track:
    def __init__(self):
        self.l = np.zeros(N + SR * 2); self.r = np.zeros(N + SR * 2)

    def add(self, sec, sig, gain=1.0, pan=0.0):
        a = int(round(sec * SR))
        if a < 0:
            sig = sig[-a:]; a = 0
        b = min(len(self.l), a + len(sig)); sig = sig[: b - a]
        self.l[a:b] += sig * gain * (1 - max(0, pan)); self.r[a:b] += sig * gain * (1 + min(0, pan))

    def stereo(self):
        return np.stack([self.l[:N], self.r[:N]], 1)


def groove(tr, b0, b1, step_beats=0.5, claps=False, stabs=False, roll=False, gain=1.0):
    """The site's 16-step pattern from beat b0 to b1. step_beats=1.0 is the half-time feel."""
    s = 0
    while b0 + s * step_beats < b1 - 1e-6:
        sec = t(b0 + s * step_beats); i = s % 16; on = i % 2 == 0; bib = (i >> 1) % 4
        if (on and bib in (0, 2)) or i == 14 or i in (5, 11):
            tr.add(sec, kick(), 0.95 * gain)
        if on and bib in (1, 3):
            tr.add(sec, snare(), 0.8 * gain)
            if claps:
                tr.add(sec, clap(), 0.55 * gain, pan=0.15)
        tr.add(sec, hat(on), 0.9 * gain, pan=-0.25 if on else 0.25)
        if BASS[i]:
            tr.add(sec, bass(BASS[i], BEAT * step_beats * 0.9), 1.0 * gain)
        if stabs and not on:
            ch = CHORDS[int((b0 + s * step_beats) // 4) % 4]
            sig = sum(osc(f * 2, 0.22, 'saw') for f in ch) / 3
            tr.add(sec, filt(sig, 'lowpass', 2400) * env(0.22, 0.16), gain, pan=0.3 if i % 4 == 1 else -0.3)
        s += 1
    if roll:                                          # 16th-note snare roll rising into the next drop
        n = int((b1 - b0) * 4)
        for k in range(n):
            tr.add(t(b0 + k / 4), snare(), 0.15 + 0.6 * (k / n) ** 2, pan=0.1)


def pad(tr, b0, b1, gain=0.10, cutoff=1400):
    """sustained chord bed, one chord per bar, pumped by the four-on-the-floor."""
    for bar in range(int(b0 // 4), int(np.ceil(b1 / 4))):
        a, b = max(b0, bar * 4), min(b1, bar * 4 + 4)
        dur = t(b - a)
        ch = CHORDS[bar % 4]
        sig = sum(osc(f * (1 + d), dur, 'saw') for f in ch for d in (-0.004, 0.004)) / 6
        sig = filt(sig, 'lowpass', cutoff)
        x = tt(dur); beat_ph = ((x + t(a)) / BEAT) % 1.0
        pump = 0.35 + 0.65 * np.minimum(1, beat_ph / 0.45)
        fade = np.minimum(1, np.minimum(x / 0.02, (dur - x) / 0.02))
        tr.add(t(a), sig * pump * fade, gain, pan=-0.2)
        tr.add(t(a) + 0.012, sig * pump * fade, gain, pan=0.2)


def riser(dur, peak=0.35):
    x = tt(dur)
    f = 400 * (8000 / 400) ** (x / dur)
    n = NOISE[: len(x)]
    out = np.zeros_like(x)
    for k in range(0, len(x), 2400):                    # a stepped sweep is plenty for a riser
        a = max(0, k - 4800)                            # 100 ms of lead-in settles the filter
        y = filt(n[a:k + 2400], 'bandpass', float(f[k]))
        out[k:k + 2400] = y[k - a:]
    return out * (x / dur) ** 2 * peak * 2


def ringtone(tr, beat, gain=1.0):
    notes = [880.0, 1318.5, 1046.5, 1318.5, 880.0, 1318.5, 1046.5, 1568.0]
    for k, hz in enumerate(notes):
        tr.add(t(beat + k * 0.25), pluck(hz), 0.55 * gain, pan=0.2 if k % 2 else -0.2)


# ---- SFX ----
def sfx(kind):
    if kind == 'swipe':
        x = tt(0.3); f = 600 * (6000 / 600) ** (x / 0.3)
        return riser(0.3, 0.6)[::-1] * 0.6 + osc(f, 0.3) * 0.03 * np.sin(np.pi * x / 0.3)
    if kind == 'impact':
        boom = osc(sweep(90, 38, 1.2, 0.3), 1.2) * env(1.2, 1.0) * 0.9
        crash = noise(1.4, 'highpass', 3500, 0.5)
        out = np.zeros(int(1.4 * SR)); out[: len(boom)] += boom; out += crash * 0.7
        k = kick(1.2); out[: len(k)] += k
        return out
    if kind in ('perfect', 'blip1', 'blip2', 'blip3'):
        hz = dict(perfect=880, blip1=660, blip2=784, blip3=988)[kind]
        a = osc(sweep(hz, hz * 1.5, 0.14, 0.06), 0.14, 'square') * env(0.14, 0.18)
        if kind != 'perfect':
            return a
        b = osc(sweep(hz * 1.5, hz * 2.25, 0.2, 0.06), 0.2, 'square') * env(0.2, 0.16)
        out = np.zeros(int(0.3 * SR)); out[: len(a)] += a; o = int(0.07 * SR); out[o:o + len(b)] += b
        return filt(out, 'lowpass', 6000)
    if kind == 'tick':
        return osc(1320, 0.08) * env(0.08, 0.3)
    if kind == 'zoom':
        return riser(0.25, 0.5) + osc(sweep(200, 900, 0.25, 0.25), 0.25) * np.linspace(0, 0.08, int(0.25 * SR))
    if kind == 'stamp':
        out = kick(1.0).copy(); s = noise(0.25, 'lowpass', 900, 0.6)
        n = min(len(out), len(s)); out[:n] += s[:n]
        return out
    if kind == 'shutter':
        out = np.zeros(int(0.2 * SR))
        for d, dur in ((0.0, 0.03), (0.07, 0.05)):
            c = noise(dur, 'highpass', 2500, 0.7); a = int(d * SR); out[a:a + len(c)] += c
        return out
    return None


def tapestop(src, dur=0.6):
    """play `src` (a slice of the mix) slowing from 1x to 0 over `dur` seconds."""
    x = tt(dur); rate = (1 - x / dur) ** 1.6
    idx = np.cumsum(rate)
    idx = idx[idx < len(src) - 1]
    out = np.interp(idx, np.arange(len(src)), src)
    return out * np.minimum(1, (len(idx) - np.arange(len(idx))) / (0.05 * SR))


def build_music():
    tr = Track()
    # 1 hook: two rings, a filtered groove sneaking in, riser, swipe, DROP on beat 8
    ringtone(tr, 0); ringtone(tr, 2, 0.9)
    pre = Track(); groove(pre, 4, 8, gain=0.9); pad(pre, 4, 8, 0.08)
    lp = filt(pre.l, 'lowpass', 500, 4); lpr = filt(pre.r, 'lowpass', 500, 4)
    a, b = int(t(4) * SR), int(t(8) * SR)
    ramp = np.linspace(0.25, 0.75, b - a)
    tr.l[a:b] += lp[a:b] * ramp; tr.r[a:b] += lpr[a:b] * ramp
    tr.add(t(6), riser(t(2)), 0.6)
    # 2-3 full groove
    groove(tr, 8, 38); pad(tr, 8, 38)
    # 4 voice control: tape stop on beat 38, half-time from 39 to 44
    mono = (tr.l + tr.r) / 2
    stop = tapestop(mono[int(t(37) * SR): int(t(38.2) * SR)], t(1))
    tr.add(t(38), stop, 0.9)
    groove(tr, 39, 44, step_beats=1.0, gain=0.85); pad(tr, 39, 44, 0.09, 900)
    # speed ramp: build into the payoff
    groove(tr, 44, 48, roll=True); pad(tr, 44, 48, 0.08, 2000)
    tr.add(t(44), riser(t(4)), 0.55)
    # 5 payoff: full energy
    groove(tr, 48, 64, claps=True, stabs=True); pad(tr, 48, 64, 0.11, 2200)
    # 6 CTA: groove under the end card, final hit on beat 72, then the ring once more
    groove(tr, 64, 72, gain=0.9); pad(tr, 64, 72, 0.09)
    hit = Track()
    hit.add(0, kick(1.3)); hit.add(0, bass(55, 2.4) * 1.6)
    ch = sum(osc(f * m, 2.6, 'saw') for f in CHORDS[0] for m in (1, 2)) / 6
    hit.add(0, filt(ch, 'lowpass', 2600) * env(2.6, 0.5)); hit.add(0, noise(2.0, 'highpass', 3500, 0.45))
    tr.add(t(72), hit.l[: int(2.8 * SR)], 1.0)
    ringtone(tr, 74, 0.5)
    for beat, kind in SFX:
        s = sfx(kind)
        if s is not None:
            tr.add(t(beat), s, 0.9)
    return tr.stereo()


def load_voice():
    """voice track (stereo, centred) and a sidechain envelope for the ducking."""
    v = np.zeros(N)
    for key, beat, max_s, _ in VOICE:
        p = os.path.join(PITCH, 'vo', key + '.wav')
        with wave.open(p) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768
            x = resample_poly(x, SR, w.getframerate())
        if max_s:
            x = x[: int(max_s * SR)]
            f = int(0.08 * SR); x[-f:] *= np.linspace(1, 0, f)
        x = filt(x, 'highpass', 90)
        a = int(t(beat) * SR); b = min(N, a + len(x)); v[a:b] += x[: b - a]
    pk = np.max(np.abs(v)) or 1
    v = v / pk * 0.85
    e = np.abs(v)
    # envelope follower: 15 ms attack, 300 ms release
    att, rel = np.exp(-1 / (0.015 * SR)), np.exp(-1 / (0.3 * SR))
    out = np.zeros_like(e); y = 0.0
    for i in range(0, len(e), 48):                     # 1 ms hops is plenty for a ducking envelope
        m = e[i:i + 48].max()
        c = att if m > y else rel
        y = c * y + (1 - c) * m
        out[i:i + 48] = y
    duck = 1 - 0.6 * np.minimum(1, out / 0.08)        # about -8 dB while Brodie talks
    return v, duck


def write(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    m = build_music()
    m = np.tanh(m * 1.2) / np.tanh(1.2)
    m = m / np.max(np.abs(m)) * 0.8
    write(os.path.join(PITCH, 'music.wav'), m)
    v, duck = load_voice()
    mix = m * duck[:, None] * 0.75 + v[:, None] * 0.95
    mix = mix / np.max(np.abs(mix)) * 0.95
    write(os.path.join(PITCH, 'mix.wav'), mix)
    print(f'music.wav + mix.wav: {DURATION:.1f}s at {SR} Hz')


if __name__ == '__main__':
    main()
