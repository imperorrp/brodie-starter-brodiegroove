"""The film's soundtrack: Lyria cues cut on timeline.py's beats, synthesised SFX (team/video/music.py voices plus a record
scratch, ringtone and whoosh), and the dialogue on top with the music ducked under it.

Run:  python -m team.video.film.audio      (after team.video.film.score and team.video.film.voice)
Writes data/tmp/film/mix.wav (48 kHz stereo).
"""
import os, wave
import numpy as np
from scipy.signal import resample_poly
from team.video import music as M
from team.video.film.timeline import MUSIC_CUES, SFX, VOICE, DURATION, MUSIC, VO, t
from team.video.film.gen import FILM

SR = M.SR
N = int(round(DURATION * SR))


def load(path):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        x = x.reshape(-1, w.getnchannels())
        if w.getframerate() != SR:
            x = resample_poly(x, SR, w.getframerate(), axis=0)
    return x if x.shape[1] == 2 else np.repeat(x, 2, 1)


def place(buf, sec, sig, gain=1.0):
    a = int(round(sec * SR)); sig = np.atleast_2d(sig.T).T if sig.ndim == 1 else sig
    if sig.ndim == 1 or sig.shape[1] == 1:
        sig = np.repeat(sig.reshape(-1, 1), 2, 1)
    b = min(len(buf), a + len(sig))
    if b > a:
        buf[a:b] += sig[: b - a] * gain


def scratch(src):
    """a record scratch: the last moment of the music dragged back and forth under the needle, then the stop."""
    d = 0.55; x = np.arange(int(d * SR)) / SR
    rate = np.interp(x, [0, 0.08, 0.2, 0.32, 0.45, d], [1.0, -2.2, 1.8, -1.6, 0.6, 0.0])
    pos = len(src) * 0.75 + np.cumsum(rate)
    pos = np.clip(pos, 0, len(src) - 2)
    out = np.stack([np.interp(pos, np.arange(len(src)), src[:, c]) for c in range(2)], 1)
    out = np.stack([M.filt(out[:, c], 'bandpass', 1400) for c in range(2)], 1) * 2.0
    return out * np.minimum(1, (len(x) - np.arange(len(x)))[:, None] / (0.05 * SR))


def ringtone():
    notes = [880.0, 1318.5, 1046.5, 1318.5, 880.0, 1318.5, 1046.5, 1568.0]
    out = np.zeros(int(2.2 * SR))
    for k, hz in enumerate(notes):
        p = M.pluck(hz, 0.5, 0.3); a = int(k * 0.143 * SR); out[a:a + len(p)] += p[: len(out) - a]
    return out


def whoosh(d=0.6):
    return M.riser(d, 0.7)[::-1] * 0.5 + M.riser(d, 0.5) * 0.4


def sparkle():
    out = np.zeros(int(0.9 * SR)); rng = np.random.default_rng(3)
    for k in range(9):
        hz = rng.choice([1760, 2093, 2637, 3136, 3520]); p = M.pluck(hz, 0.4, 0.12); a = int(k * 0.07 * SR)
        out[a:a + len(p)] += p[: len(out) - a]
    return out


def make_sfx(kind):
    if kind == 'ring': return ringtone()
    if kind == 'whoosh': return whoosh()
    if kind == 'sparkle': return sparkle()
    if kind == 'thud': return M.kick(1.0) * 0.9 + np.pad(M.noise(0.12, 'lowpass', 500, 0.5), (0, int(0.23 * SR)))[: len(M.kick())]
    if kind == 'tick': return M.osc(1320, 0.08) * M.env(0.08, 0.3)
    if kind == 'buzz':
        x = M.osc(110, 0.35, 'square') * M.env(0.35, 0.25); return M.filt(x, 'lowpass', 1200)
    if kind == 'blip': return M.sfx('blip2')
    return M.sfx(kind)


def music_bed():
    bed = np.zeros((N, 2), np.float32)
    tracks = {n: load(os.path.join(MUSIC, n + '.wav')) for n in ('party', 'bedroom')}
    for b0, b1, name, src0, speed in MUSIC_CUES:
        x = tracks[name]
        n_out = int(t(b1 - b0) * SR)
        if speed == 1.0:
            a = int(src0 * SR); seg = x[a:a + n_out]
        else:                                          # slowed: resample (pitch drops too, like a slowed tape) + muffle
            a = int(src0 * SR); n_src = int(n_out * speed) + 2
            seg = resample_poly(x[a:a + n_src], int(1000 / speed), 1000, axis=0)[:n_out]
            seg = np.stack([M.filt(seg[:, c], 'lowpass', 900) for c in range(2)], 1) * 1.4
        f = int(0.03 * SR); env = np.ones(len(seg)); env[:f] = np.linspace(0, 1, f); env[-f:] = np.linspace(1, 0, f)
        gain = 0.55 if name == 'bedroom' and speed == 1.0 else 1.0
        if name == 'bedroom' and b1 == 29:                 # fade the lo-fi out under Brodie's beat drop
            k = int(t(1) * SR); env[-k:] *= np.linspace(1, 0, k)
        place(bed, t(b0), seg * env[:, None], gain)
    # tape stop at beat 46.5: the groove grinds down before slow motion
    a = int(t(45.5) * SR); src = bed[a:int(t(46.5) * SR)].mean(1)
    stop = M.tapestop(src, t(1.2))
    place(bed, t(46.5), stop, 0.9)
    # scratch at beat 4: the party track is cut by the needle
    a, b = int(t(3) * SR), int(t(4) * SR)
    place(bed, t(4), scratch(bed[a:b].copy()), 0.9)
    # end: fade the last two beats into the impact tail
    k = int(t(3) * SR); bed[-k:] *= np.linspace(1, 0.0, k)[:, None]
    return bed


def voice_track():
    v = np.zeros((N, 2), np.float32)
    for key, beat, _, _ in VOICE:
        x = load(os.path.join(VO, key + '.wav'))
        x = np.stack([M.filt(x[:, c], 'highpass', 90) for c in range(2)], 1)
        place(v, t(beat), x)
    return v / (np.abs(v).max() or 1) * 0.85


def duck_env(v):
    e = np.abs(v).mean(1); out = np.zeros_like(e); y = 0.0
    att, rel = np.exp(-1 / (0.015 * SR)), np.exp(-1 / (0.3 * SR))
    for i in range(0, len(e), 48):
        m = e[i:i + 48].max(); c = att if m > y else rel; y = c * y + (1 - c) * m; out[i:i + 48] = y
    return 1 - 0.75 * np.minimum(1, out / 0.08)            # about -12 dB under dialogue: the counts must cut through


def write(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    bed = music_bed()
    fx = np.zeros((N, 2), np.float32)
    for beat, kind in SFX:
        s = make_sfx(kind)
        if s is not None:
            place(fx, t(beat), s.astype(np.float32), 0.8)
    v = voice_track()
    duck = duck_env(v)[:, None]
    mix = bed * duck * 0.8 + fx * (0.6 + 0.4 * duck) + v
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    mix = mix / np.abs(mix).max() * 0.95
    write(os.path.join(FILM, 'mix.wav'), mix)
    print(f'mix.wav: {DURATION:.1f}s')


if __name__ == '__main__':
    main()
