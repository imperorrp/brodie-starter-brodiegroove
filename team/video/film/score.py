"""The film's music: a Lyria party anthem and a lo-fi bedroom cue, decoded and beat-tracked so the edit can cut on their beats.

Run:  python -m team.video.film.score [party=TAKE] [bedroom=TAKE]
Writes data/tmp/film/music/{party,bedroom}.wav (48 kHz stereo) and beats.json (bpm, first beat, beat times, energy per bar).
"""
import os, sys, json, subprocess
import numpy as np
from scipy.signal import butter, sosfilt, find_peaks
from team.video.film.gen import music, FILM

OUT = os.path.join(FILM, 'music')
SR = 48000
PROMPTS = {
    'party': ('lyria-3-pro-preview',
              'Instrumental only, no vocals. An energetic modern hip-hop / dance-pop party anthem at exactly 105 BPM in '
              'A minor. Punchy 808 kick and sub bass, crisp claps and finger snaps, bright plucky synth stabs, a catchy '
              'whistled lead hook. The full beat starts right away with no long intro. Steady four-on-the-floor energy '
              'with a short breakdown in the middle and then a huge drop.'),
    'bedroom': ('lyria-3-clip-preview',
                'Instrumental lo-fi chill hip-hop at exactly 105 BPM in A minor, no vocals. Soft Rhodes keys, warm bass, '
                'lazy swung drums, vinyl crackle, a gentle hopeful mood, late night in a bedroom.'),
}


def decode(mp3, wav):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mp3, '-ar', str(SR), '-ac', '2', wav], check=True)
    import wave
    with wave.open(wav) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float32) / 32768


def onset_env(x, hop=480):
    """spectral-flux-ish onset strength at 100 Hz: low band (kick) weighted with the full band."""
    m = x.mean(1)
    lo = sosfilt(butter(4, 150, 'lowpass', fs=SR, output='sos'), m)
    out = []
    for sig, wgt in ((lo, 1.5), (m, 1.0)):
        n = len(sig) // hop
        e = np.sqrt((sig[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-9)
        d = np.maximum(0, np.diff(np.log(e), prepend=np.log(e[0])))
        out.append(d * wgt)
    return out[0] + out[1], SR / hop


def track(x):
    env, fr = onset_env(x)
    env = env - env.mean()
    best = None
    for bpm in np.arange(90, 130.01, 0.05):                    # tempo: comb correlation
        per = 60 / bpm * fr
        idx = np.arange(0, len(env) - 1, per)
        for ph in np.linspace(0, per, 24, endpoint=False):
            s = env[np.clip((idx + ph).astype(int), 0, len(env) - 1)].sum()
            if best is None or s > best[0]:
                best = (s, bpm, ph)
    _, bpm, ph = best
    first = ph / fr
    beats = np.arange(first, len(x) / SR, 60 / bpm)
    rms = np.sqrt(np.convolve(x.mean(1) ** 2, np.ones(SR // 10) / (SR // 10), 'same'))
    bar_energy = [float(rms[int(b * SR):int((b + 4 * 60 / bpm) * SR)].mean()) for b in beats[::4]]
    return dict(bpm=round(float(bpm), 3), first_beat=round(first, 4), beats=[round(float(b), 4) for b in beats],
                bar_energy=[round(e, 4) for e in bar_energy])


def main():
    takes = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for name, (model, prompt) in PROMPTS.items():
        mp3 = music(prompt, model, name=name, take=int(takes.get(name, 0)))
        x = decode(mp3, os.path.join(OUT, name + '.wav'))
        meta[name] = track(x)
        m = meta[name]
        bars = ''.join(' .:-=+*#%@'[min(9, int(e / max(m['bar_energy']) * 9.99))] for e in m['bar_energy'])
        print(f'{name}: {len(x) / SR:.1f}s  bpm {m["bpm"]}  first beat {m["first_beat"]}s\n  energy per bar: {bars}')
    json.dump(meta, open(os.path.join(OUT, 'beats.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
