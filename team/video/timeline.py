"""The pitch video's edit, in beats of the landing page's 100 BPM groove (1 beat = 0.6 s, 18 frames at 30 fps).

Everything else reads this: music.py lays the soundtrack, SFX and voice on the same grid, make_pitch.py draws the frames.
Clip times are seconds in the source file (data/tmp/pitch/src/, a frozen copy of the eval renders).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PITCH = os.path.join(ROOT, 'data', 'tmp', 'pitch')
SRC = os.path.join(PITCH, 'src')
HUD = os.path.join(ROOT, 'data', 'tmp', 'hud')
WEB = os.path.join(ROOT, 'team', 'web', 'groove', 'assets')
FONTS = os.path.join(ROOT, 'team', 'dance', 'assets', 'fonts')

BPM = 100
BEAT = 60 / BPM
FPS = 30
W, H = 1080, 1920
TOTAL_BEATS = 77                       # 76 beats of groove + one beat of tail
DURATION = TOTAL_BEATS * BEAT          # 46.2 s


def t(beat):
    return beat * BEAT


# (start beat, end beat, kind, params). kind: call | clip | freeze | still | end
# clip: src file, at (source second at the segment start), rate (source seconds per output second), zoom (x, y, scale) in
# source pixels, punch (scale kick on the first beat), chip / ring overlays.
SEGMENTS = [
    # 1 HOOK: Brodie is calling... answer, ask, drop.
    (0, 4, 'call', {}),
    (4, 8, 'clip', dict(src='space_gesture.mp4', at=0.0)),
    (8, 12, 'clip', dict(src='battle.mp4', at=13.9, punch=1.10)),
    # 2 TEACH: watch the ghost, then dance with it.
    (12, 16, 'clip', dict(src='flow.mp4', at=24.0, chip=('STEP BY STEP', 'volt', 12.5))),
    (16, 20, 'clip', dict(src='flow.mp4', at=30.6, zoom=(360, 640, 1.18), chip=('GHOST · ONE BEAT AHEAD', 'cyan', 16.25))),
    (20, 24, 'clip', dict(src='flow.mp4', at=38.2)),
    # 3 REAL-TIME FEEDBACK: count in on the beat, scored every count, one tip at a time.
    (24, 31, 'clip', dict(src='battle.mp4', at=27.0, rate=0.8333, burst=(30.0, 'PERFECT', 'volt'))),
    (31, 32.5, 'clip', dict(src='duel.mp4', at=46.6)),
    (32.5, 34, 'freeze', dict(src='duel.mp4', at=47.6, zoom_to=(360, 930, 1.45))),
    (34, 36, 'clip', dict(src='duel.mp4', at=47.6)),
    # 4 VOICE CONTROL: "Brodie, slower." -> the tempo drops to the SLOW level.
    (36, 38, 'clip', dict(src='duel.mp4', at=48.9)),
    (38, 44, 'clip', dict(src='flow.mp4', at=33.0, rate=0.75, ring=(116, 87, 160, 44), chip=('0.5×', 'cyan', 38.25))),
    # speed ramp back up, one cut per beat
    (44, 45, 'clip', dict(src='flow.mp4', at=43.0, punch=1.08, chip=('0.6×', 'cyan', 44))),
    (45, 46, 'clip', dict(src='flow.mp4', at=39.2, punch=1.08, chip=('0.75×', 'amber', 45))),
    (46, 47, 'clip', dict(src='duel.mp4', at=49.2, punch=1.08, chip=('0.9×', 'pink', 46))),
    (47, 48, 'clip', dict(src='battle.mp4', at=63.6, punch=1.08, chip=('FULL SPEED', 'volt', 47))),
    # 5 PAYOFF: one cut every two beats.
    (48, 50, 'still', dict(src='10_challenge_fire.png', zoom=(1.0, 1.12), punch=1.10)),
    (50, 52, 'clip', dict(src='battle.mp4', at=67.3, punch=1.08)),
    (52, 54, 'clip', dict(src='battle.mp4', at=68.4, punch=1.12)),
    (54, 56, 'clip', dict(src='battle.mp4', at=73.3, punch=1.08)),
    (56, 58, 'still', dict(src='17_levelup.png', zoom=(1.0, 1.10), punch=1.10)),
    (58, 64, 'still', dict(src='19_scorecard.png', zoom=(1.04, 1.0), punch=1.14, shutter=62.7)),
    # 6 CTA
    (64, TOTAL_BEATS, 'end', {}),
]

# voice: (key in data/tmp/pitch/vo, start beat, max seconds or None, caption side 'b' Brodie | 'u' user)
VOICE = [
    ('u_teach', 4.15, None, 'u'),
    ('b_ready', 7.1, None, 'b'),
    ('b_teach', 12.5, None, 'b'),
    ('b_tip', 31.4, None, 'b'),
    ('u_slower', 36.3, None, 'u'),
    ('b_slower', 39.2, None, 'b'),
    ('b_combo', 52.2, None, 'b'),
    ('b_screen', 58.3, None, 'b'),
    ('b_cta', 66.6, None, 'b'),
]

# one-shot sound effects at beats
SFX = [
    (3.55, 'swipe'), (8, 'impact'), (30, 'perfect'), (24, 'tick'), (25, 'tick'), (26, 'tick'), (27, 'tick'),
    (32.5, 'zoom'), (38, 'tapestop'), (44, 'blip1'), (45, 'blip2'), (46, 'blip3'), (47, 'perfect'), (48, 'impact'),
    (52, 'perfect'), (58, 'stamp'), (62.7, 'shutter'), (64, 'impact'), (72, 'final'),
]

# beats where the picture flashes (cuts that land on a hit)
FLASH = [8, 24, 44, 45, 46, 47, 48, 58, 64]

# captions are the spoken words; this mirrors team/video/voice.py LINES
CAPTIONS = {
    'u_teach': 'Brodie, teach me hip hop!',
    'b_ready': "Let's go! Ready to dance?",
    'b_teach': 'Counts 1 to 2, both arms go up. Follow my ghost.',
    'b_tip': 'Lift your left arm higher on 5 to 7.',
    'u_slower': 'Brodie, slower.',
    'b_slower': 'Got you. Slowing it down.',
    'b_combo': 'Combo Menace!',
    'b_screen': 'Human Metronome! Screenshot this.',
    'b_cta': None,                       # the end card says it
}
