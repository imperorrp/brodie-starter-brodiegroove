"""The film's cut, in beats of the Lyria party track (105 BPM: 1 beat = 0.5714 s). audio.py and make_film.py both read this.

Clip times are seconds into data/tmp/film/clips/<id>.mp4 (24 fps Veo output, 720x1280).
"""
import os
from team.video.film.gen import FILM

BPM = 105
BEAT = 60 / BPM
FPS = 30
W, H = 1080, 1920
TOTAL_BEATS = 96
DURATION = TOTAL_BEATS * BEAT          # 54.9 s
CLIPS = os.path.join(FILM, 'clips')
MUSIC = os.path.join(FILM, 'music')
VO = os.path.join(FILM, 'vo')


def t(beat):
    return beat * BEAT


# (start beat, end beat, clip, params). params: at (clip second at the start), rate, freeze (hold one frame), phone (show the
# clip inside a phone over a blurred copy), punch, zoom (z0, z1), desat, chips [(text, colour, beat)], labels, burst, stamp
SHOTS = [
    # cold open: the party, the flail, the splash... freeze.
    (0, 4, 's01', dict(at=2.0, rate=0.765, punch=1.06)),     # already mid-flail, easing into the crash
    (4, 9, 's01', dict(at=3.75, freeze=True, desat=0.55, zoom=(1.0, 1.10), labels=7.4)),
    # that night: the phone rings, Brodie bursts out
    (9, 14, 's03', dict(at=0.6, banner=11.8)),
    (14, 22, 's04', dict(at=0.0)),
    (22, 28, 's05', dict(at=0.3)),
    # the lesson: ghost, count-in, a miss, a tip
    (28, 38, 's06', dict(at=0.4, punch=1.06, dots=True, counts=(32, 33, 34, 35), chip=('GHOST · ONE BEAT AHEAD', 'cyan', 29))),
    (38, 43, 's07', dict(at=0.0, tip=('LEFT ARM HIGHER', 39.4))),      # the shot has its own neon MISS
    # "Brodie... slower."
    (43, 53, 's08', dict(at=0.0, chip=('0.5×', 'cyan', 47))),
    # montage: days pass, it clicks
    (53, 55, 's09a', dict(at=0.0, rate=2.0, chip=('DAY 1', 'paper', 53))),
    (55, 58, 's09c_hud', dict(at=0.4, phone=True, chip=('DAY 3', 'paper', 55))),
    (58, 60.5, 's09b', dict(at=0.0, punch=1.06, chip=('DAY 5', 'paper', 58))),
    (60.5, 64, 's09c', dict(at=1.6, punch=1.06, chip=('DAY 7', 'paper', 60.5), combo=(61, 'x8  ON FIRE'))),
    # the party again: same drop, she nails it
    (64, 75, 's10', dict(at=0.6, punch=1.10)),
    (75, 81, 's11', dict(at=0.0, burst=(75.2, 'PERFECT', 'volt'), stamp=(77.4, 'S', 'BEAT ASSASSIN'))),
    (81, 87, 's12', dict(at=0.3)),
    (87, TOTAL_BEATS, 'end', {}),
]

# dialogue: (key, start beat, side 'b' Brodie | 'm' Maya, caption or None)
VOICE = [
    ('m_thatsme', 4.2, 'm', "Yep. That's me. Two left feet."),
    ('m_help', 9.3, 'm', 'Ugh. I need a dance coach.'),       # she asks... and the phone answers
    ('b_yo', 18.0, 'b', 'Yo! Heard you need a coach.'),
    ('m_yelp', 23.9, 'm', None),
    ('b_phoneup', 25.0, 'b', 'Phone up. We dance.'),
    ('b_ghost', 29.5, 'b', 'Follow my ghost. 5, 6, 7, 8!'),
    ('b_tip', 38.7, 'b', 'Left arm higher on 5 to 7.'),
    ('m_slower', 42.9, 'm', 'Brodie... slower.'),
    ('b_slower', 48.1, 'b', 'Got you. Slowing it down.'),
    ('b_clean', 55.4, 'b', 'Clean!'),
    ('b_locked', 58.3, 'b', 'Locked in!'),
    ('b_combo', 60.7, 'b', "Combo's heating up!"),
    ('b_chef', 78.5, 'b', "Chef's kiss."),
    ('b_cta', 81.5, 'b', None),            # the end card says it
]

# music: (start beat, end beat, track, source second at the start, speed)
MUSIC_CUES = [
    (0, 4, 'party', 9.2128, 1.0),          # drums already in: it slams from frame one
    (9, 29, 'bedroom', 0.4284, 1.0),
    (28, 46.5, 'party', 11.4985, 1.0),     # Brodie drops the beat
    (46.5, 53, 'bedroom', 9.57, 0.75),     # slow motion: the lo-fi cue, slowed and muffled
    (53, TOTAL_BEATS, 'party', 103.528, 1.0),   # build... the drop lands on beat 64
]

SFX = [(4, 'scratch'), (11.7, 'ring'), (12.9, 'ring'), (15.6, 'whoosh'), (16.1, 'sparkle'), (24.4, 'thud'),
       (32, 'tick'), (33, 'tick'), (34, 'tick'), (35, 'tick'), (38.6, 'buzz'), (46.5, 'tapestop'),
       (53, 'whoosh'), (55, 'blip'), (58, 'blip'), (60.5, 'blip'), (75.2, 'perfect'), (77.4, 'stamp'), (87, 'impact')]

FLASH = [4, 14, 28, 53, 64, 75, 87]
