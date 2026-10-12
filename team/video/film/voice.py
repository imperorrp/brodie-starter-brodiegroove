"""Dialogue for the film: Brodie in his product voice (Puck), Maya in Kore. Each fresh clip is transcribed to catch ad-libs.

Run:  python -m team.video.film.voice [key ...]   (keys force a re-render)
Writes data/tmp/film/vo/<key>.wav + lines.json
"""
import os, sys
from team.video.voice import render
from team.video.film.gen import FILM

OUT = os.path.join(FILM, 'vo')
HYPE = ('Puck', '(hyped, warm, playful cartoon dance coach, big grin, punchy)')
DEADPAN = ('Puck', '(cartoon dance coach, dry deadpan, unimpressed but kind, short)')
COACH = ('Puck', '(cartoon dance coach, focused and precise, encouraging)')
CALM = ('Puck', '(cartoon dance coach, slow, relaxed, soothing)')
MAYA_WRY = ('Kore', '')                               # directions here get read aloud; plain text reads deadpan anyway
MAYA_ASK = ('Kore', '(young woman out of breath mid-dance, pleading, plain, no gasp before speaking)')
MAYA_YELP = ('Kore', '(young woman startled, a short loud yelp)')
LINES = {
    'm_thatsme': (MAYA_WRY, "Yep. That's me. Two left feet."),
    'm_help':    (('Kore', ''), 'Ugh. I need a dance coach.'),
    'b_yo':      (HYPE, "Yo! Heard you need a coach."),
    'm_yelp':    (MAYA_YELP, 'Aaah!'),
    'b_phoneup': (DEADPAN, 'Phone up. We dance.'),
    'b_ghost':   (('Puck', ''), 'Follow my ghost. Five, six, seven, eight!'),
    'b_tip':     (COACH, 'Left arm higher on five to seven.'),
    'm_slower':  (MAYA_ASK, 'Brodie... slower.'),
    'b_slower':  (CALM, 'Got you. Slowing it down.'),
    'b_clean':   (HYPE, 'Clean!'),
    'b_locked':  (HYPE, 'Locked in!'),
    'b_combo':   (HYPE, "Combo's heating up!"),
    'b_chef':    (HYPE, "Chef's kiss."),
    'b_cta':     (HYPE, 'Your first eight count is one call away.'),
}

if __name__ == '__main__':
    render(LINES, OUT, [a for a in sys.argv[1:] if not a.startswith('--')])
