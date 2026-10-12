"""The six modes, one source for the menu cards, the hello line and the picker. Copy, tags and colours are the BrodieGroove
landing page's (team/web/groove/index.html, read-only): the call shows the same cards the site advertises.

  CARDS: list of dicts (mode, title, tag, colour, body, say)
  by_mode(mode) -> the card or None
  HELLO, NEXT_LINE: what Brodie says when the cards come up
"""

CARDS = (
    dict(mode='learn', title='Learn', tag='Solo', colour='volt', icon='learn', say='learn a routine',
         body='One small move at a time. A slow demo, a short breakdown, then you try it.'),
    dict(mode='battle', title='Ghost Battle', tag='3 rounds', colour='pink', icon='ghost', say='a ghost battle',
         body='Three rounds against a ghost. Win and you get a four-digit code.'),
    dict(mode='daily', title='Daily Challenge', tag='Daily', colour='cyan', icon='daily', say='the daily',
         body='Everyone gets the same routine each day. One shot at the daily board.'),
    dict(mode='squad_battle', title='Squad Battle', tag='Up to 4', colour='amber', icon='squad', say='a squad battle',
         body='Friends in one camera. Each dancer gets a colour tag and their own score.'),
    dict(mode='telephone', title='Telephone', tag='Party', colour='pink', icon='phone', say='dance telephone',
         body='Copy a move, pass it on, and watch it change along the way.'),
    dict(mode='class', title='Class', tag='Group', colour='volt', icon='class', say='a class',
         body='Brodie leads the routine, the room follows, and everyone gets scored.'),
)
MODES = tuple(c['mode'] for c in CARDS)

HELLO = ("Hey, I'm Brodie! What are we doing: learn a routine, a ghost battle, or the daily? "
         "With friends: squad battle, telephone or a class.")
NEXT_LINE = "Nice work! What next? Pick any mode, or say bye."
CYCLE_LINE = "Or raise both hands when the one you want lights up."
NOISY_LINE = "It's a bit loud in there. Raise both hands when your mode lights up."


def by_mode(mode):
    return next((c for c in CARDS if c['mode'] == mode), None)


def index_of(mode):
    return next((i for i, c in enumerate(CARDS) if c['mode'] == mode), None)
