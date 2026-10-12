"""Dead air: no state may sit silent for long. Every wait in live.py has its own nudge, but real calls still had 50 s of silence
after a result (the choice wait reprompts once, then waits forever) and 350 s in chat when background talk kept resetting the quiet
clock. This is the fallback under all of them: when neither Brodie nor the caller has said anything for LIMITS[state] seconds, it
hands back one short nudge for that state. At most MAX_NUDGES per state entry, then the gap doubles (up to MAX_GAP_S), so a caller
who stepped away isn't nagged. Off while dancing, paused or in a safety stop. The limits sit just above each flow's own reprompt,
so the two never ask twice in a row.

Also: END_AFTER_S with nobody in view and nothing said ends the call (a forgotten call would burn the team's Live minutes), and the
silence statistics for the per-call metrics (longest silence, silences over 20 s).
"""

LIMITS = dict(CHOOSE=25.0, EXPLORE=25.0, OFFER=25.0, CONFIRM=30.0, REVIEW=22.0, PREPARE=30.0)
MAX_NUDGES = 3
MAX_GAP_S = 60.0
END_AFTER_S = 300.0
LONG_SILENCE_S = 20.0

NUDGES = dict(
    CHOOSE=["What are we doing? Learn, ghost battle, or the daily? Or raise both hands when your pick lights up.",
            "Pick a mode: say its name, or raise both hands when it lights up.",
            "I'm right here whenever you want to pick something."],
    EXPLORE=["Want to dance? Say learn, battle or daily. Or ask me anything.",
             "I'm here when you're ready. Say teach me, or ask me a question.",
             "Still there? Say menu to see what we can do."],
    OFFER=["Want to start? Say yes, or raise both hands.",
           "Say yes to start, or no for something else.",
           "Whenever you're ready, say yes or raise both hands."],
    CONFIRM=["Say ready, or raise both hands, when you're set.",
             "Take your time. Say ready when you are.",
             "I'm waiting on you. Say ready, or raise both hands."],
    REVIEW=["Say again, slower or next. Or raise both hands for yes.",
            "Still with me? Say next for a new move, or again to repeat it.",
            "Take a breather if you need it. Say next when you're ready."],
    PREPARE=["I'm still checking your camera. Tell me if you need a minute.",
             "Still setting up? Say arms only if you can't step back.",
             "No rush. I'll wait right here."],
)


class DeadAir:
    def __init__(self, now):
        self.now = now
        self.state = None; self.entered = now(); self.last_activity = now(); self.nudges = 0; self.gap = None
        self.last_seen = now()                       # someone in view
        self.last_caller = now()                     # the caller last spoke to Brodie (nudges don't count: they're ours)
        self.max_silence = 0.0; self.long_silences = 0; self._in_long = False

    def activity(self, t=None, caller=False):
        """Brodie spoke (or is playing audio), the caller spoke to him, or music runs: the silence ends."""
        t = self.now() if t is None else t
        self._close_silence(t)
        self.last_activity = max(self.last_activity, t)
        if caller: self.last_caller = max(self.last_caller, t)

    def seen(self, t=None):
        self.last_seen = self.now() if t is None else t

    def _close_silence(self, t):
        s = t - self.last_activity
        if s > self.max_silence: self.max_silence = s
        self._in_long = False

    def check(self, state, t=None):
        """('nudge', line) / ('end', line) / None. Call once per frame."""
        t = self.now() if t is None else t
        if state != self.state:
            self.state = state; self.entered = t; self.nudges = 0; self.gap = None
            self.last_activity = max(self.last_activity, t)
        limit = LIMITS.get(state)
        if limit is None:                                            # dancing, paused, safety: never nudge, never count as silence
            self.last_activity = t; return None
        quiet = t - self.last_activity
        if quiet > self.max_silence: self.max_silence = quiet
        if quiet > LONG_SILENCE_S and not self._in_long: self._in_long = True; self.long_silences += 1
        if t - self.last_seen > END_AFTER_S and t - self.last_caller > END_AFTER_S:
            return ('end', "I can't see or hear anyone, so I'll hang up for now. Call me back any time!")
        need = limit if self.gap is None else self.gap
        if quiet < need: return None
        self.nudges += 1
        if self.nudges >= MAX_NUDGES: self.gap = min(MAX_GAP_S, (self.gap or limit) * 2)
        lines = NUDGES.get(state) or NUDGES['EXPLORE']
        self.last_activity = t                                       # the nudge itself is activity (and its audio keeps it fresh)
        return ('nudge', lines[(self.nudges - 1) % len(lines)])
