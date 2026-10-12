"""How sure are we that the caller answered yes? Three sources can say so: our own reading of the transcript (classify), Gemini's
caller_answered tool (it hears the audio, not the transcript), and both hands up. Each gives a confidence; the open question
takes the best one.

  >= ACCEPT_AT          accept
  ASK_AT .. ACCEPT_AT   ask once: "Was that a yes? Say yes, or raise both hands." A second medium yes to the same question is
                        accepted, so this can never loop.
  a medium "no"         is treated as "wait", never as a cancel (stop and pain words always act at once, elsewhere)

  confidence(kind, addressed=True, agrees=False) -> float
"""
ACCEPT_AT = 0.7
ASK_AT = 0.4

BASE = dict(gesture=1.0, strong=0.9, weak=0.6, foreign=0.6, gemini=0.75)


def confidence(kind, addressed=True, agrees=False):
    """kind: 'gesture' / 'strong' (a clear yes in our words) / 'weak' ('ok I guess') / 'foreign' ('claro') / 'gemini' (the tool)."""
    if agrees: return 0.95                                   # our reading and Gemini's agree
    c = BASE.get(kind, 0.5)
    if kind == 'gemini' and not addressed: c = 0.5           # Gemini heard a yes in talk that wasn't to Brodie
    if kind in ('strong', 'weak', 'foreign') and not addressed: c -= 0.3
    return max(0.0, min(1.0, c))


def decide(conf, already_asked=False):
    """'accept' / 'ask' / 'ignore'."""
    if conf >= ACCEPT_AT or (already_asked and conf >= ASK_AT): return 'accept'
    if conf >= ASK_AT: return 'ask'
    return 'ignore'
