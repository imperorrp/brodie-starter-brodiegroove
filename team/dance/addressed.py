"""Was that said to Brodie? One phone in a living room hears everybody: friends chatting, a TV, song lyrics. Real calls had
Brodie answering a background conversation ("ask her what reasoning chain...") as if the caller had asked. A transcript gets a
score in [0, 1]; ADDRESSED_AT and above counts as talking to Brodie. Below it the line never answers a question, never makes an
offer and never resets the silence timers; well below (IGNORE_AT) and Gemini starts replying anyway, the reply is cut.

Signals (all from the text and the moment it arrived):
  + a question of ours is open and the reply is answer-shaped (yes / no / wait / a command, 6 words or fewer)
  + "Brodie", "you"/"your", a command, a mode or a style word
  + short (12 words or fewer)
  - third-person story telling ("she said", "he was", a name that isn't the caller's)
  - long (over 20 words) with none of our words in it
  - it arrived while Brodie was talking (the caller talking over him to someone else, or the speaker echo)
  - garbled (mostly non-English tokens)
"""
import re

ADDRESSED_AT = 0.5
IGNORE_AT = 0.2

_OUR_WORDS = re.compile(r"\b(brodie|dance|dancing|learn|teach|battle|ghost|daily|squad|telephone|class|routine|move|again|next|slower|faster|"
                        r"stop|pause|resume|ready|yes|yeah|yep|no|nope|start|go|menu|code|hip ?hop|house|pop|k-?pop|waack|krump|jazz|lock|"
                        r"easy|hard|beginner|rematch|skip|explain|bye)\b", re.I)
_YOU = re.compile(r"\b(you|your|you'?re|u)\b", re.I)
_THIRD = re.compile(r"\b(she|he|they|her|his|them)\s+(said|says|told|was|were|is|did|does|went|wants|thinks|got)\b|\b(my (mom|mum|dad|brother|sister|friend|boss)|"
                    r"did you see|on tv|this episode|the show)\b", re.I)
_ANSWER = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok(ay)?|no|nope|nah|not yet|wait|hold on|one sec|ready|let'?s go|go|again|next|slower|faster|"
                     r"stop|pause|skip|got it|i'?m ready|do it|bet|i'?m down)\b", re.I)
_EN = re.compile(r"^[a-z'\-]+$")
_COMMON = set("""the a an and or but i you me my we it is are was to of in on at for with that this what how can do don't not so just like
yes no yeah okay ok go let's let lets please now one more time hi hey hello thanks thank good great nice cool""".split())


def score(text, question_open=False, brodie_speaking=False, caller_name=None):
    t = (text or '').strip(); low = t.lower()
    words = re.findall(r"[a-zA-Z'\-]+", t)
    if not words: return 0.0
    n = len(words); s = 0.3
    if question_open and n <= 6 and _ANSWER.search(low): s += 0.4
    if _OUR_WORDS.search(low) or _YOU.search(low): s += 0.3
    if n <= 12: s += 0.2
    names = [w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", t) if w.lower() not in _COMMON and w != 'Brodie' and w != caller_name]
    if _THIRD.search(low) or (names and n > 6 and not re.search(r"\b(i'?m|my name|call me)\b", low)): s -= 0.4
    if n > 20 and not _OUR_WORDS.search(low): s -= 0.3
    if brodie_speaking: s -= 0.1 if _OUR_WORDS.search(low) else 0.3   # talking over Brodie, likely to someone else (unless it's our words)
    weird = sum(1 for w in words if not _EN.match(w.lower())) + sum(1 for w in words if len(w) > 3 and not re.search(r'[aeiouy]', w.lower()))
    if n >= 3 and weird / n > 0.4: s -= 0.2
    return max(0.0, min(1.0, s))


def addressed(text, **kw):
    return score(text, **kw) >= ADDRESSED_AT
