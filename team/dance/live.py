"""Dance mode inside the Brodie worker, and a simulator that runs the same code without LiveKit.

Worker use (perception off: no licensed classifier, the floor always open, Gemini for talk):
  .venv/Scripts/python -m facetime.worker --perception off --voice gemini --persona team/dance/persona.md \
      --handlers team/dance/live.py --agent-name "" dev
  env: DANCE_KEY=<library or gen_* key> (default: a generated hip-hop starter routine), DANCE_MODE=learn|battle|daily|telephone|class,
       DANCE_TEMPO=1.0 (multiplier), DANCE_RTT=0.3, DANCE_FPS=15, DANCE_POSE=full|lite, DANCE_DIFFICULTY=beginner|intermediate|advanced,
       DANCE_CALL_HANDLE=<the Chert line's FaceTime number or email, for the scorecard QR>, DANCE_TOOLS=1 (Gemini function tools),
       DANCE_UNLOCK_ALL=0, DANCE_NUDGE_S=45, DANCE_LOBBY=0 (test mode: one routine, no chat), DANCE_DEBUG_MP4=path (developer only: records what the caller sees; never on by default)

How it plugs in: the worker imports this file as the handlers module (on_event below). At import we register a consumer
coroutine with the worker; once a caller is bound the consumer publishes a video track ('coach-video'), reads the newest caller
frame the worker already decodes (job.latest), runs multi-person MediaPipe pose at up to DANCE_FPS in one background thread
(a primary-dancer lock keeps walk-ins from stealing the scoring; 2+ dancers in frame unlock squad mode), drives the PhraseEngine,
renders the game HUD, plays a synthetic metronome, and asks Gemini for short lines between phases. Gemini also gets function tools
(tools.py). The caller's words (voice events with --log-caller) and held poses (gestures.py) steer it.

Simulator (no LiveKit, no phone):
  .venv/Scripts/python -m team.dance.live --simulate data/tmp/mac_handoff/segments/run3_jacks.mp4 --mode learn --lobby \
      --say "2:I'm Maya, teach me hip hop easy" --say "8:ready" [--players 2] [--out data/tmp/sim.mp4]
It feeds the video as if it were the caller, logs every cue and every voice prompt with timestamps, and writes the composed
video track to --out so you can watch exactly what the caller would have seen.
"""
import asyncio, datetime, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, 'src')):
    if p not in sys.path: sys.path.insert(0, p)

from team.dance.reference import load_key, list_keys           # noqa: E402  (absolute imports: the worker loads this file by path)
from team.dance.engine import PhraseEngine                       # noqa: E402
from team.dance.render import compose, compose_idle, draw_card, to_canvas, canvas_size, framing_hint, visibility as render_visibility, W, H   # noqa: E402
from team.dance import audio as dance_audio                      # noqa: E402
from team.dance import progression, choreo, feedback            # noqa: E402
from team.dance import cues, pacing                              # noqa: E402
from team.dance import ghost as ghost_mod                        # noqa: E402
from team.dance.orient import Orient, facing_phrase              # noqa: E402
from team.dance import addressed as addr_mod, answers as answers_mod, audio_in   # noqa: E402
from team.dance.guards import FrameBudget, CameraShift, VideoFreeze               # noqa: E402
from team.dance.fx import FxQueue                                # noqa: E402
from team.dance.gestures import GestureDetector                  # noqa: E402
from team.dance.telemetry import Telemetry                       # noqa: E402
from team.dance.multipose import PlayerTracker                   # noqa: E402
from team.dance.squad import SquadEngine                         # noqa: E402
from team.dance.watchdog import DeadAir                          # noqa: E402
from team.dance import modes as mode_cards                       # noqa: E402

from team.dance.state import STATE                               # noqa: E402  shared across every per-job copy of this file
# The name must be capitalised as a name ("I'm down" / "I'm bored" are not names); "my name is" / "call me" are the reliable forms.
NAME_RE = re.compile(r"(?i:\b(my name is|my name's|call me|i am|i'm|im|this is))\s+([A-Z][a-z]{1,15})\b")
LTR_RE = re.compile(r"left to right[,:]?\s+(.*)", re.I)
NOT_NAMES = {'Ready', 'Here', 'Good', 'Fine', 'Done', 'Back', 'Not', 'So', 'Just', 'Going', 'Trying', 'The', 'And', 'Learning', 'Dancing', 'Okay',
             'Down', 'In', 'Bored', 'Fun', 'From', 'Gonna', 'Sorry', 'Sure', 'Cool', 'Great', 'Tired', 'Excited', 'New', 'Into', 'Like', 'Pretty',
             'Really', 'Very', 'Still', 'Also', 'Kinda', 'Kind', 'Super', 'Hip', 'Pop', 'Learning', 'Interested', 'Ok', 'Alright', 'All', 'Awesome',
             'Doing', 'Feeling', 'Happy', 'Nervous', 'Bad', 'Better', 'Busy', 'Free', 'Home', 'Late', 'Out', 'On', 'At', 'With', 'A', 'An', 'It', 'That',
             'What', 'Yes', 'Yeah', 'No', 'Not', 'Your', 'My', 'Me', 'Brodie', 'Brody', 'Hey', 'Hi', 'Hello'}
# phrase -> command; longer phrases are matched first, priority decides between several in one utterance
CMD_WORDS = {
    # safety and session
    'pain': 'pain', 'hurts': 'pain', 'hurt': 'pain', 'dizzy': 'pain', 'lightheaded': 'pain', 'light headed': 'pain', 'injured': 'pain',
    'cramp': 'pain', 'twisted my': 'pain', "can't breathe": 'pain',
    'stop': 'stop', "i'm done": 'stop', 'end the call': 'stop', "that's enough": 'stop',
    'forget me': 'forget', 'delete my data': 'forget', 'delete me': 'forget', 'delete my scores': 'forget',
    'pause': 'pause', 'hold on': 'pause', 'hang on': 'pause', 'give me a sec': 'pause', 'give me a second': 'pause',
    'resume': 'resume', 'continue': 'resume', 'keep going': 'resume', "i'm back": 'resume', 'unpause': 'resume',
    # flow
    'again': 'again', 'repeat': 'again', 'one more time': 'again', 'slower': 'slower', 'slow it down': 'slower', 'slow down': 'slower',
    'faster': 'faster', 'speed it up': 'faster', 'next': 'next', 'skip': 'next',
    'skip this move': 'skip', 'skip this one': 'skip', 'skip this step': 'skip', 'skip it': 'skip', 'skip the move': 'skip',
    'simpler': 'simplify', 'simplify': 'simplify', 'break it down': 'simplify', 'too hard': 'simplify', 'too fast': 'slower',
    'explain': 'explain', 'how do i': 'explain', 'what do i do': 'explain', 'show me how': 'explain',
    'full speed': 'full', 'whole routine': 'full', 'full routine': 'full', 'whole thing': 'full',
    'rematch': 'rematch', 'run it back': 'rematch', 'harder': 'harder', 'level up': 'harder', 'challenge me': 'harder', 'easier': 'easier',
    # modes
    'squad battle': 'squad_battle', 'battle each other': 'squad_battle', 'squad mode': 'squad_battle', 'squad': 'squad_battle',
    'with my friends': 'squad_battle', 'with my friend': 'squad_battle',
    'learn together': 'squad_learn', 'dance together': 'squad_learn', 'learn it together': 'squad_learn',
    'battle': 'battle', 'ghost battle': 'battle', 'duel': 'battle', 'freestyle': 'battle', 'learn': 'learn', 'teach me': 'learn',
    'learn a routine': 'learn', 'daily challenge': 'daily', 'daily': 'daily', 'the daily': 'daily',
    'telephone': 'telephone', 'dance telephone': 'telephone', 'party game': 'telephone',
    # class = the site's class: Brodie leads, the room follows. Only explicit phrases ("I'm a student" in passing must not switch).
    'class': 'class', 'group class': 'class', 'class mode': 'class', 'run a class': 'class', 'lead a class': 'class', 'do a class': 'class',
    # the teacher-records flow: a teacher records phrases, students join later by code, the teacher hears a report
    'record my class': 'class_record', 'teach my class': 'class_record', 'record a class': 'class_record', 'join a class': 'class_record',
    'join the class': 'class_record', 'report': 'report',
    'loop it': 'loop', 'loop that': 'loop', 'loop that part': 'loop', 'just that part': 'loop', 'that part again': 'loop', 'loop the': 'loop'}
MENU_RE = re.compile(r"\b(what can you do|what do you do|what (are|r) (my|the) (options|choices)|options|menu|what else|what (modes|games) (do you have|are there)|"
                     r"what do you have|what can we do|what can i do)\b", re.I)
VAGUE_RE = re.compile(r"\b(i ?d ?k|i don'?t know|dunno|whatever|you pick|you choose|anything( is fine)?|surprise me|dealer'?s choice|up to you|not sure)\b", re.I)
SOLO_RE = re.compile(r"\b(solo|just me|only me|by myself|on my own|alone)\b", re.I)
# Starting is never a keyword: it takes an explicit request in chat, the camera setup, and a clear yes to "ready?" (classify()).
# Callers are mostly under 25 and transcripts are noisy: fillers ("oh yeah", "um okay"), slang ("bet", "I'm down"), stretched words
# ("yesss", "let's gooo") and a garbled first word ("de Hello yes") must still read as a yes. The table in tests/test_live.py pins it.
FILLER = r"(?:(?:oh|uh|um+|hmm+|so|and|well|hell|bro|yo|hey|man|dude|i mean|like|okay|ok|alright|all right|yeah|yes|cool|right)\W+)*"
AFFIRM_WORDS = (r"(?:ye+a*h*|yes+(?:sir)?|yep+|yup+|ya+|yah|sure|o?k(?:ay+|k+)?|okie|ready|i'?m ready|ready to go|let'?s go+|lets go+|let'?s do (?:it|this)|"
                r"let'?s get it|go+|go ahead|start|do it|alright|all right|all set|i'?m set|of course|definitely|absolutely|uh[- ]?huh|mhm+|mm[- ]?hmm|"
                r"bet|i'?m (?:down|in|game)|for sure|sounds good|got it|gotcha|cool|perfect|yes please|why not|i think so|i guess|totally|deal|lfg|"
                r"makes sense|let'?s try( it)?|i'?m good to go|good to go|"
                r"claro|ja|si|sí|oui|vale|bien|okey|dale)")                       # a "yes" the transcriber heard in another language
QUESTION_RE = re.compile(r"^\W*" + FILLER + r"(what|what's|whats|where|how|which|why|who|when|should|shall|can i|could i|can you|could you|would|"
                         r"are you|is it|is that|is this|is there|do you|do i|does|did|am i)\b", re.I)
DENY_EXEMPT_RE = re.compile(r"\b(no (cap|problem|worries|doubt)|can'?t wait|don'?t stop|never stop|i'?m good to go|good to go|not bad|why not|not a problem)\b", re.I)
DENY_RE = re.compile(r"\b(no|nope|nah|not|i'?m not|wait|hold on|hang on|one sec(ond)?|give me a (sec|second|minute)|later|never ?mind|cancel|stop|"
                     r"i'?m good|no thanks?|no thank you|not today|i don'?t (want|think)|don'?t want|do not want|go away)\b", re.I)
CANCEL_RE = re.compile(r"\b(never ?mind|cancel|stop|not now|not right now|later|forget it|not today|i'?m good|no thanks?|maybe later)\b", re.I)
HOLD_RE = re.compile(r"\b(wait|hold on|hang on|one sec(ond)?|give me a (sec|second|minute)|not yet)\b", re.I)
AFFIRM_RE = re.compile(r"^\W*" + FILLER + AFFIRM_WORDS + r"\b", re.I)
STRONG_YES_RE = re.compile(r"\b(yes+|yeah+|yep|i'?m ready|let'?s go+|let'?s do it|sounds good|i'?m down|got it)\b", re.I)   # anywhere in a short utterance
REQUEST_RE = re.compile(r"\b(teach me|let'?s (do|learn|dance|try|start|battle|play)|let us (do|learn|dance|try|start)|i (want|wanna|would like|'d like|d like) to (learn|dance|do|try|battle|play|start)|"
                        r"i want (a|the) (lesson|battle|challenge|routine)|start (a |the |my )?(lesson|battle|challenge|routine|daily|dancing|class)|show me (a|the|some|how to)|"
                        r"(can|could) we (start|dance|learn|do|try|battle)|(do|play) the daily( challenge)?|rematch|run it back|battle me|go again|one more (round|time)|another round)\b", re.I)
REMATCH_RE = re.compile(r"\b(rematch|run it back|go again|one more (round|time)|another round|same (one|routine) again)\b", re.I)


def classify(text):
    """What an utterance means for the flow: {request, rematch, affirm, deny, cancel, hold, question}. Questions never request
    (except "can we start / could we dance"); a negation cancels an affirmation ("I'm not ready")."""
    t = (text or '').strip().lower()
    t_deny = DENY_EXEMPT_RE.sub(' ', t)                                   # "no worries", "can't wait", "don't stop" are not a no
    deny = bool(DENY_RE.search(t_deny))
    words = re.findall(r"[\w']+", t)
    affirm = bool(AFFIRM_RE.search(t)) or (len(words) <= 6 and bool(STRONG_YES_RE.search(t)))
    # a question starts with a question word, or ends with "?" -- except a one- or two-word answer ("Yeah?") or a yes followed by a
    # question ("Yes! Can you see me?"): those still answer
    question = bool(QUESTION_RE.search(t)) or ('?' in t and len(words) > 2 and not AFFIRM_RE.search(t))
    if re.match(r"^\W*" + AFFIRM_WORDS + r"\s*[.!]", t): question = False          # "Yes! Can you see me?" answers first
    request = bool(REQUEST_RE.search(t)) and not deny
    if request and question and not re.search(r"\b(can|could) we\b", t): request = False
    return dict(request=request, rematch=bool(REMATCH_RE.search(t)) and not deny, affirm=affirm and not deny, deny=deny,
                cancel=bool(CANCEL_RE.search(DENY_EXEMPT_RE.sub(' ', t))), hold=bool(HOLD_RE.search(t_deny)), question=question)
PAIN_STRONG_RE = re.compile(r"\b(ow+|ouch|i'?m (hurt|injured|dizzy|lightheaded|light headed)|i (hurt|twisted|pulled|sprained) my|i feel (dizzy|sick|faint|lightheaded)|"
                            r"(my|the) \w+( \w+)? (hurts|is hurting|is killing me)|i'?ve got a cramp|i got a cramp|i have a cramp|cramp|can'?t breathe|"
                            r"it hurts|that hurts|this hurts|i'?m in pain|i feel faint|(i'?m|feeling) (sick|faint)|i think i'?m injured)\b", re.I)


def pain_level(low):
    """'strong' (first person, now: stop at once), 'weak' (a pain word, but maybe not about them: ask first) or None."""
    if re.search(r"\b(doesn'?t|don'?t|didn'?t|won'?t|not|no|never) (really )?(hurt|pain)", low): return None
    if PAIN_STRONG_RE.search(low) and not re.search(r"\b(song|music|beat|track|friend|brother|sister|mom|dad)\b.{0,15}\bhurts?\b", low): return 'strong'
    if re.search(r"\b(pain|hurts?|hurting|dizzy|lightheaded|injured|cramp)\b", low): return 'weak'
    return None


def style_from(low):
    """The style the caller wants: the last one mentioned, skipping any after "instead of" / "not" / "no" ("kpop instead of hip hop")."""
    spans = []
    for w in STYLE_ORDER:
        for m in re.finditer(rf'\b{re.escape(w)}\b', low):
            if any(a < m.end() and m.start() < b for a, b, _ in spans): continue
            spans.append((m.start(), m.end(), STYLE_WORDS[w]))
    keep = [(a, code) for a, b, code in spans if not re.search(r"(instead of|rather than|not|no|don'?t want|than)\s+(the\s+|a\s+|some\s+)?$", low[:a])]
    return max(keep)[1] if keep else None


QUESTION_KEYS = ('offer', 'space', 'confirm', 'gotit', 'choice', 'challenge', 'ghostsave')     # never dropped for age: a wait depends on them
KEEP_KEYS = ('wrap', 'cancel', 'code', 'bye', 'error', 'safety')                 # survive a state change (goodbyes, the challenge code)
SETUP_CAP_S = 150.0                                                               # one camera setup never runs longer than this
SETUP_TOTAL_CAP_S = 180.0                                                         # ...and all setup rounds for one activity together
CONFIRM_BREAK_S = 5.0                                                             # the ready check goes back to setup only after this long
READY_WINDOW_S = 3.0                                                              # squad: hands-up holds that count as one "ready"
LOBBY = ('CHOOSE', 'EXPLORE', 'OFFER')                                            # before anything starts (cards, chat, an offer)
WIN_TOTAL = dict(beginner=195, intermediate=216, advanced=240)                   # a battle win vs Brodie's ghost: the pass mark x 3 rounds
CARD_HOLD_S = 6.0                                                                # the score card stays this long before the menu returns
CYCLE_START_S = 10.0                                                              # quiet after the hello: the card highlight starts cycling
CYCLE_STEP_S = 2.5                                                                # one card lit this long
BOARD_RE = re.compile(r"(leader ?board|the board|daily board|scoreboard|high ?scores?|top scores?|who'?s winning)", re.I)
BYE_RE = re.compile(r"\b(bye|goodbye|good night|i'?m done|i'?m finished|that'?s (it|all) for (today|now)|gotta go|got to go|see you|hang up|end the call)\b", re.I)
FALSE_PROGRESS_RE = re.compile(r"\b(locked in|all set|you'?re set|here we go|starting now|let'?s go|camera'?s (good|perfect|set))\b", re.I)
CMD_ORDER = sorted(CMD_WORDS, key=len, reverse=True)
MODES = ('learn', 'battle', 'telephone', 'class', 'class_record', 'daily', 'squad_battle', 'squad_learn')
SQUAD_MODES = ('squad_battle', 'squad_learn')
MENU_LINE = ("I can teach you a routine, battle you against my ghost, or run today's daily challenge. Got friends with you? We can do a "
             "squad battle, dance telephone or a class. What do you feel like?")
PRIORITY = ('pain', 'stop', 'forget', 'pause', 'resume', 'rematch', 'harder', 'easier', 'simplify', 'explain', 'full', 'slower', 'faster',
            'loop', 'again', 'skip', 'next', 'report', 'ready')
STYLE_WORDS = {'hip hop': 'lh', 'hip-hop': 'lh', 'hiphop': 'lh', 'house': 'ho', 'popping': 'po', 'pop choreo': 'po', 'pop': 'po', 'locking': 'lo', 'lock': 'lo',
               'waacking': 'wa', 'waack': 'wa', 'whack': 'wa', 'krump': 'kr', 'jazz': 'js', 'k-pop': 'kpop', 'kpop': 'kpop', 'k pop': 'kpop',
               'huntrix': 'kpop', 'demon hunters': 'kpop', 'dance fitness': 'fitness', 'dance cardio': 'fitness', 'cardio': 'fitness', 'fitness': 'fitness',
               'workout': 'fitness', 'squat': 'squats', 'breaking': 'lh', 'breakdance': 'lh', 'anything': 'any', 'surprise me': 'any', 'you pick': 'any', 'you choose': 'any'}
STYLE_ORDER = sorted(STYLE_WORDS, key=len, reverse=True)
STYLE_NAMES = dict(lh='hip-hop', mh='hip-hop', ho='house', po='pop', lo='lock', wa='waack', kr='krump', js='street jazz', kpop='K-pop', squats='squats',
                   fitness='dance cardio', any='your pick')
PERSONA_WORDS = {'hype coach': 'hype', 'hype mode': 'hype', 'hype': 'hype', 'chill coach': 'chill', 'chill mode': 'chill', 'chill': 'chill', 'calm': 'chill',
                 'strict coach': 'coach', 'strict': 'coach', 'serious coach': 'coach', 'tough coach': 'coach', 'real coach': 'coach'}
PERSONA_ORDER = sorted(PERSONA_WORDS, key=len, reverse=True)
DIFF_RE = [(re.compile(r'\b(beginner|easy mode|(?<!take it )(?<!so )(?<!too )easy)\b'), 'beginner'),
           (re.compile(r'\b(intermediate|medium)\b'), 'intermediate'),
           (re.compile(r'\b(advanced|expert|hard mode|pro mode|hard (one|routine|level|version)|something hard)\b'), 'advanced')]
PERSONA_PROMPTS = dict(
    hype="Coaching style for the rest of the call: HYPE. Big energy, playful, light slang (never cringe), celebrate wins loudly, keep corrections upbeat.",
    chill="Coaching style for the rest of the call: CHILL. Relaxed, calm, low pressure, few words, warm and encouraging.",
    coach="Coaching style for the rest of the call: COACH. Precise, direct and technical, honest about mistakes, still kind. No hype words.")
UPPER_WORDS = ('upper body', 'at my desk', 'sitting', "can't stand", 'cant stand', 'laptop', 'just my arms', 'arms only')
NUM_WORDS = {'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}
NUDGE_S = float(os.environ.get('DANCE_NUDGE_S', '45') or 45)      # quiet chat this long: Brodie offers a routine once (never starts one)
RUNGS = (0.5, 0.6, 0.75, 0.9, 1.0)                                   # the teaching ladder: tempo multipliers of the routine's BPM (0.5 = very slow)
START_RUNG = dict(beginner=0, intermediate=1, advanced=2)
MOVE_TARGET = 2                                                      # a single move is "learned" at 0.75x; the full routine climbs to 0.9x
ROUTINE_TARGET = 3
CHOICES = ('again', 'next', 'skip', 'faster', 'slower', 'simplify', 'explain', 'full', 'stop', 'loop', 'harder', 'easier')
NO_CLAIMS = (" Never say the camera is set, that you can see all of them, how many people you see, or that anything is starting, unless the latest"
             " [camera now] note or a tool result says so.")
STATE_RULES = dict(
    CHOOSE="STATE: PICK A MODE. Six mode cards are on screen: Learn, Ghost Battle, Daily Challenge, Squad Battle, Telephone, Class. Answer questions about them in one short sentence. Only when the caller clearly names one, call propose_activity with it. Do not suggest a routine yourself and do not start anything. If people are talking to each other, a TV or music, stay silent." + NO_CLAIMS,
    EXPLORE="STATE: CHAT. Get to know the caller, answer questions, recommend where to start. You cannot start dancing yourself: only when they clearly ask to dance or learn, call propose_activity. Never say a lesson or challenge is starting." + NO_CLAIMS,
    PREPARE="STATE: CAMERA SETUP. The system checks the camera and hands you one adjustment at a time; say only that, in a few words. If they talk, answer in one short sentence. Never say the lesson is starting." + NO_CLAIMS,
    OFFER="STATE: OFFER. A routine has been suggested (see the latest system note). Ask once, in your own words, whether they want to start it now, mentioning a quick camera check comes first. Then wait for a clear yes. Do not start anything, do not give camera instructions yet. If they ask questions or change their mind, answer and stay here." + NO_CLAIMS,
    CONFIRM="STATE: READY CHECK. You asked if they're ready. Wait for their answer. Do not start anything yourself and do not count in.",
    ACTIVITY="STATE: DANCING. The music is playing. Stay silent unless the system gives you a private instruction.",
    REVIEW="STATE: RESULTS. Speak only the short line the system asks for, then wait.",
    PAUSED="STATE: PAUSED. If they speak, say they can say resume or raise both hands. Nothing else.",
    SAFETY="STATE: SAFETY STOP. Tell them to rest. Say nothing about dancing.")
LEVEL_CFG = dict(beginner=dict(tempo=0.8, show=2, pass_mark=75), intermediate=dict(tempo=0.95, show=2, pass_mark=80), advanced=dict(tempo=1.05, show=1, pass_mark=85))
MAX_ATTEMPTS = 3
MODE_LABEL = dict(learn='Learn', battle='Ghost Battle', daily='Daily Challenge', telephone='Telephone', class_='Class', squad_battle='Squad Battle',
                  squad_learn='Squad Learn', **{'class': 'Class', 'class_record': 'Class'})
CARD_KW = ('grade', 'award_title', 'xp', 'achievements', 'handle', 'mode_label', 'squad', 'total_line', 'perfects', 'routine', 'photo', 'brag')


POSE_ON = (os.environ.get('DANCE_POSE_STEP', '1') or '1').strip() != '0'     # pose by pose first looks and freeze-frame practice (DANCE_POSE_STEP=0: off)
COUNT_WORDS = ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve')


def weak_span(cue):
    """(first, last) 0-based counts of the cue's weak run worth looping (2 counts at least), or None."""
    beats = sorted((cue or {}).get('beats') or [])
    if not beats or (cue or {}).get('kind') in ('praise', 'visibility'): return None
    a, b = beats[0], beats[-1]
    if b == a: b = a + 1
    return a, b


def count_words(span):
    """'counts 1 to 2' -> 'One, two'."""
    nums = [int(x) for x in re.findall(r'\d+', span)]
    if not nums: return span
    a, b = nums[0], nums[-1]
    words = [COUNT_WORDS[i - 1] if 0 < i <= len(COUNT_WORDS) else str(i) for i in range(a, b + 1)]
    return ', '.join(words).capitalize()


def key_title(key):
    """A human name for the top bar: 'Waack 01', 'K-pop Huntrix', 'Hip-Hop Starter Mix'."""
    if key.get('title') and key.get('generated'): return key['title']
    n = key.get('name', '')
    if n.startswith('aist_'):
        parts = n.split('_'); style = STYLE_NAMES.get(parts[1], key.get('style') or parts[1]); ch = parts[-1].replace('ch', '')
        return f"{style} {ch}"
    fixed = dict(cc_huntrix='K-pop Huntrix', cc_pm_dancer='festival groove', cc_street_girl='street', test_squats='squats')
    if n in fixed: return fixed[n]
    if n.startswith('class_'): return f"class {n[6:]}"
    if n.startswith('chain_'): return 'telephone'
    return key.get('title') or n.replace('_', ' ')


def parse_code(text, strict=False):
    """A four-digit code spoken as digits ("4471", "44 71") or four number words in a row ("four four seven one"). None when absent.
    strict (outside the menu): the utterance must mention a code or be short, and a counting run ("one two three four", "five six
    seven eight") is a count-in, not a code."""
    t = text.lower()
    code = None
    m = re.search(r'\b(\d)\D{0,2}(\d)\D{0,2}(\d)\D{0,2}(\d)\b', t)
    if m: code = ''.join(m.groups())
    else:
        w = '|'.join(NUM_WORDS)
        m = re.search(rf'\b({w})[\s,-]+({w})[\s,-]+({w})[\s,-]+({w})\b', t)
        if m: code = ''.join(NUM_WORDS[g] for g in m.groups())
    if code and strict:
        said_code = re.search(r'\b(code|challenge|battle|chain)\b', t)
        if not said_code and len(re.findall(r"[a-z0-9']+", t)) > 6: return None
        d = [int(c) for c in code]
        if not said_code and all(d[i + 1] - d[i] == 1 for i in range(3)): return None
    return code


def parse_commands(low):
    """All commands in an utterance (longest phrases first, each span used once)."""
    found = []; used = []
    for w in CMD_ORDER:
        for m in re.finditer(rf'\b{re.escape(w)}\b', low):
            if any(a < m.end() and m.start() < b for a, b in used): continue
            cmd = CMD_WORDS[w]
            if cmd == 'stop' and re.search(r"\b(don'?t|never) stop\b", low): continue
            if cmd == 'pain' and re.search(r"\b(doesn'?t|don'?t|didn'?t|not|no) (hurt|pain)", low): continue
            used.append((m.start(), m.end())); found.append(cmd)
    return found


def pick_command(found):
    controls = [c for c in found if c not in MODES]
    for c in PRIORITY:
        if c in controls: return c
    return controls[-1] if controls else None


class Backend:
    """The team backend over HTTP, or the same code in-process (BACKEND_URL=fake) for the simulator and tests."""
    def __init__(self, url=None):
        self.url = (url or os.environ.get('BACKEND_URL') or 'http://127.0.0.1:8787').rstrip('/')
        self.db = None
        if self.url in ('fake', 'off'):
            import importlib.util
            spec = importlib.util.spec_from_file_location('team_db', os.path.join(ROOT, 'team', 'backend', 'db.py'))
            self.db = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.db); self.db.init()

    def _http(self, method, path, body=None):
        import urllib.request
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={'content-type': 'application/json'}, method=method)
        with urllib.request.urlopen(req, timeout=3) as r: return json.loads(r.read().decode())

    async def call(self, method, path, body=None):
        loop = asyncio.get_running_loop()
        try:
            if self.db is not None: return self._fake(method, path, body)
            return await loop.run_in_executor(None, self._http, method, path, body)
        except Exception as e:
            print(f'[dance] backend {method} {path} failed: {e!r}', flush=True); return None

    def _fake(self, method, path, body):
        db = self.db; parts = path.split('?')[0].strip('/').split('/'); body = body or {}
        from urllib.parse import parse_qs, urlparse, unquote
        q = parse_qs(urlparse(path).query)
        if parts == ['events'] and method == 'POST': db.add_event(body); return {'ok': True}
        if parts[:2] == ['dance', 'result'] and method == 'POST': return db.post_result(body)
        if parts[:2] == ['dance', 'result'] and method == 'GET': return db.get_result(parts[2]) or {'error': 'no such code'}
        if parts[:2] == ['dance', 'history'] and method == 'GET': return db.history((q.get('name') or [''])[0])
        if parts[:2] == ['dance', 'leaderboard'] and method == 'GET':
            since = (q.get('since') or [None])[0]
            return db.leaderboard((q.get('key') or [None])[0], int((q.get('limit') or [20])[0]), mode=(q.get('mode') or [None])[0],
                                  since=float(since) if since else None)
        if parts[:2] == ['dance', 'code'] and len(parts) == 3: return db.resolve_code(parts[2])
        if parts[:2] == ['dance', 'ghost'] and len(parts) == 3:
            if method == 'POST': return db.save_ghost(parts[2], body.get('token'), body.get('src_key'), body.get('bpm', 100), body.get('per_beat', 4),
                                                      body.get('data') or [], body.get('total', 0), body.get('score100', 0), body.get('difficulty'))
            return db.get_ghost(parts[2]) or {'error': 'no such ghost'}
        if parts[:2] == ['dance', 'daily_played']: return dict(played=db.daily_played((q.get('name') or [''])[0], (q.get('key') or [''])[0]))
        if parts[:2] == ['dance', 'profile'] and len(parts) == 3:
            name = unquote(parts[2])
            if method == 'GET': return db.get_profile(name) or {'error': 'no such profile', 'name': name}
            if method == 'DELETE': return dict(deleted=db.delete_user(name), name=name)
        if parts[:2] == ['dance', 'profile'] and len(parts) == 4 and method == 'POST':
            name = unquote(parts[2])
            if parts[3] == 'award': return db.award(name, int(body.get('xp', 0)), style=body.get('style'), achievements=body.get('achievements') or (), day=body.get('day'), session=body.get('session'))
            if parts[3] == 'prefs': return db.set_prefs(name, body)
        if parts[:2] == ['dance', 'share'] and len(parts) == 4 and method == 'POST': db.log_share(parts[2], 'challenge_start'); return dict(code=parts[2], starts=db.share_count(parts[2]))
        if parts[:2] == ['dance', 'share'] and len(parts) == 3: return dict(code=parts[2], starts=db.share_count(parts[2]), completes=db.share_count(parts[2], 'challenge_complete'))
        if parts[:2] == ['dance', 'board'] and len(parts) == 3: return db.crew_board(parts[2]) or {'error': 'no such code'}
        if parts[:2] == ['dance', 'funnel']: return db.funnel()
        if parts[:2] == ['dance', 'chain'] and method == 'POST' and len(parts) == 2: return dict(code=db.start_chain(body.get('key', ''), body.get('link', {})), link_no=1)
        if parts[:2] == ['dance', 'chain'] and method == 'POST' and len(parts) == 4:
            n = db.add_link(parts[2], body.get('link', {})); return dict(code=parts[2], link_no=n) if n else {'error': 'no such chain'}
        if parts[:2] == ['dance', 'chain'] and method == 'GET': return db.get_chain(parts[2]) or {'error': 'no such chain'}
        if parts[:2] == ['dance', 'class'] and method == 'POST' and len(parts) == 2: return dict(code=db.start_class(body.get('teacher'), body.get('key') or {}))
        if parts[:2] == ['dance', 'class'] and method == 'POST' and len(parts) == 4: return db.add_student(parts[2], body) or {'error': 'no such class'}
        if parts[:2] == ['dance', 'class'] and method == 'GET' and len(parts) == 4: return db.class_report(parts[2]) or {'error': 'no such class'}
        if parts[:2] == ['dance', 'class'] and method == 'GET': return db.get_class(parts[2], with_key='key=0' not in path) or {'error': 'no such class'}
        if parts == ['metrics']: return db.metrics()
        return None


SAY_PREFIX = 'Say this to the caller, word for word, and nothing before or after it: '
STAGE = '[Stage direction from the system, not the caller. Do not answer it, mention it or read it out.] '


def stage_direction(prompt):
    """Our prompt as a user-role stage direction. The plugin sends generate_reply(instructions=...) as a *model* turn, so Gemini
    took our "Say this..." for its own words and continued them: lines said twice, the "(Private instruction...)" prefix read out
    (live calls 2026-10-11). A user turn is something to act on, not something already said."""
    p = prompt.strip()
    if p.startswith(SAY_PREFIX): return STAGE + 'Say exactly this to the caller, nothing before or after: ' + p[len(SAY_PREFIX):]
    return STAGE + p


def tune_realtime(rt):
    """Input transcription pinned to English (GEMINI_LANGUAGE only sets Brodie's own voice; the caller's words were auto-detected and
    came out as Hindi, German, Vietnamese), quieter speech still starts a turn (a "yes" from two metres away), and a sliding context
    window for long calls. The plugin reads these options at every (re)connect. Verified against the relay on 2026-10-11.
    DANCE_INPUT_LANG=off / DANCE_VAD=off leave the plugin defaults."""
    try:
        from google.genai import types
        opts = rt._opts
        lang = (os.environ.get('DANCE_INPUT_LANG') or 'en-US').strip()
        if lang.lower() not in ('', 'off', 'none'): opts.input_audio_transcription = types.AudioTranscriptionConfig(language_codes=[lang])
        if (os.environ.get('DANCE_VAD') or 'high').strip().lower() != 'off':
            opts.realtime_input_config = types.RealtimeInputConfig(automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH, prefix_padding_ms=300, silence_duration_ms=700))
        opts.context_window_compression = types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow())
        print(f'[dance] gemini tuned: input transcription {lang}, sensitive start of speech, sliding context window', flush=True)
    except Exception as e:
        print(f'[dance] gemini tuning skipped: {e!r}', flush=True)


def patch_voice_layer():
    """Applied once at import (the worker re-imports this file per job; the facetime.voice module is shared)."""
    if (os.environ.get('DANCE_VOICE_PATCH', '1') or '1').strip() == '0': return False
    try:
        from facetime import voice as fv
    except Exception:
        return False
    GM = getattr(fv, 'GeminiModel', None)
    if GM is None or getattr(GM, '_dance_patched', False): return False
    orig_build = GM.build

    def build(self):
        fresh = self.rt is None
        rt = orig_build(self)
        if fresh and rt is not None: tune_realtime(rt)
        return rt

    def generate(self, prompt):
        if self.session is None: return None
        self.n_generate += 1
        try: return self.session.generate_reply(user_input=stage_direction(prompt))
        except RuntimeError as e: fv.log.warning('generate_reply: %s', e); return None

    GM.build = build; GM.generate = generate; GM._dance_patched = True
    if not getattr(GM.start, '__wrapped_audio_in__', False): GM.start = audio_in.wrap_start(GM.start)   # Krisp BVC + gain control (DANCE_NC)
    FL = getattr(fv, 'Floor', None)
    if FL is not None and not getattr(FL, '_dance_patched', False):
        orig_gate = FL.gate_input

        def gate_input(self):
            """The echo guard, plus: (1) a barge-in window (force_open_until, set when the local VAD hears the caller over the music)
            keeps the caller's audio flowing to Gemini so "stop" / "pause" are heard; (2) when the guard closes while the caller was
            mid-word, 600 ms of silence goes to the model so it sees the end of their turn instead of a hanging one."""
            if self.open and self.player is not None and self.now() < getattr(self, 'force_open_until', 0.0):
                if not self.input_on: self.input_on = True; self.model.set_input_enabled(True)
                return
            was = self.input_on
            orig_gate(self)
            if was and not self.input_on:
                try:
                    m = self.model; last = float(getattr(m, 'last_caller_t', -1e9) or -1e9)
                    if self.now() - last < 0.6:
                        rt = getattr(getattr(getattr(m, 'session', None), '_activity', None), '_rt_session', None)
                        if rt is not None and hasattr(rt, 'push_audio'):
                            from livekit import rtc
                            rt.push_audio(rtc.AudioFrame(bytes(2 * 9600), 16000, 1, 9600))
                except Exception:
                    pass

        FL.gate_input = gate_input; FL._dance_patched = True
    return True


try: patch_voice_layer()
except Exception as _e: print(f'[dance] voice patch failed: {_e!r}', flush=True)


def no_power_throttling():
    """Windows: opt this process out of power throttling (EcoQoS). Measured 2026-10-11 on the team laptop: once livekit.agents
    loaded, Windows throttled the worker and every CPU step ran ~3x slower (pose, drawing, Python), which is why the video lagged.
    SetProcessInformation(ProcessPowerThrottling, ControlMask=EXECUTION_SPEED, StateMask=0) restores full speed. A no-op elsewhere;
    DANCE_NO_THROTTLE=0 leaves Windows' default."""
    if os.name != 'nt' or (os.environ.get('DANCE_NO_THROTTLE', '1') or '1').strip() == '0': return False
    try:
        import ctypes, ctypes.wintypes as wt
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wt.HANDLE
        k.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]

        class _PPT(ctypes.Structure): _fields_ = [('Version', wt.ULONG), ('ControlMask', wt.ULONG), ('StateMask', wt.ULONG)]
        st = _PPT(1, 0x1 | 0x4, 0)                       # EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, state 0 = never throttle
        return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)))
    except Exception as e:
        print(f'[dance] power throttling opt-out skipped: {e!r}', flush=True); return False


try: no_power_throttling()
except Exception: pass


def on_event(event, say_now, ask_voice):
    """The worker's handler hook: capture the verbs and forward every event to the running session."""
    STATE['say_now'] = say_now; STATE['ask_voice'] = ask_voice
    s = STATE.get('session')
    if s is not None:
        try: s.on_worker_event(event)
        except Exception as e: print(f'[dance] handler error: {e!r}', flush=True)


class DanceSession:
    def __init__(self, job, key_name=None, mode='learn', tempo=1.0, rtt_s=0.3, fps=15, pose_variant='full', debug_mp4=None,
                 now=None, sleep=None, offline_pose=None):
        self.job = job; self.mode = mode; self.tempo = float(tempo); self.rtt_s = float(rtt_s); self.fps = float(fps)
        self.pose_variant = pose_variant; self.debug_mp4 = debug_mp4; self.now = now or time.monotonic
        self.sleep = sleep or asyncio.sleep; self.offline_pose = offline_pose
        self.difficulty = (os.environ.get('DANCE_DIFFICULTY') or 'beginner').strip().lower()
        if self.difficulty not in LEVEL_CFG: self.difficulty = 'beginner'
        if key_name: self.key = choreo.load_any(key_name)
        else:
            try: self.key = choreo.generate('hiphop', self.difficulty, seed=1001)
            except Exception:
                keys = list_keys()
                if not keys: raise RuntimeError('no keys in the library: build one with team.dance.offline_test key ...')
                self.key = load_key(keys[0]['name'])
        self.engine = None; self.phrase = 0; self.attempts = 0; self.round = 0; self.rounds = []; self.name = None
        self.code = os.environ.get('DANCE_CODE') or None          # a four-digit code the caller spoke (duel or chain to join)
        self.my_code = None; self.backend = Backend()
        self.pending_cmd = None; self.stop = False; self.pose = None; self.mpose = None; self.pose_reload = False
        self.exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dance-pose')
        self.source = None; self.writer = None; self.last_seq = -1; self.frames = 0; self.pose_ms = []
        self.last_lm = None; self.log = []; self.t0 = None; self.published = False
        self.mirror_view = (os.environ.get('DANCE_MIRROR_VIEW', '1') or '1').strip().lower() in ('1', 'true', 'yes', 'on')
        self.size = (W, H)                                       # the canvas; set from the caller's first frame (portrait phone or landscape webcam)
        self.lobby_enabled = (os.environ.get('DANCE_LOBBY', '1') or '1').strip() != '0'
        # Per-call overrides from the room name, so one worker can serve the FaceTime line (full flow) and a Chrome demo room:
        # 'demo-...' = one routine, no chat/offer/ready gates; 'desk-' anywhere = arms-only camera check (laptop webcam at a desk).
        self.room_name = str(getattr(getattr(job, 'room', None), 'name', '') or '')
        if self.room_name.startswith('demo-'): self.lobby_enabled = False
        self.room_upper = ('desk-' in self.room_name) or (os.environ.get('DANCE_UPPER', '0') or '0').strip() == '1'
        self.style = None; self.upper_body = False; self.parts = {}; self.hint = None; self.hint_since = None
        self.mode_choice = None; self.mode_choice_pending = False    # a spoken mode word; the lobby reads it (learn/battle also reach pending_cmd mid-session)
        self.idle_title = 'Brodie'; self.idle_lines = []
        self.pose_px = int(os.environ.get('DANCE_POSE_PX', '384') or 384)
        self.key['title'] = key_title(self.key)
        self.history = []; self.best = None                       # scored phrases so far, kept on screen
        self.voice_lines = {}
        # game layer
        self.fx = FxQueue(); self.gest = {}; self.tracker = PlayerTracker(); self.players = []; self.n_people = 0
        self.telemetry = Telemetry(now=self.now, write=True)
        self.persona = 'hype'; self.persona_choice = None; self.diff_choice = None; self.profile = None; self.level = 0; self.xp = 0
        self.recap = None; self.caption = None; self.paused = False; self.pause_req = False; self.resume_req = False
        self.safety_stop = False; self.safety_shown = False; self.forget_until = -1.0
        self.session_rounds = []; self.extras = {}; self.call_mode = 'solo'; self.call_mode_choice = None
        self.squad_players = []; self.colours = {}; self.names = {}; self.ltr_names = []
        self.step = None; self.battle_hud = None; self.queued_key = None; self.challenge = None; self.final_card = None
        self.last_cmd = {}; self.last_result = None; self.last_cue = None; self.last_title = None; self.in_warmup = False
        self.combo_run = {}; self.last_tip_beat = {}; self.cur_engine = None; self.time_to_start = None; self.squad_since = None
        self.state = 'EXPLORE'; self.epoch = 0; self.say_q = []; self.speaker_task = None; self.activity_req = None; self.affirm_t = -1.0; self.deny_t = -1.0
        self.cancel_req = False; self.hold_until = -1.0; self.space_ok = False; self.upper_override = bool(getattr(self, 'room_upper', False)); self.luma = None; self.cur_lm = None
        if self.upper_override: self.upper_body = True
        self.setup_ui = None; self.stage = None; self.ghost_overlay = True; self.practice = False; self.last_caller_text_t = -1e9; self.in_demo = False
        self.last_mode = None; self.last_key = None; self.stop_routine = False
        self.proposal = None; self.proposal_t = -1.0; self.coach_q_t = -1.0; self.line_end = {}; self.last_variant = {}; self.last_frame_t = -1.0
        self.offers_made = 0; self.last_bgr = None; self.countdown_ui = None; self.hist_best = None
        self.line_start = {}                     # key -> when that line's audio started (answers count from just after)
        self.offer_asked_t = -1.0                # the first time the current offer was asked (by us or by Gemini)
        self.space_answer_t = -1.0               # the yes that answered "space?" (never also a yes to "ready?")
        self.confirm_first_t = None              # first "ready?" of this setup round; a "ready" after it counts even across a re-setup
        self.cmd_floor_t = -1.0; self.pending_t = -1.0    # a pending command counts only if it came after the floor (no stale "stop"/"next")
        self.declined = []                       # styles the caller passed on (never offered again right away)
        self.last_utt = None                     # (t, classify(...), command) of the caller's last line
        self.coach_q_text = ''; self.frame_aspect = None; self.frame_errors = 0; self.pain_check_t = -1e9
        self.ready_floor_t = -1.0                # answers to the ready check count only after this
        self.menu_open = False                   # the opening menu is on screen and a mode word alone picks a mode
        self.code_pending = None                 # a spoken code that matched two kinds ("battle or telephone?")
        self.report_req = False                  # "report": the teacher wants the class report
        self.solo_req = False                    # "just me": a squad offer runs solo
        self.daily_done = False                  # this call already had today's daily shot (one shot at the board)
        self.first_demo_done = False             # the very first demo of the call is pose by pose for every level
        self.slow_mode = False                   # "slower" at the slowest speed: pose by pose and freeze-frame again
        self.practice_run = False                # a daily run after the day's shot: no board post
        self.compare_ui = None                   # telephone's side-by-side "watch it change"
        self._prev_cues = {}; self.last_cues = []   # the last try's corrections per move (progress between tries)
        self.vad_last_t = -1e9; self.vad_utt_end = None; self.vad_run = 0; self.last_barge_t = -1e9   # the local VAD on the caller's audio
        self.q_open = None; self.unclear_t = -1e9    # the question waiting for an answer ("was that a yes?" when it can't be read)
        self.tools_ok = False
        self.space_ok_t = -1e9                   # when the space question was last answered (a rematch within 10 min doesn't re-ask)
        self.deadair = DeadAir(self.now)         # no state sits silent for long (watchdog.py)
        self.menu_ui = None                      # the mode cards: dict(header, lit, cycling, badges, picked) while choosing
        self.break_ui = None                     # a break between moves: dict(label, t0, dur, preview) (pacing.py)
        self.orient = Orient(); self._orient_pid = None; self.orient_state = None; self.cur_world = None   # depth: turn vs limb (orient.py)
        self.depth_gate = (os.environ.get('DANCE_DEPTH_GATE', '0') or '0').strip() == '1'               # leave ambiguous joints out of scoring
        self.vad_echo = 0.0; self.music_duck_until = 0.0; self.barge_s = 0.0; self.barge_eng = None      # voice over the music (on_vad)
        self.last_addr = 1.0; self.yes_asked_q = None                   # how addressed the last line was; the question "was that a yes?" was asked for
        self.move_card_ui = None                 # the current move's name for the step chip ("MOVE 1 · SIDE POINT")
        self.board_ui = None                     # today's daily board on screen: dict(title, rows, rank, n)
        self.budget = FrameBudget(); self.cam_shift = CameraShift(); self.camera_moved = False   # guards.py
        self.voice_was_up = False; self.voice_down_since = None; self.call_warned = False; self.end_after_activity = False
        self.call_max_s = 60.0 * float(os.environ.get('DANCE_CALL_MAX_MIN', '20') or 20)        # a soft cap per call (the team's Live minutes)
        self.subject_luma = None; self.fall_since = None
        self.last_addressed_t = -1e9             # the caller last said something short / to Brodie (gates Gemini's offers in CHOOSE)
        self.noise_utts = []                     # times of unusable utterances while choosing (a noisy room starts the card picker)
        self.played = []                         # modes played this call (PLAYED badges, the rematch shortcut)
        self.card_shown_t = None                 # when the score card went up (it stays CARD_HOLD_S before the menu)
        self.bye_req = False; self.menu_req = False; self.switch_req = None     # "bye"; "menu" between moves; a new mode named mid-activity
        self.recommended_mode = None             # a returning caller's FOR YOU card
        self.ready_hits = {}                     # squad: dancer id -> when their hands-up hold completed (majority within a window)
        self.metrics = dict(reasks=0, nudges=0, reprompts=0, choice_timeouts=0, answer_fallbacks=0, unclear_asks=0, barge_ins=0,
                            gemini_asked_skips=0, frame_errors=0, silent_lines=0, false_progress=0, dead_air_nudges=0, setup_bounces=0,
                            flipflop_breaks=0, gesture_answers=0, max_silence_s=0.0, silences_over_20=0)
        self.max_s = float(os.environ.get('DANCE_MAX_S', '0') or 0)          # simulator/test cap on the session length (0 = none)
        self.walkin_noted = False; self.max_players = 1; self._prev_item_score = {}; self.gesture_ui = None; self._last_squad = None; self.want_ready = False; self.handle = (os.environ.get('DANCE_CALL_HANDLE') or '').strip() or None
        # Spoken lines come from Brodie's own cue bank (his Gemini voice) or from Gemini live; the Windows voice lines are off
        # unless DANCE_LOCAL_VOICE=1 (2026-10-11: two different voices on one call confused the caller).
        if (os.environ.get('DANCE_LOCAL_VOICE', '0') or '0').strip() == '1':
            try:
                from team.dance import tts_local
                for name in tts_local.LINES:
                    pcm = tts_local.load(name)
                    if pcm is not None: self.voice_lines[name] = pcm
            except Exception as e:
                print(f'[dance] no local voice lines ({e!r})', flush=True)

    # ---- Brodie's own voice from the cue bank ----
    BANK_LINES = dict(step_back='Step back so I can see your whole body', see_you='Great, I can see you', wait='I will wait here, step into view when you are ready',
                      lets_go="Let's go", nice='Nice work', great='Great job', keep='Keep going', rest='Rest for a moment', done='All done, great session', rhythm='Great rhythm')

    def bank_has(self, name):
        bank = getattr(self.job, 'bank', None); text = self.BANK_LINES.get(name)
        return bool(bank is not None and text and (text in getattr(bank, 'cache', {}) or text in getattr(bank, 'manifest', {})))

    def bank_say(self, name, fallback_ask=None):
        """Play one of Brodie's recorded lines (his own voice, instant); or ask Gemini when the bank lacks it."""
        if self.bank_has(name): return self.say(self.BANK_LINES[name], 'announce')
        if fallback_ask: return self.ask(fallback_ask)
        return None

    def bank_pcm(self, name):
        bank = getattr(self.job, 'bank', None)
        if not self.bank_has(name) or not hasattr(bank, 'get'): return None
        try: return bank.get(self.BANK_LINES[name])
        except Exception: return None

    # ---- helpers ----
    def say(self, text, intent='announce'):
        p = getattr(self.job, 'player', None)
        if p is None: return None
        try: return p.say(text, intent, dict(source='dance'))
        except Exception as e: print(f'[dance] say failed: {e!r}', flush=True)

    def sfx(self, name):
        """A synthesized sound effect (registered at session start), queued after whatever is playing."""
        if getattr(self.job, 'bank', None) is None: return None
        return self.say(dance_audio.SFX[name], 'announce')

    def cut_audio(self):
        """Stop the music now: an urgent clip (the soft pause chime) preempts and clears whatever is playing."""
        p = getattr(self.job, 'player', None)
        if p is None or getattr(self.job, 'bank', None) is None: return
        try:
            if hasattr(p, 'say_now'): p.say_now(dance_audio.SFX['pause'], 'urgent', dict(source='dance'))
        except Exception as e: print(f'[dance] cut audio failed: {e!r}', flush=True)

    def ask(self, prompt, key=None, ttl=8.0, urgent=False):
        """The only way our code makes Brodie speak. One line at a time, only when the caller is quiet and nothing is playing; each
        line is tagged with the session state and dropped if the state moved on (or, for lines nobody waits on, if it waited longer
        than `ttl`). Gemini answers the caller by itself; this is never used to reply to something the caller just said."""
        v = getattr(self.job, 'voice', None)
        if v is None or getattr(v, 'ask_voice', None) is None:
            self.log.append(dict(t=round(self.now() - (self.t0 or 0), 2), ask=prompt[:90], note='no voice layer')); return None
        if not prompt.startswith(SAY_PREFIX):
            prompt = ('Speak directly to the caller as Brodie, in the second person ("you"). Reply with only the words you say aloud. What to say: '
                      + prompt.replace('One short sentence: ', '').replace('Two short sentences: ', '').replace('ask them to', 'ask the caller to'))
        if urgent: self.say_q = []
        if key: self.say_q = [q for q in self.say_q if q['key'] != key]
        self.say_q.append(dict(prompt=prompt, key=key, epoch=self.epoch, t=self.now(), ttl=ttl, urgent=urgent, tries=0))
        if self.speaker_task is None or self.speaker_task.done():
            try: self.speaker_task = asyncio.get_running_loop().create_task(self._speaker_loop())
            except RuntimeError:
                self.say_q.pop(); return v.ask_voice(prompt)
        return 'queued'

    def say_line(self, text, key=None, ttl=8.0, urgent=False):
        """Brodie says exactly this (written by us, in second person). Gemini read meta instructions aloud ("Let's ask them to step
        back...") when we described what to say; quoting the words removes that."""
        return self.ask(f'{SAY_PREFIX}"{text}"', key=key, ttl=ttl, urgent=urgent)

    def pick(self, key, variants):
        """A variant different from the last one used for this key (never the same sentence twice in a row)."""
        last = self.last_variant.get(key); opts = [v for v in variants if v != last] or list(variants)
        import random as _r
        v = _r.choice(opts); self.last_variant[key] = v
        return v

    def _model(self):
        return getattr(getattr(self.job, 'voice', None), 'model', None)

    def _caller_quiet(self, s=1.0):
        last = getattr(self._model(), 'last_caller_t', None)
        last = max(float(last) if last is not None else -1e9, self.vad_last_t)
        return (self.now() - last) >= s

    def _model_speaking(self):
        try: return bool(getattr(self._model(), 'speaking', False))
        except Exception: return False

    def _timed(self):
        """True when lines take real time to play (a live call, or the realistic simulator): wait for the audio to start and end."""
        return self.now is time.monotonic or bool(getattr(getattr(self.job, 'voice', None), 'timed', False))

    async def _tick(self):
        """One short wait inside the speaker loop: 0.1 s live; in the simulator just a yield (the main loop moves the clock)."""
        await asyncio.sleep(0.1 if self.now is time.monotonic else 0)

    def answer_after(self, key, t0, fallback=12.0):
        """From when an answer to the line `key` (queued at t0) counts, or None while it hasn't been said yet. Answers count from
        a second into the line (an answer given as the line ends, or while Gemini replies to it, still counts); if the line never
        played within `fallback` s (dropped, or the voice is down), from t0, so a wait can never hang on a line nobody heard."""
        st = self.line_start.get(key, -1.0)
        if st >= t0 - 1e-6: return st + (1.0 if self._timed() else -0.3)
        if self.now() - t0 > fallback: self.metrics['answer_fallbacks'] += 1; return t0
        return None

    def gemini_asked(self, pattern, within=8.0):
        """Gemini itself just asked this ("Ready to start?"): our own line would be a second question."""
        return self.now() - self.coach_q_t < within and bool(re.search(pattern, self.coach_q_text or '', re.I))

    SKIP_IF_ASKED = dict(confirm=r"\bready\b", space=r"\b(space|room|steps?)\b", gotit=r"\b(got it|make sense|see it again|try it with me|again)\b",
                         offer=r"\b(start|jump (in|into)|want to (try|do|start|learn|dance)|shall we|up for|down for|ready to)\b")

    async def _speaker_loop(self):
        v = getattr(self.job, 'voice', None); live = self.now is time.monotonic
        while self.say_q and not getattr(self.job, 'closed', False):
            q = self.say_q[0]; now = self.now()
            stale = q['key'] not in QUESTION_KEYS and now - q['t'] > q['ttl']
            if (q['epoch'] != self.epoch and not q['urgent']) or stale:
                self.say_q.pop(0); self.telemetry.event('speech_dropped', key=q['key'], state=self.state)
                self.log.append(dict(t=round(now - (self.t0 or 0), 2), ask=q['prompt'][:90], note='dropped: stale')); continue
            listening = now - self.last_caller_text_t < 6.0 and not q['urgent']          # the caller just spoke: Gemini answers, we wait
            model = self._model()
            thinking = getattr(getattr(model, 'session', None), 'agent_state', None) == 'thinking'
            ready = (model is None or getattr(model, 'started', False)) and self._audio_idle() and not self._model_speaking() and self._caller_quiet() \
                and not listening and not thinking
            if not ready:
                if listening:                                                    # keep every queued line alive a little past the caller's turn
                    for x in self.say_q: x['t'] = max(x['t'], now - x['ttl'] + 4.0)
                await self._tick(); continue
            pat = self.SKIP_IF_ASKED.get(q['key'])
            if pat and self.gemini_asked(pat):                                   # Gemini already asked it in its own words
                self.say_q.pop(0); self.line_start[q['key']] = self.line_end[q['key']] = now
                self.log.append(dict(t=round(now - (self.t0 or 0), 2), ask=q['prompt'][:90], note='skipped: Gemini asked')); self.metrics['gemini_asked_skips'] += 1; continue
            self.say_q.pop(0)
            if live: await asyncio.sleep(0.3)                                    # past the player's post-clip hold
            if q['epoch'] != self.epoch and not q['urgent']: continue
            try: r = v.ask_voice(q['prompt'])
            except Exception as e: print(f'[dance] ask failed: {e!r}', flush=True); r = None
            floor = getattr(v, 'floor', None)
            if r == 'queued' and floor is not None and q['tries'] < 20:          # the floor was busy: take it back and retry ourselves
                floor.queue = [(t, p) for t, p in floor.queue if p != q['prompt']]   # (its queue joins prompts into one doubled reply)
                q['tries'] += 1; self.say_q.insert(0, q); await self._tick(); continue
            self.log.append(dict(t=round(self.now() - (self.t0 or 0), 2), ask=q['prompt'][:90], result=r, key=q['key']))
            self.telemetry.event('speech', key=q['key'], state=self.state)
            t_ask = self.now(); started = None
            if self._timed():                                                    # wait for the audio to start, then to end
                while self.now() - t_ask < 6.0 and not getattr(self.job, 'closed', False):
                    if not self._audio_idle() or self._model_speaking(): started = self.now(); break
                    await self._tick()
                if started is None:
                    if not q.get('retried'):                                     # nothing played: once more
                        q['retried'] = True; self.say_q.insert(0, q)
                        self.log.append(dict(t=round(self.now() - (self.t0 or 0), 2), ask=q['prompt'][:90], note='no audio, retrying')); continue
                    self.telemetry.event('speech_silent', key=q['key']); self.metrics['silent_lines'] += 1; continue
                while (not self._audio_idle() or self._model_speaking()) and self.now() - started < 25.0 and not getattr(self.job, 'closed', False):
                    await self._tick()
            else:
                started = t_ask; await asyncio.sleep(0)
            if q['key']: self.line_start[q['key']] = started; self.line_end[q['key']] = self.now()

    async def wait_spoken(self, keys, t0, max_s=8.0, min_s=0.0):
        """Keep the picture flowing until the lines (queued at/after t0, under these keys) have been said, at most max_s and at least
        min_s: a teaching line must land before the music starts, or the state change would drop it."""
        keys = [keys] if isinstance(keys, str) else list(keys)
        keys = [k for k in keys if any(q['key'] == k for q in self.say_q) or self.line_end.get(k, -1.0) >= t0]
        while not self.stop and self.now() - t0 < max_s:
            pending = any(q['key'] in keys for q in self.say_q)
            done = not pending and (not self._timed() or all(self.line_end.get(k, -1.0) >= t0 for k in keys))
            if done and self.now() - t0 >= min_s: break
            await self.frame_step(None); await self.sleep(1.0 / self.fps)

    def transition(self, state, why=''):
        """Change the session state: queued lines from the old state are dropped (goodbyes and codes survive) and Gemini gets the new
        state's rules as a note."""
        if state == self.state: return
        prev = self.state; self.state = state; self.epoch += 1
        keep = [q for q in self.say_q if q['urgent'] or q['key'] in KEEP_KEYS]
        for q in keep: q['epoch'] = self.epoch
        self.say_q = keep
        self.telemetry.event('state', frm=prev, to=state, why=why); print(f'[dance] state {prev} -> {state}' + (f' ({why})' if why else ''), flush=True)
        if state == 'PREPARE': self.cmd_floor_t = self.now(); self.pending_cmd = None
        self._apply_state_rules(state)

    def _apply_state_rules(self, state):
        """The state's rules as a short note. (agent.update_instructions re-sent the whole persona as another fake Brodie turn on
        every change and made Gemini restart its context.)"""
        self.context(STATE_RULES.get(state, '') + ' ' + PERSONA_PROMPTS.get(self.persona, ''))

    def _audio_idle(self):
        """Mirror the floor's own notion of 'Brodie is talking' (playing, pending, urgent, mixer backlog, post-clip hold), so an ask
        is only placed when the floor would speak it at once rather than queue it."""
        v = getattr(self.job, 'voice', None); floor = getattr(v, 'floor', None)
        if floor is not None and hasattr(floor, '_audio_busy'):
            try: return not floor._audio_busy()
            except Exception: pass
        p = getattr(self.job, 'player', None)
        if p is None: return True
        if getattr(p, 'playing', None) is not None or getattr(p, 'pending', None) is not None or getattr(p, 'urgent', None) is not None: return False
        mix = getattr(p, 'mix', None)
        return not (mix is not None and (getattr(mix, 'pending', 0) > 0 or self.now() < getattr(mix, 'duck_until', 0.0)))

    def context(self, text):
        """A background note for Gemini (a user-role turn that asks for no reply). Notes state facts only: a note with a command in
        it ("ask if they want to start") got answered on the next turn, on top of our own line."""
        v = getattr(self.job, 'voice', None)
        if v is None or not hasattr(v, '_context') or not text: return
        body = re.sub(r'^\[system note\]\s*', '', text.strip())
        try: v._context('[Background note for you only. Do not reply to it or read it out.] ' + body)
        except Exception as e: print(f'[dance] context failed: {e!r}', flush=True)

    def task(self, coro):
        try: return asyncio.get_running_loop().create_task(coro)
        except RuntimeError: coro.close(); return None              # no loop (tests): drop it cleanly

    # ---- what the caller says ----
    def on_worker_event(self, ev):
        if ev.get('type') != 'voice': return
        now = self.now()
        if ev.get('role') == 'coach':
            text = ev.get('text', ''); self.caption = (text, now)
            if self.state in ('EXPLORE', 'OFFER', 'PREPARE') and FALSE_PROGRESS_RE.search(text) and not text.startswith(SAY_PREFIX):
                self.metrics['false_progress'] += 1; self.telemetry.event('false_progress', state=self.state, text=text[:80])
            if '?' in text:
                self.coach_q_t = now; self.coach_q_text = text
                if self.state == 'OFFER' and self.offer_asked_t < self.proposal_t and re.search(
                        r"\b(start|ready|jump|go|try|routine|dance|camera|want|shall|up for|down for|let'?s)\b", text, re.I):
                    self.offer_asked_t = now                                       # Gemini asked the offer question in its own words
            return
        if ev.get('role') != 'caller': return
        text = ev.get('text', ''); low = text.lower(); self.last_caller_text_t = now
        self.telemetry.event('caller_said', words=len(text.split()), state=self.state)
        brodie_busy = (not self._audio_idle()) or self._model_speaking()
        addr = addr_mod.score(text, question_open=bool(self.q_open) or self.state == 'OFFER', brodie_speaking=brodie_busy, caller_name=self.name)
        self.last_addr = addr; addressed = addr >= addr_mod.ADDRESSED_AT
        if addressed:                                                # talk that isn't to Brodie (TV, friends) never resets the silence timers
            self.deadair.activity(now, caller=True); self.last_addressed_t = now
        else:
            self.metrics['nonaddressed'] = self.metrics.get('nonaddressed', 0) + 1; self.telemetry.event('nonaddressed', state=self.state, score=round(addr, 2))
            if addr <= addr_mod.IGNORE_AT and self.state in ('CHOOSE', 'OFFER', 'CONFIRM', 'REVIEW', 'ACTIVITY') and \
                    (os.environ.get('DANCE_IGNORE_CUT', '1') or '1').strip() != '0':
                self.task(self._cut_reply_soon())                    # Gemini starting to answer the room: stop it
        m = LTR_RE.search(text)
        if m:
            names = [n for n in re.findall(r"\b([A-Z][a-z]{1,15})\b", m.group(1)) if n not in NOT_NAMES][:4]
            if names: self.ltr_names = names; self.apply_ltr_names()
        m = NAME_RE.search(text)
        if m and not self.name and m.group(2) not in NOT_NAMES and (m.group(1).lower() in ('my name is', "my name's", 'call me') or len(text.split()) <= 12):
            self.name = m.group(2); self.telemetry.caller = self.name; print(f'[dance] caller is {self.name}', flush=True)
            self.task(self.welcome_back())
        code = parse_code(text, strict=not self.menu_open)
        if code and self.state in LOBBY and code != (self.proposal or {}).get('code') and \
                not re.search(r"\b(at|born|in|year|since|o'?clock|am|pm|about|around)\s+\d", low) and not re.search(r"\b(19|20)\d\d\b", low):
            print(f'[dance] code heard: {code}', flush=True); self.task(self.handle_code(code))
        it = classify(text); st = self.state
        style_before = self.style; new_style = style_from(low)
        if new_style and st in LOBBY: self.style = new_style; print(f'[dance] style heard: {self.style}', flush=True)
        for w in PERSONA_ORDER:
            if re.search(rf'\b{re.escape(w)}\b', low): self.persona_choice = PERSONA_WORDS[w]; self.apply_persona(self.persona_choice); break
        if st in LOBBY and 'too hard' not in low:
            for rx, d in DIFF_RE:
                if rx.search(low): self.diff_choice = d; self.apply_difficulty(d, announce=False); break
        if any(w in low for w in UPPER_WORDS) or re.search(r"\b(can'?t (step|go|move) back|no (room|space) (to|behind)|as far as i can|that'?s as far|i'?m sitting|sitting down)\b", low):
            if not self.upper_override: self.telemetry.event('upper_body', source='voice')
            self.upper_body = True; self.upper_override = True
        found = parse_commands(low)
        pain = pain_level(low)
        if 'pain' in found and pain != 'strong':
            found = [c for c in found if c != 'pain']
            if pain == 'weak' and now - self.pain_check_t > 30:                 # "this song hurts", "my friend got hurt": ask, don't end the call
                self.pain_check_t = now; self.say_line("Are you okay? If anything hurts, say stop and we'll rest.", key='paincheck', urgent=False)
        elif pain == 'strong' and 'pain' not in found: found.append('pain')
        if re.search(r"\b(after|before|in|at|from|for|of|my|your) class\b", low): found = [c for c in found if c != 'class']   # "after class"
        modes = list(dict.fromkeys(c for c in found if c in MODES))
        if 'report' in found: self.report_req = True
        if self.code_pending and modes and st in LOBBY:              # "is that a battle or a telephone code?" -> "telephone"
            kind = {'battle': 'battle', 'telephone': 'telephone', 'class': 'class', 'class_record': 'class'}.get(modes[0])
            if kind: self.task(self.handle_code(self.code_pending, kind=kind)); return
        if modes and (it['request'] or st == 'OFFER' or (st in ('CHOOSE', 'EXPLORE') and not it['question'])):
            self.mode_choice = modes[0]; print(f'[dance] mode preference: {modes[0]}', flush=True)
        cmd = pick_command(found)
        self.last_utt = (now, it, cmd)
        if cmd in ('pain', 'forget'): self._cmd(cmd, 'voice'); return
        if not addressed and cmd not in ('stop', 'pause'):
            if st == 'CHOOSE': self.noise_utts.append(now)           # room talk while choosing: the card picker starts sooner
            return                                                    # not said to Brodie: no answer, no offer, no mode pick
        if it['affirm'] and not it['question'] and self.q_open and addr < answers_mod.ACCEPT_AT + 0.05 and self.state in ('OFFER', 'PREPARE', 'CONFIRM', 'REVIEW'):
            if answers_mod.decide(answers_mod.confidence('weak' if len(text.split()) > 6 else 'strong', addressed=addr >= 0.7),
                                  already_asked=self.yes_asked_q == self.q_open) == 'ask' and now - self.unclear_t > 15.0:
                self.yes_asked_q = self.q_open; self.metrics['yes_checks'] = self.metrics.get('yes_checks', 0) + 1
                self.ask_unclear(); return                            # unsure: "Was that a yes?" once, never twice for one question
        words = re.findall(r"[a-z']+", low)
        if st in LOBBY and BYE_RE.search(low) and not it['question'] and len(words) <= 10 and not re.search(r"\b(don'?t|not) (go|leave|hang)", low):
            self.bye_req = True; self.telemetry.event('bye', state=st); return            # "bye" / "I'm done": the lobby says goodbye
        if self.q_open and len(words) <= 3 and not (it['affirm'] or it['deny'] or it['request'] or it['question'] or it['hold'] or it['cancel'] or cmd or modes) \
                and not NAME_RE.search(text) and now - self.unclear_t > 15.0 and st in ('OFFER', 'PREPARE', 'CONFIRM', 'REVIEW'):
            self.ask_unclear(); return
        if re.search(r"\b(how did i do|how was that|what should i fix|what do i fix|what did i do wrong)\b", low) and not self.tools_ok and self.last_cues:
            tips = [c.get('text') for c in self.last_cues if c.get('text')][:2]       # no Gemini tools: answer from the analysis ourselves
            if tips: self.say_line(' And '.join(tips) + '.', key='howdid'); return
        if st in ('CHOOSE', 'EXPLORE'):
            if BOARD_RE.search(low) and not modes: self.task(self.show_board(wait=False)); return  # "show me the board" / "leaderboard"
            if MENU_RE.search(low) and not modes: self.say_menu(); return                # "what can you do?" -> the menu again
            if len(modes) >= 2 and not it['deny']:                                       # "battle or the daily?" -> which one
                self.say_line(f"Which one: {self.mode_name(modes[0])} or {self.mode_name(modes[1])}?", key='menu'); return
            usable = bool(modes or it['request'] or it['question'] or it['affirm'] or it['deny'] or it['hold'] or cmd or NAME_RE.search(text) or new_style)
            if st == 'CHOOSE' and not usable and (len(words) > 12 or len(words) <= 2):
                self.noise_utts.append(now)                                              # chatter, TV, a garbled word: the room may be noisy
            # a mode word picks it only when said to Brodie: a request, or a short line (long talk in the room never picks a mode)
            if it['request'] or (modes and not it['question'] and not it['deny'] and (st != 'CHOOSE' or len(words) <= 10)):
                self.request_activity(mode=modes[0] if modes else None, rematch=it['rematch'], source='voice')
            elif self.menu_open and VAGUE_RE.search(low) and not it['question']:         # "idk" / "surprise me": Brodie picks for them
                self.request_activity(mode=self.default_mode(), style=STYLE_NAMES.get(self._next_style(), 'hip-hop'), source='brodie')
            return
        if st == 'OFFER':
            p = self.proposal or {}
            if len(modes) >= 2 and not it['deny']:
                self.say_line(f"Which one: {self.mode_name(modes[0])} or {self.mode_name(modes[1])}?", key='menu'); return
            changed = (self.style != style_before and self.style != p.get('style')) or bool(modes and modes[0] != p.get('mode'))
            if changed and not it['cancel']:                                      # a change of mind ("actually K-pop"): a new offer
                self.request_activity(mode=modes[0] if modes else None, rematch=it['rematch'], source='voice'); return
            if it['cancel'] or (it['deny'] and not it['hold']): self.cancel_req = True      # "no" / "not right now": back to chat
            elif it['hold']: self.hold_until = now + 30.0
            elif (it['affirm'] and not it['question']) or it['request']: self.affirm_t = now   # "yes", or "all right let's start it"
            return
        if st == 'PREPARE':
            if it['cancel']: self.cancel_req = True; self.deny_t = now
            elif it['hold']: self.hold_until = now + 30.0                         # a hold is not a "no"
            elif it['deny']: self.deny_t = now
            elif it['affirm'] and not it['question']: self.affirm_t = now
            return
        if st == 'CONFIRM' and SOLO_RE.search(low): self.solo_req = True             # "just me": a squad becomes solo
        if st in ('CONFIRM', 'REVIEW') and modes and it['request'] and modes[0] != self.mode and not it['question']:
            # "let's do a ghost battle instead": leave this activity and offer that one (never mid-music: ACTIVITY ignores it)
            self.switch_req = modes[0]; self.telemetry.event('mode_switch', frm=self.mode, to=modes[0])
            if st == 'CONFIRM': self.cancel_req = True
            else: self.stop_routine = True; self.say_line(f"Switching to {self.mode_name(modes[0])}. We'll leave this one here.", key='cancel')
            return
        if st == 'REVIEW' and MENU_RE.search(low) and not modes:
            self.menu_req = True; self.stop_routine = True; return                     # "menu" between moves: back to the cards
        if st in ('CONFIRM', 'REVIEW'):
            if it['cancel'] and st == 'CONFIRM': self.cancel_req = True; self.deny_t = now
            elif it['hold']: self.hold_until = now + 30.0
            elif it['deny'] and not cmd: self.deny_t = now
            elif it['affirm'] and not it['question']: self.affirm_t = now
            if st == 'REVIEW' and it['hold'] and cmd == 'pause': cmd = None      # "hold on" between moves is a wait, not a pause
            if not cmd: return
        if st == 'PAUSED' and it['affirm']: cmd = 'resume'
        if cmd: self._cmd(cmd, 'voice')

    def avatar_state(self, speaking):
        """What Brodie's avatar shows, from real signals: offline (no voice), speaking (his audio is playing), listening (the
        caller is talking), thinking (a line is being prepared), else idle."""
        v = getattr(self.job, 'voice', None); model = getattr(v, 'model', None)
        if v is None or getattr(v, 'ask_voice', None) is None: return 'offline'
        if self.now is time.monotonic and model is not None and not getattr(model, 'started', False): return 'offline'
        if speaking: return 'speaking'
        last = getattr(model, 'last_caller_t', None)
        if (last is not None and time.monotonic() - float(last) < 1.0) or self.now() - self.last_caller_text_t < 1.5: return 'listening'
        if self.say_q: return 'thinking'
        return 'idle'

    def camera_note_text(self):
        from team.dance.setup import camera_note
        note = camera_note(self.cur_lm, (self.setup_ui or {}).get('issue'), upper=self.upper_override)
        o = self.orient_state or {}
        if o.get('away'): note = note[:-1] + '; they have their back to the camera]'
        elif abs(o.get('yaw') or 0) >= 35: note = note[:-1] + f"; turned about {abs(o['yaw']):.0f} degrees to their {'left' if o['yaw'] > 0 else 'right'}]"
        return note

    def mode_name(self, mode):
        return dict(learn='a lesson', battle='a ghost battle', daily="the daily challenge", squad_battle='a squad battle', squad_learn='a squad lesson',
                    telephone='dance telephone', **{'class': 'a class', 'class_record': 'class recording'}).get(mode, 'a lesson')

    def default_mode(self):
        """What Brodie suggests when the caller leaves it to him: a lesson for a new caller, a ghost battle for a returning one."""
        return self.mode_choice or ('battle' if self.hist_best is not None and self.hist_best >= 60 else 'learn')

    def say_menu(self):
        self.menu_open = True; self.telemetry.event('menu')
        if self.state in ('EXPLORE', 'REVIEW'): self.menu_req = True               # "menu" / "what else": back to the cards
        self.say_line(mode_cards.HELLO.replace("Hey, I'm Brodie! ", ''), key='menu', ttl=15)

    async def handle_code(self, code, kind=None):
        """A spoken four-digit code: a friend's battle, a telephone chain to join, or a recorded class. Each makes the matching
        offer (nothing starts without a yes). A code that matches two kinds (old codes) asks which; an unknown one asks again."""
        r = await self.backend.call('GET', f'/dance/code/{code}') or {}
        kinds = [k for k in (r.get('kinds') or []) if kind is None or k.get('kind') == kind]
        digits = ' '.join(str(code))
        self.telemetry.event('code', code=code, kinds=[k.get('kind') for k in kinds])
        if not kinds:
            self.code_pending = None
            self.context(f"Code {code} doesn't match any battle, telephone chain or class.")
            self.say_line(f"I can't find code {digits}. Say it again one digit at a time, or want to battle my ghost instead?", key='code_q'); return None
        if len(kinds) > 1:
            self.code_pending = code
            names = ' or a '.join(dict(battle='battle', telephone='telephone', **{'class': 'class'})[k['kind']] for k in kinds)
            self.say_line(f"Code {digits}: is that a {names} code?", key='code_q'); return None
        self.code_pending = None; k = kinds[0]
        if self.state not in LOBBY: return None
        if k['kind'] == 'battle':
            ch = await self.load_challenge(code)
            if ch: self.request_activity(mode='battle', source='code', code=code)
        elif k['kind'] == 'telephone':
            self.context(f"Code {code} is a dance telephone chain with {k.get('links')} link(s): the caller copies the last move.")
            self.request_activity(mode='telephone', source='code', code=code, code_info=k)
        else:
            self.context(f"Code {code} is {k.get('teacher')}'s recorded class.")
            self.request_activity(mode='class_record', source='code', code=code, code_info=k)
        return k

    def request_activity(self, mode=None, style=None, difficulty=None, rematch=False, source='voice', code=None, code_info=None):
        """A clear wish to dance (the caller, a code, Gemini's propose_activity, or Brodie's own suggestion) becomes an OFFER, never
        a start: Brodie asks if they want to begin now, and only a clear yes leads to the camera check. The same offer again changes
        nothing (it used to restart the offer and throw away the caller's yes: "All right, let's start it" looped forever)."""
        if self.state not in LOBBY:
            return dict(ok=False, state=self.state, reason='already setting up or dancing')
        if source == 'tool' and self.state == 'CHOOSE' and self.now() - self.last_addressed_t > 6.0:
            # Gemini heard a mode in background talk (a TV, people chatting): the caller hasn't asked for anything yet
            self.telemetry.event('offer_refused', why='no addressed request'); self.metrics['nonaddressed'] = self.metrics.get('nonaddressed', 0) + 1
            return dict(ok=False, state='CHOOSE', reason='No clear request from the caller yet; keep listening and say nothing about starting.')
        st = self.style
        if style:
            sc = STYLE_WORDS.get(str(style).lower()) or next((STYLE_WORDS[w] for w in STYLE_ORDER if w in str(style).lower()), None)
            if sc: st = sc
        if source != 'brodie': self.style = st                       # Brodie's own pick never overwrites what the caller said they like
        if difficulty:
            d = dict(easy='beginner', medium='intermediate', hard='advanced').get(str(difficulty).lower(), str(difficulty).lower())
            if d in LEVEL_CFG: self.apply_difficulty(d, announce=False)
        prev = self.proposal or {}
        new = dict(mode=mode or prev.get('mode') or self.mode_choice or 'learn', style=st, difficulty=self.difficulty, rematch=bool(rematch), source=source,
                   code=code if code is not None else (prev.get('code') if (mode or prev.get('mode')) == prev.get('mode') else None),
                   code_info=code_info if code_info is not None else prev.get('code_info'), practice=False)
        if new['mode'] not in MODES: new['mode'] = 'learn'
        if self.state == 'OFFER' and prev and all(new[k] == prev.get(k) for k in ('mode', 'style', 'difficulty', 'code')):
            return dict(ok=True, state='OFFER', already_offered=True, offer=self.offer_label(),
                        waiting_for="the caller's clear yes to starting it now; nothing has started")
        self.proposal = new; self.proposal_t = self.now(); self.offer_asked_t = -1.0; self.menu_open = False
        self.telemetry.event('offer', mode=self.proposal['mode'], style=st, source=source)
        if new['mode'] in SQUAD_MODES and self.tracker.stable_count() < 2:
            self.say_line("Squad needs at least two of you in the picture. Grab a friend, or I can do it with just you.", key='squadneed')
        if new['mode'] == 'daily': self.task(self._check_daily(new))
        self.context(f"Suggested: {self.offer_label()}. Nothing has started; it starts only after the caller says yes, with a quick camera check first.")
        if self.menu_ui is not None and source != 'card': self.menu_ui = dict(self.menu_ui, picked=new['mode'], lit=None)
        if self.state in ('CHOOSE', 'EXPLORE'): self.transition('OFFER', source)
        return dict(ok=True, state='OFFER', offer=self.offer_label(), note='Nothing has started. Ask once if they want to start it now; a camera check comes first.')

    async def _check_daily(self, prop):
        """One shot at the daily board: a caller who already had today's shot (this call, or by name) gets a practice run instead."""
        played = self.daily_done
        if not played and self.name:
            r = await self.backend.call('GET', f"/dance/daily_played?name={self.name}&key={choreo.daily()['name']}") or {}
            played = bool(r.get('played'))
        if played and self.proposal is prop and self.state == 'OFFER':
            prop['practice'] = True
            self.say_line("You've had your shot at today's board. Want to practise it anyway, or try a ghost battle?", key='offer', ttl=12)

    def offer_label(self):
        p = self.proposal or {}
        mode = p.get('mode') or 'learn'; style = STYLE_NAMES.get(p.get('style') or 'lh', 'hip-hop'); info = p.get('code_info') or {}
        if mode == 'daily': return "today's daily challenge" + (" as practice" if p.get('practice') else ", one shot at the board")
        if mode == 'battle':
            if p.get('code') and self.challenge: return f"a battle against {self.challenge.get('name') or 'your friend'}'s ghost"
            return f"a 3-round {style} ghost battle"
        if mode == 'squad_battle': return f"a {style} squad battle"
        if mode == 'squad_learn': return f"a {style} lesson for the whole squad"
        if mode == 'telephone':
            return f"link {int(info.get('links') or 0) + 1} of that telephone chain" if p.get('code') else "dance telephone: you make up a move, friends copy it"
        if mode == 'class': return f"a {style} class: I lead, everyone follows"
        if mode == 'class_record':
            return f"{info.get('teacher') or 'the'}'s class" if p.get('code') else "class recording: you teach the moves, I remember them"
        return f"a {p.get('difficulty') or self.difficulty} {style} routine"

    def _cmd(self, cmd, source='voice'):
        """Commands during a routine (voice regex, Gemini tool, gesture, page button): dedupe, log, act. Starting is not a command.
        Returns True when something will happen."""
        now = self.now(); last = self.last_cmd.get(cmd)
        if last is not None and (now - last[0] < 1.5 or (now - last[0] < 4.0 and last[1] != source)):
            return False                                                  # the same words heard by the regex and by Gemini's tool: once
        self.last_cmd[cmd] = (now, source)
        self.telemetry.event('command', cmd=cmd, source=source, state=self.state); print(f'[dance] command ({source}): {cmd}', flush=True)
        if cmd == 'pain': self.safety_stop = True; self.pending_cmd = 'pain'; self.task(self.safety_halt()); return True
        if cmd == 'forget': self.task(self.request_forget()); return True
        if self.state in LOBBY:
            if cmd in ('ready', 'resume') and self.want_ready and source != 'voice':      # hands up / the button: picks the lit card
                self.affirm_t = now; return True
            if cmd in ('harder', 'easier'):
                order = ['beginner', 'intermediate', 'advanced']; j = order.index(self.difficulty) + (1 if cmd == 'harder' else -1)
                if 0 <= j < 3: self.apply_difficulty(order[j], announce=False)
                return True
            return False                                                      # flow commands mean nothing before a routine
        if cmd in ('resume', 'ready'):
            if self.paused: self.resume_req = True
            elif self.state in ('CONFIRM', 'PREPARE', 'REVIEW'): self.affirm_t = now
            return True
        if self.state in ('PREPARE', 'CONFIRM'):
            if cmd == 'stop': self.cancel_req = True; return True
            return False                                                      # nothing to repeat, skip or pause yet (no stale command for later)
        if cmd == 'pause':
            if self.cur_engine is not None and self.cur_engine.state != 'done': self.pause_req = True
            else: self.pending_cmd = 'pause'; self.pending_t = now
            return True
        self.pending_cmd = cmd; self.pending_t = now
        return True

    def take_cmd(self, accept=None):
        """The pending command if it is fresh (said after the current floor) and accepted; it is consumed."""
        cmd = self.pending_cmd
        if cmd is None: return None
        if self.pending_t < self.cmd_floor_t: self.pending_cmd = None; return None      # said before this step or routine: stale
        if accept is not None and cmd not in accept: return None
        self.pending_cmd = None
        return cmd

    def apply_persona(self, p):
        if p not in PERSONA_PROMPTS: return False
        self.persona = p; self.context(PERSONA_PROMPTS[p]); self.telemetry.event('persona', persona=p)
        if self.name: self.task(self.backend.call('POST', f'/dance/profile/{self.name}/prefs', dict(persona=p)))
        return True

    def apply_difficulty(self, d, announce=True):
        """Set the difficulty if unlocked; else tell Gemini (as a note) how far away it is. Returns (applied, reason)."""
        if d not in LEVEL_CFG: return False, f'unknown difficulty {d!r}'
        if not progression.difficulty_unlocked(d, self.level):
            need = progression.xp_to_unlock(d, self.xp); lv = progression.LEVELS[progression.DIFF_UNLOCK_LEVEL[d]][1]
            reason = f'{d} unlocks at level {lv}, {need} XP away'
            self.context(f"[system note] {d} is locked: it unlocks at {lv}, {need} XP away; staying at {self.difficulty}.")
            if announce and self.state not in ('EXPLORE', 'PREPARE', 'CONFIRM'): self.say_line(f"{d.capitalize()} unlocks at {lv}, {need} XP away. We'll stay on {self.difficulty} for now.", key='diff')
            return False, reason
        changed = d != self.difficulty; self.difficulty = d; self.telemetry.event('difficulty', difficulty=d)
        if self.name: self.task(self.backend.call('POST', f'/dance/profile/{self.name}/prefs', dict(difficulty=d)))
        return changed, ''

    async def welcome_back(self):
        """The caller said their name: load their history and profile as a note for Gemini (it greets them itself; we don't ask)."""
        if not self.name or getattr(self, '_welcomed', False): return
        self._welcomed = True
        hist = await self.backend.call('GET', f'/dance/history?name={self.name}')
        prof = await self.backend.call('GET', f'/dance/profile/{self.name}')
        if prof and not prof.get('error'):
            self.profile = prof; self.level = int(prof.get('level') or 0); self.xp = int(prof.get('xp') or 0)
            prefs = prof.get('prefs') or {}
            if prefs.get('persona') and not self.persona_choice: self.persona = prefs['persona']; self._apply_state_rules(self.state)
            if prefs.get('difficulty') and not self.diff_choice and progression.difficulty_unlocked(prefs['difficulty'], self.level): self.difficulty = prefs['difficulty']
        if not hist or not isinstance(hist, list):
            self.context(f"[system note] The caller's name is {self.name}. First time here: level Rookie. Good starting points: the beginner hip-hop or pop routine.")
            return
        last = hist[0]; best = max(float(h.get('score100') if h.get('score100') is not None else (h.get('total') or 0)) for h in hist)
        self.hist_best = best
        # FOR YOU: a rematch of what they battled last time, else today's daily if they haven't had their shot
        lm_ = last.get('mode')
        self.recommended_mode = lm_ if lm_ in ('battle', 'squad_battle') else ('daily' if not self.daily_done else 'learn')
        if self.state == 'CHOOSE' and self.menu_ui is not None and not self.menu_ui.get('picked'): self._open_menu(self.menu_ui.get('header') or 'WHAT ARE WE DOING?')
        share = await self.backend.call('GET', f"/dance/share/{last.get('code')}") if last.get('code') else None
        board = await self.backend.call('GET', f"/dance/board/{last.get('code')}") if last.get('code') else None
        line = (f"[system note] The caller is {self.name}, back for visit {len(hist) + 1}. Last time: {last.get('mode')} on {str(last.get('key', '')).replace('_', ' ')}. "
                f"Best {best:.0f}. Last code {last.get('code')}.")
        if self.profile: line += f" Level {self.profile.get('level_name')} with {self.xp} XP, streak {self.profile.get('streak', 0)} day(s)."
        if share and share.get('starts'): line += f" {share['starts']} friend(s) took their challenge since."
        if board and not board.get('error') and board.get('n', 0) > 1:
            top = board['entries'][0]; line += f" Their board leader is {top['name']} with {top['score100']:.0f}."
        rec = self.recommendation(hist)
        if rec: line += f" Good next step for them: {rec}."
        self.context(line)
        self.telemetry.event('welcome_back', visits=len(hist), level=self.level)

    def recommendation(self, hist):
        scores = [float(h.get('score100') if h.get('score100') is not None else (h.get('total') or 0)) for h in hist[:3]]
        order = ['beginner', 'intermediate', 'advanced']; i = order.index(self.difficulty)
        if len(scores) >= 2 and np.mean(scores) >= 85 and i < 2 and progression.difficulty_unlocked(order[i + 1], self.level):
            return f"moving up to {order[i + 1]} (they've been crushing it)"
        if len(scores) >= 2 and np.mean(scores) < 50 and i > 0: return f"{order[i - 1]}, to build confidence"
        return None

    async def load_challenge(self, code):
        """A friend's code: fetch it and load that routine. Saying a code is a clear request to battle it, so it leads to the
        camera setup (never straight to music)."""
        r = await self.backend.call('GET', f'/dance/result/{code}')
        if not r or r.get('error'):
            self.context(f"Code {code} doesn't match any challenge.")
            self.code = None; return None
        self.challenge = r
        g = await self.backend.call('GET', f'/dance/ghost/{code}')
        if g and not g.get('error') and g.get('data') and not ghost_mod.valid(g.get('data'), g.get('bpm'), g.get('per_beat')):
            print(f'[dance] ghost {code} failed validation; the routine instead', flush=True); g = None
        if g and not g.get('error') and g.get('data'):                       # a saved ghost: battle what the friend really danced
            try:
                k = ghost_mod.ghost_key(code, r.get('name'), ghost_mod.unpack(g['data']), float(g.get('bpm') or 100), int(g.get('per_beat') or 4))
                k['default_tempo'] = 1.0; self.queued_key = k; self.challenge = dict(r, ghost=True)
                self.telemetry.event('ghost_loaded', code=code)
            except Exception as e:
                print(f'[dance] ghost {code} not loadable ({e!r}); the routine instead', flush=True); g = None
        if not (g and not g.get('error') and g.get('data')):
            try:
                k = choreo.load_any(r['key']) if r.get('key') else None
                if k is not None:
                    k['title'] = key_title(k); self.queued_key = k
            except Exception as e:
                print(f'[dance] challenge routine {r.get("key")!r} not loadable ({e!r}); keeping the current one', flush=True)
        await self.backend.call('POST', f'/dance/share/{code}/start')
        s100 = r.get('score100') if r.get('score100') is not None else r.get('total')
        self.context(f"The caller is taking on {r.get('name')}'s challenge {code}: {float(s100 or 0):.0f} on {str(r.get('key', '')).replace('_', ' ')}. "
                     "Nothing has started yet.")
        self.telemetry.event('challenge_start', code=code)
        return r

    # ---- safety, pause, forget ----
    async def safety_halt(self):
        if self.safety_shown: return
        self.safety_shown = True; self.stop = True
        self.cut_audio()
        if self.cur_engine is not None and self.cur_engine.state != 'done': self.cur_engine.abort(self.now())
        self.transition('SAFETY', 'pain words'); self.telemetry.event('safety_stop')
        self.say_line("Let's stop here. Sit down, breathe and drink some water. If the pain or dizziness doesn't go away, please talk to a doctor or someone nearby.",
                      key='safety', ttl=20, urgent=True)

    async def hold_safety(self, seconds):
        """After a pain/dizziness stop: keep the calm safety card on screen while Brodie says to rest (no music, no score)."""
        await self.sleep(0.05)                                   # let the scheduled stop line start
        t_end = self.now() + seconds
        while self.now() < t_end and not getattr(self.job, 'closed', False):
            await self.frame_step(None); await self.sleep(1.0 / self.fps)

    async def paused_wait(self):
        """Paused until the caller says resume (or raises both hands) or stop. It never resumes by itself (the music used to restart
        after 120 s with nobody there); if nobody has been in view for two minutes the routine ends."""
        self.transition('PAUSED', 'pause'); self.paused = True; self.resume_req = False; self.telemetry.event('paused')
        self.say_line("Paused. Say resume, or raise both hands, when you're ready.", key='paused')
        t_seen = self.now()
        while not self.stop and not self.safety_stop:
            c = self.take_cmd(('ready', 'again', 'stop'))
            if c == 'stop': self.stop_routine = True; break
            if self.resume_req or c: break
            if self.cur_lm is not None: t_seen = self.now()
            if self.now() - t_seen > 120.0:
                self.stop_routine = True; self.say_line("I'll stop here for now. Say rematch when you're back.", key='cancel'); break
            await self.frame_step(None); await self.sleep(1.0 / self.fps)
        self.paused = False; self.resume_req = False; self.telemetry.event('resumed')
        self.transition('REVIEW', 'resume')
        if not self.stop: await self.idle(1.0)

    async def request_forget(self):
        now = self.now()
        if now > self.forget_until:
            self.forget_until = now + 15.0
            self.say_line("Say forget me once more and I'll delete your scores and progress.", key='forget')
            return dict(needs_confirmation=True)
        return await self.do_forget()

    async def do_forget(self):
        self.forget_until = -1.0
        if not self.name:
            self.say_line("I don't have anything saved for you, and your video is never stored.", key='forget')
            return dict(deleted=None, reason='no name on this call')
        out = await self.backend.call('DELETE', f'/dance/profile/{self.name}')
        self.profile = None; self.level = 0; self.xp = 0; self.telemetry.event('forget_me', ok=bool(out))
        self.say_line("Done. Your scores and progress are deleted, and your video was never stored.", key='forget')
        return dict(deleted=(out or {}).get('deleted'))

    # ---- tools (called by Gemini through tools.py) ----
    def tool_command(self, action):
        """Gemini's control_session. Only a running routine (or the results between moves, or a pause) can be changed; 'stop' during
        the offer, camera check or ready check cancels back to chat; harder/easier before a routine change the level. The result says
        what really happens, so Gemini never claims something that didn't."""
        st = self.state
        if st in LOBBY and action not in ('harder', 'easier', 'stop'):
            return dict(will_happen='nothing', state=st, reason='no routine is running; nothing starts until the caller says yes to an offer')
        if st in ('OFFER', 'PREPARE', 'CONFIRM') and action == 'stop':
            self.cancel_req = True; return dict(will_happen='going back to chat', state=st)
        if st in ('PREPARE', 'CONFIRM'):
            return dict(will_happen='nothing', state=st, reason='the camera check / ready check is running; only the caller saying ready or yes moves it on')
        done = self._cmd(action, 'tool')
        if not done: return dict(will_happen='nothing', state=st, reason='already handled or not possible right now')
        what = dict(again='repeating the step', slower='slowing down', faster='speeding up', next='moving on', pause='pausing', resume='resuming',
                    stop='ending the routine', simplify='breaking it down', explain='explaining the move', full='going to the full routine',
                    harder='going harder', easier='going easier').get(action, action)
        return dict(will_happen=what, state=st, mid_phrase=bool(self.cur_engine is not None and self.cur_engine.state not in ('done', 'idle')))

    async def _cut_reply_soon(self):
        """Gemini may start a reply to talk that wasn't to Brodie: if it does within a second, interrupt it (our own queued lines are
        untouched; urgent safety lines go through the speaker loop, not Gemini's free replies)."""
        model = self._model()
        if model is None or not hasattr(model, 'interrupt'): return
        t = self.now()
        while self.now() - t < 1.5:
            if self._model_speaking() and not self.say_q:
                try: model.interrupt(); self.telemetry.event('reply_cut', state=self.state)
                except Exception as e: print(f'[dance] interrupt failed: {e!r}', flush=True)
                return
            await asyncio.sleep(0.1 if self.now is time.monotonic else 0)
            if self.now is not time.monotonic: return

    def tool_answer(self, answer):
        """Gemini's caller_answered: Gemini hears the audio far better than the transcript ("de Hello", "c'est quand"), so its reading
        of a clear yes / no / ready counts like the caller's own words, but only while one of our questions is open (the offer, the
        space question, the ready check, a choice between moves)."""
        a = (answer or '').strip().lower(); now = self.now()
        if a not in ('yes', 'no', 'ready'): return dict(ok=False, error="answer must be yes, no or ready")
        if self.state not in ('OFFER', 'PREPARE', 'CONFIRM', 'REVIEW'):
            return dict(ok=True, counted=False, state=self.state, reason='no question is waiting for an answer')
        self.telemetry.event('caller_answered', answer=a, state=self.state)
        recent_addr = self.last_addr if (self.last_utt is not None and now - self.last_utt[0] < 4.0) else None
        conf = answers_mod.confidence('gemini', addressed=recent_addr is None or recent_addr >= addr_mod.ADDRESSED_AT)
        if a != 'no' and answers_mod.decide(conf, already_asked=self.yes_asked_q == self.q_open) == 'ask':
            self.yes_asked_q = self.q_open; self.ask_unclear()
            return dict(ok=True, counted=False, state=self.state, reason='unclear; the system asks "was that a yes?"')
        if a == 'no':
            if self.state == 'OFFER': self.cancel_req = True
            else: self.deny_t = now
        else: self.affirm_t = now
        return dict(ok=True, counted=True, state=self.state, next='the system moves on by itself; say at most a few words')

    def tool_confirm(self):
        """Gemini's confirm_start: it only reports; the start itself needs the caller's own clear yes, heard by the system."""
        if self.state != 'CONFIRM': return dict(started=False, state=self.state, reason='not at the ready check yet')
        return dict(started=False, state=self.state, waiting_for="the caller's own clear yes ('ready', 'let's go') or both hands up")

    def set_preferences(self, personality='', difficulty='', style=''):
        applied = {}; locked = None
        p = PERSONA_WORDS.get((personality or '').strip().lower(), (personality or '').strip().lower())
        if p in PERSONA_PROMPTS and self.apply_persona(p): applied['personality'] = p
        d = (difficulty or '').strip().lower(); d = dict(easy='beginner', medium='intermediate', hard='advanced').get(d, d)
        if d:
            ok, reason = self.apply_difficulty(d, announce=False)
            if reason: locked = reason
            else: applied['difficulty'] = d; self.diff_choice = d
        s = (style or '').strip().lower()
        if s:
            code = STYLE_WORDS.get(s) or next((STYLE_WORDS[w] for w in STYLE_ORDER if w in s), None)
            if code: self.style = code; applied['style'] = STYLE_NAMES.get(code, code)
        return dict(applied=applied, locked_reason=locked, persona=self.persona, difficulty=self.difficulty)

    def camera_status(self):
        tm = self.telemetry
        out = dict(parts=self.parts, hint=self.hint, dancers=self.tracker.stable_count(), overlapping=self.tracker.overlapping(),
                   tracking_rate=round(tm.frames_pose / tm.frames, 3) if tm.frames else None, full_body=all(self.parts.get(p, False) for p in ('head', 'arms', 'torso', 'legs')))
        if self.state in LOBBY:                         # never let Brodie announce "camera's locked in" before the check
            out['note'] = 'Nothing has started. The camera check only happens after the caller says yes to a routine; do not say the camera is set.'
        return out

    def last_attempt_summary(self):
        r = self.last_result
        if not r: return dict(attempted=False)
        if not r.get('scored'): return dict(attempted=True, scored=False, reason=r.get('reason') or 'not enough of the body visible')
        cue = self.last_cue or {}
        corr = [dict(text=c.get('text'), kind=c.get('kind'), body_part=c.get('body_part'), counts=[b + 1 for b in c.get('beats') or []]) for c in (self.last_cues or [])]
        return dict(attempted=True, scored=True, score=r['score'], grade=progression.grade(r['score']), title=self.last_title, shape=round(100 * r['spatial']),
                    timing=round(100 * r['timing']), energy=round(100 * r['energy']), weak_counts=[b + 1 for b in r.get('weak_beats', [])],
                    cue=dict(text=cue.get('text'), body_part=cue.get('body_part'), direction=cue.get('direction'), counts=[b + 1 for b in cue.get('beats', [])]),
                    corrections=corr, size=r.get('size'))

    async def tool_compare(self, code):
        code = parse_code(str(code)) or str(code).strip()
        if not re.fullmatch(r'\d{4}', code): return json.dumps(dict(ok=False, error='a challenge code is four digits'))
        k = await self.handle_code(code)
        if not k: return json.dumps(dict(ok=False, error='no such code, or it matches more than one kind (Brodie asked which)'))
        return json.dumps(dict(ok=True, kind=k.get('kind'), name=k.get('name') or k.get('teacher'), score100=k.get('score100'),
                               offered=self.offer_label() if self.state == 'OFFER' else None))

    def queue_routine(self, style, difficulty='beginner'):
        code = STYLE_WORDS.get((style or '').strip().lower()) or next((STYLE_WORDS[w] for w in STYLE_ORDER if w in (style or '').lower()), 'lh')
        d = dict(easy='beginner', medium='intermediate', hard='advanced').get((difficulty or '').lower(), (difficulty or '').lower())
        if d in LEVEL_CFG and not progression.difficulty_unlocked(d, self.level): d = self.difficulty
        if d not in LEVEL_CFG: d = self.difficulty
        k = self.make_routine(code, d); self.queued_key = k; self.style = code
        return dict(title=k['title'], sections=len([p for p in k['phrases'] if not p.get('combined')]), seconds=round(choreo.routine_seconds(k, LEVEL_CFG[d]['tempo']), 1),
                    difficulty=d, starts='at the next step')

    def profile_summary(self):
        lv = progression.level_for(self.xp)
        return dict(name=self.name, level=lv[0], level_name=lv[1], xp=self.xp, next_level_at=lv[3], streak=(self.profile or {}).get('streak', 0),
                    achievements=(self.profile or {}).get('achievements', []), unlocked=sorted(progression.unlocked(self.level)), saved=bool(self.name))

    async def tool_forget(self, confirm):
        if not confirm: return json.dumps(dict(ok=True, needs_confirmation=True))
        if self.now() > self.forget_until: r = await self.request_forget(); return json.dumps(dict(ok=True, **r))
        r = await self.do_forget(); return json.dumps(dict(ok=True, **(r or {})), default=str)

    def squad_status(self):
        sq = getattr(self.cur_engine, 'squad', None) if isinstance(self.cur_engine, SquadEngine) else None
        players = [dict(colour=c, name=n) for _, c, n in self.squad_players]
        out = dict(mode=self.call_mode, dancers_in_frame=self.tracker.stable_count(), players=players)
        if sq: out.update(ranking=[dict(label=r['label'], score=r['score'], grade=r['grade']) for r in sq['ranking']], mvp=(sq['mvp'] or {}).get('label'), crew_sync=sq['crew_sync'])
        counts = getattr(self, '_last_squad_counts', None)
        if counts:                                                        # "who nailed count 6?"
            n = max(len(v) for v in counts.values())
            out['per_count'] = {lab: [v[b] if b < len(v) else None for b in range(n)] for lab, v in counts.items()}
            out['nailed'] = {f'count {b + 1}': [lab for lab, v in counts.items() if b < len(v) and v[b] == 'green'] for b in range(n)}
        return out

    def set_call_mode(self, mode):
        mode = (mode or '').strip().lower()
        if mode == 'solo': self.call_mode_choice = 'solo'; self.mode_choice = 'learn'; return dict(mode='solo')
        if mode in ('squad_battle', 'squad_learn'):
            if self.tracker.stable_count() < 2: return dict(mode=self.call_mode, note='only one dancer in frame; squad needs two or more')
            self.mode_choice = mode; return dict(mode=mode)
        return dict(error=f'unknown mode {mode!r}')

    def assign_name(self, colour, name):
        colour = (colour or '').strip().lower(); name = (name or '').strip().capitalize()
        if colour not in ('volt', 'pink', 'cyan', 'amber') or not name: return dict(error='colour must be volt, pink, cyan or amber, and a name is needed')
        self.names[colour] = name
        self.squad_players = [(pid, c, self.names.get(c)) for pid, c, _ in self.squad_players]
        return dict(colour=colour, name=name)

    def apply_ltr_names(self):
        order = [c for _, c, _ in self.squad_players] or [c for _, c in self.tracker.assign_colours(self.mirror_view)]
        for c, n in zip(order, self.ltr_names): self.names[c] = n
        self.squad_players = [(pid, c, self.names.get(c)) for pid, c, _ in self.squad_players]

    # ---- video out ----
    async def publish(self):
        room = getattr(self.job, 'room', None)
        if room is None: return
        try:
            from livekit import rtc
            cw, ch = self.size
            self.source = rtc.VideoSource(cw, ch)
            track = rtc.LocalVideoTrack.create_video_track('coach-video', self.source)
            await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA, simulcast=False,
                                                                                      video_encoding=rtc.VideoEncoding(max_framerate=int(self.fps), max_bitrate=1_800_000)))
            self.published = True; print(f'[dance] published video track coach-video {cw}x{ch} at {self.fps:g} fps', flush=True)
        except Exception as e:
            print(f'[dance] video publish failed ({e!r}); running without video out', flush=True)

    def push_frame(self, bgr):
        """Loop-side push of an already composed BGR frame (the simulator and small paths); the live loop uses render_push."""
        vf = self._encode_sync(bgr)
        if vf is not None: self.source.capture_frame(vf)

    def _encode_sync(self, bgr):
        """Thread or loop: write the debug file and encode an rtc.VideoFrame (I420) for the published track."""
        if self.writer is not None: self.writer.write(bgr)
        if self.source is None: return None
        from livekit import rtc
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); h, w = rgb.shape[:2]
        return rtc.VideoFrame(w, h, rtc.VideoBufferType.RGB24, rgb.tobytes()).convert(rtc.VideoBufferType.I420)

    def _prepare_sync(self, frame, ms):
        """Thread: the caller's raw frame (rtc.VideoFrame or BGR ndarray) -> (bgr, landmarks, world landmarks)."""
        if isinstance(frame, np.ndarray): bgr = frame
        else:
            from facetime.coach_agent import frame_to_rgb
            bgr = cv2.cvtColor(frame_to_rgb(frame), cv2.COLOR_RGB2BGR)
        lm, w = self._pose_sync(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ms)
        return bgr, lm, w

    def _render_sync(self, bgr, lm, hud, ghost, card, extra=None):
        """Thread: compose the overlay (or the card) and encode it. DANCE_MIRROR_VIEW=1 (default) shows the caller a mirror image
        with the ghost mirrored to match; the scoring landmarks are never flipped."""
        extra = dict(extra or {})
        if self.mirror_view:
            from team.dance.render import mirror_view
            from team.dance.features import mirror_landmarks
            bgr, lm, ghost = mirror_view(bgr, lm, ghost)
            if extra.get('squad_lms'): extra['squad_lms'] = {pid: (mirror_landmarks(l)[0] if l is not None else None, c) for pid, (l, c) in extra['squad_lms'].items()}
            if extra.get('players'): extra['players'] = [(mirror_landmarks(l)[0] if l is not None else None, c) for l, c in extra['players']]
        if card is not None:
            kw = {k: card[k] for k in CARD_KW if k in card}
            if kw.get('photo') is not None:                       # the finish pose, shown the way the caller saw themselves
                ph = cv2.flip(kw['photo'], 1) if self.mirror_view else kw['photo']; kw['photo'] = to_canvas(ph, self.size)[0]
            img = draw_card(to_canvas(bgr, self.size)[0], card.get('result'), card.get('key', ''), extra_lines=card.get('lines', ()), code=card.get('code'),
                            title=card.get('title'), **kw)
        elif hud is not None:
            img = compose(bgr, lm, hud, ghost=ghost, show_live=True, size=self.size, squad_lms=extra.get('squad_lms'))
        else:
            img = compose_idle(bgr, lm, extra.get('title', self.idle_title), extra.get('lines', self.idle_lines), size=self.size, history=self.history, best=self.best,
                               recap=extra.get('recap'), caption=extra.get('caption'), footer=extra.get('footer'), fxq=self.fx, t=extra.get('t', 0.0),
                               gesture=extra.get('gesture'), paused=extra.get('paused', False), safety=extra.get('safety', False), players=extra.get('players'),
                               hints=extra.get('hints', False), setup=extra.get('setup'), speaking=extra.get('speaking', False), state=extra.get('state'),
                               avatar_state=extra.get('avatar_state', 'idle'), countdown=extra.get('countdown'), menu=extra.get('menu', False),
                               compare=extra.get('compare'), brk=extra.get('brk'), board=extra.get('board'))
        return self._encode_sync(img)

    async def prepare(self, frame, ms):
        if self.offline_pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if isinstance(frame, np.ndarray) else None
            lm, w = self.offline_pose(rgb, ms); return frame, lm, w
        return await asyncio.get_running_loop().run_in_executor(self.exec, self._prepare_sync, frame, ms)

    async def render_push(self, bgr, lm, hud, ghost=None, card=None, extra=None):
        vf = await asyncio.get_running_loop().run_in_executor(self.exec, self._render_sync, bgr, lm, hud, ghost, card, extra)
        if vf is not None: self.source.capture_frame(vf)
        self.frames += 1

    # ---- pose: several people, one primary ----
    def _close_pose_sync(self):
        """Pose thread: close the landmarkers ourselves. Left to the garbage collector, MediaPipe's __del__ -> close() can run inside
        one of its own ThreadPoolExecutor.submit calls (which holds the global executor lock) and deadlock the whole worker: video,
        voice and the job all froze on 2026-10-12 after the budget guard dropped the full model. close() is idempotent, so the later
        __del__ does nothing."""
        for m in (self.mpose, self.pose):
            try:
                if m is not None: m.close()
            except Exception as e:
                print(f'[dance] pose close: {e!r}', flush=True)
        self.mpose = None; self.pose = None

    def _pose_sync(self, rgb, ms):
        if self.pose_reload:                                 # the budget guard asked for another model: close the old one here, on this thread
            self.pose_reload = False; self._close_pose_sync()
        if self.mpose is None and self.pose is None:
            try:
                from team.dance.multipose import MultiPose
                self.mpose = MultiPose(self.pose_variant, video=True, max_poses=int(os.environ.get('DANCE_MAX_POSES', '4') or 4))
            except Exception as e:
                print(f'[dance] multi-person pose unavailable ({e!r}); single person', flush=True)
                from rtcoach.pose import Pose
                self.pose = Pose(self.pose_variant, video=True)
        s = min(rgb.shape[:2])
        if s > self.pose_px:
            f = self.pose_px / s; rgb = cv2.resize(rgb, (max(1, round(rgb.shape[1] * f)), max(1, round(rgb.shape[0] * f))), interpolation=cv2.INTER_AREA)
        t = time.perf_counter()
        if self.mpose is not None: poses = self.mpose(np.ascontiguousarray(rgb), ms)
        else:
            lm, w = self.pose(np.ascontiguousarray(rgb), ms); poses = [(lm, w)] if lm is not None else []
        self.pose_ms.append((time.perf_counter() - t) * 1000)
        self.n_people = len(poses)
        self.tracker.squad = self.call_mode == 'squad' or self.mode in SQUAD_MODES or self.mode == 'class'
        self.tracker.activity = self.state in ('ACTIVITY', 'REVIEW', 'PAUSED')            # mid-routine: walk-ins never take the lock
        self.tracker.update(self.now(), poses, frame=rgb)                                 # rgb: the pose input (colours re-identify people)
        self.players = [(p.pid, p.lm, p.world) for p in self.tracker.stable_players()]
        return self.tracker.solo_pose()

    async def detect(self, rgb, ms):
        if self.offline_pose is not None: return self.offline_pose(rgb, ms)
        return await asyncio.get_running_loop().run_in_executor(self.exec, self._pose_sync, rgb, ms)

    def newest_frame(self):
        """(raw frame, t_mono) of the newest caller frame (rtc.VideoFrame live, BGR ndarray in the simulator), or (None, None).
        No pixel work here: the loop must stay light (audio underruns otherwise)."""
        slot = getattr(self.job, 'latest', None)
        if slot is None or slot[4] == self.last_seq: return None, None
        frame, ts_us, t_mono, t_wall, seq = slot; self.last_seq = seq
        return frame, t_mono

    # ---- the call ----
    async def run(self):
        job = self.job
        while getattr(job, 'bound', None) is None or getattr(job, 'latest', None) is None:
            if getattr(job, 'closed', False): return
            await self.sleep(0.05)
        self.t0 = self.now()
        first = job.latest[0]
        try:
            sw, sh = (first.shape[1], first.shape[0]) if isinstance(first, np.ndarray) else (int(first.width), int(first.height))
            self.size = canvas_size(sw, sh)
            mode = (os.environ.get('DANCE_CANVAS') or 'auto').strip().lower()
            chert = str(getattr(job, 'bound', '') or '').startswith('chert-bridge')
            if mode == 'portrait' or (mode == 'auto' and chert): self.size = (720, 1280)      # FaceTime on a portrait iPhone: fill the screen, nothing cropped off
            elif mode == 'landscape': self.size = (1280, 720)
            print(f'[dance] caller video {sw}x{sh} ({"FaceTime via Chert" if chert else "browser"}) -> canvas {self.size[0]}x{self.size[1]}', flush=True)
        except Exception as e:
            print(f'[dance] could not read the first frame size ({e!r}); portrait canvas', flush=True)
        await self.publish()
        v = getattr(job, 'voice', None); model = getattr(v, 'model', None) if v is not None else None
        for _ in range(100):                                   # give the Gemini session up to 5 s to come up, so the first line is spoken, not dropped
            if model is None or getattr(model, 'started', False) or getattr(job, 'closed', False): break
            await self.sleep(0.05)
        await self.register_tools(model)
        self.task(self.vad_listener())
        if self.now is time.monotonic and model is not None: await self.sleep(1.5)     # update_tools reconnects Gemini: let it settle before the hello
        self.check_single_agent()
        self.listen_buttons()
        if self.debug_mp4:
            os.makedirs(os.path.dirname(os.path.abspath(self.debug_mp4)) or '.', exist_ok=True)
            if self.now is time.monotonic:                       # live: one file per call (two calls at once were overwriting each other)
                base, ext = os.path.splitext(self.debug_mp4); self.debug_mp4 = f"{base}_{time.strftime('%H%M%S')}{ext or '.mp4'}"
            self.writer = cv2.VideoWriter(self.debug_mp4, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
            print(f'[dance] DEV RECORDING ON: what the caller sees goes to {self.debug_mp4} (DANCE_DEBUG_MP4; never enable for real callers)', flush=True)
        STATE['session'] = self
        print(f"[dance] session: key {self.key['name']} ({self.key['bpm']:.0f} bpm, {len(self.key['phrases'])} phrases), mode {self.mode}, tempo {self.tempo}, rtt {self.rtt_s} s; "
              f"difficulty {self.difficulty}; handle {'set' if self.handle else 'unset'}; room '{self.room_name}' -> "
              f"{'DEMO (no chat gates)' if not self.lobby_enabled else 'full flow'}, {'arms only' if self.upper_override else 'whole body'}", flush=True)
        bank = getattr(self.job, 'bank', None)
        if bank is not None: dance_audio.register_sfx(bank)            # non-verbal sounds only: Gemini is the one voice
        self._apply_state_rules('EXPLORE')
        if self.code and self.lobby_enabled: await self.handle_code(self.code)          # a code given up front (DANCE_CODE): offer it
        elif self.code and self.mode == 'battle': await self.load_challenge(self.code)
        completed = True
        try:
            await self.play_loop()
            if self.safety_stop: await self.hold_safety(12.0)
        except Exception as e:
            completed = False; self.telemetry.error('session', e); print(f'[dance] session error: {e!r}', flush=True)
            import traceback; traceback.print_exc()
        finally:
            STATE['session'] = None
            if self.writer is not None: self.writer.release()
            if self.source is not None and hasattr(self.source, 'aclose'):
                try: await asyncio.wait_for(self.source.aclose(), 1.0)
                except Exception as e: print(f'[dance] video source close: {e!r}', flush=True)
            try: self.exec.submit(self._close_pose_sync)        # after any frame still in flight; never left to the garbage collector
            except RuntimeError: pass
            self.exec.shutdown(wait=False)
            if self.pose_ms: print(f'[dance] pose {np.median(self.pose_ms):.0f} ms median over {len(self.pose_ms)} frames; {self.frames} frames composed', flush=True)
            self.telemetry.mode = self.mode
            self.telemetry.extra.update(time_to_start_s=self.time_to_start, call_mode=self.call_mode, max_dancers=self.max_players, difficulty=self.difficulty,
                                        persona=self.persona, safety_stop=self.safety_stop, flow_metrics=dict(self.metrics))
            summary = self.telemetry.close(completed=completed)
            try: await asyncio.wait_for(self.backend.call('POST', '/events', dict(type='session_summary', t_wall=round(time.time(), 3), **summary)), 4.0)
            except Exception: pass
            try:
                log_path = os.environ.get('DANCE_SESSION_LOG') or os.path.join(ROOT, 'data', 'tmp', 'dance_session_log.json')
                os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(dict(key=self.key['name'], mode=self.mode, name=self.name, rounds=self.rounds, log=self.log, summary=summary), f, indent=1, default=str)
            except Exception as e: print(f'[dance] session log not written: {e!r}', flush=True)

    async def vad_listener(self):
        """The local voice-activity detector (livekit-local-inference, on-device, free) on the caller's own audio track. That track is
        separate from Gemini's input (which the echo guard mutes while Brodie's audio plays), so we know the caller is talking even
        during the music or when the transcript is unreadable. Speech held 250 ms during a phrase opens a 4 s barge-in window."""
        if (os.environ.get('DANCE_BARGE_IN', '1') or '1').strip() == '0': return
        room = getattr(self.job, 'room', None)
        if room is None or not hasattr(room, 'remote_participants'): return
        try:
            from livekit import rtc
            from livekit.local_inference import VAD, VAD_WINDOW_SAMPLES as N
        except Exception as e:
            print(f'[dance] local VAD unavailable ({e!r})', flush=True); return
        track = None
        for _ in range(200):
            if getattr(self.job, 'closed', False): return
            p = room.remote_participants.get(getattr(self.job, 'bound', None) or '')
            if p is not None:
                for pub in p.track_publications.values():
                    if getattr(pub, 'kind', None) == rtc.TrackKind.KIND_AUDIO and getattr(pub, 'track', None) is not None: track = pub.track; break
            if track is not None: break
            await asyncio.sleep(0.25)
        if track is None: print('[dance] local VAD: no caller audio track', flush=True); return
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1); vad = VAD(); buf = np.zeros(0, np.int16)
        print('[dance] local VAD listening to the caller', flush=True)
        try:
            async for ev in stream:
                if getattr(self.job, 'closed', False) or self.stop: break
                buf = np.concatenate([buf, np.frombuffer(bytes(ev.frame.data), dtype=np.int16)])
                while len(buf) >= N:
                    prob = float(vad.predict(np.ascontiguousarray(buf[:N]))); buf = buf[N:]
                    self.on_vad(prob, self.now())
        except Exception as e:
            print(f'[dance] local VAD stopped: {e!r}', flush=True)
        finally:
            try: await stream.aclose()
            except Exception: pass

    def on_vad(self, prob, now):
        """One 32 ms window of the caller's audio: P(speech). During the count-in (clicks playing, the caller usually quiet) it learns
        how much of the speaker's own sound leaks back as "speech" (echo), and during the music speech must clear that by a margin."""
        eng = self.cur_engine; st = getattr(eng, 'state', None)
        if self.state == 'ACTIVITY' and st == 'countin': self.vad_echo = 0.9 * self.vad_echo + 0.1 * prob
        thr = max(0.6, min(0.9, self.vad_echo + 0.25)) if self.state == 'ACTIVITY' else 0.6
        if prob > thr:
            self.vad_run += 1; self.vad_last_t = now
        elif prob < 0.35:
            if self.vad_run >= 12: self.vad_utt_end = now                        # an utterance of ~0.4 s or more just ended
            self.vad_run = 0
        if self.vad_run * 0.032 >= 0.25: self.barge_in(now)

    def barge_in(self, now):
        """The caller is talking over the music: keep their audio flowing to Gemini for 4 s so "stop", "pause", "slower" get heard."""
        eng = self.cur_engine
        if self.state != 'ACTIVITY' or eng is None or getattr(eng, 'state', None) not in ('show', 'countin', 'attempt'): return
        floor = getattr(getattr(self.job, 'voice', None), 'floor', None)
        if floor is None: return
        if eng is not self.barge_eng: self.barge_eng = eng; self.barge_s = 0.0
        span = max(4.0, float(getattr(eng, 'attempt_s', 0) or 0) + float(getattr(eng, 'show_s', 0) or 0))
        if self.barge_s > 0.2 * span: return                     # singing along / counting with it: the gesture still works, the mic stays shut
        if now - self.last_barge_t < 6.0 and time.monotonic() < getattr(floor, 'force_open_until', 0.0): return
        self.barge_s += 2.5
        floor.force_open_until = time.monotonic() + 2.5
        self._install_music_duck(); self.music_duck_until = time.monotonic() + 2.5   # the music drops about 18 dB while they talk
        if now - self.last_barge_t > 5.0:
            self.last_barge_t = now; self.metrics['barge_ins'] += 1; self.telemetry.event('barge_in', state=self.state)

    def _install_music_duck(self):
        """Duck the playing clip (the dance track) while music_duck_until: an instance patch on the cue player's clip path. The beat
        clock is the engine's, so timing is unaffected; only the loudness drops."""
        p = getattr(self.job, 'player', None)
        if p is None or getattr(p, '_dance_duck', False) or not hasattr(p, '_under'): return
        orig = p._under; sess = self
        def _under(chunk):
            out = orig(chunk)
            try:
                if time.monotonic() < sess.music_duck_until:
                    a = np.asarray(out); return (a.astype(np.float32) * 0.125).astype(a.dtype)
            except Exception: pass
            return out
        p._under = _under; p._dance_duck = True

    def check_unclear(self, now):
        """Speech came after a question but no transcript followed (too quiet, garbled): ask once instead of guessing."""
        end = self.vad_utt_end
        if end is None or self.q_open is None or now - end < 3.5: return
        self.vad_utt_end = None
        if self.last_caller_text_t >= end - 1.0 or now - self.unclear_t < 15.0: return
        self.ask_unclear()

    def ask_unclear(self):
        self.unclear_t = self.now(); self.metrics['unclear_asks'] += 1; self.telemetry.event('unclear', q=self.q_open)
        line = "Sorry, I didn't catch that. Was that a yes? You can also raise both hands." if self.q_open in ('offer', 'space', 'confirm') else \
               "Sorry, I didn't catch that. Say again, next, or slower."
        self.say_line(line, key='unclear', ttl=10)

    def listen_buttons(self):
        """The Chrome test page's Pause / Repeat / Slower / Stop buttons arrive as LiveKit data messages {"type": "dance_cmd", "cmd": ...}
        and go through the same state-gated command path as voice. (FaceTime callers use their voice and gestures.)"""
        room = getattr(self.job, 'room', None)
        if room is None or not hasattr(room, 'on'): return
        def on_data(pkt, *a):
            try:
                raw = getattr(pkt, 'data', pkt)
                msg = json.loads(bytes(raw).decode('utf-8'))
                if msg.get('type') == 'dance_cmd': self.button_cmd(str(msg.get('cmd', '')))
            except Exception as e:
                print(f'[dance] data message ignored: {e!r}', flush=True)
        try: room.on('data_received', on_data)
        except Exception as e: print(f'[dance] no data channel ({e!r})', flush=True)

    def button_cmd(self, cmd):
        cmd = {'repeat': 'again'}.get(cmd, cmd)
        if cmd not in ('pause', 'again', 'slower', 'stop', 'resume'): return False
        self.telemetry.event('button', cmd=cmd, state=self.state)
        if cmd == 'stop' and self.state in ('OFFER', 'PREPARE', 'CONFIRM'): self.cancel_req = True; return True
        if cmd == 'pause' and self.paused: cmd = 'resume'
        return self._cmd(cmd, 'button')

    def check_single_agent(self):
        """Exactly one coach: flag any other agent participant in the room (a stale worker would be a second voice)."""
        room = getattr(self.job, 'room', None)
        if room is None: return
        try:
            from livekit import rtc
            others = [p.identity for p in room.remote_participants.values() if getattr(p, 'kind', None) == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT]
            if others:
                self.telemetry.event('second_agent', identities=others); print(f'[dance] WARNING: another agent is in this room: {others} (two voices). Stop the stale worker.', flush=True)
        except Exception as e:
            print(f'[dance] agent check skipped: {e!r}', flush=True)

    async def register_tools(self, model):
        if (os.environ.get('DANCE_TOOLS', '1') or '1').strip() == '0' or model is None: return
        agent = getattr(model, 'agent', None)
        if agent is None: return
        try:
            from team.dance import tools as dance_tools
            n = await dance_tools.register(self, agent); self.tools_ok = n > 0
            self.telemetry.event('tools_registered', n=n); print(f'[dance] {n} Gemini tools registered', flush=True)
        except Exception as e:
            self.telemetry.event('tools_unavailable', error=repr(e)); print(f'[dance] tools unavailable ({e!r}); voice commands by keyword only', flush=True)

    async def play_loop(self):
        """Chat until the caller says yes to an offer; camera setup; a ready check; the routine; back to chat. Every stage runs inside
        an error boundary: a failure is logged, Brodie apologises, and the session returns to chat instead of ending."""
        first = True; after = False
        while not self.stop:
            req = await self.guard(self.explore(first=first, after=after), 'explore'); first = False; after = False
            if self.stop: break
            if not req: continue
            if not await self.guard(self._start_activity(req), 'activity'): self.proposal = None; continue
            ok = await self.guard(self.prepare_and_confirm('start'), 'setup')
            if not ok or self.stop: self.proposal = None; continue
            self.transition('REVIEW', 'lesson')                     # from here "skip" / "stop" work; nothing plays until the first move
            self.cmd_floor_t = self.now(); self.pending_cmd = None
            self.call_mode = 'squad' if (self.mode in SQUAD_MODES or (self.mode == 'class' and self.tracker.stable_count() >= 2)) else 'solo'
            if self.call_mode == 'squad':
                try: self.start_squad()
                except Exception as e:
                    print(f'[dance] squad start failed ({e!r}); solo', flush=True); self.call_mode = 'solo'
                    self.mode = 'battle' if self.mode == 'squad_battle' else 'learn'
            run = dict(battle=self.battle, telephone=self.telephone, daily=self.daily, squad_battle=self.squad_battle, class_record=self.class_mode,
                       **{'class': self.class_lead}).get(self.mode)
            if run is None: run = self.learn
            if self.time_to_start is None: self.time_to_start = round(self.now() - self.t0, 2)
            self.telemetry.event('mode_started', mode=self.mode, call_mode=self.call_mode)
            await self.guard(run(), f'mode {self.mode}')
            self.played.append(self.mode)
            self.reset_activity_state(); after = True
            if self.end_after_activity: self.bye_req = True             # the call cap was reached during the activity                    # every feature ends back on the menu ("What's next?")
            if self.safety_stop: break
            if not self.lobby_enabled: break                              # test mode (DANCE_LOBBY=0): one routine, then end

    def reset_activity_state(self):
        """Between features: nothing from the last activity leaks into the next (a squad roster, a battle HUD, a latched gesture, a
        stale "next"). Kept: the caller's name, profile, coach style, arms-only choice, today's daily, what they played."""
        if self.cur_engine is not None and getattr(self.cur_engine, 'state', 'done') != 'done': self.cur_engine.abort(self.now())
        self.cur_engine = None; self.engine_holder = None
        self.proposal = None; self.code = None; self.challenge = None; self.report_req = False; self.solo_req = False; self.mode_choice = None
        self.call_mode = 'solo'; self.squad_players = []; self.colours = {}; self.squad_since = None
        self.battle_hud = None; self.compare_ui = None; self.countdown_ui = None; self.setup_ui = None; self.recap = None
        self.pending_cmd = None; self.pause_req = False; self.resume_req = False; self.stop_routine = False; self.cancel_req = False
        self.paused = False; self.in_warmup = False; self.in_demo = False; self.practice = False; self.stage = None; self.step = None
        self.slow_mode = False; self.last_tip_beat = {}; self.combo_run = {}; self.ready_hits = {}; self.want_ready = False
        for g in self.gest.values(): g.reset()
        self.affirm_t = -1.0; self.deny_t = -1.0; self.hold_until = -1.0; self.q_open = None
        self.menu_req = False; self.cmd_floor_t = self.now()

    async def _start_activity(self, req):
        self.setup_activity(req)                       # a squad with one dancer is settled at the ready check (a friend may still walk in)
        return True

    async def guard(self, coro, stage):
        """Run one stage; any exception is logged and turned into a graceful return to chat (the call never dies here)."""
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.telemetry.error(stage, e); print(f'[dance] {stage} failed: {e!r}', flush=True)
            import traceback; traceback.print_exc()
            try:
                if self.cur_engine is not None: self.cur_engine.abort(self.now()); self.cur_engine = None
                self.cut_audio(); self.setup_ui = None; self.paused = False; self.want_ready = False
                self.transition('CHOOSE', 'error'); self.say_line("Sorry, I glitched for a second. Pick something and we'll go again.", key='error', urgent=True)
            except Exception as e2:
                print(f'[dance] recovery failed: {e2!r}', flush=True)
            return None

    def _next_style(self):
        """Brodie's own suggestion: what the caller said they like, else the next style they haven't passed on."""
        last = (self.proposal or {}).get('style')
        if self.style and self.style not in self.declined and self.style != 'any' and self.offers_made <= 1: return self.style
        for c in ('lh', 'po', 'kpop', 'fitness', 'ho', 'wa'):
            if c not in self.declined and c != last: return c
        return 'lh'

    def _open_menu(self, header):
        """The six cards with their badges: FOR YOU (a returning caller's pick), REMATCH (the mode just played), PLAYED, DONE TODAY."""
        badges = {m: 'PLAYED' for m in self.played}
        if self.played: badges[self.played[-1]] = 'REMATCH'
        if self.daily_done: badges['daily'] = 'DONE TODAY'
        rec = getattr(self, 'recommended_mode', None)
        if rec and rec not in badges: badges[rec] = 'FOR YOU'
        self.menu_ui = dict(header=header, lit=None, cycling=False, badges=badges, picked=None)
        self.menu_open = True

    async def explore(self, first=False, after=False):
        """CHOOSE: the six mode cards (modes.py) and one question, "what are we doing?". Brodie never suggests or starts anything by
        himself here. The caller names a mode (or says a code); if nothing usable comes for CYCLE_START_S, or the room is noisy, a
        highlight steps through the cards and both hands up picks the lit one (the card lit when the hold *began*, so a slow hold
        never picks the next one). After a feature the score card stays CARD_HOLD_S (hands up or "rematch" there = the same again),
        then the cards come back with "What's next?". A declined offer returns here, cards only, no line."""
        if self.state != 'OFFER': self.transition('CHOOSE')
        self.cancel_req = False
        if not self.lobby_enabled:
            self.proposal = dict(mode=self.mode, rematch=False, source='test', code=self.code); return self.proposal
        if self.final_card is None: self.idle_title = 'Brodie'
        danced = after and self.final_card is not None
        self._open_menu("WHAT'S NEXT?" if after else 'WHAT ARE WE DOING?')
        if first: self.say_line(mode_cards.HELLO, key='hello', ttl=20); self.telemetry.event('menu')
        t_card = self.now() if danced else None
        t_open = self.now(); cyc = None; t_choose = self.now()
        while not self.stop:
            if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; break
            if self.bye_req: await self.say_bye(); return None
            if getattr(self, 'switch_req', None) and self.state != 'OFFER':
                m = self.switch_req; self.switch_req = None; t_card = None; self.final_card = None
                self.request_activity(mode=m, source='voice')
            if self.state == 'OFFER':
                try: r = await self.offer_loop()
                finally: self.want_ready = False; self.q_open = None
                if r == 'yes':
                    self.telemetry.event('mode_chosen', mode=(self.proposal or {}).get('mode'), how=(self.proposal or {}).get('source'),
                                         s=round(self.now() - t_choose, 1))
                    self.menu_ui = None; self.menu_open = False; return self.proposal
                self.proposal = None
                if self.bye_req: continue
                self.transition('CHOOSE', r); self._open_menu(self.menu_ui['header'] if self.menu_ui else 'WHAT ARE WE DOING?')
                t_open = self.now(); cyc = None; t_card = None; self.final_card = None
                continue
            await self.frame_step(None); await self.sleep(1.0 / self.fps)
            now = self.now()
            if t_card is not None:                                      # the score card first; hands up on it = rematch
                # ...but only a fresh hold: the finish pose is often arms up, so the hands must come down once after the card shows
                if not self.want_ready and now - t_card > 1.5 and self._hands_down(): self._arm_gesture()
                if self.affirm_t > t_card and self.last_mode:
                    self.want_ready = False; t_card = None; self.final_card = None
                    self.request_activity(mode=self.last_mode, rematch=True, source='card'); continue
                wrap_busy = any(q['key'] in ('wrap', 'pose') for q in self.say_q) or self.line_end.get('wrap', -1.0) < self.line_start.get('wrap', -2.0)
                if now - t_card < CARD_HOLD_S or (wrap_busy and now - t_card < CARD_HOLD_S + 15.0): continue
                t_card = None; self.final_card = None; self.want_ready = False; t_open = now
                self.say_line(mode_cards.NEXT_LINE, key='menu', ttl=12)
                continue
            said = max(self.line_end.get('hello', -1.0), self.line_end.get('menu', -1.0), t_open)
            noisy = sum(1 for x in self.noise_utts if now - x < 15.0) >= 2
            quiet = now - max(said, self.last_caller_text_t)
            if cyc is None and (quiet > CYCLE_START_S or noisy) and not self.say_q and self._audio_idle() and not self._model_speaking():
                start = mode_cards.index_of(getattr(self, 'recommended_mode', None) or '') or 0
                cyc = dict(t0=now, start=start, frozen=None); self.noise_utts = []
                if self._hands_down(): self._arm_gesture()
                self.menu_ui = dict(self.menu_ui or {}, cycling=True, lit=start)
                self.telemetry.event('card_cycle', noisy=noisy)
                self.say_line(mode_cards.NOISY_LINE if noisy else mode_cards.CYCLE_LINE, key='cycle', ttl=10)
            if cyc is not None:
                if not self.want_ready and self._hands_down(): self._arm_gesture()     # arms already up (a stretch) never pick a card
                g = self.gesture_ui or {}
                holding = g.get('kind') == 'ready' and float(g.get('p') or 0.0) > 0.15
                if holding and cyc['frozen'] is None:
                    cyc['frozen'] = self.menu_ui['lit'] if self.menu_ui.get('lit') is not None else cyc['start']
                if not holding and cyc['frozen'] is not None and self.affirm_t <= cyc['t0']:      # the hold was dropped: go on from there
                    cyc['start'] = cyc['frozen']; cyc['t0'] = now; cyc['frozen'] = None
                lit = cyc['frozen'] if cyc['frozen'] is not None else (cyc['start'] + int((now - cyc['t0']) / CYCLE_STEP_S)) % len(mode_cards.CARDS)
                self.menu_ui['lit'] = lit
                if self.affirm_t > cyc['t0']:
                    pick = mode_cards.CARDS[lit]['mode']; self.want_ready = False
                    self.menu_ui = dict(self.menu_ui, picked=pick, lit=None, cycling=False)
                    self.metrics['card_picks'] = self.metrics.get('card_picks', 0) + 1
                    cyc = None; self.request_activity(mode=pick, source='card')
        return None

    def _hands_down(self):
        """No dancer in view has both hands up right now (so a hold that starts later is a deliberate answer)."""
        from team.dance.gestures import classify as gesture_of
        lms = [l for _, l, _ in self.players] if self.call_mode == 'squad' and self.players else [self.cur_lm]
        return all(gesture_of(l) != 'ready' for l in lms)

    async def say_bye(self):
        """The caller is done: a goodbye (and the wrap card stays up if they danced), then the call ends."""
        self.bye_req = False; self.menu_ui = None
        t = self.now()
        line = "Thanks for dancing with me! Come back any time, and bring your friends." if self.played else "No worries. Call me back whenever you want to dance!"
        self.say_line(line, key='bye', ttl=10, urgent=True)
        await self.wait_spoken('bye', t, 8.0, min_s=2.0)
        self.stop = True

    async def offer_loop(self):
        """OFFER: one concrete suggestion waits for a clear yes. Gemini often asks the question itself; if it hasn't within a few
        quiet seconds, we ask (and ask again, at most 3 times, if nobody heard it). A yes counts only after the question was asked;
        "no" / "not now" goes back to chat; questions keep us here; 60 quiet seconds go back to chat."""
        self.affirm_t = -1.0; self.deny_t = -1.0; self.cancel_req = False; asked_t = -1.0; asks = 0; seen_prop = self.proposal_t
        while not self.stop:
            if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; return 'timeout'
            await self.frame_step(None); await self.sleep(1.0 / self.fps)
            now = self.now()
            if self.proposal_t != seen_prop: seen_prop = self.proposal_t; asked_t = -1.0; asks = 0   # a different offer (a change of mind)
            if self.cancel_req: self.cancel_req = False; return 'cancel'
            st_line = self.line_start.get('offer', -1.0)                  # when our own question started playing
            if st_line >= self.proposal_t and (self.offer_asked_t < self.proposal_t or st_line < self.offer_asked_t): self.offer_asked_t = st_line
            q_t = self.offer_asked_t if self.offer_asked_t >= self.proposal_t else -1.0
            if q_t >= 0 and not self.want_ready: self._arm_gesture()          # asked: both hands up answers it too (the screen says so)
            self.q_open = 'offer' if q_t >= 0 else None
            if q_t >= 0 and self.affirm_t > q_t - 0.5:
                self.telemetry.event('offer_accepted', offer=self.offer_label()); return 'yes'
            src = (self.proposal or {}).get('source')
            mine = src == 'brodie'                                          # Brodie's own suggestion: he asks right away
            card = src == 'card'                                            # picked on the cards: confirm it right away
            wait = (1.0 if (mine or card) else 5.0) if asks == 0 else 15.0
            if q_t < 0 and asks < 3 and now >= self.hold_until and not any(q['key'] == 'offer' for q in self.say_q) and \
                    now - max(asked_t, self.proposal_t, self.last_caller_text_t) > wait:
                asks += 1; asked_t = now
                if asks > 1: self.metrics['reasks'] += 1
                if card and asks == 1:
                    c = mode_cards.by_mode((self.proposal or {}).get('mode')) or {}
                    line = (f"{c.get('title', 'Got it')}! Want to start {self.offer_label()}? Say yes, or raise both hands again. "
                            "Quick camera check first.")
                elif mine:
                    line = self.pick('suggest', [f"Let's get you moving! How about {self.offer_label()}? Want to start now?",
                                                 f"I've got a good one for you: {self.offer_label()}. Shall we start?",
                                                 f"Here's an idea: {self.offer_label()}. Up for it?"])
                else:
                    line = self.pick('offer', [f"Want to start {self.offer_label()} now? We'll do a quick camera check first.",
                                               f"Shall we jump into {self.offer_label()}? First a quick camera check."])
                self.say_line(line, key='offer', ttl=12)
            if now - max(self.proposal_t, self.last_caller_text_t, self.coach_q_t, asked_t) > 60.0: return 'timeout'
        return 'timeout'

    def setup_activity(self, req):
        mode = req.get('mode') or 'learn'
        if req.get('rematch') and self.final_card is not None: mode = self.last_mode or mode
        self.reset_run(); self.pending_cmd = None; self.cmd_floor_t = self.now(); self.pause_req = False; self.stop_routine = False
        if not (req.get('rematch') and self.now() - self.space_ok_t < 600.0): self.space_ok = False   # every new activity asks about space again
        self.code = req.get('code') or (self.code if not self.lobby_enabled else None); self.practice_run = bool(req.get('practice'))
        if mode != 'battle' or not self.code: self.challenge = None
        if mode == 'daily': self.key = choreo.daily()
        elif self.queued_key is not None: self.key = self.queued_key; self.queued_key = None
        elif req.get('rematch') and getattr(self, 'last_key', None) is not None: self.key = self.last_key
        elif mode not in ('telephone', 'class_record'):
            style = req.get('style') or self.style or ('wa' if self.upper_body else 'lh')
            try: self.key = self.make_routine(style)
            except Exception as e:                                       # never lose the yes to a routine the library can't build
                print(f'[dance] routine for {style!r} failed ({e!r}); hip-hop instead', flush=True)
                self.key = choreo.generate('hiphop', self.difficulty)
        self.mode = mode; self.last_mode = mode; self.last_key = self.key
        self.key['title'] = key_title(self.key); self.idle_title = self.key['title']; self.final_card = None
        self.context(f"They said yes to {MODE_LABEL.get(mode, mode)}: '{self.key['title']}' at {self.difficulty}. The camera check runs next; nothing is playing yet.")
        self.telemetry.event('activity', mode=mode, key=self.key['name'], difficulty=self.difficulty)

    async def prepare_and_confirm(self, why='start'):
        """Camera setup, then the ready check. True to start; False goes back to chat (always with a spoken line)."""
        self.confirm_first_t = None; bounces = 0; self.setup_started_t = self.now()     # SETUP_TOTAL_CAP_S counts across bounces
        while not self.stop and not self.safety_stop:
            if not await self.setup_camera(why): return False
            r = await self.ready_check()
            if r == 'go': return True
            if r == 'explore': return False
            bounces += 1; why = 'setup again'; self.metrics['setup_bounces'] += 1
            if bounces == 3: self.say_line("Let's get the camera steady first. Find a spot where I can see you, and stay there.", key='setup')
            if bounces >= 6:
                self.say_line("Let's try again in a bit. Tell me when you're set up.", key='cancel'); return False
        return False

    def _arm_gesture(self):
        """Hands up means yes from now on: a hold that started earlier fired into nothing, so the detectors start fresh."""
        if not self.want_ready:
            for g in self.gest.values(): g.reset()
            self.ready_hits = {}
        self.want_ready = True

    async def wait_answer(self, seconds, gesture=False, tracker=None, key=None):
        """'yes' / 'no' / 'setup' (the framing broke while waiting) / None. With `key`, only answers after that line was actually
        spoken count (a "yes" to something else never starts anything); if it was never spoken, answers count from the start."""
        t0 = self.now(); bad_since = None
        if gesture: self._arm_gesture()
        try:
            while self.now() - t0 < seconds + (6.0 if key else 0.0) and not self.stop:
                if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; break
                after = t0 if not key else self.answer_after(key, t0)
                if self.cancel_req: return 'no'
                if after is not None:
                    if self.deny_t > after and self.now() >= self.hold_until: return 'no'
                    if self.affirm_t > after: return 'yes'
                await self.frame_step(None); await self.sleep(1.0 / self.fps)
                if tracker is not None:
                    res = tracker.update(self.now(), self.cur_lm, self.luma, self.frame_aspect); self.setup_ui = res
                    bad_since = None if res['issue'] is None else (bad_since or self.now())
                    if bad_since is not None and self.now() - bad_since > 3.0: return 'setup'
        finally:
            self.want_ready = False
        return None

    async def setup_camera(self, why='start'):
        try: return await self._setup_camera(why)
        finally: self.want_ready = False; self.q_open = None   # hands up means yes only while a question is open

    def _space_answer(self, sq, now):
        """'yes' / 'no' / None for the open space question: answers count from just after it started playing (or from when it was
        queued, if it never played), so a "yes" to something earlier never counts, and a late one always does."""
        if sq['after'] is None:
            sq['after'] = self.answer_after('space', sq['t'])
            if sq['after'] is None: return None
        if self.affirm_t > sq['after'] and self.affirm_t > sq['seen']: sq['seen'] = self.affirm_t; return 'yes'
        if self.deny_t > sq['after'] and self.deny_t > sq['seen']: sq['seen'] = self.deny_t; return 'no'
        return None

    async def _setup_camera(self, why):
        """One stable problem at a time, said once; then wait for the camera to show the fix (an event, not a timer). The same problem
        is never repeated within 20 s, check-ins come after 15 quiet seconds (at most 3 per problem), and every exit is guaranteed:
        an arms-only offer after 20 s of no hips (or 90 s without progress), back to chat after 120 s without progress or 150 s in
        all. Once the framing holds we ask about space once and keep the question open: "yes" or both hands up answers it at any
        time, two gentle nudges if nothing comes, then Brodie trusts them and moves on. True when ready to go on; False otherwise."""
        from team.dance.setup import SetupTracker, lines_for, ORDER
        self.transition('PREPARE', why); self.cancel_req = False; self.hold_until = -1.0
        tr = SetupTracker(upper=self.upper_body or self.upper_override)
        t_enter = self.now(); spoken_issue = None; t_issue = self.now(); checkins = {}; last_said = {}
        best = -1; t_progress = self.now(); offered_upper = False; lost_since = None; lost_said = False
        no_hips_since = None; vague_since = None; note_issue = 'start'; note_t = -1e9
        sq = None                      # the open space question: asked once, answerable any time after (voice or both hands up)
        flips = []; relaxed = False    # issue changes in the last minute; relaxed: head and hips are in, a flip-flop is accepted
        rank = lambda i: len(ORDER) + 1 if i is None else ORDER.index(i)     # lower = further from done
        if not self.lobby_enabled: self.space_ok = True
        while not self.stop and not self.safety_stop:
            if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; return False
            await self.frame_step(None); await self.sleep(1.0 / self.fps)
            now = self.now()
            if self.cancel_req:
                self.cancel_req = False; self.setup_ui = None
                self.say_line("No problem. Tell me whenever you want to dance.", key='cancel'); return False
            if (self.upper_override or self.upper_body) and not tr.upper:
                tr = SetupTracker(upper=True); spoken_issue = None
                self.say_line("Got it, arms only. I'll just watch your upper body.", key='setup')
            if self.last_frame_t > 0 and now - self.last_frame_t > 5.0:
                self.setup_ui = dict(ok_now=False, ok_stable=False, issue='lost', checks={}, hold=0.0)
                lost_since = lost_since or now
                if not lost_said: lost_said = True; self.say_line("I've lost your video for a moment. I'll wait right here.", key='setup')
                if now - lost_since > 60.0:
                    self.setup_ui = None; self.say_line("Looks like your video stopped. Tell me when you're back.", key='cancel'); return False
                continue
            lost_since = None; lost_said = False
            res = tr.update(now, self.cur_lm, self.luma, self.frame_aspect)
            if relaxed and self.cur_lm is not None and ((res.get('metrics') or {}).get('hips_in') or tr.upper) and res['issue'] != 'no_person':
                res = dict(res, ok_stable=True, issue=None, hold=1.0)     # after a flip-flop break: head and hips in is good enough
            o = self.orient_state or {}
            if res['issue'] is None and (o.get('away') or abs(o.get('yaw') or 0.0) > 50):    # back to the camera, or turned well away
                res = dict(res, ok_stable=False, issue='not_facing')
            backlit = self.subject_luma is not None and self.luma is not None and self.luma > 110 and self.subject_luma < 0.55 * self.luma
            if res['issue'] is None and backlit and (self.cur_lm is None or float(np.mean(np.asarray(self.cur_lm)[:, 3])) < 0.6):
                res = dict(res, ok_stable=False, issue='dark')                # a window behind them: tracking suffers
            self.setup_ui = res
            if res['issue'] != note_issue and now - note_t > 4.0:          # Gemini's picture of the camera, only when it changes
                note_issue = res['issue']; note_t = now; self.context(self.camera_note_text())
            if sq is not None and not self.space_ok:
                self.setup_ui = dict(res, prompt='Say "yes" or raise both hands')
                ans = self._space_answer(sq, now)
                if ans == 'yes':
                    self.space_ok = True; self.space_ok_t = now; self.want_ready = False; self.space_answer_t = self.affirm_t
                    self.telemetry.event('space_confirmed', late=round(now - sq['t'], 1))
                elif ans == 'no' and now >= self.hold_until:
                    sq['last'] = now; sq['nudges'] = 0
                    self.say_line("Okay, clear a little space and tell me when it's done.", key='space_no')
            if rank(res['issue']) > best or res['ok_stable']:              # real progress (a better state than ever before)
                if rank(res['issue']) > best: best = rank(res['issue'])
                t_progress = now
            if res['ok_stable']:
                if spoken_issue is not None: spoken_issue = None
                if self.space_ok: self.setup_ui = None; return True
                if sq is None:
                    if now - self.last_caller_text_t > 3:
                        sq = dict(t=now, after=None, seen=-1e9, last=now, nudges=0); self._arm_gesture(); self.q_open = 'space'
                        self.telemetry.event('space_asked')
                        self.say_line(self.pick('space', ["Perfect, I can see all of you. Do you have about two big steps of clear space around you? Say yes, or raise both hands.",
                                                          "Looking good! Is there about two big steps of space around you? Say yes, or raise both hands."]), key='space')
                    continue
                if not self.want_ready: self._arm_gesture()
                if self.line_end.get('space', -1.0) < sq['t'] and now - sq['t'] < 20.0: continue      # the question is still being said
                last = max(sq['last'], self.line_end.get('space', -1.0), self.line_end.get('space_no', -1.0), self.line_end.get('space_nudge', -1.0), self.last_caller_text_t)
                if now >= self.hold_until and now - last > (8.0 if sq['nudges'] == 0 else 20.0):
                    if sq['nudges'] < 2:
                        sq['nudges'] += 1; sq['last'] = now; self.telemetry.event('space_nudge', n=sq['nudges']); self.metrics['nudges'] += 1
                        self.say_line("If you've got the space, just say yes, or raise both hands." if sq['nudges'] == 1 else
                                      "Take your time. Raise both hands when you're clear.", key='space_nudge')
                    else:                                                  # space can't be seen anyway: never stall on it
                        self.space_ok = True; self.space_ok_t = now; self.telemetry.event('space_assumed')
                        self.say_line("I'll trust you've got room. Let's go!", key='space_nudge')
                        self.setup_ui = None; return True
                continue
            total_t = now - getattr(self, 'setup_started_t', t_enter)
            if now - t_enter > SETUP_CAP_S or total_t > SETUP_TOTAL_CAP_S:
                self.setup_ui = None; self.say_line("Let's try the camera again in a bit. Tell me when you're set up and we'll go.", key='cancel'); return False
            issue = res['issue']
            if issue is not None and issue != (flips[-1][1] if flips else None): flips.append((now, issue))
            flips[:] = [f for f in flips if now - f[0] <= 60.0]
            alternating = len(flips) >= 4 and len({f[1] for f in flips[-4:]}) == 2       # A, B, A, B within a minute
            if alternating and not relaxed and self.cur_lm is not None and ((res.get('metrics') or {}).get('hips_in') or tr.upper):
                relaxed = True; self.metrics['flipflop_breaks'] += 1; self.telemetry.event('setup_flipflop', issues=[f[1] for f in flips[-4:]])
                self.say_line("Stay right there, that works.", key='setup')
                continue                                                   # the next frames see `relaxed` and go on to the space question
            if issue is None and res.get('top_raw'):                       # no single problem leads, but it isn't stable either
                vague_since = vague_since or now
                if now - vague_since > 5.0: issue = res['top_raw']
            else:
                vague_since = None
            hips = (res.get('metrics') or {}).get('hips_in')
            no_hips_since = (no_hips_since or now) if (issue == 'too_close' and not hips and not tr.upper) else None
            if not offered_upper and not tr.upper and ((no_hips_since is not None and now - no_hips_since > 20.0) or now - t_progress > 90):
                offered_upper = True; t_progress = now
                self.say_line("If you can't step back, say arms only and we'll start with the upper body. Or say never mind to chat.", key='setup')
                continue
            if now - t_progress > 120:
                self.setup_ui = None
                self.say_line("No rush. Tell me when the camera's set up and we'll try again.", key='cancel'); return False
            if issue is None or now < self.hold_until: continue
            quiet = now - self.last_caller_text_t > 6
            if issue != spoken_issue and quiet and now - last_said.get(issue, -1e9) > 20.0:
                spoken_issue = issue; t_issue = now; last_said[issue] = now
                self.telemetry.event('setup_issue', issue=issue)
                fp = facing_phrase((self.orient_state or {}).get('yaw') or 0.0, (self.orient_state or {}).get('away')) if issue == 'not_facing' else None
                if issue == 'dark' and self.subject_luma is not None and self.luma is not None and self.luma > 110 and self.subject_luma < 0.55 * self.luma:
                    fp = "There's bright light behind you. Turn so the window or lamp is in front of you." 
                self.say_line(fp or self.pick(f'first_{issue}', lines_for(issue, tr.upper)), key='setup')
            elif issue == spoken_issue and now - max(t_issue, self.last_caller_text_t) > 15 and checkins.get(issue, 0) < 3:
                t_issue = now; last_said[issue] = now; checkins[issue] = checkins.get(issue, 0) + 1
                self.say_line(self.pick(f'checkin_{issue}', lines_for(issue, tr.upper, checkin=True)), key='setup')
        return False

    async def ready_check(self):
        """'go' after a clear yes (or both hands up) to "ready?"; 'prepare' if the framing really breaks (the smoothed issue held 3 s);
        'explore' after a cancel or a long silence (with a spoken line). A "ready" said during a re-setup still counts; the yes to the
        space question never does; "wait" just waits."""
        if not self.lobby_enabled: return 'go'
        from team.dance.setup import SetupTracker
        self.transition('CONFIRM'); tr = SetupTracker(upper=self.upper_body or self.upper_override)
        self.deny_t = -1.0; bad_since = None; asks = 0; last_ask = -1e9; t_in = self.now(); squad_said = None
        self._arm_gesture(); self.q_open = 'confirm'
        try:
            while not self.stop and not self.safety_stop:
                if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; return 'explore'
                now = self.now()
                if self.cancel_req:
                    self.cancel_req = False; self.say_line("No problem. Tell me whenever you want to dance.", key='cancel'); return 'explore'
                need_two = self.mode in SQUAD_MODES and not self.solo_req and self.tracker.stable_count() < 2
                if self.mode in SQUAD_MODES and self.solo_req:
                    self.mode = 'battle' if self.mode == 'squad_battle' else 'learn'; self.solo_req = False
                    self.say_line("Solo it is.", key='squadneed'); self.telemetry.event('squad_to_solo', why='caller')
                if need_two:
                    if squad_said is None:
                        squad_said = now
                        self.say_line("Squad needs at least two of you in the picture. Grab a friend, or say just me and we'll go solo.", key='squadneed')
                    elif now - squad_said > 45.0:
                        self.mode = 'battle' if self.mode == 'squad_battle' else 'learn'; self.telemetry.event('squad_to_solo', why='no friend')
                        self.say_line("No friend yet? Let's do it solo. You can call a squad any time.", key='squadneed'); squad_said = -1e9
                if self.confirm_first_t is not None and not need_two:
                    after = self.answer_after('confirm', self.confirm_first_t)
                    if after is not None and self.affirm_t > max(after, self.space_answer_t):
                        self.telemetry.event('confirmed', asks=asks); self.setup_ui = None; return 'go'
                quiet = now - self.last_caller_text_t > 3.0
                if asks < 2 and now >= self.hold_until and quiet and now - last_ask > 20.0 and not (asks == 0 and now - self.line_end.get('confirm', -1e9) < 20.0):
                    asks += 1; last_ask = now
                    if self.confirm_first_t is None: self.confirm_first_t = now
                    self.say_line(self.pick('confirm', ["You're all set. Ready when you are: say ready, or raise both hands.",
                                                        "All good on camera. Say ready, or raise both hands, and we'll start."]) if asks == 1 else
                                  "Whenever you're ready, just say ready, or raise both hands.", key='confirm', ttl=10)
                elif asks == 0 and self.confirm_first_t is None and now - self.line_end.get('confirm', -1e9) < 20.0:
                    self.confirm_first_t = self.line_start.get('confirm', now); asks = 1; last_ask = now   # asked just before a re-setup
                if asks >= 2 and now - max(last_ask, self.last_caller_text_t) > 40.0 and now >= self.hold_until:
                    self.say_line("No worries. Tell me whenever you want to dance.", key='cancel'); return 'explore'
                await self.frame_step(None); await self.sleep(1.0 / self.fps)
                res = tr.update(self.now(), self.cur_lm, self.luma, self.frame_aspect); self.setup_ui = res
                # only a real break sends them back to setup: nobody in view, or too close with the hips gone; small drifts
                # (a step off centre, a turn) during "ready?" are fine and used to loop setup <-> ready for minutes
                hips = (res.get('metrics') or {}).get('hips_in')
                broken = res['issue'] == 'no_person' or (res['issue'] == 'too_close' and not hips and not tr.upper)
                bad_since = (bad_since or self.now()) if broken else None
                if bad_since is not None and self.now() - bad_since > CONFIRM_BREAK_S and self.now() - t_in > 2.0: return 'prepare'
        finally:
            self.want_ready = False; self.q_open = None
        return 'explore'

    def reset_run(self):
        self.rounds = []; self.history = []; self.best = None; self.phrase = 0; self.attempts = 0; self.my_code = None
        self.session_rounds = []; self.extras = {}; self.recap = None; self.battle_hud = None; self.step = None; self.fx.clear(); self.stage = None; self._last_squad = None

    async def idle(self, seconds):
        """Keep the video flowing (live frame, no overlay) while waiting for speech to land."""
        t_end = self.now() + seconds
        while self.now() < t_end and not self.stop:
            await self.frame_step(None)
            await self.sleep(1.0 / self.fps)
        if self.safety_stop:                                   # keep the safety card up for a moment after a stop
            t_end = self.now() + min(seconds, 2.0)
            while self.now() < t_end: await self.frame_step(None); await self.sleep(1.0 / self.fps)

    def _gestures(self, now, lm, in_attempt):
        """Held poses from the primary dancer (solo) or any tracked dancer (squad). Hands up means yes only when a yes is wanted
        (the ready check, paused); crossed arms pause the routine."""
        sources = [(pid, l) for pid, l, _ in self.players] if self.call_mode == 'squad' and self.players else [(None, lm)]
        fired = []; best = (None, 0.0)
        for pid, l in sources:
            g = self.gest.setdefault(pid, GestureDetector())
            k = g.update(now, l)
            if k: fired.append(k)
            if k == 'ready': self.ready_hits[pid] = now
            kind, p = g.progress(now)
            if kind and p > best[1] and (kind == 'pause' and in_attempt or kind == 'ready' and (self.want_ready or self.paused)): best = (kind, p)
        self.gesture_ui = dict(kind=best[0], p=best[1]) if best[0] else None
        if 'pause' in fired and in_attempt: self.pause_req = True; self.telemetry.event('gesture', kind='pause')
        elif 'ready' in fired and (self.want_ready or self.paused) and not in_attempt:
            # squad: each detector fires at the end of its own hold, so count dancers whose hold landed within READY_WINDOW_S
            live_ids = {pid for pid, _ in sources}
            self.ready_hits = {pid: t for pid, t in self.ready_hits.items() if pid in live_ids and now - t <= READY_WINDOW_S}
            need = max(1, (len(sources) + 1) // 2)
            if len(self.ready_hits) >= need or self.call_mode != 'squad':
                self.telemetry.event('gesture', kind='ready', state=self.state, dancers=len(self.ready_hits)); self.metrics['gesture_answers'] += 1
                self.ready_hits = {}
                if self.paused: self.resume_req = True
                else: self.affirm_t = now

    async def frame_step(self, engine):
        """One frame: pose, gestures, the engine, the composed picture. A bad frame (a decode or pose glitch) is logged and skipped;
        only a run of 30 in a row escalates (it used to abort the whole lesson on the first one)."""
        try:
            await self._frame_step(engine); self.frame_errors = 0
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.frame_errors += 1; self.metrics['frame_errors'] += 1
            if self.frame_errors in (1, 10): print(f'[dance] frame skipped ({self.frame_errors}): {e!r}', flush=True)
            if self.frame_errors >= 30: self.frame_errors = 0; raise

    def _dead_air_step(self, now, lm, engine):
        """The no-dead-air fallback (watchdog.py): Brodie's audio, queued lines and music count as activity; a nudge is spoken only
        when everything is quiet, nobody asked for a moment ("one sec"), and the caller isn't mid-turn."""
        wd = self.deadair
        if lm is not None: wd.seen(now)
        if self.say_q or self._model_speaking() or not self._audio_idle() or (engine is not None and engine.state != 'done'): wd.activity(now)
        if not self.lobby_enabled or now < self.hold_until or now - self.last_caller_text_t < 6.0: return
        r = wd.check(self.state, now)
        self.metrics['max_silence_s'] = round(wd.max_silence, 1); self.metrics['silences_over_20'] = wd.long_silences
        if r is None: return
        kind, line = r
        self.metrics['dead_air_nudges'] += 1; self.telemetry.event('dead_air', state=self.state, kind=kind)
        self.say_line(line, key='bye' if kind == 'end' else 'nudge', ttl=10)
        if kind == 'end': self.task(self._end_soon(8.0))

    def _guards(self, now, bgr, lm, engine):
        """guards.py checks once a frame: work per frame vs the budget, the phone moving mid-try, a backlit caller, someone on the floor,
        the voice dropping out, the per-call cap. Each acts at most once per occurrence and never throws."""
        try:
            ms = (time.perf_counter() - getattr(self, '_frame_t0', time.perf_counter())) * 1000.0
            step = self.budget.add(now, ms) if self.now is time.monotonic else None
            if step == 'lite' and self.pose_variant != 'lite':
                self.pose_variant = 'lite'; self.pose_reload = True; self.telemetry.event('budget', step='lite', ms=round(ms, 1))   # swapped on the pose thread
                print('[dance] frames over budget: lite pose model', flush=True)
            elif step == 'fps12' and self.fps > 12:
                self.fps = 12.0; self.telemetry.event('budget', step='fps12'); print('[dance] frames over budget: 12 fps', flush=True)
            in_try = engine is not None and getattr(engine, 'state', None) == 'attempt' and isinstance(engine, PhraseEngine)
            if bgr is not None and self.cam_shift.update(now, bgr) and in_try and not self.camera_moved:
                self.camera_moved = True; self.telemetry.event('camera_moved')
            if bgr is not None and lm is not None:                       # backlight: the person much darker than the picture
                a = np.asarray(lm, np.float32); v = a[:, 3] > 0.5
                if v.sum() >= 8:
                    h, w_ = bgr.shape[:2]; xs = np.clip(a[v, 0] * w_, 0, w_ - 1); ys = np.clip(a[v, 1] * h, 0, h - 1)
                    roi = bgr[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]
                    if roi.size: self.subject_luma = float(roi[::4, ::4].mean())
            if lm is not None and self.state == 'ACTIVITY' and not self.safety_stop:      # someone down on the floor and still
                a = np.asarray(lm, np.float32)
                if min(a[[11, 12, 23, 24], 3]) > 0.5:
                    d = 0.5 * (a[11, :2] + a[12, :2]) - 0.5 * (a[23, :2] + a[24, :2])
                    flat = abs(d[0]) > 1.8 * abs(d[1]) and float(np.mean(a[[23, 24], 1])) > 0.7
                    self.fall_since = (self.fall_since or now) if flat else None
                    if self.fall_since is not None and now - self.fall_since > 2.0:
                        self.fall_since = None; self.pause_req = True; self.telemetry.event('fall_check')
                        self.say_line("Whoa, everyone okay? Take a second. Say resume, or raise both hands, when you're good.", key='safety', urgent=True)
            model = self._model()
            if model is not None and self.now is time.monotonic:          # the voice dropped (session closed, relay limit)
                up = bool(getattr(model, 'started', False))
                if up: self.voice_was_up = True; self.voice_down_since = None
                elif self.voice_was_up:
                    self.voice_down_since = self.voice_down_since or now
                    self.caption = ("Brodie lost his voice for a second. Hang on!", now)
                    if now - self.voice_down_since > 30.0 and not self.stop:
                        self.telemetry.event('voice_lost'); print('[dance] voice lost for 30 s: ending the call', flush=True); self.stop = True
            if self.t0 is not None and self.lobby_enabled and self.call_max_s > 0:
                age = now - self.t0
                if age > self.call_max_s - 120 and not self.call_warned:
                    self.call_warned = True; self.say_line("Heads up: two minutes left on this call. Let's make them count!", key='bye')
                if age > self.call_max_s and not self.stop:
                    if self.state in LOBBY: self.bye_req = True
                    else: self.end_after_activity = True
        except Exception as e:
            print(f'[dance] guard skipped: {e!r}', flush=True)

    async def _end_soon(self, s):
        t = self.now()
        while self.now() - t < s and not self.stop: await self.frame_step(None); await self.sleep(1.0 / self.fps)
        self.stop = True

    async def _frame_step(self, engine):
        self._frame_t0 = time.perf_counter()
        if self.vad_utt_end is not None: self.check_unclear(self.now())
        frame, t = self.newest_frame()
        if frame is None: return
        self.last_frame_t = self.now()
        bgr, lm, w = await self.prepare(frame, int((self.now() - self.t0) * 1000))
        now = self.now(); self.cur_lm = lm; self.cur_world = w
        self._guards(now, bgr, lm, engine)
        if self.board_ui is not None and now > getattr(self, 'board_until', 0.0) + 0.5: self.board_ui = None
        self._dead_air_step(now, lm, engine)
        if self._orient_pid != self.tracker.primary_id:                   # a new main dancer: calibrate their proportions afresh
            self._orient_pid = self.tracker.primary_id; self.orient = Orient()
        try: self.orient_state = self.orient.update(lm, w, self.frame_aspect or 1.0, now) if lm is not None else None
        except Exception as e: self.orient_state = None; print(f'[dance] orient skipped: {e!r}', flush=True)
        try: self.frame_aspect = bgr.shape[1] / float(bgr.shape[0])
        except Exception: pass
        try: self.luma = float(cv2.resize(bgr, (24, 24), interpolation=cv2.INTER_AREA).mean())
        except Exception: self.luma = None
        if lm is not None: self.last_lm = lm
        self.parts = render_visibility(lm)[1]
        hint = framing_hint(self.parts) if lm is not None else "I can't see you"
        if hint != self.hint: self.hint = hint; self.hint_since = now
        self.telemetry.frame(lm is not None, full_body=all(self.parts.get(p, False) for p in ('head', 'arms', 'torso', 'legs')),
                             upper=self.parts.get('torso', False) and not self.parts.get('legs', False))
        self.max_players = max(self.max_players, self.tracker.stable_count())
        for ev in self.tracker.pop_events():
            kind = ev.pop('type'); self.telemetry.event(kind, **ev)
            if kind == 'crowd' and (self.call_mode == 'squad' or self.mode in SQUAD_MODES or self.mode == 'class') and self.state != 'ACTIVITY':
                self.say_line("I can score four dancers at a time. The rest of you can cheer!", key='squad')
        in_attempt = engine is not None and engine.state in ('show', 'countin', 'attempt')
        self._gestures(now, lm, in_attempt)
        speaking = self._model_speaking() or (engine is None and not self._audio_idle())
        self.last_bgr = bgr
        av_state = self.avatar_state(speaking)
        hud = None; ghost = None; card = None; extra = dict(t=now, speaking=speaking, state=self.state, avatar_state=av_state, countdown=self.countdown_ui)
        if engine is not None:
            if isinstance(engine, SquadEngine):
                engine.on_frame(now, {pid: (l, ww) for pid, l, ww in self.players})
            else:
                engine.on_frame(now, lm, w)
            engine.tick(now)
            if engine.state != 'done':
                hud = engine.hud(now); ghost = engine.ghost_sample(now)
                hud.update(history=self.history, best=self.best, fx=self.fx, difficulty=self.difficulty, step=self.step, battle=self.battle_hud,
                           gesture=getattr(self, 'gesture_ui', None), paused=self.paused, stage=self.stage, ghost_overlay=self.ghost_overlay,
                           practice=self.practice, speaking=speaking, demo=self.in_demo, avatar_state=av_state)
                if self.in_warmup or self.practice: hud['fx'] = None
                if self.call_mode == 'squad':
                    extra['squad_lms'] = {pid: (l, self.colours.get(pid, 'volt')) for pid, l, _ in self.players if pid in self.colours}
                    ov = self.tracker.overlapping()
                    if ov and ov[0][0] and ov[0][1]: hud['squad_hint'] = f"Spread out: {ov[0][0].upper()} and {ov[0][1].upper()} are overlapping"
        if hud is None:
            if self.final_card is not None and not self.safety_stop and self.state in ('EXPLORE', 'REVIEW', 'CHOOSE'):
                card = dict(self.final_card); card.setdefault('key', self.key['name'])
            else:
                cap = self.caption[0] if self.caption and now - self.caption[1] < 6.0 else None
                hints = self.state == 'EXPLORE' and self.last_caller_text_t < (self.t0 or 0) + 0.01
                extra.update(menu=self.menu_ui if self.state == 'CHOOSE' else None, compare=getattr(self, 'compare_ui', None), brk=self.break_ui, board=self.board_ui)
                extra.update(title=self.idle_title, lines=[], recap=self.recap if self.state == 'REVIEW' else None, caption=cap, footer=None,
                             gesture=getattr(self, 'gesture_ui', None), paused=self.paused, safety=self.safety_stop, hints=hints,
                             setup=self.setup_ui if self.state in ('PREPARE', 'CONFIRM') else None)
                if self.call_mode == 'squad' or self.tracker.stable_count() >= 2:
                    extra['players'] = [(l, self.colours.get(pid, 'volt')) for pid, l, _ in self.players]
        await self.render_push(bgr, lm if lm is not None else self.last_lm, hud, ghost=ghost, card=card, extra=extra)

    # ---- engine events: judgments, tips, combo, battle damage, telemetry ----
    def on_engine_event(self, ev):
        typ = ev.get('type'); now = self.now(); pid = ev.get('pid')
        if typ == 'pose_hint':
            self.fx.add('tip', now, short=ev.get('short'), text=ev.get('text')); return
        if typ == 'pose_step':
            if ev.get('why') == 'matched': self.fx.add('judgment', now, verdict='green', pid=None)
            self.telemetry.event('pose_count', count=ev.get('count'), why=ev.get('why'), err=ev.get('err')); return
        if typ in ('phrase_state', 'attempt_started', 'phrase_aborted', 'move_shown'):
            self.telemetry.event(typ, **{k: v for k, v in ev.items() if k in ('to', 'phrase', 'pid', 'at_state', 'bpm', 'tempo', 'n_beats')})
        if typ == 'beat' and not self.in_warmup:
            v = ev.get('live_verdict'); b = ev.get('beat', 0)
            if v: self.fx.add('judgment', now, verdict=v, pid=pid, colour=ev.get('colour'), small=pid is not None)
            c = self.combo_run.get(pid, 0); c = c + 1 if v == 'green' else 0; self.combo_run[pid] = c
            if c in (4, 8, 16) and pid in (None, (self.squad_players[0][0] if self.squad_players else None)): self.fx.add('combo', now, n=c)
            if self.battle_hud is not None and v in ('green', 'amber'):
                floor_hp = self.battle_hud.get('round_floor', 0)
                self.battle_hud['ghost_hp'] = max(floor_hp, self.battle_hud['ghost_hp'] - (12 if v == 'green' else 6))
            eng = self.cur_engine.engines.get(pid) if isinstance(self.cur_engine, SquadEngine) else self.cur_engine
            if eng is not None and b >= 1 and v == 'red' and eng.live_verdicts[b - 1] == 'red' and b - self.last_tip_beat.get(pid, -99) >= 4:
                rows = [eng.live_channel_err[b - 1], eng.live_channel_err[b]]
                amb = (self.orient_state or {}).get('ambiguous') if pid is None else None
                if amb and all(r is not None for r in rows):                     # never a tip about a joint the camera can't judge (depth)
                    rows = [np.where(np.isin(np.arange(len(r)), list(amb)), np.nan, r) for r in rows]
                if all(r is not None for r in rows):
                    tip = feedback.live_tip(rows)
                    if tip:
                        self.last_tip_beat[pid] = b; label = ev.get('colour')
                        self.fx.add('tip', now, short=(f"{label.upper()}: {tip['short']}" if label else tip['short']), text=tip['text'], colour=label)
                        self.telemetry.cue(ev.get('t_beat_end', now), now, 'tip', tip['short'])
        if typ == 'phrase_scored' and pid in (None, (self.squad_players[0][0] if self.squad_players else None)) and not self.in_warmup:
            r = {k: ev.get(k) for k in ('score', 'scored', 'spatial', 'timing', 'energy', 'max_combo', 'visible_fraction')}
            self.telemetry.score(r, phrase=ev.get('phrase'), pid=pid)
            if r.get('scored') and (r.get('score') or 0) >= 80: self.fx.add('confetti', now + 1.0, seed=int(now * 10) % 9973)
            if r.get('scored'): self.sfx('stamp')

    # ---- the opening conversation ----
    def library_keys(self):
        try: return [k for k in list_keys() if 'error' not in k]
        except Exception: return []

    def pick_key(self, style):
        """A library key for a spoken style (kept for the K-pop signature routine and upper-body fallbacks)."""
        keys = {k['name'] for k in self.library_keys()}
        if style == 'kpop' and 'cc_huntrix' in keys: return 'cc_huntrix'
        if style == 'squats' and 'test_squats' in keys: return 'test_squats'
        if style == 'any' or style is None:
            for c in (('wa', 'po', 'lo') if self.upper_body else ('lh', 'kpop', 'ho', 'po')):
                k = self.pick_key(c)
                if k: return k
            return None
        cands = sorted(k for k in keys if k.startswith(f'aist_{style}_'))
        if not cands and style == 'lh': cands = sorted(k for k in keys if k.startswith('aist_mh_'))
        return cands[0] if cands else None

    def make_routine(self, style, difficulty=None):
        """A routine for a spoken style: generated from the library by difficulty (the K-pop routine is a fixed signature key)."""
        d = difficulty or self.difficulty
        if style == 'kpop' and 'cc_huntrix' in {k['name'] for k in self.library_keys()}:
            k = load_key('cc_huntrix'); k['title'] = 'K-pop Huntrix'; return k
        if style == 'squats':
            k = load_key('test_squats'); k['title'] = 'squats'; return k
        if style in ('any', None): style = 'wa' if self.upper_body else 'lh'
        k = choreo.generate(choreo.STYLE_CODE.get(style, 'hiphop'), d)
        return k

    async def wait_cmd(self, seconds, accept=None):
        """Keep the picture flowing while waiting for a spoken command (or any). Returns it, or None on timeout."""
        t_end = self.now() + seconds
        while self.now() < t_end and not self.stop:
            cmd = self.pending_cmd
            if cmd and (accept is None or cmd in accept): self.pending_cmd = None; return cmd
            await self.frame_step(None); await self.sleep(1.0 / self.fps)
        return None

    # ---- one phrase (with pause/resume) ----
    async def run_phrase(self, phrase_index, tempo, show_repeats=2, countin_beats=4, freestyle=False, focus=None, step=None, key=None,
                         stage=None, practice=False, demo=False, ghost_overlay=True):
        """One phrase: metronome + ghost + scoring. demo=True: the reference only (no count-in, no attempt). practice=True: dance with
        the ghost on your body, no score shown. A pause (voice between phrases, crossed arms during one) stops the music; on resume the
        same phrase runs again from its count-in."""
        key = key or self.key
        if focus is None and (self.upper_body or self.upper_override) and not freestyle: focus = 'arms'   # arms only: never score the legs
        self.stage = stage; self.practice = practice; self.in_demo = demo; self.ghost_overlay = ghost_overlay
        while True:
            self.transition('ACTIVITY', stage or 'phrase')
            if self.call_mode == 'squad' and self.squad_players and not demo: self.refresh_squad()
            if self.call_mode == 'squad' and len(self.squad_players) >= 2 and not demo:
                eng = SquadEngine(key, self.squad_players, on_event=self.on_engine_event, phrase_index=phrase_index, tempo=tempo, show_repeats=show_repeats,
                                  countin_beats=countin_beats, rtt_s=self.rtt_s, freestyle=freestyle, focus=focus)
            else:
                eng = PhraseEngine(key, phrase_index=phrase_index, tempo=tempo, show_repeats=max(1, show_repeats) if demo else show_repeats,
                                   countin_beats=countin_beats, rtt_s=self.rtt_s, on_event=self.on_engine_event, freestyle=freestyle, focus=focus, demo_only=demo)
            eng.step = step; self.step = step; self.cur_engine = eng; self.combo_run = {}; self.last_tip_beat = {}; self.recap = None
            if self.depth_gate and isinstance(eng, PhraseEngine):
                def _gate(m):
                    amb = (self.orient_state or {}).get('ambiguous')
                    if not amb: return m
                    m = np.array(m, bool, copy=True); m[list(amb)] = False; return m
                eng.mask_hook = _gate
            bank = getattr(self.job, 'bank', None)
            if bank is not None:                                   # clicks only: no spoken words inside the track (one voice)
                show_beats = 0 if freestyle else eng.show_s / eng.beat_s
                dance_audio.register_phrase_track(bank, eng.bpm, int(round(show_beats)), 0 if demo else countin_beats,
                                                  0 if demo else eng.n_beats + eng.extra_beats, voice={})
            t_start = self.now()
            holder = getattr(self, 'engine_holder', None)
            if holder is not None: holder['engine'] = eng
            eng.start(t_start); self.say(dance_audio.TRACK, 'announce')
            self.log.append(dict(t=round(t_start - self.t0, 2), phrase=phrase_index, tempo=tempo, event='start', stage=stage))
            paused = False
            while eng.state != 'done' and not self.stop:
                await self.frame_step(eng)
                if self.safety_stop: eng.abort(self.now()); break
                # warm-up, demo and practice can be skipped at once; during a scored try "next" stays pending and acts after the score
                # ("next" at "Got it?" means "try it with me", so practice only ends early on "skip")
                c = self.take_cmd(('stop', 'pause', 'next', 'skip') if (self.in_warmup or demo) else (('stop', 'pause', 'skip') if practice else ('stop', 'pause')))
                if c == 'stop': self.stop_routine = True; self.cut_audio(); eng.abort(self.now()); break
                if c in ('next', 'skip'): self.cut_audio(); eng.abort(self.now()); break          # "skip" the warm-up
                if self.pause_req or c == 'pause':
                    self.pause_req = False
                    eng.abort(self.now()); self.cut_audio(); paused = True; break
                frozen = VideoFreeze.status(self.now(), self.last_frame_t) == 'abort' and eng.state == 'attempt'
                if (self.camera_moved or frozen) and isinstance(eng, PhraseEngine) and not demo and not practice:
                    why = 'camera_moved' if self.camera_moved else 'video_frozen'; self.camera_moved = False
                    eng.abort(self.now()); self.cut_audio(); eng.result = dict(scored=False, score=0.0, reason=why)
                    self.telemetry.event('try_stopped', why=why)
                    self.say_line("Looks like the phone moved. Let's check the camera." if why == 'camera_moved' else
                                  "Your video froze. Let's try that again.", key='setup', urgent=True)
                    break
                await self.sleep(1.0 / self.fps)
            if paused and not self.stop:
                await self.paused_wait()
                if self.stop or self.safety_stop or self.stop_routine: break
                show_repeats = min(show_repeats, 1); continue
            break
        self.cur_engine = None; self.in_demo = False
        if isinstance(eng, SquadEngine):
            self._last_squad_counts = {(nm or c).upper(): eng.engines[pid].verdicts() for pid, c, nm in eng.players if pid in eng.engines}
        if not self.safety_stop: self.transition('REVIEW', 'phrase done')
        r = eng.result or {}
        self.log.append(dict(t=round(self.now() - self.t0, 2), phrase=phrase_index, event='scored', stage=stage, result={k: r.get(k) for k in ('score', 'spatial', 'timing', 'energy', 'scored')} if r else None))
        if not demo:
            print(f"[dance] {stage or 'phrase'} {phrase_index + 1} @x{tempo:.2f} {'scored' if r.get('scored') else 'unscored'}: score {r.get('score')} timing {r.get('timing')} "
                  f"visible {r.get('visible_fraction')} cue {eng.cue['text'] if eng.cue else None}", flush=True)
        if eng.result is not None and not self.in_warmup and not practice and not demo:
            self.history.append(dict(phrase=phrase_index, score=(r.get('score') if r.get('scored') else None), scored=bool(r.get('scored')), tempo=tempo,
                                     partial=bool(r.get('partial')), freestyle=bool(freestyle), verdicts=[row['verdict'] for row in r.get('per_beat', [])] if r.get('scored') else []))
            scored = [h['score'] for h in self.history if h['score'] is not None]
            self.best = max(scored) if scored else None
        return eng

    async def run_pose_step(self, key, pi, mode='watch', step=None, stage=None, speak=True, focus=None):
        """Pose by pose (posestep.py). WATCH: the ghost steps through the counts, ~3.5 s each, with a soft tick and Brodie naming each
        pair of counts. FREEZE: the caller matches each pose (a tick and the next pose when it's held; on after 4 s regardless).
        "faster" / "next" end it early (on to the slow run); "stop" and "pause" work as in a phrase. Returns the engine."""
        from team.dance.posestep import PoseStepEngine
        stage = stage or ('WATCH' if mode == 'watch' else 'MATCH IT')
        if focus is None and (self.upper_body or self.upper_override): focus = 'arms'
        self.stage = stage; self.practice = mode == 'freeze'; self.in_demo = False; self.ghost_overlay = True
        self.transition('ACTIVITY', stage)
        busy = lambda: (any(q['key'] == 'posecount' for q in self.say_q) or not self._audio_idle() or self._model_speaking())
        eng = PoseStepEngine(key, pi, mode=mode, on_event=self.on_engine_event, focus=focus, busy=busy if speak else None)
        eng.step = step; self.step = step; self.cur_engine = eng
        holder = getattr(self, 'engine_holder', None)
        if holder is not None: holder['engine'] = eng
        lines = []                                     # one short cue per count from the move card (cues.py), e.g. "Three: arms up in a V"
        if speak:
            try: lines = cues.count_lines(cues.move_card(key, pi))
            except Exception as e:
                print(f'[dance] count cues failed ({e!r}); angle breakdown', flush=True); lines = []
            if not lines:
                try:
                    for p in feedback.describe_phrase(key, pi, per=1).split('; '):
                        if ': ' in p: lines.append(p.split(': ', 1)[1].capitalize())
                except Exception: lines = []
        eng.start(self.now()); last = -1
        self.log.append(dict(t=round(self.now() - self.t0, 2), phrase=pi, event='pose_step', mode=mode, n=eng.n_beats))
        while eng.state != 'done' and not self.stop:
            await self.frame_step(eng)
            if self.safety_stop: eng.abort(self.now()); break
            c = self.take_cmd(('stop', 'pause', 'next', 'skip', 'faster'))
            if c == 'stop': self.stop_routine = True; eng.abort(self.now()); break
            if c in ('next', 'skip', 'faster'): eng.abort(self.now()); break
            if self.pause_req or c == 'pause':
                self.pause_req = False; eng.abort(self.now()); await self.paused_wait(); break
            if eng.cur != last and eng.state != 'done':
                last = eng.cur
                if last > 0 and self._audio_idle() and not self._model_speaking(): self.sfx('tick')   # a tick under a line pauses the voice
                if speak and last < len(lines):
                    word = COUNT_WORDS[last].capitalize() if last < len(COUNT_WORDS) else str(last + 1)
                    self.say_line(f"{word}: {lines[last][:1].lower() + lines[last][1:]}.", key='posecount', ttl=max(4.0, eng.hold_s))
            await self.sleep(1.0 / self.fps)
        self.cur_engine = None; self.practice = False
        if not self.safety_stop and not self.paused: self.transition('REVIEW', 'pose done')
        self.telemetry.event('pose_step_done', mode=mode, verdicts=eng.verdicts(), n=eng.n_beats)
        return eng

    def phrase_outcome(self, eng):
        """(scored, score, result, cue) for a solo engine or a squad (median of the dancers, cue for the one furthest off)."""
        if isinstance(eng, SquadEngine) and eng.squad:
            sc = [x['score'] for x in eng.squad['players'] if x['scored']]
            if not sc: return False, 0.0, eng.lead.result or {}, eng.cue
            return True, float(np.median(sc)), eng.lead.result or {}, eng.cue
        r = eng.result or {}
        return bool(r.get('scored')), float(r.get('score') or 0) if r.get('scored') else 0.0, r, eng.cue

    def summarise(self, eng, round_label='', xp=None, item=None):
        scored, score, r, cue = self.phrase_outcome(eng)
        if eng.result is None: return None
        if isinstance(eng, SquadEngine) and eng.squad:
            sq = eng.squad; self._last_squad = sq
            if sq.get('mvp'): self.fx.add('mvp', self.now(), label=sq['mvp']['label'], colour=sq['mvp']['colour']); self.sfx('mvp')
            self.recap = dict(result=dict(scored=scored, score=score, spatial=r.get('spatial', 0), timing=r.get('timing', 0), energy=r.get('energy', 0)),
                              title=sq['squad_title'], cue=(cue or {}).get('text'))
            lead = sq['ranking'][0] if sq.get('ranking') else None
            line = (f"{(lead.get('name') or lead['colour']).capitalize()} leads with {lead['score']:.0f}! " if lead else '') + ((cue or {}).get('text', '') + '.' if cue else '')
            return self.say_line(line.strip(), key='result')
        if not scored:
            self.last_result = r; self.last_cue = cue
            self.recap = dict(result=r, cue='Step back so I can see all of you')
            return self.say_line("I couldn't see enough of you that time. Let's fix the camera first.", key='result')
        # up to three ranked corrections (shape, depth / facing, timing per count, size), and whether the last one landed
        layered = (os.environ.get('DANCE_LAYERED_CUES', '1') or '1').strip() != '0'
        cues = feedback.layered_cues(r, getattr(eng, 'key', None), getattr(eng, 'pi', 0), arms_only=bool(self.upper_body or self.upper_override)) if layered else [cue or {}]
        main = cues[0] if cues else (cue or {})
        prev = self._prev_cues.get(item) if item is not None else None
        prog = feedback.progress_note(prev[0], r, prev[1]) if prev else None
        if item is not None: self._prev_cues[item] = (r, main)
        self.last_result = r; self.last_cue = main; self.last_cues = cues
        glow = getattr(self, '_glow', False)
        title = progression.title_for(r, dict(glow_up=glow)); self.last_title = title; g = progression.grade(score)
        cue_txt = main.get('text', '') if main.get('kind') != 'praise' else ''
        tips = [c.get('text') for c in cues if c.get('kind') != 'praise' and c.get('text')]
        self.recap = dict(result=r, title=title, cue=cue_txt, tips=tips, xp=xp)
        react = dict(S=["Unreal!", "That was clean!"], A=["So good!", "Nice one!"], B=["Solid!", "Getting there!"], C=["Good effort!", "Nice try!"],
                     D=["Good start!", "We'll get it!"])[g]
        tail = (cue_txt + '.') if cue_txt else "Keep that energy."
        if len(tips) > 1:
            second = tips[1]; second = second[0].lower() + second[1:]
            tail += f" And {second}."
        return self.say_line(f"{self.pick('react_' + g, react)} {score:.0f}. {(prog + ' ') if prog else ''}{tail}", key='result')

    def xp_for_attempt(self, score):
        return int(round(score / 4.0 * progression.DIFF_MULT.get(self.difficulty, 1.0)))

    def record_round(self, eng, item=None):
        """Book one attempt for XP and achievements; marks a glow-up (15+ points better than the last try at the same step)."""
        scored, score, r, cue = self.phrase_outcome(eng)
        prev = self._prev_item_score.get(item) if item is not None else None
        glow = bool(scored and prev is not None and score - prev >= 15)
        self._glow = glow
        if item is not None and scored: self._prev_item_score[item] = score
        self.session_rounds.append(dict(score=score if scored else None, scored=scored, max_combo=r.get('max_combo', 0), glow_up=glow))
        return scored, score, r, cue

    # ---- learn: sections, then combine; adapt to the dancer ----
    def learn_plan(self, key):
        """Moves in teaching order. Beginners learn each 8-count step in two 4-count halves first (one small move at a time), then
        the whole step; steps are put together after they are learned. Items: (kind, section, half) with kind H(alf) / S(tep) / C(ombine)."""
        n = len([p for p in key['phrases'] if not p.get('combined') and 'half' not in p])
        n = min(n, int(os.environ.get('DANCE_MAX_PHRASES', '0') or 0) or n)
        halves = self.difficulty == 'beginner'
        plan = []
        for s in range(n):
            if halves: plan += [('H', s, 0), ('H', s, 1)]
            plan.append(('S', s, None))
            if s >= 1: plan.append(('C', s, None))
        return plan, n

    async def explain(self, key, pi):
        """Walk the move through pose by pose, each pair of counts named (from the reference itself); the caller can say "faster" to
        skip ahead. Without pose by pose: the spoken breakdown."""
        self.telemetry.event('explain', phrase=pi)
        if POSE_ON:
            await self.run_pose_step(key, pi, 'watch', step=self.step)
        else:
            t = self.now(); self.say_breakdown(key, pi, intro="Here's the move, step by step:"); await self.wait_spoken('breakdown', t, 15.0, min_s=2.0)

    def say_breakdown(self, key, pi, intro="Here's the move:"):
        """The move in a few counts, in position words (cues.py): "One, arms down. Three, left arm up. Five, sink low." Counts with
        nothing new are skipped; at most four are said."""
        try:
            card = cues.move_card(key, pi); lines = cues.count_lines(card)
            said = [(i, l) for i, l in enumerate(lines) if l.lower() not in ('hold it', 'bounce')][:4]
            parts = [f"{COUNT_WORDS[i].capitalize() if i < len(COUNT_WORDS) else i + 1}, {l[:1].lower() + l[1:]}" for i, l in said]
            if parts: return self.say_line(f"{intro} " + '. '.join(parts) + f". {card['feel']}", key='breakdown', ttl=15)
        except Exception as e:
            print(f'[dance] breakdown cues failed ({e!r})', flush=True)
        desc = feedback.describe_phrase(key, pi)
        steps = [p for p in desc.split('; ') if ': ' in p][:4]
        if not steps: return None
        parts = [f"{p.split(': ', 1)[0]}, {p.split(': ', 1)[1]}" for p in steps]
        return self.say_line(f"{intro} " + '. '.join(parts) + '.', key='breakdown', ttl=15)

    async def show_break(self, label, seconds, key=None, pi=None):
        """A short break between moves (pacing.py): a label, a 3-2-1 ring, and the first pose of what comes next, so the caller can
        reset and breathe. "stop" and "pause" still work; nothing is spoken."""
        if seconds <= 0 or self.stop or self.stop_routine: return
        preview = None
        if key is not None and pi is not None:
            try:
                from team.dance.reference import phrase_slice
                preview = phrase_slice(key, pi)[3][0]
                if self.mirror_view:                                     # drawn like the on-body ghost: mirrored with the picture
                    from team.dance.render import mirror_skel
                    preview = mirror_skel(preview)
            except Exception: preview = None
        self.break_ui = dict(label=label, t0=self.now(), dur=float(seconds), preview=preview)
        self.telemetry.event('break', label=label, s=seconds)
        try:
            t_end = self.now() + seconds
            while self.now() < t_end and not self.stop and not self.stop_routine and not self.safety_stop:
                c = self.take_cmd(('stop',))
                if c == 'stop': self.stop_routine = True; break
                await self.frame_step(None); await self.sleep(1.0 / self.fps)
        finally:
            self.break_ui = None

    async def wait_choice(self, line, key, seconds=18.0, reprompt=None, timeout=None, on_timeout=None, any_reply=False):
        """Ask (or, with line=None, just keep listening) and wait patiently for the caller: a choice (again / next / skip / faster /
        slower / simplify / explain / full / stop) or a yes / no. One gentle reprompt after `seconds` of quiet; then, with `timeout`,
        `on_timeout` after that many more quiet seconds (only where going ahead is safe, e.g. from a demo to practising with the
        ghost); otherwise Brodie keeps waiting quietly. any_reply: any answer that isn't a no, a wait or a command means yes ("got
        it", "makes sense", "let's try"). A choice made while the move was playing counts; one from an earlier step never does. If
        the question itself never got said, answers count from when it was asked, so this can never hang."""
        cmd = self.take_cmd(CHOICES)
        if cmd:
            self.cmd_floor_t = self.now()
            if cmd == 'stop': self.stop_routine = True
            return cmd
        if line: self.say_line(line, key=key, ttl=12)
        t0 = self.now(); reprompted = False; t_re = None; self.q_open = key if line else self.q_open
        if line: self._arm_gesture()            # both hands up answers yes here too (voice was the only way after a result)
        try:
            return await self._wait_choice_loop(line, key, seconds, reprompt, timeout, on_timeout, any_reply, t0)
        finally:
            self.q_open = None; self.want_ready = False

    async def _wait_choice_loop(self, line, key, seconds, reprompt, timeout, on_timeout, any_reply, t0):
        reprompted = False; t_re = None
        while not self.stop and not self.stop_routine and not self.safety_stop:
            if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; break
            now = self.now()
            cmd = self.take_cmd(CHOICES)
            if cmd:
                self.cmd_floor_t = now
                if cmd == 'stop': self.stop_routine = True
                return cmd
            after = self.answer_after(key, t0, fallback=14.0) if line else t0
            if after is not None:
                if self.affirm_t > after: self.cmd_floor_t = now; return 'yes'
                if self.deny_t > after: self.cmd_floor_t = now; return 'no'
                if any_reply and self.last_utt is not None and self.last_utt[0] > after:
                    _, it, c = self.last_utt
                    if not (it['deny'] or it['hold'] or it['question'] or c): self.cmd_floor_t = now; return 'yes'
            await self.frame_step(None); await self.sleep(1.0 / self.fps)
            now = self.now()
            base = t0 if after is None else max(after, t0)
            quiet = now - max(base, self.last_caller_text_t)
            if reprompt and not reprompted and after is not None and quiet > seconds and now >= self.hold_until:
                reprompted = True; t_re = now; self.say_line(reprompt, key=key + '_re', ttl=12); self.metrics['reprompts'] += 1
            if timeout is not None and reprompted and now - max(t_re, self.last_caller_text_t) > timeout and now >= self.hold_until:
                self.telemetry.event('choice_timeout', key=key, went=on_timeout); self.metrics['choice_timeouts'] += 1; return on_timeout
            if not line and quiet > seconds * 3: return None              # listening-only call: hand back so the caller can be asked again
        return 'stop'

    async def learn(self, mode_name='learn', skip_warmup=False, finish=True, challenge=True):
        """A patient lesson at the caller's pace. Each move: a short spoken breakdown and a very slow demo (0.5x), then "got it, or
        again?"; practice with the ghost; then the caller's own tries, each followed by one direct, actionable tip and a question
        (again / a little faster / next). Speed rises one notch at a time and only when the caller says yes; two misses simplify
        the move (halves, slower, arms first). Silence never skips ahead. Steps are combined only after they are learned, and the
        full-speed challenge is offered only after the routine is solid at 0.9x."""
        pass_mark = dict(beginner=65, intermediate=72, advanced=80)[self.difficulty]
        start_rung = START_RUNG[self.difficulty]; self.stop_routine = False; self._prev_item_score = {}; self._passed_halves = set()
        if self.difficulty != 'beginner' and not skip_warmup and self.call_mode == 'solo': await self.warmup()
        plan, n = self.learn_plan(self.key); i = 0; completed_all = False
        t = self.now()
        self.say_line(self.pick('lesson', ["We'll learn this one move at a time, nice and slow. No rush.",
                                           "Let's take it one small move at a time. Super slow to start."]), key='lesson')
        await self.wait_spoken('lesson', t, 8.0, min_s=1.0)
        t_lesson = self.now(); cap_offered = False
        while i < len(plan) and not self.stop and not self.stop_routine:
            item = plan[i]
            if i > 0:                                                         # a 3-4 s reset between moves, showing what's next
                try: k_, p_ = self.item_key(item)
                except Exception: k_, p_ = None, None
                await self.show_break('NEXT MOVE', pacing.MOVE_GAP_S, k_, p_)
                if self.stop or self.stop_routine: break
            if not cap_offered and self.now() - t_lesson > pacing.LESSON_SOFT_CAP_S and i < len(plan) - 1:
                cap_offered = True                                            # a long lesson: offer the full routine instead of more steps
                a = await self.wait_choice("We've been at it a while! Want to try the full routine now, or keep practising step by step?",
                                           key='choice', reprompt="Say full routine, or keep going.")
                if a == 'stop': break
                if a in ('full', 'yes'): i = len(plan) - 1; item = plan[i]
            res = await self.teach_item(item, n, start_rung, pass_mark)
            if res == 'passed' and item[0] == 'H': self._passed_halves.add((item[1], item[2]))
            if res == 'restart':
                plan, n = self.learn_plan(self.key); i = 0; start_rung = START_RUNG[self.difficulty]
                pass_mark = dict(beginner=65, intermediate=72, advanced=80)[self.difficulty]; continue
            if res == 'stop': break
            if res == 'full': i = len(plan) - 1; continue
            kind, idx = item[0], item[1]
            if res == 'passed' and ((kind == 'C' and idx == n - 1) or (n == 1 and kind == 'S')): completed_all = True
            i += 1; self.phrase = max(self.phrase, i)
        if completed_all and not self.stop and not self.stop_routine and challenge:
            self.extras['combo_run'] = True
            ans = await self.wait_choice("You've got the whole routine! Want to try it at full speed for a score?", key='challenge',
                                         reprompt="No pressure. Say yes for the full-speed challenge, or no to finish here.")
            if ans in ('yes', 'faster'):
                key, pi = choreo.with_combined(self.key, n - 1) if n > 1 else (self.key, 0)
                eng = await self.run_phrase(pi, RUNGS[-1] * self.tempo, show_repeats=0, step=dict(label='CHALLENGE'), key=key, stage='CHALLENGE', ghost_overlay=False)
                if eng.result is not None:
                    self.record_round(eng, item='challenge'); self.rounds.append(dict(phrase=pi, kind='CH', tempo=RUNGS[-1] * self.tempo, result=eng.result))
        if not finish: return
        if self.call_mode == 'squad' and self._last_squad:
            totals = {x['label']: 3.0 * (x['score'] or 0.0) for x in self._last_squad['players']}
            await self.finish_squad(totals, self._last_squad, mode='squad_learn')
        else:
            await self.finish_session(mode_name)

    def item_key(self, item):
        """(key, phrase index) a learn item teaches: a half, the whole step, or the steps combined."""
        kind, idx = item[0], item[1]; half = item[2] if len(item) > 2 else None
        if kind == 'H': return choreo.with_half(self.key, idx, half)
        if kind == 'C': return choreo.with_combined(self.key, idx)
        return self.key, idx

    async def teach_item(self, item, n, start_rung, pass_mark):
        """One move, taught and practised until the caller chooses to move on. Returns 'passed', 'skipped', 'full', 'restart' or 'stop'."""
        kind, idx = item[0], item[1]; half = item[2] if len(item) > 2 else None
        final = (kind == 'C' and idx == n - 1) or (n == 1 and kind == 'S')
        mult = self.tempo
        if kind == 'H':
            key, pi = choreo.with_half(self.key, idx, half); label = f'STEP {idx + 1}/{n}'; part = f'counts {1 if half == 0 else 5} to {4 if half == 0 else 8}'
        elif kind == 'C':
            key, pi = choreo.with_combined(self.key, idx); label = 'FULL ROUTINE' if idx == n - 1 else f'STEPS 1-{idx + 1}'; part = None
        else:
            key, pi = self.key, idx; label = f'STEP {idx + 1}/{n}'; part = None
        rung = start_rung if kind != 'C' else max(start_rung, 1)
        target = ROUTINE_TARGET if final else MOVE_TARGET
        step = lambda: dict(label=label, rung=rung)
        # 1. explain and demonstrate, slowly, until the caller says they've got it
        self.cmd_floor_t = self.now()                                     # nothing said before this move can skip or stop it
        if kind == 'C':
            t = self.now(); self.say_line(f"Now let's put steps 1 to {idx + 1} together. Nice and slow.", key='step')
            await self.wait_spoken('step', t, 8.0, min_s=1.0)
        else:
            t = self.now()
            first_move = kind == 'H' and idx == 0 and half == 0
            try: card = cues.move_card(key, pi)
            except Exception as e: print(f'[dance] move card failed ({e!r})', flush=True); card = None
            if card is not None:                                         # the full card as a note: Gemini can answer "which foot?"
                self.context(cues.card_note(card, f'{idx + 1}' + (f' ({part})' if part else '')))
                self.move_card_ui = dict(name=card['name'], label=label)
            if kind == 'H':
                lead = f"First move, {part}" if first_move else f"Next move, {part}"
                self.say_line((cues.intro_line(card, lead) if card else lead + '.') + (" Watch me, really slow." if first_move else " Watch first."), key='step')
            elif self.difficulty == 'beginner': self.say_line("Now the whole step, both moves together. Watch.", key='step')
            else: self.say_line((cues.intro_line(card, f"Step {idx + 1}") if card else f"Step {idx + 1}.") + " Watch.", key='step')
            await self.wait_spoken('step', t, 16.0, min_s=1.0)
            feel_said = False
            demos = 0; ans = None
            pose_next = POSE_ON and (not self.first_demo_done or (self.difficulty == 'beginner' and kind == 'H') or self.slow_mode)
            while not self.stop and not self.stop_routine:
                if pose_next:                                                # the first look: one pose at a time, ~3.5 s each
                    pose_next = False; self.first_demo_done = True
                    await self.run_pose_step(key, pi, 'watch', step=step())
                else:
                    await self.run_phrase(pi, RUNGS[rung] * mult, show_repeats=2 if kind == 'H' else 1, step=step(), key=key, stage='WATCH', demo=True)
                    self.first_demo_done = True
                demos += 1
                if self.stop or self.stop_routine: break
                await self.idle(pacing.DEMO_TO_ASK_S)                        # a breath after the demo before the question
                if card is not None and not feel_said:                       # how it should feel, once per move
                    feel_said = True; t = self.now(); self.say_line(card['feel'], key='feel'); await self.wait_spoken('feel', t, 8.0, min_s=1.0)
                ans = await self.wait_choice(self.pick('gotit', ["Got it? Say again to see it once more, or ready to try it with me.",
                                                                 "Make sense? Want to see it again, or ready to try it with me?"]),
                                             key='gotit', reprompt="Take your time. Say again, or ready.", any_reply=True, timeout=25.0, on_timeout='quiet')
                if ans == 'stop': return 'stop'
                if ans == 'skip': return 'skipped'                           # only "skip this move" skips; "next" here means try it
                if ans == 'explain': await self.explain(key, pi); continue
                if ans in ('again', 'no'):
                    if demos >= 2 and rung > 0: rung -= 1
                    continue
                if ans == 'slower':
                    if rung == 0 and POSE_ON: pose_next = True; self.slow_mode = True        # slower than the slowest speed: pose by pose
                    rung = max(0, rung - 1); continue
                break                                                        # yes / ready / next / quiet after the reprompt: try it together
            if self.stop or self.stop_routine: return 'stop'
            halves_ok = kind == 'S' and {(idx, 0), (idx, 1)} <= getattr(self, '_passed_halves', set())
            if POSE_ON and (self.difficulty == 'beginner' or self.slow_mode) and not halves_ok:   # freeze-frame (skipped once both halves passed)
                t = self.now(); self.say_line("Now you. Match each pose and hold it. I'll move on when you've got it.", key='step')
                await self.wait_spoken('step', t, 8.0, min_s=1.0)
                await self.run_pose_step(key, pi, 'freeze', step=step(), focus='arms' if (self.upper_body or self.upper_override) else None)
                if self.stop or self.stop_routine: return 'stop'
                t = self.now(); self.say_line("Nice. Now in time, slow.", key='step'); await self.wait_spoken('step', t, 6.0, min_s=1.0)
            # 2. practise with the ghost, no score
            t = self.now(); self.say_line(self.pick('withme', ["Do it with me. Follow the ghost on your body.", "Together now. Just follow the ghost."]), key='step')
            await self.wait_spoken('step', t, 8.0, min_s=1.0)
            await self.run_phrase(pi, RUNGS[rung] * mult, show_repeats=0, step=step(), key=key, stage='WITH ME', practice=True)
            if self.stop or self.stop_routine: return 'stop'
            if ans == 'quiet':                                           # they went quiet: practising was safe, a scored try waits for them
                a2 = await self.wait_choice("Nice. Ready to try it on your own? Say ready when you are.", key='choice', reprompt="Whenever you're ready, just say ready.")
                while a2 is None and not self.stop and not self.stop_routine: a2 = await self.wait_choice(None, key='choice')
                if a2 in ('stop', None): return 'stop'
                if a2 in ('skip', 'next'): return 'skipped'
            else:
                t = self.now(); self.say_line("Nice. Now on your own, same speed.", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0)
            await self.show_break('BACK TO START', pacing.TURN_GAP_S, key, pi)  # reset to the first pose before your own try
        # 3. the caller's own tries: one actionable tip each, then their choice
        passed_at = -1; fails = 0; tries = 0; halves_done = kind != 'S'; focus = None
        while not self.stop and not self.stop_routine:
            eng = await self.run_phrase(pi, RUNGS[rung] * mult, show_repeats=0, step=step(), key=key, stage='YOUR TURN', ghost_overlay=False, focus=focus)
            if self.stop or self.safety_stop or self.stop_routine: return 'stop'
            if eng.result is None: return 'stop'
            tries += 1; self.attempts = tries
            scored, score, r, cue = self.record_round(eng, item=f'{kind}{idx}{half}')
            self.rounds.append(dict(phrase=pi, kind=kind, tempo=RUNGS[rung] * mult, rung=rung, result=r))
            if not scored:
                self.recap = dict(result=r, cue='Step back so I can see you')
                if not await self.prepare_and_confirm('camera'): self.stop_routine = True; return 'stop'
                continue
            t = self.now(); self.summarise(eng, xp=self.xp_for_attempt(score), item=f'{kind}{idx}{half}')   # grade, up to 3 tips, progress
            await self.wait_spoken('result', t, 10.0, min_s=2.0)
            if self.queued_key is not None:
                self.key = self.queued_key; self.queued_key = None; self.idle_title = self.key['title']; return 'restart'
            passed = score >= pass_mark
            if passed: passed_at = max(passed_at, rung); fails = 0; focus = None
            elif score < 50: fails += 1
            if fails >= 2:                                                   # struggling twice: simplify, never push on
                fails = 0
                if not halves_done:
                    halves_done = True; self.telemetry.event('adapt', action='halves'); await self.practice_halves(key, idx, label); continue
                if rung > 0:
                    rung -= 1; self.telemetry.event('adapt', action='slower', rung=rung)
                    t = self.now(); self.say_line("Let's slow it right down and try again.", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0); continue
                if focus is None:
                    focus = 'arms'; self.telemetry.event('adapt', action='arms_first')
                    t = self.now(); self.say_line("Let's just do the arms this time. The legs can come later.", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0); continue
            if passed and rung >= target:
                q = "That's it! Want to go again, or are you ready for the next move?" if not final else "You've got it! Ready to finish up?"
                yes_means = 'next'
            elif passed:
                q = self.pick('fasterq', ["Nice! Ready to try it a little faster?", "That was solid. Want a little more speed?"])
                yes_means = 'faster'
            elif tries >= 6:
                q = "This one's tricky. Want to keep practising, slow it down, or move on for now?"
                yes_means = 'again'
            elif weak_span(cue) and kind != 'C':                             # loop just the weak counts, slower (the site's promise)
                a_, b_ = weak_span(cue)
                q = f"Want to loop counts {a_ + 1} to {b_ + 1} slower, or go again?"
                yes_means = 'loop'
            else:
                q = self.pick('againq', ["Let's go again. Ready?", "Once more? Say ready when you are."])
                yes_means = 'again'
            ans = await self.wait_choice(q, key='choice', reprompt="Whenever you're ready. Say again, slower, loop it, or next.")
            while ans is None and not self.stop and not self.stop_routine:     # still quiet: keep waiting, never skip ahead
                ans = await self.wait_choice(None, key='choice')
            if ans == 'yes': ans = yes_means
            if ans == 'skip': ans = 'next'
            if ans == 'no': ans = 'again' if yes_means != 'again' else 'wait'
            if ans in ('stop', None): return 'stop'
            if ans == 'next': return 'passed' if passed_at >= 0 else 'skipped'
            if ans == 'full': return 'full'
            if ans == 'again': continue
            if ans == 'wait':
                self.say_line("No rush. Say ready when you are.", key='step')
                ans2 = await self.wait_choice(None, key='choice')
                while ans2 is None and not self.stop and not self.stop_routine: ans2 = await self.wait_choice(None, key='choice')
                if ans2 == 'next': return 'passed' if passed_at >= 0 else 'skipped'
                if ans2 == 'stop': return 'stop'
                continue
            if ans == 'faster':
                if passed_at >= rung and rung < len(RUNGS) - 2:
                    rung += 1; self.telemetry.event('adapt', action='faster', rung=rung)
                    t = self.now(); self.say_line("A little faster. You've got this.", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0)
                else:
                    t = self.now(); self.say_line("Let's nail it at this speed first, then we'll speed up.", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0)
                continue
            if ans == 'slower':
                rung = max(0, rung - 1); t = self.now(); self.say_line("Sure, slower.", key='step'); await self.wait_spoken('step', t, 6.0, min_s=1.0); continue
            if ans == 'simplify':
                if not halves_done: halves_done = True; await self.practice_halves(key, idx, label)
                else:
                    rung = max(0, rung - 1); focus = 'arms'; t = self.now(); self.say_line("Arms only, slower. Legs later.", key='step')
                    await self.wait_spoken('step', t, 6.0, min_s=1.0)
                continue
            if ans == 'explain':
                await self.explain(key, pi)
                await self.run_phrase(pi, RUNGS[0] * mult, show_repeats=1, step=step(), key=key, stage='WATCH', demo=True)
                continue
            if ans == 'loop':
                span = weak_span(cue) or (0, min(3, max(1, eng.n_beats - 1)))
                await self.loop_counts(key, pi, span, rung, label)
                continue
            if ans in ('harder', 'easier'):
                order = ['beginner', 'intermediate', 'advanced']; j = order.index(self.difficulty) + (1 if ans == 'harder' else -1)
                if 0 <= j < 3:
                    ok, reason = self.apply_difficulty(order[j], announce=False)
                    line = (f"{order[j].capitalize()} from the next move on." if ok else
                            f"{order[j].capitalize()} {'unlocks with more XP' if reason else 'is already on'}. We'll keep going.")
                else:
                    line = "That's already the " + ('hardest' if ans == 'harder' else 'easiest') + " level."
                t = self.now(); self.say_line(line, key='step'); await self.wait_spoken('step', t, 6.0, min_s=1.0)
                continue
        return 'stop'

    async def loop_counts(self, key, pi, span, rung, label):
        """Loop just the weak counts, slower: pose by pose (or a slow demo), then with the ghost, then back to the whole move."""
        a, b = span
        k2, p2 = choreo.with_counts(key, pi, a, b)
        slow = max(0, rung - 1); self.telemetry.event('loop', counts=[a + 1, b + 1], rung=slow)
        t = self.now(); self.say_line(f"Counts {a + 1} to {b + 1}, slower. Watch.", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0)
        st = dict(label=f'{label} · COUNTS {a + 1}-{b + 1}')
        if POSE_ON: await self.run_pose_step(k2, p2, 'watch', step=st)
        else: await self.run_phrase(p2, RUNGS[slow] * self.tempo, show_repeats=2, step=st, key=k2, stage='WATCH', demo=True)
        if self.stop or self.stop_routine: return
        t = self.now(); self.say_line("With me, twice.", key='step'); await self.wait_spoken('step', t, 6.0, min_s=1.0)
        for _ in range(2):
            if self.stop or self.stop_routine: return
            await self.run_phrase(p2, RUNGS[slow] * self.tempo, show_repeats=0, step=st, key=k2, stage='WITH ME', practice=True)
        if not (self.stop or self.stop_routine):
            t = self.now(); self.say_line("Now the whole move again.", key='step'); await self.wait_spoken('step', t, 6.0, min_s=1.0)

    async def practice_halves(self, key, section, label):
        """The step in two 4-count halves, very slow: watch each half, try it with the ghost, then the whole step again."""
        t = self.now(); self.say_line("Let's break it into two small pieces, really slow.", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0)
        for h in (0, 1):
            if self.stop or self.stop_routine: return
            k2, hp = choreo.with_half(key, section, h)
            await self.run_phrase(hp, RUNGS[0] * self.tempo, show_repeats=2, step=dict(label=f'{label} · PART {h + 1}'), key=k2, stage='WATCH', demo=True)
            await self.run_phrase(hp, RUNGS[0] * self.tempo, show_repeats=0, step=dict(label=f'{label} · PART {h + 1}'), key=k2, stage='WITH ME', practice=True)
        t = self.now(); self.say_line("Now the whole step again.", key='step'); await self.wait_spoken('step', t, 6.0, min_s=1.0)

    async def warmup(self):
        try: wk = choreo.generate('fitness', 'beginner', n_phrases=1, seed=7)
        except Exception: return
        wk['title'] = 'Warm-up'; saved = self.key
        t = self.now(); self.say_line("Quick eight-count warm-up to get loose, no score. Say skip if you're already warm.", key='warmup')
        await self.wait_spoken('warmup', t, 8.0, min_s=1.0); await self.idle(2.0)          # a moment to say "skip"
        if self.take_cmd(('next', 'skip')) or self.stop: return
        self.in_warmup = True; self.telemetry.event('warmup')
        try: await self.run_phrase(0, 0.7, show_repeats=1, step=dict(label='WARM-UP'), key=wk, stage='WARM-UP', practice=True)
        finally: self.in_warmup = False; self.key = saved; self.recap = None
        if not self.stop:
            t = self.now(); self.say_line("Nice, you're warm. Now the real thing.", key='warmup'); await self.wait_spoken('warmup', t, 6.0, min_s=1.0)

    # ---- battle: three rounds against the reference ghost ----
    async def battle(self):
        """Ghost battle, three rounds against the ghost (the site's promise): round 1 the first phrase, round 2 the next, round 3 a
        speed round (the first phrase again, about 15% faster). An HP bar shows the ghost getting chipped by every clean count; 285 of
        300 is a K.O. Everyone gets a code friends can battle."""
        if not self.in_warmup and self.call_mode == 'solo': await self.warmup()
        n_ph = len([p for p in self.key['phrases'] if not p.get('combined') and 'half' not in p and 'loop' not in p])
        tempo = (self.key.get('default_tempo') or LEVEL_CFG[self.difficulty]['tempo']) * self.tempo
        plan = [(0, tempo, False), (min(1, n_ph - 1), tempo, False), (0, round(min(tempo * 1.15, 1.2), 3), False)]
        # the win line: against Brodie's ghost the level's pass mark three times; against a friend's ghost, beat their total
        ch = self.challenge if (self.challenge and self.code) else None
        target = float(ch.get('total') or 0) + 0.5 if ch else float(os.environ.get('DANCE_WIN_TOTAL') or WIN_TOTAL[self.difficulty])
        total = 0.0; self.battle_hud = dict(ghost_hp=target, ghost_max=target, round=1, rounds=3, round_floor=max(0.0, target - 100.0))
        best_r = None; self.ghost_rounds = []; self.battle_won = False
        for i, (pi, tmp, freestyle) in enumerate(plan):
            if self.stop: break
            self.battle_hud.update(round=i + 1, round_floor=max(0.0, target - total - 100.0))
            t = self.now(); self.say_line(f"Round {i + 1} of 3!" + (" Speed round: same move, faster. Beat the ghost!" if i == 2 else " Follow the ghost."), key='round')
            await self.wait_spoken('round', t, 8.0, min_s=1.5)
            eng = await self.run_phrase(pi, tmp, show_repeats=1 if i else 2, freestyle=freestyle, step=dict(label=f'ROUND {i + 1}/3' + (' · SPEED' if i == 2 else '')), stage='CHALLENGE')
            if eng.result is None: break
            scored, s, r, cue = self.record_round(eng, item=f'R{i}')
            total += s if scored else 0.0; self.battle_hud['ghost_hp'] = max(0.0, target - total)
            if scored and (best_r is None or s > (best_r.get('score') or 0)): best_r = r
            if i < 2 and scored and isinstance(eng, PhraseEngine):         # your ghost: rounds 1 and 2 as body points (kept only on a yes)
                try:
                    arr = ghost_mod.live_to_arrays(eng)
                    if arr is not None: self.ghost_rounds.append(arr)
                except Exception as e: print(f'[dance] ghost record skipped: {e!r}', flush=True)
            self.rounds.append(dict(round=i + 1, phrase=pi, tempo=tmp, result=r))
            t = self.now(); self.summarise(eng, round_label=f"Round {i + 1} of 3. ", xp=self.xp_for_attempt(s) if scored else None)
            await self.wait_spoken('result', t, 10.0, min_s=2.0)
            if self.take_cmd(('stop',)) or self.stop_routine: break
            if i < len(plan) - 1:                                            # a 4 s breather, showing the next round's first pose
                await self.show_break(f'ROUND {i + 2}', pacing.ROUND_GAP_S, self.key, plan[i + 1][0])
        won = total >= target and len(self.rounds) == 3; self.battle_won = won
        self.fx.add('ko', self.now(), text='K.O.!' if won else 'GHOST WINS'); self.sfx('ko' if won else 'stamp')
        self.extras['battle_done'] = len(self.rounds) == 3
        self.battle_target = target
        await self.finish_session('battle', total=total, best_r=best_r)

    async def daily(self):
        """Today's routine, same for everyone: learn the first two steps, then ONE shot at today's board (the full routine at 0.9x;
        that run alone is posted and ranked). After the day's shot it's practice: the same flow, nothing posted."""
        if not self.key.get('daily'): self.key = choreo.daily()
        self.extras['daily'] = True
        await self.warmup()
        saved_plan = self.learn_plan
        n_ph = len([p for p in self.key['phrases'] if not p.get('combined') and 'half' not in p and 'loop' not in p])
        self.learn_plan = lambda key: ([('S', 0, None), ('S', 1, None), ('C', 1, None)], 2) if n_ph >= 2 else ([('S', 0, None)], 1)
        try:
            await self.learn(mode_name='daily', skip_warmup=True, finish=False, challenge=False)
        finally:
            self.learn_plan = saved_plan
        if self.stop or self.safety_stop: return
        shot = None
        if not self.stop_routine:
            line = ("Practice run of today's routine. No pressure, it won't go on the board." if self.practice_run else
                    "Your one shot at today's board. The full routine, one take. Make it count!")
            t = self.now(); self.say_line(line, key='round'); await self.wait_spoken('round', t, 8.0, min_s=1.5)
            key, pi = choreo.with_combined(self.key, min(1, n_ph - 1)) if n_ph >= 2 else (self.key, 0)
            eng = await self.run_phrase(pi, RUNGS[3] * self.tempo, show_repeats=1, step=dict(label='DAILY · PRACTICE' if self.practice_run else 'DAILY · YOUR SHOT'),
                                        key=key, stage='CHALLENGE', ghost_overlay=False)
            if eng.result is not None:
                self.record_round(eng, item='daily_shot'); self.rounds.append(dict(phrase=pi, kind='DAILY', tempo=RUNGS[3] * self.tempo, result=eng.result))
                shot = eng.result if eng.result.get('scored') else None
        if shot is None and not self.practice_run:                         # the shot didn't count (camera): nothing goes on the board
            self.practice_run = True; self.telemetry.event('daily_shot_unscored')
        await self.finish_session('daily', best_r=shot)

    async def squad_battle(self):
        """Squad battle: every dancer in frame against the ghost and each other, two phrases plus freestyle; MVP, crew sync, one crew code."""
        n_ph = len([p for p in self.key['phrases'] if not p.get('combined')])
        tempo = (self.key.get('default_tempo') or LEVEL_CFG[self.difficulty]['tempo']) * self.tempo
        plan = [(0, tempo, False), (min(1, n_ph - 1), tempo, False), (0, tempo, True)]
        totals = {}; sums = {}; danced = {}; last_sq = None
        self.say_line("Squad battle! Two big steps between each of you. Everyone follows the ghost, and the highest total wins.", key='round')
        await self.idle(3.5)
        for i, (pi, tmp, freestyle) in enumerate(plan):
            if self.stop: break
            self.say_line(f"Round {i + 1} of 3!" + (" Freestyle!" if freestyle else ''), key='round')
            await self.idle(2.0)
            eng = await self.run_phrase(pi, tmp, show_repeats=1, freestyle=freestyle, step=dict(label=f'SQUAD ROUND {i + 1}/3'), stage='CHALLENGE')
            if eng.result is None: break
            sq = eng.squad if isinstance(eng, SquadEngine) else None
            if sq:
                last_sq = sq
                for x in sq['players']:                               # a round someone stepped out of is unscored for them, not a zero
                    if x.get('scored', x.get('score') is not None):
                        sums[x['label']] = sums.get(x['label'], 0.0) + (x['score'] or 0.0); danced[x['label']] = danced.get(x['label'], 0) + 1
            self.record_round(eng, item=f'Q{i}')
            self.rounds.append(dict(round=i + 1, phrase=pi, squad={k: v for k, v in (sq or {}).items() if k in ('ranking', 'crew_sync', 'crew_mean', 'squad_title')}))
            self.summarise(eng, round_label=f"Squad round {i + 1} of 3. ")
            await self.idle(4.5)
        self.extras['squad'] = True
        totals = {lab: 3.0 * sums[lab] / danced[lab] for lab in sums if danced.get(lab)}   # ranked by the mean of the rounds each one danced
        await self.finish_squad(totals, last_sq)

    def start_squad(self):
        pairs = self.tracker.assign_colours(self.mirror_view)
        self.colours = dict(pairs)
        if self.ltr_names:
            for (_, c), nm in zip(pairs, self.ltr_names): self.names[c] = nm
        self.squad_players = [(pid, c, self.names.get(c)) for pid, c in pairs]
        self.telemetry.event('squad_mode_started', n=len(pairs), colours=[c for _, c in pairs], mode=self.mode)
        if pairs:
            self.context("Squad mode: dancers are tagged by colour, left to right as they see themselves: " + ', '.join(c for _, c in pairs) +
                         ". Dancers are addressed by colour or name only; nobody's looks are described.")

    def refresh_squad(self):
        """Keep colours for dancers still here, give new ones the next free colour, drop the ones who left."""
        present = {pid for pid, _, _ in self.players}
        keep = [(pid, c, self.names.get(c)) for pid, c, _ in self.squad_players if pid in present]
        used = {c for _, c, _ in keep}
        for pid, _, _ in self.players:
            if pid not in {p for p, _, _ in keep}:
                c = next((c for c in ('volt', 'pink', 'cyan', 'amber') if c not in used), None)
                if c: keep.append((pid, c, self.names.get(c))); used.add(c); self.colours[pid] = c; self.say_line(f"{c.capitalize()}, you're in!", key='squad')
        self.squad_players = keep
        if len(keep) < 2:
            self.call_mode = 'solo'; self.say_line("Just one dancer now, so we'll go solo.", key='squad')
            self.telemetry.event('squad_to_solo')

    async def class_lead(self):
        """The site's class: Brodie leads, the room follows, everyone gets scored. Everyone in frame dances together (the squad
        engine with 2+ dancers; one dancer is a class of one): each step pose by pose, then slow with the ghost, then the whole
        routine together, scored per dancer, with a class leaderboard on the card."""
        n_ph = len([p for p in self.key['phrases'] if not p.get('combined') and 'half' not in p and 'loop' not in p])
        n_steps = max(1, min(2, n_ph, int(os.environ.get('DANCE_MAX_PHRASES', '0') or 0) or 2))
        many = self.call_mode == 'squad' and len(self.squad_players) >= 2
        t = self.now()
        self.say_line("Class is in session! I'll lead, you all follow. Watch first." if many else "Class of one! I'll lead, you follow. Watch first.", key='round')
        await self.wait_spoken('round', t, 8.0, min_s=1.5)
        self.stop_routine = False
        for st in range(n_steps):
            if self.stop or self.stop_routine: break
            label = f'CLASS · STEP {st + 1}/{n_steps}'
            t = self.now(); self.say_line(f"Step {st + 1}. Watch me, one pose at a time.", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0)
            if POSE_ON: await self.run_pose_step(self.key, st, 'watch', step=dict(label=label))
            else: await self.run_phrase(st, RUNGS[0] * self.tempo, show_repeats=1, step=dict(label=label), stage='WATCH', demo=True)
            if self.stop or self.stop_routine: break
            t = self.now(); self.say_line("Now with me, slow. Follow the ghost.", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0)
            await self.run_phrase(st, RUNGS[1] * self.tempo, show_repeats=0, step=dict(label=label), stage='WITH ME', practice=True)
        if not (self.stop or self.stop_routine):
            key, pi = choreo.with_combined(self.key, n_steps - 1) if n_steps > 1 else (self.key, 0)
            t = self.now(); self.say_line("Now all together, the whole routine. Give it everything!", key='step'); await self.wait_spoken('step', t, 8.0, min_s=1.0)
            eng = await self.run_phrase(pi, RUNGS[2] * self.tempo, show_repeats=1, step=dict(label='CLASS · FULL ROUTINE'), key=key, stage='CHALLENGE')
            if eng.result is not None:
                self.record_round(eng, item='class'); self.rounds.append(dict(phrase=pi, kind='CLASS', tempo=RUNGS[2] * self.tempo, result=eng.result))
                t = self.now(); self.summarise(eng, round_label='Class. '); await self.wait_spoken('result', t, 10.0, min_s=2.0)
        if self.call_mode == 'squad' and self._last_squad:
            totals = {x['label']: 3.0 * (x['score'] or 0.0) for x in self._last_squad['players']}
            await self.finish_squad(totals, self._last_squad, mode='class')
        else:
            await self.finish_session('class')

    # ---- end of a run: XP, level-up, achievements, the scorecard ----
    async def finish_session(self, mode, total=None, best_r=None):
        scored_rounds = [x for x in self.rounds if (x.get('result') or {}).get('scored')]
        best_r = best_r or max((x['result'] for x in scored_rounds), key=lambda r: r.get('score') or 0, default=None)
        best = float((best_r or {}).get('score') or 0)
        if (self.stop and not scored_rounds) or not self.rounds:            # nothing was danced: no finish pose, XP or card
            self.final_card = None
            if not self.stop and not self.safety_stop: self.say_line("No worries, we'll dance next time. Just say when.", key='wrap')
            return
        score100 = (total / 3.0) if total is not None else best
        photo = await self.finish_pose()
        if mode == 'daily' and self.practice_run:
            posted = None                                                  # practice after the day's shot: nothing goes on the board
        else:
            win_only = mode == 'battle' and not getattr(self, 'battle_won', False)       # the site: "win and you get a code"
            posted = await self.backend.call('POST', '/dance/result', dict(name=self.name, key=self.key['name'], mode=mode, total=total if total is not None else best,
                                                                          score100=round(score100, 1), challenge=self.code if self.challenge else None,
                                                                          issue_code=not win_only,
                                                                          rounds=[dict(phrase=x.get('phrase'), tempo=x.get('tempo'), score=(x.get('result') or {}).get('score')) for x in self.rounds]))
            if mode == 'daily' and posted: self.daily_done = True
        self.my_code = (posted or {}).get('code'); self.ghost_token = (posted or {}).get('ghost_token')
        versus = (posted or {}).get('versus'); winner = (posted or {}).get('winner'); board = (posted or {}).get('board')
        if versus and winner and winner != 'tie' and winner == (posted or {}).get('name'): self.extras['duel_win'] = True
        daily_rank = None
        if mode == 'daily' and self.my_code:
            midnight = time.mktime(datetime.date.today().timetuple())
            lb = await self.backend.call('GET', f"/dance/leaderboard?key={self.key['name']}&mode=daily&since={midnight:.0f}&limit=10000") or []
            daily_rank = next((i + 1 for i, row in enumerate(lb) if row.get('code') == self.my_code), None)
            if daily_rank: board = dict(rank=daily_rank, n=len(lb), owner="today's daily")
            if lb and not self.stop: await self.show_board(lb, mine=self.my_code, seconds=8.0)
        aw = await self.award(mode)
        title = progression.title_for(best_r or {}, dict(duel_win=self.extras.get('duel_win'), glow_up=any(x.get('glow_up') for x in self.session_rounds)))
        brag = await self.brag_lines(score100, board, mode)
        line = f"Session over ({mode}). Best step {best:.0f}, grade {progression.grade(best)}, title '{title}'."
        if total is not None:
            tgt = getattr(self, 'battle_target', WIN_TOTAL[self.difficulty])
            line += f" Battle total {total:.0f} of 300; a win needed {tgt:.0f}" + (", so it's a K.O. and they earned a code." if getattr(self, 'battle_won', False) else ", so no code this time.")
        if versus: line += f" They challenged {versus['name']} who had {float(versus.get('score100') or versus['total']):.0f}; the winner is {winner}."
        if board and board.get('rank'): line += f" They are #{board['rank']} of {board['n']} on {board.get('owner')}'s board."
        if aw: line += f" XP +{aw.get('gained', 0)}, level {aw.get('level_name')}" + (" (LEVEL UP)" if aw.get('leveled_up') else '') + \
                       (f", new: {', '.join(a['name'] for a in aw.get('new_achievements', []))}" if aw.get('new_achievements') else '') + '.'
        if self.my_code: line += f" Their challenge code is {self.my_code}."
        self.context(line)
        digits = ' '.join(str(self.my_code)) if self.my_code else None
        bits = [f"That's a wrap! You're a {title}."]
        if aw and aw.get('leveled_up'): bits.append(f"Level up: you're a {aw.get('level_name')} now!")
        elif aw and aw.get('gained'): bits.append(f"Plus {aw['gained']} XP.")
        if digits: bits.append(f"Your code is {digits}. Screenshot this and send it to the friend who thinks they can out-dance you.")
        elif mode == 'battle' and total is not None:
            bits.append(f"So close! You needed {getattr(self, 'battle_target', WIN_TOTAL[self.difficulty]):.0f} to beat the ghost and earn a code.")
        bits.append("Say rematch to go again.")
        self.say_line(' '.join(bits), key='wrap', ttl=20)
        perfects = sum(1 for row in (best_r or {}).get('per_beat', []) if row.get('verdict') == 'green') if best_r and best_r.get('scored') else None
        self.final_card = dict(result=best_r, key=self.key['name'], title=self.key.get('title'), code=self.my_code, award_title=title, xp=aw,
                               achievements=(aw or {}).get('new_achievements', []), handle=self.handle, mode_label=MODE_LABEL.get(mode, mode.title()),
                               perfects=perfects, total_line=(f"{total:.0f}/300" if total is not None else None),
                               lines=([f"vs {versus['name']}: winner {winner}"] if versus else []), photo=photo, brag=brag)
        self.log.append(dict(t=round(self.now() - self.t0, 2), event='result_posted', code=self.my_code, versus=versus, winner=winner, xp=(aw or {}).get('gained')))
        await self.idle(9.0)
        if mode == 'battle' and self.my_code and len(getattr(self, 'ghost_rounds', []) or []) >= 2 and not self.stop:
            await self.offer_ghost_save(total)

    def board_rows(self, lb, mine=None):
        return [dict(rank=i + 1, name=r.get('name'), score=r.get('score100') if r.get('score100') is not None else r.get('total'),
                     me=bool(mine) and r.get('code') == mine) for i, r in enumerate(lb)]

    async def show_board(self, lb=None, mine=None, seconds=8.0, wait=True):
        """Today's daily board (top five; the caller's own row in volt, added below if they're further down) with one spoken line."""
        if lb is None:
            midnight = time.mktime(datetime.date.today().timetuple())
            lb = await self.backend.call('GET', f"/dance/leaderboard?key={choreo.daily()['name']}&mode=daily&since={midnight:.0f}&limit=10000") or []
        rows = self.board_rows(lb, mine); me = next((r for r in rows if r['me']), None)
        top = rows[:5]
        if me is not None and me not in top: top = rows[:4] + [me]
        self.board_ui = dict(title="Today's daily", rows=top, n=len(rows))
        self.telemetry.event('board_shown', n=len(rows), rank=me['rank'] if me else None)
        if me: line = f"You're number {me['rank']} of {len(rows)} today!" + (" Top of the board!" if me['rank'] == 1 else '')
        elif rows: line = f"Here's today's board. {str(rows[0]['name']).split()[0]} leads with {float(rows[0]['score'] or 0):.0f}."
        else: line = "Nobody's on today's board yet. Take the daily and you're first!"
        t = self.now(); self.say_line(line, key='board', ttl=10); self.board_until = t + seconds
        if not wait: return                                            # the running loop keeps drawing it until board_until
        try:
            while self.now() - t < seconds and not self.stop: await self.frame_step(None); await self.sleep(1.0 / self.fps)
        finally:
            self.board_ui = None

    async def offer_ghost_save(self, total):
        """A winner's ghost is saved only with a yes (body points, never video; 30 days; "forget me" deletes it)."""
        ans = await self.wait_choice("Want me to save your dance as a ghost, so friends with your code battle the real you? "
                                     "Just body points, no video. Say yes, or raise both hands.", key='ghostsave', seconds=12.0,
                                     reprompt="Save your ghost for your friends? Yes or no.", timeout=12.0, on_timeout='no')
        if ans != 'yes':
            self.telemetry.event('ghost_saved', ok=False); self.ghost_rounds = []
            self.say_line("No problem. Your code still works with my version of the routine.", key='wrap'); return
        rounds = self.ghost_rounds[:2]; self.ghost_rounds = []
        tempo = (self.key.get('default_tempo') or LEVEL_CFG[self.difficulty]['tempo']) * self.tempo
        r = await self.backend.call('POST', f'/dance/ghost/{self.my_code}', dict(token=getattr(self, 'ghost_token', None), src_key=self.key['name'], bpm=float(self.key['bpm']) * tempo,
                                                                               per_beat=4, data=ghost_mod.pack(rounds), total=float(total or 0),
                                                                               score100=float(total or 0) / 3.0, difficulty=self.difficulty))
        ok = bool(r and r.get('ok')); self.telemetry.event('ghost_saved', ok=ok)
        self.say_line("Saved! Anyone with your code now battles your ghost." if ok else "Hmm, I couldn't save it this time, but your code still works.", key='wrap')

    async def finish_pose(self):
        """'Strike your finish pose!' with a 3-2-1 on screen; returns that camera frame for the share card (kept only in memory,
        drawn into the live video, never saved)."""
        if self.stop or self.safety_stop: return None
        self.say_line(self.pick('pose', ["Strike your finish pose! Three, two, one!", "Finish pose, hold it! Three, two, one!"]), key='pose', ttl=6)
        for n in (3, 2, 1):
            self.countdown_ui = n; await self.idle(0.9)
        self.countdown_ui = None
        photo = None if self.last_bgr is None else self.last_bgr.copy()
        self.fx.add('flash', self.now()); self.telemetry.event('finish_pose', captured=photo is not None)
        return photo

    async def brag_lines(self, score100, board, mode):
        """Up to two short brag lines for the card: a new personal best, the daily rank, or 'Top N% today' on this routine."""
        out = []
        try:
            if self.hist_best is not None and score100 > self.hist_best + 0.5:
                out.append('NEW PERSONAL BEST'); self.fx.add('ko', self.now() + 0.4, text='PERSONAL BEST!'); self.sfx('levelup')
            if mode == 'daily' and board and board.get('rank'):
                out.append(f"#{board['rank']} TODAY")
            else:
                lb = await self.backend.call('GET', f"/dance/leaderboard?key={self.key['name']}&limit=200") or []
                day = time.time() - 86400; rows = [r for r in lb if (r.get('created') or 0) >= day]
                if len(rows) >= 3 and self.my_code:
                    rank = next((i + 1 for i, r in enumerate(rows) if r.get('code') == self.my_code), None)
                    if rank:
                        pct = max(1, int(round(100.0 * rank / len(rows))))
                        if pct <= 50: out.append(f"TOP {pct}% TODAY")
        except Exception as e:
            print(f'[dance] brag lines skipped: {e!r}', flush=True)
        return out[:2]

    async def award(self, mode, name=None, rounds=None, style=None):
        """XP for the run: saved to the caller's profile when we know their name; level-up and achievements as overlays and sounds."""
        name = name or self.name; rounds = self.session_rounds if rounds is None else rounds
        first_today = not self.profile or (self.profile or {}).get('last_day') != datetime.date.today().isoformat()
        xp, br = progression.xp_for_session(rounds, mode, self.difficulty, dict(self.extras, first_today=first_today))
        sess = dict(rounds=rounds, **{k: v for k, v in self.extras.items() if isinstance(v, bool)})
        st = style or choreo.STYLE_CODE.get(self.style or '', self.key.get('style') or 'hiphop')
        if name:
            aw = await self.backend.call('POST', f'/dance/profile/{name}/award', dict(xp=xp, style=st, session=sess))
        else:
            aw = None
        if not aw:
            lb = progression.level_for(self.xp); la = progression.level_for(self.xp + xp)
            aw = dict(gained=xp, xp_before=self.xp, xp_after=self.xp + xp, level_before=lb[0], level_after=la[0], level_name=la[1], level_floor=la[2], next_at=la[3],
                      leveled_up=la[0] > lb[0], new_achievements=progression.new_achievements([], dict(sess, styles={st})),
                      newly_unlocked=sorted(progression.unlocked(la[0]) - progression.unlocked(lb[0])), saved=False, streak=1)
        if name and self is not None and name == self.name:
            self.xp = int(aw.get('xp_after') or self.xp); self.level = int(aw.get('level_after') or self.level)
            if self.profile is not None: self.profile['last_day'] = datetime.date.today().isoformat()
        t = self.now()
        if aw.get('leveled_up'):
            self.fx.add('levelup', t + 0.3, level_name=aw.get('level_name'), unlocks=aw.get('newly_unlocked') or []); self.sfx('levelup')
        for k, a in enumerate(aw.get('new_achievements') or []):
            self.fx.add('achievement', t + 2.8 + 0.9 * k, name=a['name']); self.sfx('achievement')
        self.telemetry.event('award', xp=xp, breakdown=br, leveled_up=aw.get('leveled_up'), achievements=[a['id'] for a in aw.get('new_achievements') or []])
        return aw

    async def finish_squad(self, totals, last_sq, mode='squad_battle'):
        ranking = sorted(totals.items(), key=lambda kv: -kv[1])
        crew_mean = float(np.mean(list(totals.values()))) / 3.0 if totals else 0.0
        names = [lab for lab, _ in ranking]
        posted = await self.backend.call('POST', '/dance/result', dict(name=' & '.join(names) or 'crew', key=self.key['name'], mode=mode, total=crew_mean * 3,
                                                                      score100=round(crew_mean, 1), challenge=self.code if self.challenge else None,
                                                                      rounds=[dict(player=lab, total=round(v, 1)) for lab, v in ranking]))
        self.my_code = (posted or {}).get('code'); self.ghost_token = (posted or {}).get('ghost_token')
        for pid, c, nm in self.squad_players:                    # XP to every named dancer
            if nm: await self.award(mode, name=nm, rounds=self.session_rounds)
        sq = dict(last_sq or {})
        if ranking:
            labels = {(x.get('name') or x['colour'].upper()): x for x in (last_sq or {}).get('players', [])}
            sq['ranking'] = [dict(colour=(labels.get(lab) or {}).get('colour', 'volt'), name=(labels.get(lab) or {}).get('name'), rank=i + 1,
                                  grade=progression.grade(v / 3.0), score=round(v / 3.0, 1), title=(labels.get(lab) or {}).get('title', '')) for i, (lab, v) in enumerate(ranking)]
        self.context(f"Squad battle over. Totals: {', '.join(f'{lab} {v:.0f}' for lab, v in ranking)}. Crew average {crew_mean:.0f}. "
                     f"Squad title {sq.get('squad_title', '')}. Crew code {self.my_code}: another crew can battle it.")
        win = ranking[0][0].capitalize() if ranking else 'The crew'
        if mode == 'class':
            self.say_line(f"Class dismissed! Top of the class: {win}." + (f" Your class code is {' '.join(str(self.my_code))}. Another crew can battle it." if self.my_code else ''), key='wrap', ttl=20)
        else:
            self.say_line(f"{win} wins! You're {sq.get('squad_title', 'a crew')}." + (f" Your crew code is {' '.join(str(self.my_code))}. Screenshot this and challenge another crew." if self.my_code else ''), key='wrap', ttl=20)
        self.final_card = dict(result=None, key=self.key['name'], title=self.key.get('title'), code=self.my_code, squad=sq, handle=self.handle, mode_label=MODE_LABEL.get(mode, 'Squad'))
        await self.idle(9.0)

    # ---- telephone ----
    async def record_phrase(self, n_beats=8, countin_beats=4, bpm=None, label='RECORDING'):
        """Telephone and class recording: record the caller's own eight counts to a metronome and build a key from them (features,
        mask, energy, skeleton on the beat grid). Runs in ACTIVITY ("stop" ends it, "pause" restarts it). Returns the key dict, or
        None when the camera saw too little or it was stopped."""
        from team.dance.features import features_from_landmarks, visibility_mask, hip_energy, skeleton2d_normalized, resample_to_grid, resample_mask
        from team.dance.reference import key_from_arrays
        bpm = float(bpm or self.key['bpm']) * self.tempo; beat_s = 60.0 / bpm; pb = 4
        self.stage = 'RECORD'; self.practice = True; self.transition('ACTIVITY', 'record')
        bank = getattr(self.job, 'bank', None)
        if bank is not None: dance_audio.register_phrase_track(bank, bpm, 0, countin_beats, n_beats + 1, voice={})
        t_start = self.now(); self.say(dance_audio.TRACK, 'announce')
        rows = []; prev = None; prev_t = None; t_attempt = t_start + countin_beats * beat_s; stopped = False
        holder = getattr(self, 'engine_holder', None)
        if holder is not None:                       # the simulator's stand-in caller follows this clock, like an engine's
            class _RecClock:
                state = 'countin'; attempt_s = (n_beats + 1) * beat_s
                def attempt_t(_s, t): return t - t_attempt - self.rtt_s
            rc = _RecClock(); holder['engine'] = rc
        hud_base = dict(state='countin', key='your move', title='your move', phrase=0, n_beats=n_beats, current_beat=-1, countdown=None, verdicts=[None] * n_beats,
                        running_score=None, combo=0, result=None, cue=None, bpm=round(bpm, 1), tempo=self.tempo, fx=self.fx, step=dict(label=label), stage='RECORD')
        while self.now() < t_attempt + (n_beats + 1) * beat_s and not self.stop:
            c = self.take_cmd(('stop', 'pause'))
            if c or self.safety_stop or self.pause_req:
                self.pause_req = False; self.cut_audio(); stopped = True; break
            if holder is not None: holder['engine'].state = 'attempt' if self.now() >= t_attempt else 'countin'
            frame, t = self.newest_frame()
            if frame is not None:
                try:
                    bgr, lm, w = await self.prepare(frame, int((self.now() - self.t0) * 1000))
                    tp = self.now() - t_attempt - self.rtt_s
                    if lm is not None and tp >= 0:
                        rows.append((tp, features_from_landmarks(lm, w), visibility_mask(lm).astype(np.float32), hip_energy(prev, lm, (t - prev_t) if prev_t else 0.0), skeleton2d_normalized(lm)))
                        prev = lm; prev_t = t
                    hud = dict(hud_base); b = (self.now() - t_attempt) / beat_s; hud['t'] = self.now()
                    if b < 0:
                        hud['state'] = 'countin'; hud['countdown'] = max(1, int(np.ceil(-b))); hud['countin_label'] = max(5, min(8, 8 - int(np.ceil(-b)) + 1)); hud['beat_pos'] = b
                    else: hud['state'] = 'attempt'; hud['current_beat'] = min(n_beats - 1, int(b))      # no fake verdicts: nothing is judged here
                    await self.render_push(bgr, lm, hud, ghost=None, extra=dict(t=self.now()))
                except Exception as e:
                    print(f'[dance] record frame skipped: {e!r}', flush=True)
            await self.sleep(1.0 / self.fps)
        self.practice = False
        if not self.safety_stop: self.transition('REVIEW', 'recorded')
        if stopped or len(rows) < n_beats * 2: return None
        grid = np.arange(n_beats * pb) * beat_s / pb
        ts = [r[0] for r in rows]
        F = resample_to_grid(ts, [r[1] for r in rows], grid); M = resample_mask(ts, [r[2] for r in rows], grid)
        E = resample_to_grid(ts, np.asarray([r[3] for r in rows], np.float32), grid)[:, 0]
        S = resample_to_grid(ts, np.asarray([r[4] for r in rows], np.float32).reshape(len(rows), -1), grid).reshape(len(grid), 33, 2)
        if M.mean() < 0.6: return None
        return key_from_arrays(f"chain_{self.name or 'anon'}", bpm / self.tempo, F, M=M, E=E, S=S, per_beat=pb, counts_per_phrase=n_beats, source='telephone', save=False)

    async def say_wait(self, text, key='step', max_s=10.0, min_s=1.0):
        t = self.now(); self.say_line(text, key=key, ttl=max(8.0, max_s)); await self.wait_spoken(key, t, max_s, min_s=min_s)

    async def show_compare(self, a_key, b_key, la='LAST DANCER', lb='YOU', seconds=5.0):
        """"Watch it change": the previous link's move and the caller's own, side by side, looping, for a few seconds."""
        try:
            A = np.asarray(a_key['skeleton'], np.float32); B = np.asarray(b_key['skeleton'], np.float32)
        except Exception:
            return
        if not len(A) or not len(B): return
        self.compare_ui = dict(a=A, b=B, la=la, lb=lb, t0=self.now(), beat_s=60.0 / float(a_key.get('bpm') or 100), pb=int(a_key.get('per_beat') or 4))
        self.telemetry.event('compare', frames=int(min(len(A), len(B))))
        try: await self.idle(seconds)
        finally: self.compare_ui = None

    async def telephone(self):
        """Dance telephone: copy a move, pass it on, watch it change. No code: the caller invents an eight-count move, it becomes
        link 1 and they get the chain code to pass on. With a code: they copy the last link's move, get scored against it and
        against the original, see both moves side by side, and become the next link (the code stays the same)."""
        if not self.code:
            await self.say_wait("Dance telephone! Make up any eight-count move. I'll count you in, and I'll remember it.", key='tel')
            key = await self.record_phrase(label='YOUR MOVE')
            if key is None:
                if not self.stop and not self.stop_routine: await self.say_wait("I couldn't catch that move. Step back so I can see you, and say telephone to try again.", key='tel')
                return
            posted = await self.backend.call('POST', '/dance/chain', dict(key=self.key['name'], link=dict(name=self.name or 'anonymous', key=key, score=100.0, fidelity_prev=None, fidelity_origin=None)))
            self.my_code = (posted or {}).get('code'); self.ghost_token = (posted or {}).get('ghost_token')
            self.rounds.append(dict(mode='telephone', link=1, code=self.my_code))
            self.context(f"The caller's move is link 1 of a new telephone chain, code {self.my_code}.")
            digits = ' '.join(str(self.my_code)) if self.my_code else None
            self.say_line(f"Got your move! Your chain code is {digits}. Send it to a friend: they copy your move, then pass it on, and we watch it change." if digits
                          else "Got your move, but I couldn't save the chain just now. Try again in a minute.", key='code', ttl=20)
            self.final_card = dict(result=None, title='telephone, link 1', code=self.my_code, lines=['pass this code on: copy the move, keep the chain going'],
                                   handle=self.handle, mode_label='Telephone')
            await self.wait_spoken('code', self.now() - 0.01, 14.0, min_s=6.0); return
        chain = await self.backend.call('GET', f'/dance/chain/{self.code}')
        if not chain or chain.get('error') or not chain.get('links'):
            await self.say_wait(f"Code {' '.join(str(self.code))} doesn't match a telephone chain. Say the code again, one digit at a time.", key='tel'); return
        links = chain['links']; prev_key = links[-1]['key']; origin_key = links[0]['key']
        self.key = prev_key; self.key.setdefault('title', 'telephone')
        await self.say_wait(f"This is link {len(links) + 1}. Watch the last dancer's move twice, then copy it.", key='tel')
        eng = await self.run_phrase(0, self.tempo, step=dict(label=f'LINK {len(links) + 1}'), stage='CHALLENGE')
        if eng.result is None: return
        r = eng.result or {}
        fidelity_prev = r.get('score', 0.0) if r.get('scored') else 0.0
        fidelity_origin = None
        if r.get('scored') and eng.live:
            from team.dance.scoring import score_phrase
            from team.dance.features import resample_to_grid, resample_mask
            from team.dance.reference import phrase_slice
            oF, oM, oE, _ = phrase_slice(origin_key, 0); pb = eng.pb
            grid = np.arange(len(oF) + pb) * eng.beat_s / pb; ts = [x[0] for x in eng.live]
            F = resample_to_grid(ts, [x[1] for x in eng.live], grid); M = resample_mask(ts, [x[2].astype(np.float32) for x in eng.live], grid)
            E = resample_to_grid(ts, np.asarray([x[3] for x in eng.live], np.float32), grid)[:, 0]
            fidelity_origin = score_phrase(F, M, E, oF, oM, oE, per_beat=pb).get('score')
        my_key = None
        if r.get('scored') and eng.live:
            from team.dance.reference import key_from_arrays
            from team.dance.features import resample_to_grid, resample_mask
            pb = eng.pb; grid = np.arange(eng.n_key) * eng.beat_s / pb; ts = [x[0] for x in eng.live]
            F = resample_to_grid(ts, [x[1] for x in eng.live], grid); M = resample_mask(ts, [x[2].astype(np.float32) for x in eng.live], grid)
            E = resample_to_grid(ts, np.asarray([x[3] for x in eng.live], np.float32), grid)[:, 0]
            if eng.live_skel:                        # the next link's ghost is THIS dancer's own skeleton
                st = [x[0] for x in eng.live_skel]
                S = resample_to_grid(st, np.asarray([x[1] for x in eng.live_skel], np.float32).reshape(len(st), -1), grid).reshape(len(grid), 33, 2)
            else:
                S = np.asarray(prev_key['skeleton'], np.float32)[:eng.n_key]
            my_key = key_from_arrays(f"chain_{self.name or 'anon'}_{len(links) + 1}", float(prev_key['bpm']), F, M=M, E=E, S=S, per_beat=pb, counts_per_phrase=eng.n_beats, source='telephone', save=False)
        posted = await self.backend.call('POST', f'/dance/chain/{self.code}/link', dict(link=dict(name=self.name or 'anonymous', key=my_key or prev_key, score=r.get('score'), fidelity_prev=fidelity_prev, fidelity_origin=fidelity_origin)))
        link_no = (posted or {}).get('link_no') or len(links) + 1
        self.rounds.append(dict(mode='telephone', link=link_no, fidelity_prev=fidelity_prev, fidelity_origin=fidelity_origin, result=r))
        self.context(f"Telephone link {link_no}: {fidelity_prev:.0f} of 100 to the last dancer, "
                     f"{'unknown' if fidelity_origin is None else round(fidelity_origin)} to the original. The code stays {self.code}.")
        if my_key is not None: await self.show_compare(prev_key, my_key)
        orig = '' if fidelity_origin is None else f" and {fidelity_origin:.0f} to the very first move"
        self.say_line(f"You matched the last dancer {fidelity_prev:.0f} out of 100{orig}. Pass code {' '.join(str(self.code))} on and watch it keep changing!", key='code', ttl=20)
        self.final_card = dict(result=r, title=f"telephone link {link_no}", code=self.code, handle=self.handle, mode_label='Telephone',
                               lines=[f"vs last dancer {fidelity_prev:.0f}", f"vs original {'-' if fidelity_origin is None else round(fidelity_origin)}"])
        await self.wait_spoken('code', self.now() - 0.01, 14.0, min_s=6.0)

    # ---- class recording: the teacher records, students are tested by code, the teacher hears the report ----
    async def class_mode(self):
        """No code: the teacher records the class. With a code: a student, unless the caller is the teacher (by name) or said "report"."""
        if not self.code:
            await self.class_teacher(); return
        cls = await self.backend.call('GET', f'/dance/class/{self.code}')
        if not cls or cls.get('error'):
            await self.say_wait(f"Code {' '.join(str(self.code))} doesn't match a class. Say it again, one digit at a time.", key='tel'); return
        is_teacher = bool(self.name) and self.name.lower() == str(cls.get('teacher', '')).lower()
        if is_teacher or self.report_req or (os.environ.get('DANCE_CLASS_ROLE') == 'teacher'):
            self.report_req = False; await self.class_report(cls); return
        await self.class_student(cls)

    async def class_teacher(self):
        from team.dance.reference import key_from_arrays, save_key, phrase_slice
        from team.dance.scoring import score_phrase
        n_phrases = int(os.environ.get('DANCE_CLASS_PHRASES', '2') or 2); demos = int(os.environ.get('DANCE_CLASS_DEMOS', '1') or 1)
        await self.say_wait(f"Class recording! I'll count you in for {n_phrases} eight-counts. You dance each one and I'll remember it. Your students get a code.", key='tel', max_s=12.0)
        parts = []; agreements = []
        for i in range(n_phrases):
            if self.stop or self.stop_routine: break
            await self.say_wait(f"Eight-count {i + 1}. Four clicks, then show me.", key='tel')
            key = await self.record_phrase(label=f'PHRASE {i + 1}/{n_phrases}'); tries = 1
            while key is None and tries < 2 and not self.stop and not self.stop_routine:
                await self.say_wait("I couldn't see all of you. Step back, and I'll count you in again.", key='tel')
                key = await self.record_phrase(label=f'PHRASE {i + 1}/{n_phrases}'); tries += 1
            if key is None: continue
            if demos >= 2 and not self.stop and not self.stop_routine:
                await self.say_wait("Nice. Same eight-count once more, so I can check it.", key='tel')
                key2 = await self.record_phrase(label=f'PHRASE {i + 1} · CHECK')
                if key2 is not None:
                    aF, aM, aE, _ = phrase_slice(key, 0); bF, bM, bE, _ = phrase_slice(key2, 0)
                    agree = score_phrase(bF, bM, bE, aF, aM, aE, per_beat=key['per_beat'])
                    agreements.append(agree.get('score', 0.0) if agree.get('scored') else 0.0)
                    key = key2
            parts.append(key)
            await self.say_wait(f"Eight-count {i + 1} saved.", key='tel', max_s=6.0)
        if not parts:
            await self.say_wait("I couldn't capture a phrase this time. Call back with the phone a bit lower and further away.", key='tel'); return
        F = np.concatenate([np.asarray(k['features'], np.float32) for k in parts]); M = np.concatenate([np.asarray(k['mask'], bool) for k in parts])
        E = np.concatenate([np.asarray(k['energy'], np.float32) for k in parts]); S = np.concatenate([np.asarray(k['skeleton'], np.float32) for k in parts])
        bpm = float(parts[0]['bpm']); teacher = self.name or 'teacher'
        key = key_from_arrays(f"class_{teacher.lower()}", bpm, F, M=M, E=E, S=S, per_beat=parts[0]['per_beat'], counts_per_phrase=parts[0]['counts_per_phrase'], source=f'class mode, taught by {teacher}', save=False)
        key['style'] = 'class'; key['difficulty'] = 2
        posted = await self.backend.call('POST', '/dance/class', dict(teacher=teacher, key=key))
        self.my_code = (posted or {}).get('code'); self.ghost_token = (posted or {}).get('ghost_token')
        if self.my_code:
            key['name'] = f'class_{self.my_code}'
            try: save_key(key)
            except Exception as e: print(f'[dance] class key not saved locally: {e!r}', flush=True)
        self.rounds.append(dict(mode='class', role='teacher', phrases=len(parts), code=self.my_code, agreements=agreements))
        self.extras['teacher'] = True
        self.context(f"Class recorded: {len(parts)} phrase(s) at {bpm:.0f} bpm by {teacher}. Class code {self.my_code}. Students call and say the code; "
                     "the teacher calls back, says their name or 'report' with the code.")
        digits = ' '.join(str(self.my_code)) if self.my_code else None
        self.say_line(f"Your class is saved. The class code is {digits}. Share it with your students: they call me and say the code." if digits
                      else "Your class is recorded, but I couldn't save it just now. Try again in a minute.", key='code', ttl=20)
        if self.name: await self.award('class', rounds=[])
        self.final_card = dict(result=None, title=f'class by {teacher}', code=self.my_code, lines=[f'{len(parts)} phrase(s) at {bpm:.0f} bpm', 'students: say this code'],
                               handle=self.handle, mode_label='Class')
        await self.wait_spoken('code', self.now() - 0.01, 14.0, min_s=6.0)

    async def class_student(self, cls):
        key = cls.get('key')
        if not key or not key.get('phrases'):
            await self.say_wait("That class has no phrases yet.", key='tel'); return
        self.key = key; teacher = cls.get('teacher') or 'the teacher'; self.key.setdefault('title', f"{teacher}'s class")
        n_ph = len(key['phrases']); max_attempts = int(os.environ.get('DANCE_CLASS_ATTEMPTS', '2') or 2)
        await self.say_wait(f"This is {teacher}'s class: {n_ph} eight-count{'s' if n_ph != 1 else ''}. The ghost is {teacher} dancing. Watch twice, then it's you.", key='tel', max_s=12.0)
        best = {}; results = {}
        pi = 0
        while pi < n_ph and not self.stop and not self.stop_routine:
            attempts = 0; tempo = self.tempo
            while attempts < max_attempts and not self.stop and not self.stop_routine:
                self.cmd_floor_t = self.now()
                eng = await self.run_phrase(pi, tempo, step=dict(label=f'PHRASE {pi + 1}/{n_ph}'), stage='CHALLENGE'); attempts += 1
                r = eng.result or {}
                if eng.result is None: break
                self.record_round(eng, item=f'K{pi}')
                self.rounds.append(dict(mode='class', role='student', phrase=pi, tempo=tempo, result=r))
                if r.get('scored') and r['score'] > best.get(pi, -1): best[pi] = r['score']; results[pi] = (eng, r)
                t = self.now(); self.summarise(eng, round_label=f"Class phrase {pi + 1} of {n_ph}. "); await self.wait_spoken('result', t, 10.0, min_s=2.0)
                cmd = self.take_cmd(('stop', 'slower', 'next', 'skip'))
                if cmd == 'stop': self.stop_routine = True; break                  # ends this class, never the whole call
                if cmd == 'slower': tempo = max(0.6, round(tempo - 0.15, 2)); continue
                if cmd in ('next', 'skip') or (r.get('scored') and r['score'] >= 80): break
                if attempts < max_attempts: await self.say_wait("Once more on that phrase.", key='step', max_s=6.0)
            pi += 1
        phrases = []
        for i in range(n_ph):
            if i in results:
                eng, r = results[i]
                phrases.append(dict(phrase=i, score=r['score'], scored=True, verdicts=[row['verdict'] for row in r.get('per_beat', [])],
                                    per_group=r.get('per_group'), weak_beats=r.get('weak_beats')))
            elif any(x.get('phrase') == i for x in self.rounds):
                phrases.append(dict(phrase=i, score=None, scored=False, verdicts=[], per_group={}, weak_beats=[]))
        posted = await self.backend.call('POST', f'/dance/class/{self.code}/student', dict(name=self.name or 'anonymous', phrases=phrases))
        total = (posted or {}).get('total'); rank = (posted or {}).get('rank'); n = (posted or {}).get('n_students'); mean = (posted or {}).get('class_mean')
        scores_txt = ', '.join(f"phrase {i + 1}: {best[i]:.0f}" for i in sorted(best)) or 'nothing scored'
        self.context(f"Class {self.code} ({teacher}): the student's best per phrase: {scores_txt}. "
                     + (f"Their average {total:.0f}; the class average is {mean:.0f} across {n} student(s); they rank {rank} of {n}." if total is not None else ''))
        if best:
            line = f"Your best: {scores_txt}." + (f" You're number {rank} of {n} in {teacher}'s class." if total is not None and rank else '')
        else:
            line = "Nothing scored this time. Step back so I can see you, and try the class again."
        self.say_line(line, key='code', ttl=20)
        aw = await self.award('class') if results else None
        lines = [f"phrase {i + 1}: {best[i]:.0f}" for i in sorted(best)] + ([f"class average {mean:.0f}, you are {rank} of {n}"] if total is not None else [])
        last = results[max(results)][1] if results else None
        self.final_card = dict(result=last, title=f"{teacher}'s class", code=self.code, lines=lines, xp=aw, handle=self.handle, mode_label='Class')
        await self.wait_spoken('code', self.now() - 0.01, 14.0, min_s=6.0)

    async def class_report(self, cls):
        rep = await self.backend.call('GET', f'/dance/class/{self.code}/report')
        if not rep or rep.get('error'):
            await self.say_wait("The report isn't available right now.", key='tel'); return
        n = rep.get('n_students', 0); hm = rep.get('heatmap', [])
        rows = '; '.join(f"{r['name']}: " + ', '.join('-' if v is None else f'{v:.0f}' for v in r['scores']) for r in hm) or 'no students yet'
        self.rounds.append(dict(mode='class', role='teacher_report', code=self.code, n_students=n, suggestion=rep.get('suggestion')))
        self.context(f"Class report for code {self.code} ({rep.get('n_phrases')} phrases): {n} student(s). Scores by phrase: {rows}. "
                     f"Phrase means: {rep.get('phrase_mean')}. Suggestion: {rep.get('suggestion')}"
                     + (f" Not attempted: {', '.join(rep['not_attempted'])}." if rep.get('not_attempted') else ''))
        weak = ''
        if rep.get('weakest_phrase') is not None:
            weak = f" The class drops on phrase {rep['weakest_phrase'] + 1}" + (f", count {rep['weakest_beat'] + 1}" if rep.get('weakest_beat') is not None else '') + '.'
        self.say_line(f"{n} student{'s' if n != 1 else ''} danced your class.{weak} {rep.get('suggestion') or ''}".strip(), key='code', ttl=20)
        lines = [f"{n} student(s)"] + [f"{r['name']}: " + '  '.join('-' if v is None else f'{v:.0f}' for v in r['scores']) for r in hm[:6]]
        if rep.get('weakest_phrase') is not None: lines.append(f"weak: phrase {rep['weakest_phrase'] + 1}" + (f", count {rep['weakest_beat'] + 1}" if rep.get('weakest_beat') is not None else '') + (f", {rep['weakest_group']}" if rep.get('weakest_group') else ''))
        self.final_card = dict(result=None, title='class report', code=self.code, lines=lines, mode_label='Class')
        await self.wait_spoken('code', self.now() - 0.01, 14.0, min_s=6.0)


# ---- worker registration ----
async def dance_consumer(job):
    s = DanceSession(job, key_name=os.environ.get('DANCE_KEY') or None, mode=os.environ.get('DANCE_MODE', 'learn'),
                     tempo=float(os.environ.get('DANCE_TEMPO', '1.0')), rtt_s=float(os.environ.get('DANCE_RTT', '0.3')),
                     fps=float(os.environ.get('DANCE_FPS', '15')), pose_variant=os.environ.get('DANCE_POSE', 'full'),
                     debug_mp4=os.environ.get('DANCE_DEBUG_MP4') or None)
    await s.run()


def _register_consumer():
    """Register once on every live copy of the worker module. `python -m facetime.worker` runs the worker as `__main__`, so
    `facetime.worker` imported from here would be a second copy with its own CONSUMERS list; register on both, never twice
    (the worker re-imports this file per job through load_handlers)."""
    done = []
    for name in ('__main__', 'facetime.worker'):
        m = sys.modules.get(name)
        if m is None and name == 'facetime.worker':
            try:
                from facetime import worker as m   # noqa: N816
            except Exception:
                m = None
        lst = getattr(m, 'CONSUMERS', None)
        if m is None or not isinstance(lst, list) or not hasattr(m, 'consumer'): continue
        if any(getattr(f, '__name__', '') == 'dance_consumer' for f in lst): done.append(name + ' (already)'); continue
        lst.append(dance_consumer); done.append(name)
    return done


try:
    _registered_on = _register_consumer()
    _worker = sys.modules.get('facetime.worker')
    if _registered_on: STATE['registered'] = True
except Exception:          # imported outside the worker (simulator, tests)
    _worker = None; _registered_on = []


# ---- simulator ----
class _FakePlayer:
    def __init__(self, log, now): self.log = log; self.now = now; self.playing = None
    def say(self, text, intent='other', meta=None): self.log.append(dict(t=round(self.now(), 2), say=text, intent=intent)); return dict(status='ok')
    def say_now(self, text, intent='probe', meta=None): return self.say(text, intent, meta)


class _FakeModel:
    """The realistic simulator's Gemini: a line plays ~0.9 s after it is asked, at ~65 ms per character; the caller's speech gets
    its own ~1.6 s reply a second later. Only timing: no words."""
    def __init__(self, now): self.now = now; self.started = True; self.busy = []; self.last_caller_t = None; self.last_coach_t = -1e9
    @property
    def speaking(self):
        t = self.now(); return any(a <= t < b for a, b in self.busy)
    def reply(self, start, dur):
        self.busy = [x for x in self.busy if x[1] > self.now() - 1.0] + [(start, start + dur)]; self.last_coach_t = start + dur


class _FakeVoice:
    def __init__(self, log, now, timed=False):
        self.log = log; self.now = now; self.timed = timed; self.model = _FakeModel(now) if timed else None
    def ask_voice(self, prompt):
        self.log.append(dict(t=round(self.now(), 2), ask=prompt[:100]))
        if self.timed:
            text = prompt.split(': ', 1)[-1]
            start = max([self.now() + 0.9] + [b for a, b in self.model.busy])
            self.model.reply(start, 0.065 * len(text))
        return 'spoken'
    def _context(self, text): self.log.append(dict(t=round(self.now(), 2), context=text[:140]))


class _FakeJob:
    def __init__(self): self.bound = 'sim-caller'; self.latest = None; self.closed = False; self.room = None; self.player = None; self.voice = None; self.bank = dance_audio.FakeBank()


async def simulate(video, key, mode, out, offset=0.0, fast=True, tempo=1.0, rtt=0.0, fps=15.0, pose_variant='full', name=None, lobby=False, says=(),
                   players=1, background=False, gestures=(), crop_bottom=0.0, realistic=False, answers=None, flicker=0.0):
    """players=2: the stand-in caller is the clip twice side by side, the right copy a quarter second late (a squad).
    background=True: a small copy of the clip pasted in the corner (someone walking behind a solo dancer).
    gestures: [(t, 'pause'|'ready', seconds)] synthetic held poses injected into the landmarks.
    realistic=True: lines take time to play, Gemini answers the caller, transcripts arrive late (the timing of a real call).
    answers: {line key: [reply, ...]} the stand-in caller answers Brodie's lines a second after each one ends (in order; the last
    reply repeats); '@hands' raises both hands instead of speaking. flicker: share of frames with the lower body cut off."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): raise SystemExit(f'cannot open {video}')
    vfps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    clock = dict(t=0.0); log = []
    def now(): return clock['t']
    job = _FakeJob(); job.player = _FakePlayer(log, now); job.voice = _FakeVoice(log, now, timed=realistic)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        frames.append(bgr)
    cap.release()
    print(f'[sim] {video}: {len(frames)} frames at {vfps:g} fps; the caller "starts" at t=0 and the clip plays once (loops if the session is longer)', flush=True)
    dur = len(frames) / vfps
    holder = {}
    import random as _rnd
    rng = _rnd.Random(7)

    def clip_t(t):
        eng = holder.get('engine')
        if eng is not None and eng.state == 'attempt': return offset + eng.attempt_t(t)
        if eng is not None and eng.state in ('scored', 'feedback', 'done'): return offset + eng.attempt_s
        return max(0.0, offset - 1.0)

    def frame_at(t):
        """The stand-in caller: during an attempt the clip plays from `offset` in step with the attempt clock (so the clip's
        dancing lands in the scoring window); otherwise a still frame just before `offset` (standing, waiting)."""
        vt = clip_t(t); f = frames[int((vt % dur) * vfps) % len(frames)]
        if players >= 2:
            g = frames[int(((vt - 0.25) % dur) * vfps) % len(frames)]
            f = np.hstack([f, g])
        cb = crop_bottom if crop_bottom > 0 else (0.5 if flicker > 0 and rng.random() < flicker else 0.0)
        if cb > 0:                                             # the camera too high / too close: the lower body is out of frame
            h = f.shape[0]; f = cv2.resize(f[:int(h * (1 - cb))], (f.shape[1], h))
        if background:
            f = f.copy(); h, w = f.shape[:2]; small = cv2.resize(frames[int(((vt + 1.3) % dur) * vfps) % len(frames)], (w // 4, h // 4))
            f[h // 12:h // 12 + small.shape[0], w - small.shape[1] - 4:w - 4] = small
        return f
    sim = {}; pending_says = sorted(says, key=lambda x: x[0]); deliveries = []
    answers = {k: list(v) for k, v in (answers or {}).items()}; seen_end = {}; used = {}
    gest_list = list(gestures)

    def speak(t_say, text):
        """The caller talks at t_say: the transcript lands ~0.8 s later (realistic), and Gemini starts its own short reply."""
        log.append(dict(t=round(clock['t'], 2), caller=text))
        if text == '@hands':
            gest_list.append((clock['t'], 'ready', 1.5)); return
        if realistic:
            job.voice.model.last_caller_t = clock['t']; job.voice.model.reply(clock['t'] + 1.0, 1.6)
            deliveries.append((clock['t'] + 0.8, text))
        else:
            sim['s'].on_worker_event(dict(type='voice', role='caller', text=text))

    async def fake_sleep(s_):
        clock['t'] += s_
        job.latest = (frame_at(clock['t']), 0, clock['t'], clock['t'], int(clock['t'] * vfps))
        s = sim.get('s')
        while pending_says and pending_says[0][0] <= clock['t'] and s is not None:   # scripted caller lines ("3:I'm Maya")
            t_say, text = pending_says.pop(0); speak(t_say, text)
        if s is not None and answers:                                                   # the reactive caller answers Brodie's lines
            for k, replies in answers.items():
                end = s.line_end.get(k)
                if end is not None and end > seen_end.get(k, -1.0) and clock['t'] >= end + 1.0:
                    seen_end[k] = end; i = used.get(k, 0); used[k] = i + 1
                    speak(clock['t'], replies[min(i, len(replies) - 1)])
        while deliveries and deliveries[0][0] <= clock['t'] and s is not None:
            _, text = deliveries.pop(0); s.on_worker_event(dict(type='voice', role='caller', text=text))
        if not fast: await asyncio.sleep(s_)
        else: await asyncio.sleep(0)
    job.latest = (frame_at(0.0), 0, 0.0, 0.0, 0)
    s = DanceSession(job, key_name=key, mode=mode, tempo=tempo, rtt_s=rtt, fps=fps, pose_variant=pose_variant, debug_mp4=out, now=now, sleep=fake_sleep)
    s.log = log; s.engine_holder = holder; s.lobby_enabled = bool(lobby); sim['s'] = s
    real = s._gestures; done = set()                       # scripted held poses; like the real detector, one hold fires once
    def scripted(now_, lm, in_attempt):
        for (tg, kind, secs) in gest_list:
            if tg <= now_ <= tg + secs and (tg, kind) not in done:
                if kind == 'pause' and in_attempt and now_ >= tg + 1.3: done.add((tg, kind)); s.pause_req = True; s.telemetry.event('gesture', kind='pause', scripted=True)
                if kind == 'ready' and now_ >= tg + 1.1:
                    done.add((tg, kind)); s.telemetry.event('gesture', kind='ready', scripted=True, state=s.state)
                    if s.want_ready or s.paused: s._cmd('resume' if s.paused else 'ready', 'gesture')
        return real(now_, lm, in_attempt)
    s._gestures = scripted
    if name: s.name = name                                   # what the caller would have said ("I'm Maya")
    await s.run()
    print(json.dumps([e for e in log if 'say' in e or 'ask' in e or 'caller' in e][:80], indent=0))
    for r in s.rounds:
        res = r.get('result') or {}
        print(f"  {r.get('kind', '')}{r.get('phrase', r.get('round'))} tempo {r.get('tempo')}: score {res.get('score')} shape {res.get('spatial')} timing {res.get('timing')} "
              f"energy {res.get('energy')} scored {res.get('scored')}" + (f" squad {r['squad'].get('ranking')}" if r.get('squad') else ''))
    print(f'[sim] composed video -> {out}; session log -> data/tmp/dance_session_log.json; telemetry -> {s.telemetry.path}')
    return s


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Dance mode simulator (no LiveKit).')
    ap.add_argument('--simulate', required=True, help='a video standing in for the caller')
    ap.add_argument('--key', default=None); ap.add_argument('--mode', default='learn', choices=list(MODES))
    ap.add_argument('--code', default=None, help='a four-digit code the caller "spoke" (duel, chain or class to join)')
    ap.add_argument('--name', default=None, help='the name the caller "said" (class mode: the teacher\'s name opens the report)')
    ap.add_argument('--lobby', action='store_true', help='run the opening conversation (otherwise the simulator starts the music directly)')
    ap.add_argument('--say', action='append', default=[], help='scripted caller line at a time, "3:I am Maya, let us learn waacking"; repeatable')
    ap.add_argument('--answer', action='append', default=[], help='the caller answers a line key a second after it ends, "space:yes" or "gotit:@hands"; repeatable (in order)')
    ap.add_argument('--gesture', action='append', default=[], help='scripted held pose "t:pause:2" or "t:ready:1.5"; repeatable')
    ap.add_argument('--realistic', action='store_true', help='real-call timing: lines take time to play, Gemini replies, transcripts arrive late')
    ap.add_argument('--flicker', type=float, default=0.0, help='share of frames with the lower body cut off (a jittery framing)')
    ap.add_argument('--players', type=int, default=1, help='2 = two dancers side by side (squad)')
    ap.add_argument('--background', action='store_true', help='a small second person in the corner (walk-in test)')
    ap.add_argument('--crop-bottom', type=float, default=0.0, help='cut this share off the bottom of the picture (feet out of frame)')
    ap.add_argument('--backend', default='fake', help="'fake' = in-process backend code (default for the simulator); or the server URL")
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'tmp', 'dance_sim.mp4')); ap.add_argument('--offset', type=float, default=0.0)
    ap.add_argument('--tempo', type=float, default=1.0); ap.add_argument('--rtt', type=float, default=0.0); ap.add_argument('--fps', type=float, default=15.0)
    ap.add_argument('--pose', default='full'); ap.add_argument('--realtime', action='store_true', help='pace the simulation at wall clock')
    a = ap.parse_args(argv)
    os.environ['BACKEND_URL'] = a.backend
    os.environ.setdefault('DANCE_MAX_S', '400')                        # a simulated call ends when the chat goes quiet
    if a.code: os.environ['DANCE_CODE'] = a.code
    says = []
    for s in a.say:
        t, _, text = s.partition(':'); says.append((float(t), text.strip()))
    answers = {}
    for s in a.answer:
        k, _, text = s.partition(':'); answers.setdefault(k.strip(), []).append(text.strip())
    gests = []
    for g in a.gesture:
        t, kind, secs = (g.split(':') + ['1.5'])[:3]; gests.append((float(t), kind, float(secs)))
    asyncio.run(simulate(a.simulate, a.key, a.mode, a.out, offset=a.offset, fast=not a.realtime, tempo=a.tempo, rtt=a.rtt, fps=a.fps, pose_variant=a.pose, name=a.name,
                         lobby=a.lobby, says=says, players=a.players, background=a.background, gestures=gests, crop_bottom=a.crop_bottom,
                         realistic=a.realistic, answers=answers, flicker=a.flicker))


if __name__ == '__main__':
    main()
