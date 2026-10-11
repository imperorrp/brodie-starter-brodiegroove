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
from team.dance.fx import FxQueue                                # noqa: E402
from team.dance.gestures import GestureDetector                  # noqa: E402
from team.dance.telemetry import Telemetry                       # noqa: E402
from team.dance.multipose import PlayerTracker                   # noqa: E402
from team.dance.squad import SquadEngine                         # noqa: E402

from team.dance.state import STATE                               # noqa: E402  shared across every per-job copy of this file
NAME_RE = re.compile(r"\b(?:i am|i'm|im|my name is|this is|call me)\s+([A-Z][a-z]{1,15})", re.I)
LTR_RE = re.compile(r"left to right[,:]?\s+(.*)", re.I)
NOT_NAMES = {'Ready', 'Here', 'Good', 'Fine', 'Done', 'Back', 'Not', 'So', 'Just', 'Going', 'Trying', 'The', 'And', 'Learning', 'Dancing', 'Okay'}
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
    'simpler': 'simplify', 'simplify': 'simplify', 'break it down': 'simplify', 'too hard': 'simplify', 'too fast': 'slower',
    'explain': 'explain', 'how do i': 'explain', 'what do i do': 'explain', 'show me how': 'explain',
    'full speed': 'full', 'whole routine': 'full', 'full routine': 'full', 'whole thing': 'full',
    'rematch': 'rematch', 'run it back': 'rematch', 'harder': 'harder', 'level up': 'harder', 'challenge me': 'harder', 'easier': 'easier',
    # modes
    'squad battle': 'squad_battle', 'battle each other': 'squad_battle', 'squad': 'squad_battle', 'all of us': 'squad_battle',
    'learn together': 'squad_learn', 'dance together': 'squad_learn', 'learn it together': 'squad_learn',
    'battle': 'battle', 'duel': 'battle', 'learn': 'learn', 'teach me': 'learn', 'daily challenge': 'daily', 'daily': 'daily',
    'telephone': 'telephone', 'class': 'class', 'teacher': 'class', 'student': 'class', 'report': 'report'}
# Starting is never a keyword: it takes an explicit request in chat, the camera setup, and a clear yes to "ready?" (classify()).
QUESTION_RE = re.compile(r"^\W*(what|what's|whats|where|how|which|why|who|when|should|shall|can i|could i|would|is|are|do|does|did|will|am i)\b|\?", re.I)
DENY_RE = re.compile(r"\b(no|nope|nah|not (yet|ready|now|really)|i'?m not|wait|hold on|hang on|one sec(ond)?|give me a (sec|second|minute)|later|never ?mind|cancel|stop)\b", re.I)
CANCEL_RE = re.compile(r"\b(never ?mind|cancel|stop|not now|later|forget it)\b", re.I)
HOLD_RE = re.compile(r"\b(wait|hold on|hang on|one sec(ond)?|give me a (sec|second|minute)|not yet)\b", re.I)
AFFIRM_RE = re.compile(r"^\W*(yes|yeah|yep|yup|ya|sure|ok|okay|ready|i'?m ready|ready to go|let'?s go|lets go|let'?s do (it|this)|go|start|do it|alright|all right|all set|i'?m set|of course|definitely|absolutely|uh huh|mhm|i do|i have)\b", re.I)
REQUEST_RE = re.compile(r"\b(teach me|let'?s (do|learn|dance|try|start|battle|play)|let us (do|learn|dance|try|start)|i (want|wanna|would like|'d like|d like) to (learn|dance|do|try|battle|play|start)|"
                        r"i want (a|the) (lesson|battle|challenge|routine)|start (a |the |my )?(lesson|battle|challenge|routine|daily|dancing|class)|show me (a|the|some|how to)|"
                        r"(can|could) we (start|dance|learn|do|try|battle)|(do|play) the daily( challenge)?|rematch|run it back|battle me|go again|one more (round|time)|another round)\b", re.I)
REMATCH_RE = re.compile(r"\b(rematch|run it back|go again|one more (round|time)|another round|same (one|routine) again)\b", re.I)


def classify(text):
    """What an utterance means for the flow: {request, rematch, affirm, deny, cancel, hold, question}. Questions never request
    (except "can we start / could we dance"); a negation cancels an affirmation ("I'm not ready")."""
    t = (text or '').strip().lower()
    deny = bool(DENY_RE.search(t)); question = bool(QUESTION_RE.search(t))
    request = bool(REQUEST_RE.search(t)) and not deny
    if request and question and not re.search(r"\b(can|could) we\b", t): request = False
    return dict(request=request, rematch=bool(REMATCH_RE.search(t)) and not deny, affirm=bool(AFFIRM_RE.search(t)) and not deny, deny=deny,
                cancel=bool(CANCEL_RE.search(t)), hold=bool(HOLD_RE.search(t)), question=question)
CMD_ORDER = sorted(CMD_WORDS, key=len, reverse=True)
MODES = ('learn', 'battle', 'telephone', 'class', 'daily', 'squad_battle', 'squad_learn')
PRIORITY = ('pain', 'stop', 'forget', 'pause', 'resume', 'rematch', 'harder', 'easier', 'simplify', 'explain', 'full', 'slower', 'faster',
            'again', 'next', 'report', 'ready')
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
DIFF_RE = [(re.compile(r'\b(beginner|easy mode|easy)\b'), 'beginner'), (re.compile(r'\b(intermediate|medium)\b'), 'intermediate'),
           (re.compile(r'\b(advanced|expert|hard mode|pro mode|(?<!too )hard)\b'), 'advanced')]
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
CHOICES = ('again', 'next', 'faster', 'slower', 'simplify', 'explain', 'full', 'stop')
STATE_RULES = dict(
    EXPLORE="STATE: CHAT. Get to know the caller, answer questions, recommend where to start. You cannot start dancing yourself: only when they clearly ask to dance or learn, call propose_activity. Never say a lesson or challenge is starting.",
    PREPARE="STATE: CAMERA SETUP. The system checks the camera and hands you one adjustment at a time; say only that, in a few words. If they talk, answer in one short sentence. Never say the lesson is starting.",
    OFFER="STATE: OFFER. A routine has been suggested (see the latest system note). Ask once, in your own words, whether they want to start it now, mentioning a quick camera check comes first. Then wait for a clear yes. Do not start anything, do not give camera instructions yet. If they ask questions or change their mind, answer and stay here.",
    CONFIRM="STATE: READY CHECK. You asked if they're ready. Wait for their answer. Do not start anything yourself and do not count in.",
    ACTIVITY="STATE: DANCING. The music is playing. Stay silent unless the system gives you a private instruction.",
    REVIEW="STATE: RESULTS. Speak only the short line the system asks for, then wait.",
    PAUSED="STATE: PAUSED. If they speak, say they can say resume or raise both hands. Nothing else.",
    SAFETY="STATE: SAFETY STOP. Tell them to rest. Say nothing about dancing.")
LEVEL_CFG = dict(beginner=dict(tempo=0.8, show=2, pass_mark=75), intermediate=dict(tempo=0.95, show=2, pass_mark=80), advanced=dict(tempo=1.05, show=1, pass_mark=85))
MAX_ATTEMPTS = 3
MODE_LABEL = dict(learn='Learn', battle='Ghost Battle', daily='Daily Challenge', telephone='Telephone', class_='Class', squad_battle='Squad Battle', squad_learn='Squad Learn')
CARD_KW = ('grade', 'award_title', 'xp', 'achievements', 'handle', 'mode_label', 'squad', 'total_line', 'perfects', 'routine', 'photo', 'brag')


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


def parse_code(text):
    """A four-digit code spoken as digits ("4471", "44 71") or four number words in a row ("four four seven one"). None when absent."""
    t = text.lower()
    m = re.search(r'\b(\d)\D{0,2}(\d)\D{0,2}(\d)\D{0,2}(\d)\b', t)
    if m: return ''.join(m.groups())
    w = '|'.join(NUM_WORDS)
    m = re.search(rf'\b({w})[\s,-]+({w})[\s,-]+({w})[\s,-]+({w})\b', t)
    if m: return ''.join(NUM_WORDS[g] for g in m.groups())
    return None


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
        if parts[:2] == ['dance', 'leaderboard'] and method == 'GET': return db.leaderboard((q.get('key') or [None])[0])
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
        self.pending_cmd = None; self.stop = False; self.pose = None; self.mpose = None; self.exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dance-pose')
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
        self.offer_yes = False; self.offers_made = 0; self.last_bgr = None; self.countdown_ui = None; self.hist_best = None
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
        line is tagged with the session state and dropped if the state moved on or it waited longer than `ttl`. Gemini answers the
        caller by itself; this is never used to reply to something the caller just said (that made two voices answer at once)."""
        v = getattr(self.job, 'voice', None)
        if v is None or getattr(v, 'ask_voice', None) is None:
            self.log.append(dict(t=round(self.now() - (self.t0 or 0), 2), ask=prompt[:90], note='no voice layer')); return None
        if not prompt.startswith('Say this to the caller'):
            prompt = ('Speak directly to the caller as Brodie, in the second person ("you"). Reply with only the words you say aloud. What to say: '
                      + prompt.replace('One short sentence: ', '').replace('Two short sentences: ', '').replace('ask them to', 'ask the caller to'))
        if urgent: self.say_q = []
        if key: self.say_q = [q for q in self.say_q if q['key'] != key]
        self.say_q.append(dict(prompt=prompt, key=key, epoch=self.epoch, t=self.now(), ttl=ttl, urgent=urgent))
        if self.speaker_task is None or self.speaker_task.done():
            try: self.speaker_task = asyncio.get_running_loop().create_task(self._speaker_loop())
            except RuntimeError:
                self.say_q.pop(); return v.ask_voice(prompt)
        return 'queued'

    def say_line(self, text, key=None, ttl=8.0, urgent=False):
        """Brodie says exactly this (written by us, in second person). Gemini read meta instructions aloud ("Let's ask them to step
        back...") when we described what to say; quoting the words removes that."""
        return self.ask(f'Say this to the caller, word for word, and nothing before or after it: "{text}"', key=key, ttl=ttl, urgent=urgent)

    def pick(self, key, variants):
        """A variant different from the last one used for this key (never the same sentence twice in a row)."""
        last = self.last_variant.get(key); opts = [v for v in variants if v != last] or list(variants)
        import random as _r
        v = _r.choice(opts); self.last_variant[key] = v
        return v

    def _caller_quiet(self, s=1.0):
        model = getattr(getattr(self.job, 'voice', None), 'model', None)
        last = getattr(model, 'last_caller_t', None)
        return last is None or (time.monotonic() - float(last)) >= s

    def _model_speaking(self):
        model = getattr(getattr(self.job, 'voice', None), 'model', None)
        try: return bool(getattr(model, 'speaking', False))
        except Exception: return False

    async def _speaker_loop(self):
        v = getattr(self.job, 'voice', None); model = getattr(v, 'model', None); live = self.now is time.monotonic
        while self.say_q and not getattr(self.job, 'closed', False):
            q = self.say_q[0]
            if (q['epoch'] != self.epoch and not q['urgent']) or self.now() - q['t'] > q['ttl']:
                self.say_q.pop(0); self.telemetry.event('speech_dropped', key=q['key'], state=self.state)
                self.log.append(dict(t=round(self.now() - (self.t0 or 0), 2), ask=q['prompt'][:90], note='dropped: stale')); continue
            listening = self.now() - self.last_caller_text_t < 6.0 and not q['urgent']          # the caller just spoke: Gemini answers, we wait
            ready = (model is None or getattr(model, 'started', False)) and self._audio_idle() and not self._model_speaking() and self._caller_quiet() and not listening
            if not ready:
                if listening: q['t'] = max(q['t'], self.now() - q['ttl'] + 4.0)           # keep it alive a little past the caller's turn
                await (asyncio.sleep(0.1) if live else asyncio.sleep(0)); continue
            self.say_q.pop(0)
            if live: await asyncio.sleep(0.3)                               # past the player's post-clip hold
            if q['epoch'] != self.epoch and not q['urgent']: continue
            try: r = v.ask_voice(q['prompt'])
            except Exception as e: print(f'[dance] ask failed: {e!r}', flush=True); r = None
            self.log.append(dict(t=round(self.now() - (self.t0 or 0), 2), ask=q['prompt'][:90], result=r, key=q['key']))
            self.telemetry.event('speech', key=q['key'], state=self.state)
            if live:                                                         # wait for that line to be spoken before the next one
                t_end = time.monotonic() + 3.0
                while time.monotonic() < t_end and self._audio_idle() and not self._model_speaking(): await asyncio.sleep(0.1)
                t_end = time.monotonic() + 15.0
                while time.monotonic() < t_end and (not self._audio_idle() or self._model_speaking()): await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0)
            if q['key']: self.line_end[q['key']] = self.now()              # answers to this line count from here

    def transition(self, state, why=''):
        """Change the session state: queued lines from the old state are dropped and Gemini gets the new state's rules."""
        if state == self.state: return
        prev = self.state; self.state = state; self.epoch += 1
        self.say_q = [q for q in self.say_q if q['urgent']]
        self.telemetry.event('state', frm=prev, to=state, why=why); print(f'[dance] state {prev} -> {state}' + (f' ({why})' if why else ''), flush=True)
        self.task(self._apply_state_rules(state))

    async def _apply_state_rules(self, state):
        model = getattr(getattr(self.job, 'voice', None), 'model', None); agent = getattr(model, 'agent', None)
        if agent is None or not hasattr(agent, 'update_instructions'): return
        try:
            await agent.update_instructions((getattr(model, 'persona', '') or '') + '\n\n' + STATE_RULES.get(state, '') + '\n' + PERSONA_PROMPTS.get(self.persona, ''))
        except Exception as e:
            print(f'[dance] update_instructions failed: {e!r}', flush=True)

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
        v = getattr(self.job, 'voice', None)
        if v is not None and hasattr(v, '_context'):
            try: v._context(text)
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
            if '?' in text: self.coach_q_t = now
            return
        if ev.get('role') != 'caller': return
        text = ev.get('text', ''); low = text.lower(); self.last_caller_text_t = now
        self.telemetry.event('caller_said', words=len(text.split()), state=self.state)
        m = LTR_RE.search(text)
        if m:
            names = [n for n in re.findall(r"\b([A-Z][a-z]{1,15})\b", m.group(1)) if n not in NOT_NAMES][:4]
            if names: self.ltr_names = names; self.apply_ltr_names()
        m = NAME_RE.search(text)
        if m and not self.name and m.group(1).capitalize() not in NOT_NAMES:
            self.name = m.group(1).capitalize(); self.telemetry.caller = self.name; print(f'[dance] caller is {self.name}', flush=True)
            self.task(self.welcome_back())
        code = parse_code(text)
        if code and not self.code and self.state in ('EXPLORE', 'OFFER'): self.code = code; print(f'[dance] code heard: {code}', flush=True); self.task(self.load_challenge(code))
        style_before = self.style
        for w in STYLE_ORDER:
            if re.search(rf'\b{re.escape(w)}\b', low): self.style = STYLE_WORDS[w]; print(f'[dance] style heard: {self.style}', flush=True); break
        for w in PERSONA_ORDER:
            if re.search(rf'\b{re.escape(w)}\b', low): self.persona_choice = PERSONA_WORDS[w]; self.apply_persona(self.persona_choice); break
        if 'too hard' not in low:
            for rx, d in DIFF_RE:
                if rx.search(low): self.diff_choice = d; self.apply_difficulty(d, announce=False); break
        if any(w in low for w in UPPER_WORDS) or re.search(r"\b(can'?t (step|go|move) back|no (room|space)|as far as i can|that'?s as far|i'?m sitting|sitting down)\b", low):
            if not self.upper_override: self.telemetry.event('upper_body', source='voice')
            self.upper_body = True; self.upper_override = True
        found = parse_commands(low); it = classify(text)
        modes = [c for c in found if c in MODES]
        if modes: self.mode_choice = modes[0]; print(f'[dance] mode preference: {modes[0]}', flush=True)
        cmd = pick_command(found)
        if cmd in ('pain', 'forget'): self._cmd(cmd, 'voice'); return
        st = self.state
        if st == 'EXPLORE':
            if it['request']: self.request_activity(mode=modes[0] if modes else None, rematch=it['rematch'], source='voice')
            return
        if st == 'OFFER':
            if it['cancel']: self.cancel_req = True
            elif it['affirm'] and not it['question']: self.affirm_t = now
            elif it['deny']: self.deny_t = now
            if self.style != style_before or modes or it['request']:                 # a change of mind: update the offer
                self.request_activity(mode=modes[0] if modes else None, rematch=it['rematch'], source='voice')
            return
        if st == 'PREPARE':
            self.context(self.camera_note_text())                                     # so Gemini's reply describes what the camera really sees
            if it['cancel']: self.cancel_req = True; self.deny_t = now
            elif it['hold']: self.hold_until = now + 30.0; self.deny_t = now
            elif it['deny']: self.deny_t = now
            elif it['affirm']: self.affirm_t = now
            return
        if st in ('CONFIRM', 'REVIEW'):
            if it['cancel'] and st == 'CONFIRM': self.cancel_req = True; self.deny_t = now
            elif it['hold']: self.hold_until = now + 30.0; self.deny_t = now
            elif it['deny']: self.deny_t = now
            elif it['affirm'] and not it['question']: self.affirm_t = now
            if st != 'REVIEW' or not cmd: return
        if st == 'PAUSED' and it['affirm']: cmd = 'resume'
        if cmd: self._cmd(cmd, 'voice')
        elif modes and self.mode_choice_pending: self.pending_cmd = modes[0]

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
        return camera_note(self.cur_lm, (self.setup_ui or {}).get('issue'), upper=self.upper_override)

    def request_activity(self, mode=None, style=None, difficulty=None, rematch=False, source='voice'):
        """A clear wish to dance (the caller, a code, or Gemini's propose_activity) becomes an OFFER, never a start: Brodie asks if
        they want to begin now, and only a clear yes leads to the camera check."""
        if self.state not in ('EXPLORE', 'OFFER'):
            return dict(ok=False, state=self.state, reason='already setting up or dancing')
        if style:
            code = STYLE_WORDS.get(str(style).lower()) or next((STYLE_WORDS[w] for w in STYLE_ORDER if w in str(style).lower()), None)
            if code: self.style = code
        if difficulty:
            d = dict(easy='beginner', medium='intermediate', hard='advanced').get(str(difficulty).lower(), str(difficulty).lower())
            if d in LEVEL_CFG: self.apply_difficulty(d, announce=False)
        prev = self.proposal or {}
        self.proposal = dict(mode=mode or prev.get('mode') or self.mode_choice or 'learn', style=self.style, difficulty=self.difficulty,
                             rematch=bool(rematch), source=source)
        self.proposal_t = self.now(); self.offer_yes = False
        self.telemetry.event('offer', mode=self.proposal['mode'], style=self.style, source=source)
        self.context(f"[system note] Suggested: {self.offer_label()}. Ask if they want to start it now (a quick camera check comes first). Nothing has started.")
        if self.state == 'EXPLORE': self.transition('OFFER', source)
        return dict(ok=True, state='OFFER', offer=self.offer_label(), note='Ask if they want to start now; a camera check comes first. Nothing has started.')

    def offer_label(self):
        p = self.proposal or {}
        mode = p.get('mode') or 'learn'; style = STYLE_NAMES.get(p.get('style') or 'lh', 'hip-hop')
        if mode == 'daily': return "today's daily challenge"
        if mode in ('battle', 'squad_battle'): return f"a {style} ghost battle"
        return f"a {p.get('difficulty') or self.difficulty} {style} routine"

    def _cmd(self, cmd, source='voice'):
        """Commands during a routine (voice regex, Gemini tool, gesture): dedupe, log, act. Starting is not a command."""
        now = self.now(); last = self.last_cmd.get(cmd)
        if last is not None and now - last < 2.5: return False
        self.last_cmd[cmd] = now
        self.telemetry.event('command', cmd=cmd, source=source, state=self.state); print(f'[dance] command ({source}): {cmd}', flush=True)
        if cmd == 'pain': self.safety_stop = True; self.pending_cmd = 'pain'; self.task(self.safety_halt()); return True
        if cmd == 'forget': self.task(self.request_forget()); return True
        if self.state in ('EXPLORE', 'OFFER') and cmd not in ('harder', 'easier'): return False      # flow commands mean nothing before a routine
        if cmd == 'pause':
            if self.cur_engine is not None and self.cur_engine.state != 'done': self.pause_req = True
            else: self.pending_cmd = 'pause'
            return True
        if cmd in ('resume', 'ready'):
            if self.paused: self.resume_req = True
            elif self.state in ('CONFIRM', 'PREPARE', 'REVIEW'): self.affirm_t = now
            return True
        if self.state == 'EXPLORE' and cmd in ('harder', 'easier'):
            order = ['beginner', 'intermediate', 'advanced']; j = order.index(self.difficulty) + (1 if cmd == 'harder' else -1)
            if 0 <= j < 3: self.apply_difficulty(order[j], announce=False)
            return True
        if self.state == 'EXPLORE': return False                          # flow commands mean nothing before a routine
        self.pending_cmd = cmd
        return True

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
            if prefs.get('persona') and not self.persona_choice: self.persona = prefs['persona']; self.task(self._apply_state_rules(self.state))
            if prefs.get('difficulty') and not self.diff_choice and progression.difficulty_unlocked(prefs['difficulty'], self.level): self.difficulty = prefs['difficulty']
        if not hist or not isinstance(hist, list):
            self.context(f"[system note] The caller's name is {self.name}. First time here: level Rookie. Recommend the beginner hip-hop or pop starter.")
            return
        last = hist[0]; best = max(float(h.get('score100') if h.get('score100') is not None else (h.get('total') or 0)) for h in hist)
        self.hist_best = best
        share = await self.backend.call('GET', f"/dance/share/{last.get('code')}") if last.get('code') else None
        board = await self.backend.call('GET', f"/dance/board/{last.get('code')}") if last.get('code') else None
        line = (f"[system note] The caller is {self.name}, back for visit {len(hist) + 1}. Last time: {last.get('mode')} on {str(last.get('key', '')).replace('_', ' ')}. "
                f"Best {best:.0f}. Last code {last.get('code')}.")
        if self.profile: line += f" Level {self.profile.get('level_name')} with {self.xp} XP, streak {self.profile.get('streak', 0)} day(s)."
        if share and share.get('starts'): line += f" {share['starts']} friend(s) took their challenge since."
        if board and not board.get('error') and board.get('n', 0) > 1:
            top = board['entries'][0]; line += f" Their board leader is {top['name']} with {top['score100']:.0f}."
        rec = self.recommendation(hist)
        if rec: line += f" Recommendation: {rec}."
        self.context(line)
        self.telemetry.event('welcome_back', visits=len(hist), level=self.level)

    def recommendation(self, hist):
        scores = [float(h.get('score100') if h.get('score100') is not None else (h.get('total') or 0)) for h in hist[:3]]
        order = ['beginner', 'intermediate', 'advanced']; i = order.index(self.difficulty)
        if len(scores) >= 2 and np.mean(scores) >= 85 and i < 2 and progression.difficulty_unlocked(order[i + 1], self.level):
            return f"they've been crushing it, suggest moving up to {order[i + 1]}"
        if len(scores) >= 2 and np.mean(scores) < 50 and i > 0: return f"suggest {order[i - 1]} to build confidence"
        return None

    async def load_challenge(self, code):
        """A friend's code: fetch it and load that routine. Saying a code is a clear request to battle it, so it leads to the
        camera setup (never straight to music)."""
        r = await self.backend.call('GET', f'/dance/result/{code}')
        if not r or r.get('error'):
            self.context(f"[system note] Code {code} doesn't exist; ask the caller to say it again, digit by digit.")
            self.code = None; return None
        self.challenge = r
        try:
            k = choreo.load_any(r['key']) if r.get('key') else None
            if k is not None:
                k['title'] = key_title(k); self.queued_key = k
        except Exception as e:
            print(f'[dance] challenge routine {r.get("key")!r} not loadable ({e!r}); keeping the current one', flush=True)
        await self.backend.call('POST', f'/dance/share/{code}/start')
        s100 = r.get('score100') if r.get('score100') is not None else r.get('total')
        self.context(f"[system note] The caller is taking on {r.get('name')}'s challenge {code}: {float(s100 or 0):.0f} on {str(r.get('key', '')).replace('_', ' ')}. "
                     "Camera setup comes next; don't say it's starting.")
        self.telemetry.event('challenge_start', code=code)
        if self.mode not in ('telephone', 'class') and self.state == 'EXPLORE': self.request_activity(mode='battle', source='code')
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

    async def paused_wait(self, max_s=120.0):
        self.transition('PAUSED', 'pause'); self.paused = True; self.resume_req = False; self.telemetry.event('paused')
        self.say_line("Paused. Say resume, or raise both hands, when you're ready.", key='paused')
        t_end = self.now() + max_s
        while self.now() < t_end and not self.stop and not self.safety_stop:
            if self.resume_req or self.pending_cmd in ('ready', 'again'):
                if self.pending_cmd in ('ready', 'again'): self.pending_cmd = None
                break
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
        if self.state in ('EXPLORE',) and action not in ('harder', 'easier'):
            return dict(will_happen='nothing yet', state=self.state, reason='no routine is running; use propose_activity when they clearly ask to dance')
        self._cmd(action, 'tool')
        what = dict(again='repeating the step', slower='slowing down', faster='speeding up', next='moving on', pause='pausing', resume='resuming',
                    stop='ending the routine', simplify='breaking it down', explain='explaining the move', full='going to the full routine',
                    harder='going harder', easier='going easier').get(action, action)
        return dict(will_happen=what, state=self.state, mid_phrase=bool(self.cur_engine is not None and self.cur_engine.state not in ('done', 'idle')))

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
        return dict(parts=self.parts, hint=self.hint, dancers=self.tracker.stable_count(), overlapping=self.tracker.overlapping(),
                    tracking_rate=round(tm.frames_pose / tm.frames, 3) if tm.frames else None, full_body=all(self.parts.get(p, False) for p in ('head', 'arms', 'torso', 'legs')))

    def last_attempt_summary(self):
        r = self.last_result
        if not r: return dict(attempted=False)
        if not r.get('scored'): return dict(attempted=True, scored=False, reason=r.get('reason') or 'not enough of the body visible')
        cue = self.last_cue or {}
        return dict(attempted=True, scored=True, score=r['score'], grade=progression.grade(r['score']), title=self.last_title, shape=round(100 * r['spatial']),
                    timing=round(100 * r['timing']), energy=round(100 * r['energy']), weak_counts=[b + 1 for b in r.get('weak_beats', [])],
                    cue=dict(text=cue.get('text'), body_part=cue.get('body_part'), direction=cue.get('direction'), counts=[b + 1 for b in cue.get('beats', [])]))

    async def tool_compare(self, code):
        code = parse_code(str(code)) or str(code).strip()
        if not re.fullmatch(r'\d{4}', code): return json.dumps(dict(ok=False, error='a challenge code is four digits'))
        self.code = code; r = await self.load_challenge(code)
        if not r: return json.dumps(dict(ok=False, error='no such code'))
        return json.dumps(dict(ok=True, name=r.get('name'), score100=r.get('score100'), routine=str(r.get('key', '')).replace('_', ' '), queued='battle'))

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
                               avatar_state=extra.get('avatar_state', 'idle'), countdown=extra.get('countdown'))
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
    def _pose_sync(self, rgb, ms):
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
        self.tracker.update(self.now(), poses)
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
        self.task(self._apply_state_rules('EXPLORE'))
        if self.code and self.mode not in ('telephone', 'class'):     # a code given up front (DANCE_CODE): battle it
            await self.load_challenge(self.code)
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
            self.exec.shutdown(wait=False)
            if self.pose_ms: print(f'[dance] pose {np.median(self.pose_ms):.0f} ms median over {len(self.pose_ms)} frames; {self.frames} frames composed', flush=True)
            self.telemetry.mode = self.mode
            self.telemetry.extra.update(time_to_start_s=self.time_to_start, call_mode=self.call_mode, max_dancers=self.max_players, difficulty=self.difficulty,
                                        persona=self.persona, safety_stop=self.safety_stop)
            summary = self.telemetry.close(completed=completed)
            try: await asyncio.wait_for(self.backend.call('POST', '/events', dict(type='session_summary', t_wall=round(time.time(), 3), **summary)), 4.0)
            except Exception: pass
            try:
                log_path = os.environ.get('DANCE_SESSION_LOG') or os.path.join(ROOT, 'data', 'tmp', 'dance_session_log.json')
                os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(dict(key=self.key['name'], mode=self.mode, name=self.name, rounds=self.rounds, log=self.log, summary=summary), f, indent=1, default=str)
            except Exception as e: print(f'[dance] session log not written: {e!r}', flush=True)

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
            n = await dance_tools.register(self, agent)
            self.telemetry.event('tools_registered', n=n); print(f'[dance] {n} Gemini tools registered', flush=True)
        except Exception as e:
            self.telemetry.event('tools_unavailable', error=repr(e)); print(f'[dance] tools unavailable ({e!r}); voice commands by keyword only', flush=True)

    async def play_loop(self):
        """Chat until the caller says yes to an offer; camera setup; a ready check; the routine; back to chat. Every stage runs inside
        an error boundary: a failure is logged, Brodie apologises, and the session returns to chat instead of ending."""
        first = True
        while not self.stop:
            req = await self.guard(self.explore(first=first), 'explore'); first = False
            if self.stop: break
            if not req: continue
            self.setup_activity(req)
            if self.mode in ('squad_battle', 'squad_learn') and self.tracker.stable_count() < 2:
                self.context("[system note] Only one dancer in view, so this is solo."); self.mode = 'battle' if self.mode == 'squad_battle' else 'learn'
            ok = await self.guard(self.prepare_and_confirm('start'), 'setup')
            if not ok or self.stop: continue
            self.call_mode = 'squad' if self.mode in ('squad_battle', 'squad_learn') else 'solo'
            if self.call_mode == 'squad': self.start_squad()
            run = dict(battle=self.battle, telephone=self.telephone, daily=self.daily, squad_battle=self.squad_battle).get(self.mode)
            if self.mode == 'class': run = self.class_mode
            if run is None: run = self.learn
            if self.time_to_start is None: self.time_to_start = round(self.now() - self.t0, 2)
            await self.guard(run(), f'mode {self.mode}')
            self.proposal = None
            if self.safety_stop: break
            if not self.lobby_enabled: break                              # test mode (DANCE_LOBBY=0): one routine, then end

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
                self.cut_audio(); self.setup_ui = None; self.paused = False
                self.transition('EXPLORE', 'error'); self.say_line("Sorry, I glitched for a second. Tell me when you want to go again.", key='error', urgent=True)
            except Exception as e2:
                print(f'[dance] recovery failed: {e2!r}', flush=True)
            return None

    async def explore(self, first=False):
        """Chat (Gemini talks). A clear wish to dance becomes an offer; we start only after a yes to it."""
        if self.state != 'OFFER': self.transition('EXPLORE')
        self.cancel_req = False
        if not self.lobby_enabled:
            self.proposal = dict(mode=self.mode, rematch=False, source='test'); return self.proposal
        if self.final_card is None: self.idle_title = 'Brodie'
        if first:
            self.say_line("Hey, I'm Brodie! What's your name, and what kind of dancing are you into?", key='hello', ttl=15)
        t_in = self.now(); t_hello = self.now()
        while not self.stop:
            if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; break
            if self.state == 'OFFER':
                r = await self.offer_loop()
                if r == 'yes': return self.proposal
                self.transition('EXPLORE', r); t_in = self.now()
                continue
            await self.frame_step(None); await self.sleep(1.0 / self.fps)
            # proactive: Brodie leads. After the caller has answered (or 14 s of quiet after hello), he suggests a routine himself;
            # if they pass, he suggests a different one after 25 quiet seconds (at most 3 suggestions). Starting still needs a yes.
            now = self.now(); quiet = now - max(self.last_caller_text_t, self.coach_q_t if self.coach_q_t > 0 else -1e9)
            spoke = self.last_caller_text_t > t_hello
            due = (self.offers_made == 0 and ((spoke and quiet > 4.0) or (not spoke and now - t_hello > 14.0))) or                   (0 < self.offers_made < 3 and now - t_in > 25.0 and quiet > 8.0)
            if due and not self._model_speaking():
                self.offers_made += 1
                style = self.style if (self.style and self.offers_made == 1) else ['lh', 'po', 'kpop', 'fitness'][(self.offers_made - 1) % 4]
                self.request_activity(mode=self.mode_choice or 'learn', style=STYLE_NAMES.get(style, 'hip-hop'), source='brodie')
        return None

    async def offer_loop(self):
        """OFFER: one concrete suggestion waits for a clear yes. Gemini usually asks the question itself; if it hasn't within 5 quiet
        seconds, we ask once. A yes only counts after a question was asked. Questions keep us here; cancel or 60 s silence go back."""
        self.affirm_t = -1.0; self.deny_t = -1.0; self.cancel_req = False; asked_t = -1.0; seen_prop = self.proposal_t
        while not self.stop:
            if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; return 'timeout'
            await self.frame_step(None); await self.sleep(1.0 / self.fps)
            now = self.now()
            if self.proposal_t != seen_prop: seen_prop = self.proposal_t; asked_t = -1.0; self.affirm_t = -1.0
            if self.cancel_req: self.cancel_req = False; return 'cancel'
            q_t = max(self.coach_q_t if self.coach_q_t > self.proposal_t else -1.0, self.line_end.get('offer', -1.0) if self.line_end.get('offer', -1.0) > self.proposal_t else -1.0)
            mine = (self.proposal or {}).get('source') == 'brodie'          # Brodie's own suggestion: he asks right away
            if q_t < 0 and asked_t < 0 and now - max(self.proposal_t, self.last_caller_text_t) > (1.0 if mine else 5.0):
                asked_t = now
                if mine:
                    line = self.pick('suggest', [f"Let's get you moving! How about {self.offer_label()}? Want to start now?",
                                                 f"I've got a good one for you: {self.offer_label()}. Shall we start?",
                                                 f"Here's an idea: {self.offer_label()}. Up for it?"])
                else:
                    line = self.pick('offer', [f"Want to start {self.offer_label()} now? We'll do a quick camera check first.",
                                               f"Shall we jump into {self.offer_label()}? First a quick camera check."])
                self.say_line(line, key='offer', ttl=12)
            if q_t > 0 and self.affirm_t > q_t - 0.3:
                self.telemetry.event('offer_accepted', offer=self.offer_label()); return 'yes'
            if self.deny_t > max(q_t, self.proposal_t) and not self.cancel_req and self.deny_t > 0:
                self.deny_t = -1.0                                                  # "not yet": stay, the offer stands
            if now - max(self.proposal_t, self.last_caller_text_t, self.coach_q_t) > 60.0: return 'timeout'
        return 'timeout'

    def setup_activity(self, req):
        mode = req.get('mode') or 'learn'
        if req.get('rematch') and self.final_card is not None: mode = self.last_mode or mode
        self.reset_run()
        if mode == 'daily': self.key = choreo.daily()
        elif self.queued_key is not None: self.key = self.queued_key; self.queued_key = None
        elif req.get('rematch') and getattr(self, 'last_key', None) is not None: self.key = self.last_key
        elif mode not in ('telephone', 'class'): self.key = self.make_routine(req.get('style') or self.style or ('wa' if self.upper_body else 'lh'))
        self.mode = mode; self.last_mode = mode; self.last_key = self.key
        self.key['title'] = key_title(self.key); self.idle_title = self.key['title']; self.final_card = None
        self.context(f"[system note] They said yes to {MODE_LABEL.get(mode, mode)}: '{self.key['title']}' at {self.difficulty}. Now the camera check; don't say it's starting.")
        self.telemetry.event('activity', mode=mode, key=self.key['name'], difficulty=self.difficulty)

    async def prepare_and_confirm(self, why='start'):
        """Camera setup, then the ready check. True to start; False goes back to chat."""
        for _ in range(3):
            if not await self.setup_camera(why): return False
            r = await self.ready_check()
            if r == 'go': return True
            if r == 'explore': return False
            why = 'setup again'
        return False

    async def wait_answer(self, seconds, gesture=False, tracker=None, key=None):
        """'yes' / 'no' / 'setup' (the framing broke while waiting) / None. With `key`, only answers after that line was actually
        spoken count (a "yes" to something else never starts anything)."""
        t0 = self.now(); self.want_ready = gesture; bad_since = None
        try:
            while self.now() - t0 < seconds + (6.0 if key else 0.0) and not self.stop:
                if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; break
                after = t0 if not key else (self.line_end.get(key, -1.0) - 0.3 if self.line_end.get(key, -1.0) >= t0 else None)
                if self.cancel_req: return 'no'
                if after is not None:
                    if self.deny_t > after and self.now() < self.hold_until: pass
                    elif self.deny_t > after: return 'no'
                    if self.affirm_t > after: return 'yes'
                await self.frame_step(None); await self.sleep(1.0 / self.fps)
                if tracker is not None:
                    res = tracker.update(self.now(), self.cur_lm, self.luma); self.setup_ui = res
                    bad_since = None if res['ok_now'] else (bad_since or self.now())
                    if bad_since is not None and self.now() - bad_since > 3.0: return 'setup'
        finally:
            self.want_ready = False
        return None

    async def setup_camera(self, why='start'):
        """One stable problem at a time, said once; then wait for the camera to show the fix (an event, not a timer). We speak again
        only after 25 quiet seconds without progress, with a different, contextual check-in (at most 3 per problem). What the caller
        says always gets Gemini's answer first, with a fresh note of what the camera sees. Ends True when the framing holds and they
        confirmed space; False on cancel or long no-progress."""
        from team.dance.setup import SetupTracker, FIRST, CHECKIN
        self.transition('PREPARE', why); self.cancel_req = False; self.hold_until = -1.0
        tr = SetupTracker(upper=self.upper_body or self.upper_override)
        spoken_issue = None; t_issue = self.now(); checkins = {}; t_progress = self.now(); offered_upper = False; asked_space = -1e9; lost_said = False
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
                if not lost_said: lost_said = True; self.say_line("I've lost your video for a moment. I'll wait right here.", key='setup')
                continue
            lost_said = False
            res = tr.update(now, self.cur_lm, self.luma); self.setup_ui = res
            if res['ok_stable']:
                if spoken_issue is not None: spoken_issue = None; t_progress = now
                if self.space_ok: self.setup_ui = None; return True
                if now - asked_space > 25 and now - self.last_caller_text_t > 3:
                    asked_space = now
                    self.say_line(self.pick('space', ["Perfect, I can see all of you. Do you have about two big steps of clear space around you?",
                                                      "Looking good! Is there about two big steps of space around you?"]), key='space')
                    ans = await self.wait_answer(15, key='space')
                    if ans == 'yes': self.space_ok = True; self.telemetry.event('space_confirmed'); self.setup_ui = None; return True
                    if ans == 'no' and not self.cancel_req:
                        self.say_line("Okay, clear a little space and tell me when it's done.", key='space')
                continue
            issue = res['issue']
            if issue is None or now < self.hold_until: continue
            quiet = now - self.last_caller_text_t > 6
            if issue != spoken_issue and quiet:
                spoken_issue = issue; t_issue = now; t_progress = now
                self.telemetry.event('setup_issue', issue=issue)
                self.say_line(self.pick(f'first_{issue}', FIRST[issue]), key='setup')
            elif issue == spoken_issue and now - max(t_issue, self.last_caller_text_t) > 15 and checkins.get(issue, 0) < 3:
                t_issue = now; checkins[issue] = checkins.get(issue, 0) + 1
                self.say_line(self.pick(f'checkin_{issue}', CHECKIN[issue]), key='setup')
            if now - t_progress > 90 and not offered_upper and not tr.upper:
                offered_upper = True; t_progress = now
                self.say_line("If the camera can't see all of you, say arms only and we'll start with the upper body. Or say never mind to chat.", key='setup')
            elif now - t_progress > 120:
                self.setup_ui = None
                self.say_line("No rush. Tell me when the camera's set up and we'll try again.", key='cancel'); return False
        return False

    async def ready_check(self):
        """'go' after a clear yes (or both hands up) to "ready?"; 'prepare' if the framing breaks; 'explore' if they never answer."""
        if not self.lobby_enabled: return 'go'
        from team.dance.setup import SetupTracker
        self.transition('CONFIRM'); tr = SetupTracker(upper=self.upper_body or self.upper_override)
        for n in range(2):
            self.say_line(self.pick('confirm', ["You're all set. Ready when you are: say ready, or raise both hands.",
                                                "All good on camera. Say ready, or raise both hands, and we'll start."]) if n == 0 else
                          "Whenever you're ready, just say ready.", key='confirm', ttl=10)
            ans = await self.wait_answer(20, gesture=True, tracker=tr, key='confirm')
            if ans == 'yes': self.telemetry.event('confirmed'); self.setup_ui = None; return 'go'
            if ans == 'setup': return 'prepare'
            if ans == 'no':
                if self.cancel_req: self.cancel_req = False; return 'explore'
                ans = await self.wait_answer(40, gesture=True, tracker=tr)
                if ans == 'yes': self.telemetry.event('confirmed'); self.setup_ui = None; return 'go'
                return 'prepare' if ans == 'setup' else 'explore'
        self.say_line("No worries. Tell me whenever you want to dance.", key='confirm')
        return 'explore'

    async def ask_yes_no(self, line, seconds=8.0, key='question'):
        self.say_line(line, key=key, ttl=seconds)
        return await self.wait_answer(seconds + 4.0, key=key)


    def reset_run(self):
        self.rounds = []; self.history = []; self.best = None; self.phrase = 0; self.attempts = 0; self.my_code = None
        self.session_rounds = []; self.extras = {}; self.recap = None; self.battle_hud = None; self.step = None; self.fx.clear(); self.stage = None
        if self.challenge is None: self.code = None

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
            kind, p = g.progress(now)
            if kind and p > best[1] and (kind == 'pause' and in_attempt or kind == 'ready' and (self.want_ready or self.paused)): best = (kind, p)
        self.gesture_ui = dict(kind=best[0], p=best[1]) if best[0] else None
        if 'pause' in fired and in_attempt: self.pause_req = True; self.telemetry.event('gesture', kind='pause')
        elif 'ready' in fired and (self.want_ready or self.paused) and not in_attempt:
            need = max(1, (len(sources) + 1) // 2)
            if fired.count('ready') >= need or self.call_mode != 'squad':
                self.telemetry.event('gesture', kind='ready', state=self.state)
                if self.paused: self.resume_req = True
                else: self.affirm_t = now

    async def frame_step(self, engine):
        frame, t = self.newest_frame()
        if frame is None: return
        self.last_frame_t = self.now()
        bgr, lm, w = await self.prepare(frame, int((self.now() - self.t0) * 1000))
        now = self.now(); self.cur_lm = lm
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
            self.telemetry.event(ev.pop('type'), **ev)
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
            if self.final_card is not None and not self.safety_stop and self.state in ('EXPLORE', 'REVIEW'):
                card = dict(self.final_card); card.setdefault('key', self.key['name'])
            else:
                cap = self.caption[0] if self.caption and now - self.caption[1] < 6.0 else None
                hints = self.state == 'EXPLORE' and self.last_caller_text_t < (self.t0 or 0) + 0.01
                extra.update(title=self.idle_title, lines=[], recap=self.recap if self.state == 'REVIEW' else None, caption=cap, footer=None,
                             gesture=getattr(self, 'gesture_ui', None), paused=self.paused, safety=self.safety_stop, hints=hints,
                             setup=self.setup_ui if self.state in ('PREPARE', 'CONFIRM') else None)
                if self.call_mode == 'squad' or self.tracker.stable_count() >= 2:
                    extra['players'] = [(l, self.colours.get(pid, 'volt')) for pid, l, _ in self.players]
        await self.render_push(bgr, lm if lm is not None else self.last_lm, hud, ghost=ghost, card=card, extra=extra)

    # ---- engine events: judgments, tips, combo, battle damage, telemetry ----
    def on_engine_event(self, ev):
        typ = ev.get('type'); now = self.now(); pid = ev.get('pid')
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
        self.stage = stage; self.practice = practice; self.in_demo = demo; self.ghost_overlay = ghost_overlay
        while True:
            self.transition('ACTIVITY', stage or 'phrase')
            if self.call_mode == 'squad' and self.squad_players and not demo:
                self.refresh_squad()
                eng = SquadEngine(key, self.squad_players, on_event=self.on_engine_event, phrase_index=phrase_index, tempo=tempo, show_repeats=show_repeats,
                                  countin_beats=countin_beats, rtt_s=self.rtt_s, freestyle=freestyle, focus=focus)
            else:
                eng = PhraseEngine(key, phrase_index=phrase_index, tempo=tempo, show_repeats=max(1, show_repeats) if demo else show_repeats,
                                   countin_beats=countin_beats, rtt_s=self.rtt_s, on_event=self.on_engine_event, freestyle=freestyle, focus=focus, demo_only=demo)
            eng.step = step; self.step = step; self.cur_engine = eng; self.combo_run = {}; self.last_tip_beat = {}; self.recap = None
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
                if self.pending_cmd == 'stop': self.stop_routine = True; self.pending_cmd = None; self.cut_audio(); eng.abort(self.now()); break
                if self.pause_req or self.pending_cmd == 'pause':
                    self.pause_req = False; self.pending_cmd = None if self.pending_cmd == 'pause' else self.pending_cmd
                    eng.abort(self.now()); self.cut_audio(); paused = True; break
                await self.sleep(1.0 / self.fps)
            if paused and not self.stop:
                await self.paused_wait()
                if self.stop or self.safety_stop: break
                show_repeats = min(show_repeats, 1); continue
            break
        self.cur_engine = None; self.in_demo = False
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

    def phrase_outcome(self, eng):
        """(scored, score, result, cue) for a solo engine or a squad (median of the dancers, cue for the one furthest off)."""
        if isinstance(eng, SquadEngine) and eng.squad:
            sc = [x['score'] for x in eng.squad['players'] if x['scored']]
            if not sc: return False, 0.0, eng.lead.result or {}, eng.cue
            return True, float(np.median(sc)), eng.lead.result or {}, eng.cue
        r = eng.result or {}
        return bool(r.get('scored')), float(r.get('score') or 0) if r.get('scored') else 0.0, r, eng.cue

    def summarise(self, eng, round_label='', xp=None):
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
        self.last_result = r; self.last_cue = cue
        if not scored:
            self.recap = dict(result=r, cue='Step back so I can see all of you')
            return self.say_line("I couldn't see enough of you that time. Let's fix the camera first.", key='result')
        glow = getattr(self, '_glow', False)
        title = progression.title_for(r, dict(glow_up=glow)); self.last_title = title; g = progression.grade(score)
        cue_txt = (cue or {}).get('text', '')
        self.recap = dict(result=r, title=title, cue=cue_txt, xp=xp)
        react = dict(S=["Unreal!", "That was clean!"], A=["So good!", "Nice one!"], B=["Solid!", "Getting there!"], C=["Good effort!", "Nice try!"],
                     D=["Good start!", "We'll get it!"])[g]
        tail = cue_txt + '.' if cue and cue.get('kind') != 'praise' else "Keep that energy."
        return self.say_line(f"{self.pick('react_' + g, react)} {score:.0f}. {tail}", key='result')

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
        """Say the move as simple steps (from the reference itself), then show it again slowly."""
        self.say_breakdown(key, pi, intro="Here's the move, step by step:")
        self.telemetry.event('explain', phrase=pi)
        await self.idle(5.0)

    def say_breakdown(self, key, pi, intro="Here's the move:"):
        desc = feedback.describe_phrase(key, pi)
        steps = [p for p in desc.split('; ') if ': ' in p][:4]
        if not steps: return None
        parts = [f"{p.split(': ', 1)[0]}, {p.split(': ', 1)[1]}" for p in steps]
        return self.say_line(f"{intro} " + '. '.join(parts) + '.', key='breakdown', ttl=15)

    async def wait_choice(self, line, key, seconds=18.0, reprompt=None):
        """Ask (or, with line=None, just keep listening) and wait patiently for the caller: a choice (again / next / faster / slower /
        simplify / explain / full / stop) or a yes / no. One gentle reprompt after `seconds` of quiet; after that Brodie keeps waiting
        quietly. Nothing is chosen for the caller: silence never skips ahead. A choice made while the move was playing (a button, or
        "next" said between counts) is honoured, not thrown away."""
        if self.pending_cmd in CHOICES:
            cmd = self.pending_cmd; self.pending_cmd = None
            if cmd == 'stop': self.stop_routine = True
            return cmd
        self.pending_cmd = None
        if line: self.say_line(line, key=key, ttl=12)
        t0 = self.now(); reprompted = False
        while not self.stop and not self.stop_routine and not self.safety_stop:
            if self.max_s and self.now() - self.t0 > self.max_s: self.stop = True; break
            cmd = self.pending_cmd
            if cmd in CHOICES:
                self.pending_cmd = None
                if cmd == 'stop': self.stop_routine = True
                return cmd
            end = self.line_end.get(key, -1.0) if line else t0
            after = (end - 0.3) if (end >= t0 or not line) else None
            if after is not None:
                if self.affirm_t > after: return 'yes'
                if self.deny_t > after and self.now() >= self.hold_until: return 'no'
            await self.frame_step(None); await self.sleep(1.0 / self.fps)
            quiet = self.now() - max(t0 if after is None else max(after, t0), self.last_caller_text_t)
            if reprompt and not reprompted and after is not None and quiet > seconds and self.now() >= self.hold_until:
                reprompted = True; self.say_line(reprompt, key=key + '_re', ttl=12)
            if not line and quiet > seconds * 3: return None              # listening-only call: hand back so the caller can be asked again
        return 'stop'

    async def learn(self, mode_name='learn', skip_warmup=False):
        """A patient lesson at the caller's pace. Each move: a short spoken breakdown and a very slow demo (0.5x), then "got it, or
        again?"; practice with the ghost; then the caller's own tries, each followed by one direct, actionable tip and a question
        (again / a little faster / next). Speed rises one notch at a time and only when the caller says yes; two misses simplify
        the move (halves, slower, arms first). Silence never skips ahead. Steps are combined only after they are learned, and the
        full-speed challenge is offered only after the routine is solid at 0.9x."""
        pass_mark = dict(beginner=65, intermediate=72, advanced=80)[self.difficulty]
        start_rung = START_RUNG[self.difficulty]; self.stop_routine = False; self._prev_item_score = {}
        if self.difficulty != 'beginner' and not skip_warmup and self.call_mode == 'solo': await self.warmup()
        plan, n = self.learn_plan(self.key); i = 0; completed_all = False
        self.say_line(self.pick('lesson', ["We'll learn this one move at a time, nice and slow. No rush.",
                                           "Let's take it one small move at a time. Super slow to start."]), key='lesson')
        await self.idle(3.0)
        while i < len(plan) and not self.stop and not self.stop_routine:
            item = plan[i]
            res = await self.teach_item(item, n, start_rung, pass_mark)
            if res == 'restart':
                plan, n = self.learn_plan(self.key); i = 0; start_rung = START_RUNG[self.difficulty]
                pass_mark = dict(beginner=65, intermediate=72, advanced=80)[self.difficulty]; continue
            if res == 'stop': break
            if res == 'full': i = len(plan) - 1; continue
            kind, idx = item[0], item[1]
            if res == 'passed' and ((kind == 'C' and idx == n - 1) or (n == 1 and kind == 'S')): completed_all = True
            i += 1; self.phrase = max(self.phrase, i)
        if completed_all and not self.stop and not self.stop_routine:
            self.extras['combo_run'] = True
            ans = await self.wait_choice("You've got the whole routine! Want to try it at full speed for a score?", key='challenge',
                                         reprompt="No pressure. Say yes for the full-speed challenge, or no to finish here.")
            if ans in ('yes', 'faster'):
                key, pi = choreo.with_combined(self.key, n - 1) if n > 1 else (self.key, 0)
                eng = await self.run_phrase(pi, RUNGS[-1] * self.tempo, show_repeats=0, step=dict(label='CHALLENGE'), key=key, stage='CHALLENGE', ghost_overlay=False)
                if eng.result is not None:
                    self.record_round(eng, item='challenge'); self.rounds.append(dict(phrase=pi, kind='CH', tempo=RUNGS[-1] * self.tempo, result=eng.result))
        if self.call_mode == 'squad' and self._last_squad:
            totals = {x['label']: 3.0 * (x['score'] or 0.0) for x in self._last_squad['players']}
            await self.finish_squad(totals, self._last_squad, mode='squad_learn')
        else:
            await self.finish_session(mode_name)

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
        if kind == 'C':
            self.say_line(f"Now let's put steps 1 to {idx + 1} together. Nice and slow.", key='step'); await self.idle(3.0)
        else:
            if kind == 'H': self.say_line(self.pick('part', [f"First move: {part}. Watch me, really slow.", f"Next little move: {part}. Watch first."]), key='step')
            elif self.difficulty == 'beginner': self.say_line("Now the whole step, both moves together. Watch.", key='step')
            await self.idle(2.5)
            if kind != 'S' or self.difficulty != 'beginner': self.say_breakdown(key, pi)
            demos = 0
            while not self.stop and not self.stop_routine:
                await self.run_phrase(pi, RUNGS[rung] * mult, show_repeats=2 if kind == 'H' else 1, step=step(), key=key, stage='WATCH', demo=True)
                demos += 1
                ans = await self.wait_choice(self.pick('gotit', ["Got it? Say again to see it once more, or ready to try it with me.",
                                                                 "Make sense? Want to see it again, or ready to try it with me?"]),
                                             key='gotit', reprompt="Take your time. Say again, or ready.")
                if ans == 'stop': return 'stop'
                if ans == 'next': return 'skipped'
                if ans == 'explain': await self.explain(key, pi); continue
                if ans in ('again', 'no'):
                    if demos >= 2 and rung > 0: rung -= 1
                    continue
                if ans == 'slower': rung = max(0, rung - 1); continue
                break                                                        # yes / ready / still quiet after the reprompt: try it together
            if self.stop or self.stop_routine: return 'stop'
            # 2. practise with the ghost, no score
            self.say_line(self.pick('withme', ["Do it with me. Follow the ghost on your body.", "Together now. Just follow the ghost."]), key='step')
            await self.idle(2.0)
            await self.run_phrase(pi, RUNGS[rung] * mult, show_repeats=0, step=step(), key=key, stage='WITH ME', practice=True)
            if self.stop or self.stop_routine: return 'stop'
            self.say_line("Nice. Now on your own, same speed.", key='step'); await self.idle(2.0)
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
            self.summarise(eng, xp=self.xp_for_attempt(score))            # the grade and one direct tip ("Lift your left arm higher on 5 to 7")
            await self.idle(4.0)
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
                    self.say_line("Let's slow it right down and try again.", key='step'); await self.idle(2.5); continue
                if focus is None:
                    focus = 'arms'; self.telemetry.event('adapt', action='arms_first')
                    self.say_line("Let's just do the arms this time. The legs can come later.", key='step'); await self.idle(2.5); continue
            if passed and rung >= target:
                q = "That's it! Want to go again, or are you ready for the next move?" if not final else "You've got it! Ready to finish up?"
                yes_means = 'next'
            elif passed:
                q = self.pick('fasterq', ["Nice! Ready to try it a little faster?", "That was solid. Want a little more speed?"])
                yes_means = 'faster'
            elif tries >= 6:
                q = "This one's tricky. Want to keep practising, slow it down, or move on for now?"
                yes_means = 'again'
            else:
                q = self.pick('againq', ["Let's go again. Ready?", "Once more? Say ready when you are."])
                yes_means = 'again'
            ans = await self.wait_choice(q, key='choice', reprompt="Whenever you're ready. Say again, slower, or next.")
            while ans is None and not self.stop and not self.stop_routine:     # still quiet: keep waiting, never skip ahead
                ans = await self.wait_choice(None, key='choice')
            if ans == 'yes': ans = yes_means
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
                    self.say_line("A little faster. You've got this.", key='step'); await self.idle(2.0)
                else:
                    self.say_line("Let's nail it at this speed first, then we'll speed up.", key='step'); await self.idle(2.5)
                continue
            if ans == 'slower':
                rung = max(0, rung - 1); self.say_line("Sure, slower.", key='step'); await self.idle(1.5); continue
            if ans == 'simplify':
                if not halves_done: halves_done = True; await self.practice_halves(key, idx, label)
                else: rung = max(0, rung - 1); focus = 'arms'; self.say_line("Arms only, slower. Legs later.", key='step'); await self.idle(2.0)
                continue
            if ans == 'explain':
                await self.explain(key, pi)
                await self.run_phrase(pi, RUNGS[0] * mult, show_repeats=1, step=step(), key=key, stage='WATCH', demo=True)
                continue
        return 'stop'

    async def practice_halves(self, key, section, label):
        """The step in two 4-count halves, very slow: watch each half, try it with the ghost, then the whole step again."""
        self.say_line("Let's break it into two small pieces, really slow.", key='step'); await self.idle(2.5)
        for h in (0, 1):
            if self.stop or self.stop_routine: return
            k2, hp = choreo.with_half(key, section, h)
            await self.run_phrase(hp, RUNGS[0] * self.tempo, show_repeats=2, step=dict(label=f'{label} · PART {h + 1}'), key=k2, stage='WATCH', demo=True)
            await self.run_phrase(hp, RUNGS[0] * self.tempo, show_repeats=0, step=dict(label=f'{label} · PART {h + 1}'), key=k2, stage='WITH ME', practice=True)
        self.say_line("Now the whole step again.", key='step'); await self.idle(2.0)

    async def warmup(self):
        try: wk = choreo.generate('fitness', 'beginner', n_phrases=1, seed=7)
        except Exception: return
        wk['title'] = 'Warm-up'; saved = self.key
        self.say_line("Quick eight-count warm-up to get loose, no score. Say skip if you're already warm.", key='warmup')
        await self.idle(3.0)
        if self.pending_cmd == 'next' or self.stop: self.pending_cmd = None; return
        self.in_warmup = True; self.telemetry.event('warmup')
        try: await self.run_phrase(0, 0.7, show_repeats=1, step=dict(label='WARM-UP'), key=wk, stage='WARM-UP', practice=True)
        finally: self.in_warmup = False; self.key = saved; self.recap = None
        if not self.stop: self.say_line("Nice, you're warm. Now the real thing.", key='warmup'); await self.idle(2.5)

    # ---- battle: three rounds against the reference ghost ----
    async def battle(self):
        """Ghost battle: round 1 the first phrase, round 2 the next, round 3 freestyle (no ghost, groove scoring). An HP bar shows the
        ghost getting chipped by every clean count; 285 of 300 is a K.O."""
        if not self.in_warmup and self.call_mode == 'solo': await self.warmup()
        n_ph = len([p for p in self.key['phrases'] if not p.get('combined')])
        tempo = (self.key.get('default_tempo') or LEVEL_CFG[self.difficulty]['tempo']) * self.tempo
        plan = [(0, tempo, False), (min(1, n_ph - 1), tempo, False), (0, tempo, True)]
        total = 0.0; self.battle_hud = dict(ghost_hp=300.0, ghost_max=300.0, round=1, rounds=3, round_floor=200.0)
        best_r = None
        for i, (pi, tmp, freestyle) in enumerate(plan):
            if self.stop: break
            self.battle_hud.update(round=i + 1, round_floor=max(0.0, 300.0 - total - 100.0))
            self.say_line(f"Round {i + 1} of 3!" + (" Freestyle: no ghost, dance however you like, right on the beat." if freestyle else " Follow the ghost."), key='round')
            await self.idle(2.5)
            eng = await self.run_phrase(pi, tmp, show_repeats=1 if i else 2, freestyle=freestyle, step=dict(label=f'ROUND {i + 1}/3'), stage='CHALLENGE')
            if eng.result is None: break
            scored, s, r, cue = self.record_round(eng, item=f'R{i}')
            total += s if scored else 0.0; self.battle_hud['ghost_hp'] = max(0.0, 300.0 - total)
            if scored and (best_r is None or s > (best_r.get('score') or 0)): best_r = r
            self.rounds.append(dict(round=i + 1, phrase=pi, tempo=tmp, result=r))
            self.summarise(eng, round_label=f"Round {i + 1} of 3. ", xp=self.xp_for_attempt(s) if scored else None)
            await self.idle(4.5)
            if self.pending_cmd == 'stop' or self.stop_routine: break
        ko = total >= 285
        self.fx.add('ko', self.now(), text='K.O.!' if ko else 'GHOST WINS'); self.sfx('ko' if ko else 'stamp')
        self.extras['battle_done'] = len(self.rounds) == 3
        await self.finish_session('battle', total=total, best_r=best_r)

    async def daily(self):
        """Today's routine, same for everyone: the first two steps, then together, then a rank on today's board."""
        if not self.key.get('daily'): self.key = choreo.daily()
        self.extras['daily'] = True
        await self.warmup()
        saved_plan = self.learn_plan
        self.learn_plan = lambda key: ([('S', 0, None), ('S', 1, None), ('C', 1, None)], 2)
        try:
            await self.learn(mode_name='daily', skip_warmup=True)
        finally:
            self.learn_plan = saved_plan

    async def squad_battle(self):
        """Squad battle: every dancer in frame against the ghost and each other, two phrases plus freestyle; MVP, crew sync, one crew code."""
        n_ph = len([p for p in self.key['phrases'] if not p.get('combined')])
        tempo = (self.key.get('default_tempo') or LEVEL_CFG[self.difficulty]['tempo']) * self.tempo
        plan = [(0, tempo, False), (min(1, n_ph - 1), tempo, False), (0, tempo, True)]
        totals = {}; last_sq = None
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
                for x in sq['players']:
                    totals[x['label']] = totals.get(x['label'], 0.0) + (x['score'] or 0.0)
            self.record_round(eng, item=f'Q{i}')
            self.rounds.append(dict(round=i + 1, phrase=pi, squad={k: v for k, v in (sq or {}).items() if k in ('ranking', 'crew_sync', 'crew_mean', 'squad_title')}))
            self.summarise(eng, round_label=f"Squad round {i + 1} of 3. ")
            await self.idle(4.5)
        self.extras['squad'] = True
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
                         ". Address dancers by colour or name only; never describe anyone's looks or guess who they are.")

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

    # ---- end of a run: XP, level-up, achievements, the scorecard ----
    async def finish_session(self, mode, total=None, best_r=None):
        scored_rounds = [x for x in self.rounds if (x.get('result') or {}).get('scored')]
        best_r = best_r or max((x['result'] for x in scored_rounds), key=lambda r: r.get('score') or 0, default=None)
        best = float((best_r or {}).get('score') or 0)
        if self.stop and not scored_rounds:
            self.final_card = None; return
        score100 = (total / 3.0) if total is not None else best
        photo = await self.finish_pose()
        posted = await self.backend.call('POST', '/dance/result', dict(name=self.name, key=self.key['name'], mode=mode, total=total if total is not None else best,
                                                                      score100=round(score100, 1), challenge=self.code if self.challenge else None,
                                                                      rounds=[dict(phrase=x.get('phrase'), tempo=x.get('tempo'), score=(x.get('result') or {}).get('score')) for x in self.rounds]))
        self.my_code = (posted or {}).get('code')
        versus = (posted or {}).get('versus'); winner = (posted or {}).get('winner'); board = (posted or {}).get('board')
        if versus and winner and winner != 'tie' and winner == (posted or {}).get('name'): self.extras['duel_win'] = True
        daily_rank = None
        if mode == 'daily' and self.my_code:
            lb = await self.backend.call('GET', f"/dance/leaderboard?key={self.key['name']}") or []
            daily_rank = next((i + 1 for i, row in enumerate(lb) if row.get('code') == self.my_code), None)
            if daily_rank: board = dict(rank=daily_rank, n=len(lb), owner="today's daily")
        aw = await self.award(mode)
        title = progression.title_for(best_r or {}, dict(duel_win=self.extras.get('duel_win'), glow_up=any(x.get('glow_up') for x in self.session_rounds)))
        brag = await self.brag_lines(score100, board, mode)
        line = f"Session over ({mode}). Best step {best:.0f}, grade {progression.grade(best)}, title '{title}'."
        if total is not None: line += f" Battle total {total:.0f} of 300; the ghost scores 285" + (", so it's a K.O." if total >= 285 else '.')
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
        bits.append("Say rematch to go again.")
        self.say_line(' '.join(bits), key='wrap', ttl=20)
        perfects = sum(1 for row in (best_r or {}).get('per_beat', []) if row.get('verdict') == 'green') if best_r and best_r.get('scored') else None
        self.final_card = dict(result=best_r, key=self.key['name'], title=self.key.get('title'), code=self.my_code, award_title=title, xp=aw,
                               achievements=(aw or {}).get('new_achievements', []), handle=self.handle, mode_label=MODE_LABEL.get(mode, mode.title()),
                               perfects=perfects, total_line=(f"{total:.0f}/300" if total is not None else None),
                               lines=([f"vs {versus['name']}: winner {winner}"] if versus else []), photo=photo, brag=brag)
        self.log.append(dict(t=round(self.now() - self.t0, 2), event='result_posted', code=self.my_code, versus=versus, winner=winner, xp=(aw or {}).get('gained')))
        await self.idle(9.0)

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
        self.my_code = (posted or {}).get('code')
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
        self.say_line(f"{win} wins! You're {sq.get('squad_title', 'a crew')}." + (f" Your crew code is {' '.join(str(self.my_code))}. Screenshot this and challenge another crew." if self.my_code else ''), key='wrap', ttl=20)
        self.final_card = dict(result=None, key=self.key['name'], title=self.key.get('title'), code=self.my_code, squad=sq, handle=self.handle, mode_label=MODE_LABEL.get(mode, 'Squad'))
        await self.idle(9.0)

    # ---- telephone ----
    async def record_phrase(self, n_beats=8, countin_beats=4, bpm=None):
        """Telephone: record the caller's own eight counts to a metronome and build a key from them (features, mask, energy,
        skeleton on the beat grid). Returns the key dict or None when the camera saw too little."""
        from team.dance.features import features_from_landmarks, visibility_mask, hip_energy, skeleton2d_normalized, resample_to_grid, resample_mask
        from team.dance.reference import key_from_arrays
        bpm = float(bpm or self.key['bpm']) * self.tempo; beat_s = 60.0 / bpm; pb = 4
        bank = getattr(self.job, 'bank', None)
        if bank is not None:
            yt = self.bank_pcm('lets_go') if self.bank_has('lets_go') else self.voice_lines.get('your_turn')
            dance_audio.register_phrase_track(bank, bpm, 0, countin_beats, n_beats + 1, voice={'your_turn': yt} if yt is not None else {})
        t_start = self.now(); self.say(dance_audio.TRACK, 'announce')
        rows = []; prev = None; prev_t = None; t_attempt = t_start + countin_beats * beat_s
        holder = getattr(self, 'engine_holder', None)
        if holder is not None:                       # the simulator's stand-in caller follows this clock, like an engine's
            class _RecClock:
                state = 'countin'; attempt_s = (n_beats + 1) * beat_s
                def attempt_t(_s, t): return t - t_attempt - self.rtt_s
            rc = _RecClock(); holder['engine'] = rc
        hud_base = dict(state='countin', key='your move', title='your move', phrase=0, n_beats=n_beats, current_beat=-1, countdown=None, verdicts=[None] * n_beats,
                        running_score=None, combo=0, result=None, cue=None, bpm=round(bpm, 1), tempo=self.tempo, fx=self.fx, step=dict(label='RECORDING'))
        while self.now() < t_attempt + (n_beats + 1) * beat_s and not self.stop:
            if holder is not None: holder['engine'].state = 'attempt' if self.now() >= t_attempt else 'countin'
            frame, t = self.newest_frame()
            if frame is not None:
                bgr, lm, w = await self.prepare(frame, int((self.now() - self.t0) * 1000))
                tp = self.now() - t_attempt - self.rtt_s
                if lm is not None and tp >= 0:
                    rows.append((tp, features_from_landmarks(lm, w), visibility_mask(lm).astype(np.float32), hip_energy(prev, lm, (t - prev_t) if prev_t else 0.0), skeleton2d_normalized(lm)))
                    prev = lm; prev_t = t
                hud = dict(hud_base); b = (self.now() - t_attempt) / beat_s; hud['t'] = self.now()
                if b < 0:
                    hud['state'] = 'countin'; hud['countdown'] = max(1, int(np.ceil(-b))); hud['countin_label'] = max(5, min(8, 8 - int(np.ceil(-b)) + 1)); hud['beat_pos'] = b
                else: hud['state'] = 'attempt'; hud['current_beat'] = int(b); hud['verdicts'] = ['green' if k < int(b) else None for k in range(n_beats)]
                await self.render_push(bgr, lm, hud, ghost=None, extra=dict(t=self.now()))
            await self.sleep(1.0 / self.fps)
        if len(rows) < n_beats * 2: return None
        grid = np.arange(n_beats * pb) * beat_s / pb
        ts = [r[0] for r in rows]
        F = resample_to_grid(ts, [r[1] for r in rows], grid); M = resample_mask(ts, [r[2] for r in rows], grid)
        E = resample_to_grid(ts, np.asarray([r[3] for r in rows], np.float32), grid)[:, 0]
        S = resample_to_grid(ts, np.asarray([r[4] for r in rows], np.float32).reshape(len(rows), -1), grid).reshape(len(grid), 33, 2)
        if M.mean() < 0.6: return None
        return key_from_arrays(f"chain_{self.name or 'anon'}", bpm / self.tempo, F, M=M, E=E, S=S, per_beat=pb, counts_per_phrase=n_beats, source='telephone', save=False)

    async def telephone(self):
        """Dance telephone. No code: record a move, start a chain, speak the code. With a code: imitate the previous link, get
        scored against it (and against the original), become the next link."""
        if not self.code:
            self.ask("One short sentence: say you'll count them in and they should invent any eight-count move, and you'll remember it.")
            await self.idle(4.0)
            key = await self.record_phrase()
            if key is None:
                self.ask("One short sentence: say you couldn't see enough of them; ask them to step back and say 'again'."); await self.idle(5.0); return
            posted = await self.backend.call('POST', '/dance/chain', dict(key=self.key['name'], link=dict(name=self.name or 'anonymous', key=key, score=100.0, fidelity_prev=None, fidelity_origin=None)))
            self.my_code = (posted or {}).get('code')
            self.context(f"Their move is recorded as link 1 of a new chain. The chain code is {self.my_code}. Read it digit by digit, twice, and tell them to pass it to the next person.")
            self.ask("Two short sentences: say you've got their move and read the chain code digit by digit, twice.")
            self.rounds.append(dict(mode='telephone', link=1, code=self.my_code))
            self.final_card = dict(result=None, title='telephone, link 1', code=self.my_code, lines=['pass this code to the next dancer'], handle=self.handle, mode_label='Telephone')
            await self.idle(8.0); return
        chain = await self.backend.call('GET', f'/dance/chain/{self.code}')
        if not chain or chain.get('error'):
            self.ask("One short sentence: say that code doesn't match a chain and ask them to repeat it digit by digit."); await self.idle(5.0); return
        links = chain['links']; prev_key = links[-1]['key']; origin_key = links[0]['key']
        self.key = prev_key; self.key.setdefault('title', 'telephone')
        self.ask(f"One short sentence: say this is link {len(links) + 1}; watch the previous dancer's move twice, then it's their turn.")
        await self.idle(3.0)
        eng = await self.run_phrase(0, self.tempo, step=dict(label=f'LINK {len(links) + 1}'))
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
        self.rounds.append(dict(mode='telephone', link=(posted or {}).get('link_no'), fidelity_prev=fidelity_prev, fidelity_origin=fidelity_origin, result=r))
        self.context(f"Telephone link {(posted or {}).get('link_no')}: fidelity to the previous dancer {fidelity_prev:.0f} of 100, to the original {fidelity_origin if fidelity_origin is None else round(fidelity_origin)}. "
                     f"Chain code stays {self.code}; the drift is the joke. The one cue: {eng.cue['text'] if eng.cue else 'none'}.")
        self.ask("Two short sentences: say how close they got to the previous dancer and to the original, with the numbers, and tell them to pass the same code on.")
        self.final_card = dict(result=r, title=f"telephone link {(posted or {}).get('link_no')}", code=self.code, handle=self.handle, mode_label='Telephone',
                               lines=[f"vs previous {fidelity_prev:.0f}", f"vs original {'' if fidelity_origin is None else round(fidelity_origin)}"])
        await self.idle(8.0)

    # ---- class mode: the teacher records, students are tested by code, the teacher hears the report ----
    async def class_mode(self):
        """No code: the teacher. Each phrase is demonstrated to a count-in (DANCE_CLASS_PHRASES, default 2; DANCE_CLASS_DEMOS=2
        asks for a second demo and reports how well the two agree), the phrases become one key stored with the class, the
        teacher hears the class code. With a code: a student, unless the caller is the teacher or says "report"."""
        if not self.code:
            await self.class_teacher(); return
        cls = await self.backend.call('GET', f'/dance/class/{self.code}')
        if not cls or cls.get('error'):
            self.ask("One short sentence: say that code doesn't match a class and ask them to repeat it digit by digit."); await self.idle(5.0); return
        is_teacher = bool(self.name) and self.name.lower() == str(cls.get('teacher', '')).lower()
        if is_teacher or self.pending_cmd == 'report' or (os.environ.get('DANCE_CLASS_ROLE') == 'teacher'):
            self.pending_cmd = None; await self.class_report(cls); return
        await self.class_student(cls)

    async def class_teacher(self):
        from team.dance.reference import key_from_arrays, save_key, phrase_slice
        from team.dance.scoring import score_phrase
        n_phrases = int(os.environ.get('DANCE_CLASS_PHRASES', '2') or 2); demos = int(os.environ.get('DANCE_CLASS_DEMOS', '1') or 1)
        self.ask(f"One or two short sentences: say you're in class mode, you'll count them in for {n_phrases} phrases of eight counts, "
                 f"they demonstrate each one and you'll remember it; their students will get a code.")
        await self.idle(6.0)
        parts = []; agreements = []
        for i in range(n_phrases):
            if self.stop: break
            self.ask(f"One short sentence: phrase {i + 1}, get ready, four clicks then demonstrate."); await self.idle(3.5)
            key = await self.record_phrase(); tries = 1
            while key is None and tries < 2 and not self.stop:
                self.ask("One short sentence: say you couldn't see all of them; step back, and you'll count in again."); await self.idle(5.0)
                key = await self.record_phrase(); tries += 1
            if key is None: continue
            if demos >= 2 and not self.stop:
                self.ask("One short sentence: say that was good and ask for the same phrase once more so you can check it."); await self.idle(4.0)
                key2 = await self.record_phrase()
                if key2 is not None:
                    aF, aM, aE, _ = phrase_slice(key, 0); bF, bM, bE, _ = phrase_slice(key2, 0)
                    agree = score_phrase(bF, bM, bE, aF, aM, aE, per_beat=key['per_beat'])
                    agreements.append(agree.get('score', 0.0) if agree.get('scored') else 0.0)
                    worst = max((agree.get('per_group') or {}).items(), key=lambda kv: (kv[1] if kv[1] is not None else -1), default=(None, None))
                    if agree.get('scored') and agree['score'] < 60 and worst[0]:
                        self.ask(f"One short sentence: say their two demos differed, mostly in the {worst[0]}, and you're keeping the second one.")
                    key = key2
            parts.append(key)
            self.ask(f"One short sentence: say phrase {i + 1} is saved" + (", and move on" if i + 1 < n_phrases else '') + "."); await self.idle(3.0)
        if not parts:
            self.ask("One short sentence: say you couldn't capture a phrase this time and to call back with the phone propped lower."); await self.idle(5.0); return
        F = np.concatenate([np.asarray(k['features'], np.float32) for k in parts]); M = np.concatenate([np.asarray(k['mask'], bool) for k in parts])
        E = np.concatenate([np.asarray(k['energy'], np.float32) for k in parts]); S = np.concatenate([np.asarray(k['skeleton'], np.float32) for k in parts])
        bpm = float(parts[0]['bpm']); teacher = self.name or 'teacher'
        key = key_from_arrays(f"class_{teacher.lower()}", bpm, F, M=M, E=E, S=S, per_beat=parts[0]['per_beat'], counts_per_phrase=parts[0]['counts_per_phrase'], source=f'class mode, taught by {teacher}', save=False)
        key['style'] = 'class'; key['difficulty'] = 2
        posted = await self.backend.call('POST', '/dance/class', dict(teacher=teacher, key=key))
        self.my_code = (posted or {}).get('code')
        if self.my_code:
            key['name'] = f'class_{self.my_code}'
            try: save_key(key)
            except Exception as e: print(f'[dance] class key not saved locally: {e!r}', flush=True)
        self.rounds.append(dict(mode='class', role='teacher', phrases=len(parts), code=self.my_code, agreements=agreements))
        self.extras['teacher'] = True
        self.context(f"Class recorded: {len(parts)} phrase(s) at {bpm:.0f} bpm by {teacher}. The class code is {self.my_code}. "
                     + (f"Demo agreement scores: {[round(a) for a in agreements]}. " if agreements else '')
                     + "Students call, say 'class' and the code; the teacher calls back, says their name and 'report'.")
        self.ask("Two short sentences: say the phrases are saved, then read the class code digit by digit, twice, and tell them to share it with their students.")
        if self.name: await self.award('class', rounds=[])
        self.final_card = dict(result=None, title=f'class by {teacher}', code=self.my_code, lines=[f'{len(parts)} phrase(s) at {bpm:.0f} bpm', 'students: say "class" and this code'],
                               handle=self.handle, mode_label='Class')
        await self.idle(8.0)

    async def class_student(self, cls):
        key = cls.get('key')
        if not key or not key.get('phrases'):
            self.ask("One short sentence: say that class has no phrases yet."); await self.idle(4.0); return
        self.key = key; teacher = cls.get('teacher') or 'the teacher'; self.key.setdefault('title', f"{teacher}'s class")
        n_ph = len(key['phrases']); max_attempts = int(os.environ.get('DANCE_CLASS_ATTEMPTS', '2') or 2)
        self.ask(f"One short sentence: say this is {teacher}'s class, {n_ph} phrase(s); the ghost is {teacher}'s own body, watch it twice, then it's their turn.")
        await self.idle(4.0)
        best = {}; results = {}
        pi = 0
        while pi < n_ph and not self.stop:
            attempts = 0; tempo = self.tempo
            while attempts < max_attempts and not self.stop:
                eng = await self.run_phrase(pi, tempo, step=dict(label=f'PHRASE {pi + 1}/{n_ph}')); attempts += 1
                r = eng.result or {}
                if eng.result is None: break
                self.record_round(eng, item=f'K{pi}')
                self.rounds.append(dict(mode='class', role='student', phrase=pi, tempo=tempo, result=r))
                if r.get('scored') and r['score'] > best.get(pi, -1): best[pi] = r['score']; results[pi] = (eng, r)
                self.summarise(eng, round_label=f"Class phrase {pi + 1} of {n_ph}. ")
                await self.idle(5.0)
                cmd = self.pending_cmd; self.pending_cmd = None
                if cmd == 'stop': self.stop = True; break
                if cmd == 'slower': tempo = max(0.6, round(tempo - 0.15, 2)); continue
                if cmd == 'next' or (r.get('scored') and r['score'] >= 80): break
                if attempts < max_attempts: self.ask("One short sentence: say you'll run that phrase once more."); await self.idle(2.0)
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
                     + (f"Their average {total:.0f}; the class average is {mean:.0f} across {n} student(s); they rank {rank} of {n}. " if total is not None else '')
                     + "Never name other students.")
        self.ask("Two short sentences: give their phrase scores, then say how they stand against the class average, warmly. No names of others.")
        aw = await self.award('class') if results else None
        lines = [f"phrase {i + 1}: {best[i]:.0f}" for i in sorted(best)] + ([f"class average {mean:.0f}, you are {rank} of {n}"] if total is not None else [])
        last = results[max(results)][1] if results else None
        self.final_card = dict(result=last, title=f"{teacher}'s class", code=self.code, lines=lines, xp=aw, handle=self.handle, mode_label='Class')
        await self.idle(8.0)

    async def class_report(self, cls):
        rep = await self.backend.call('GET', f'/dance/class/{self.code}/report')
        if not rep or rep.get('error'):
            self.ask("One short sentence: say the report isn't available right now."); await self.idle(4.0); return
        n = rep.get('n_students', 0); hm = rep.get('heatmap', [])
        rows = '; '.join(f"{r['name']}: " + ', '.join('-' if v is None else f'{v:.0f}' for v in r['scores']) for r in hm) or 'no students yet'
        self.rounds.append(dict(mode='class', role='teacher_report', code=self.code, n_students=n, suggestion=rep.get('suggestion')))
        self.context(f"Class report for code {self.code} ({rep.get('n_phrases')} phrases): {n} student(s). Scores by phrase: {rows}. "
                     f"Phrase means: {rep.get('phrase_mean')}. Suggestion: {rep.get('suggestion')}"
                     + (f" Not attempted: {', '.join(rep['not_attempted'])}." if rep.get('not_attempted') else ''))
        self.ask("Two or three short sentences to the teacher: how many students danced, the phrase where the class drops and the weak count, then the suggestion.")
        lines = [f"{n} student(s)"] + [f"{r['name']}: " + '  '.join('-' if v is None else f'{v:.0f}' for v in r['scores']) for r in hm[:6]]
        if rep.get('weakest_phrase') is not None: lines.append(f"weak: phrase {rep['weakest_phrase'] + 1}" + (f", count {rep['weakest_beat'] + 1}" if rep.get('weakest_beat') is not None else '') + (f", {rep['weakest_group']}" if rep.get('weakest_group') else ''))
        self.final_card = dict(result=None, title='class report', code=self.code, lines=lines, mode_label='Class')
        await self.idle(10.0)


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


class _FakeVoice:
    def __init__(self, log, now): self.log = log; self.now = now
    def ask_voice(self, prompt): self.log.append(dict(t=round(self.now(), 2), ask=prompt[:100])); return 'spoken'
    def _context(self, text): self.log.append(dict(t=round(self.now(), 2), context=text[:140]))


class _FakeJob:
    def __init__(self): self.bound = 'sim-caller'; self.latest = None; self.closed = False; self.room = None; self.player = None; self.voice = None; self.bank = dance_audio.FakeBank()


async def simulate(video, key, mode, out, offset=0.0, fast=True, tempo=1.0, rtt=0.0, fps=15.0, pose_variant='full', name=None, lobby=False, says=(),
                   players=1, background=False, gestures=(), crop_bottom=0.0):
    """players=2: the stand-in caller is the clip twice side by side, the right copy a quarter second late (a squad).
    background=True: a small copy of the clip pasted in the corner (someone walking behind a solo dancer).
    gestures: [(t, 'pause'|'ready', seconds)] synthetic held poses injected into the landmarks."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): raise SystemExit(f'cannot open {video}')
    vfps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    clock = dict(t=0.0); log = []
    def now(): return clock['t']
    job = _FakeJob(); job.player = _FakePlayer(log, now); job.voice = _FakeVoice(log, now)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        frames.append(bgr)
    cap.release()
    print(f'[sim] {video}: {len(frames)} frames at {vfps:g} fps; the caller "starts" at t=0 and the clip plays once (loops if the session is longer)', flush=True)
    dur = len(frames) / vfps
    holder = {}

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
        if crop_bottom > 0:                                    # the camera too high / too close: the feet are out of frame
            h = f.shape[0]; f = cv2.resize(f[:int(h * (1 - crop_bottom))], (f.shape[1], h))
        if background:
            f = f.copy(); h, w = f.shape[:2]; small = cv2.resize(frames[int(((vt + 1.3) % dur) * vfps) % len(frames)], (w // 4, h // 4))
            f[h // 12:h // 12 + small.shape[0], w - small.shape[1] - 4:w - 4] = small
        return f
    sim = {}; pending_says = sorted(says, key=lambda x: x[0])

    async def fake_sleep(s_):
        clock['t'] += s_
        job.latest = (frame_at(clock['t']), 0, clock['t'], clock['t'], int(clock['t'] * vfps))
        while pending_says and pending_says[0][0] <= clock['t'] and sim.get('s') is not None:   # scripted caller lines ("3:I'm Maya")
            t_say, text = pending_says.pop(0); log.append(dict(t=round(clock['t'], 2), caller=text))
            sim['s'].on_worker_event(dict(type='voice', role='caller', text=text))
        if not fast: await asyncio.sleep(s_)
        else: await asyncio.sleep(0)
    job.latest = (frame_at(0.0), 0, 0.0, 0.0, 0)
    s = DanceSession(job, key_name=key, mode=mode, tempo=tempo, rtt_s=rtt, fps=fps, pose_variant=pose_variant, debug_mp4=out, now=now, sleep=fake_sleep)
    s.log = log; s.engine_holder = holder; s.lobby_enabled = bool(lobby); sim['s'] = s
    if gestures:
        from team.dance.gestures import GestureDetector as _GD
        real = s._gestures
        def scripted(now_, lm, in_attempt):
            for (tg, kind, secs) in gestures:
                if tg <= now_ <= tg + secs:
                    if kind == 'pause' and in_attempt and now_ >= tg + 1.3: s.pause_req = True; s.telemetry.event('gesture', kind='pause', scripted=True)
                    if kind == 'ready' and now_ >= tg + 1.1: s._cmd('resume' if s.paused else 'ready', 'gesture')
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
    ap.add_argument('--gesture', action='append', default=[], help='scripted held pose "t:pause:2" or "t:ready:1.5"; repeatable')
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
    gests = []
    for g in a.gesture:
        t, kind, secs = (g.split(':') + ['1.5'])[:3]; gests.append((float(t), kind, float(secs)))
    asyncio.run(simulate(a.simulate, a.key, a.mode, a.out, offset=a.offset, fast=not a.realtime, tempo=a.tempo, rtt=a.rtt, fps=a.fps, pose_variant=a.pose, name=a.name,
                         lobby=a.lobby, says=says, players=a.players, background=a.background, gestures=gests, crop_bottom=a.crop_bottom))


if __name__ == '__main__':
    main()
