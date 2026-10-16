# Dance skill for Brodie (team)

A choreography coach inside the Brodie FaceTime agent. The caller sees their own video with a ghost of the move one beat
ahead, a beat timeline that colours as they dance, a running score and a combo; the agent plays a metronome at an exact BPM,
scores every count against a reference performance, speaks one cue, and loops the weak bars slower. Modes: learn, ghost battle,
daily challenge, squad battle (several friends in one camera), telephone, class. It plays like a rhythm game: PERFECT / NICE /
MISS on every count, combos, grades, playful titles, XP and levels, unlocks, a scorecard with a challenge code and a QR.
Design: `research/ideation/DANCE_AGENT_PLAN.md`. Progress and checklist: `team/PROGRESS.md`.

## How it works

- **Pose**: MediaPipe Pose Landmarker (Google, Apache 2.0), the same model the starter uses, 33 landmarks per frame. No licensed
  classifier is involved: the dance skill runs the worker with `--perception off` and does its own perception.
- **Features** (`features.py`): 14 view- and size-invariant joint-angle channels from the world landmarks, a per-channel
  visibility mask, hip energy, and a normalised 2D skeleton for the ghost. Mirroring swaps left and right.
- **Keys** (`reference.py`): a reference performance as feature samples on a beat grid (4 per beat), built from a video
  (MediaPipe offline, beat grid from the audio with librosa or a given BPM) or from arrays (motion files, tests). JSON in `library/`
  with provenance (source, licence).
- **Scoring** (`scoring.py`): banded dynamic time warp between key and attempt; spatial error on the aligned pairs, timing
  offsets from the warp path, energy from the hip-speed envelope. Score = 60 shape + 30 timing + 10 energy. Per-beat verdicts
  green/amber/red; combo = longest green run. Timing credit requires the shape to be roughly right.
- **Engine** (`engine.py`): show (ghost twice), count-in, attempt, scored, feedback. Injected clock, lag compensation, live
  per-beat verdicts, cue policy (visibility > timing > worst joint group on its longest bad run > praise).
- **Render** (`render.py`): the caller's frame to 720x1280 with the ghost placed on their hips at their torso scale, timeline,
  score, combo, state text, end card.
- **Audio** (`audio.py`, `tts_local.py`): a synthetic metronome with count-in at the exact BPM and spoken lines ("Watch", "Your
  turn") mixed in at exact offsets, registered into the starter's cue bank at runtime. Local speech is Windows System.Speech;
  re-render with the starter's Gemini TTS when the relay token is available.
- **Live** (`live.py`): a worker consumer, loaded by `--handlers team/dance/live.py`. Reads the caller's newest frame from the
  worker, runs pose at up to 15 fps in one thread, drives the engine, publishes the composed `coach-video` track, plays the
  track through the cue player, pushes the result into Gemini as context and asks for one line. Voice commands from the caller's
  transcript (`--log-caller`): a name, again, slower, faster, next, stop.

## How a call flows (v3, 2026-10-11): one coach, nothing starts by accident

States (`live.py`): **EXPLORE** (chat) -> **OFFER** -> **PREPARE** (camera) -> **CONFIRM** (ready check) -> **ACTIVITY** /
**REVIEW** -> back to EXPLORE; **PAUSED** and **SAFETY** on the side. Each state change updates Gemini's instructions (`STATE_RULES`).
- **Brodie leads (proactive).** After the hello he suggests a routine himself, once you've answered or after 14 quiet seconds.
  If you pass, he suggests a different one later, up to 3 times. Starting still needs your yes.
- **Feet are ignored.** The camera check needs head to hips. Ankle-based moves (knee bends, stance width) aren't scored or drawn
  (`DANCE_IGNORE_FEET=1`, the default).
- **Chat, then an offer.** Talking about dances, questions ("hip hop, is that possible?"), "maybe" and curiosity never move the
  session on. A clear wish ("teach me hip hop", a challenge code, or Gemini's `propose_activity` tool) becomes an **offer**: Brodie
  asks "want to start it now? a quick camera check comes first". Only a clear yes, said after that question, goes to setup.
  Changing your mind updates the offer; "not yet" keeps it; "never mind" goes back to chat.
- **One voice, exact words.** Gemini Live is the only voice and answers you itself. Our system lines are exact second-person
  sentences ("Take a couple of big steps back from your camera."), with variants, never the same line twice in a row. They go
  through one queue: one at a time, and never within 6 s of you speaking. That's why Brodie no longer reads instructions aloud.
- **Setup that listens** (`setup.py`):
  - One stable problem at a time; it must hold 2.5 s, so there are no step-back / step-in flips.
  - Said once, then Brodie waits. A check-in comes only after 15 quiet seconds, at most 3, each different.
  - Whatever you say gets Gemini's answer with a fresh "[camera now: ...]" note. "I can't step back" or "I'm sitting" switches to
    arms-only. "One sec" holds for 30 s. A lost video is noticed and waited out.
  - A person counts when both shoulders are visible, so desk and seated callers are tracked.
- **Ready check:** a clear yes or both hands up. Every stage runs inside an error boundary, so a failure returns you to chat with
  an apology, never a dead call.
- **Patient teaching, at your pace:**
  - Brodie teaches one small move at a time; beginners get each 8-count step as two 4-count moves first.
  - Each move starts with a short spoken breakdown ("counts 1 to 2, both arms go up…") and a very slow demo (0.5x, about 55 BPM).
  - Then "Got it? Again, or ready to try it with me?", and he waits for your answer.
  - Practise with the ghost on your body, then on your own. After every try: the grade, one direct tip ("Lift your left arm higher
    on counts 5 to 7"), then your choice: again, a little faster, slower, explain, or next move.
  - Speed rises one notch at a time (0.5 -> 0.6 -> 0.75 -> 0.9), only when you've passed at the current speed and said yes.
  - Two misses simplify the move: halves, then slower, then arms first.
  - Silence never skips ahead: Brodie checks in once, then keeps waiting. Steps are combined after they are learned, and full speed
    is only offered as a final challenge.
- **Avatar** (`avatar.py`): Brodie from the official character kit (`assets/brodie/`, kit art only, passes the kit's
  `check_cream.py`). States: idle, listening (volt ring), speaking (grin / closed-smile mouth), thinking (a typing bubble from the caller's last word until his voice starts), offline (sleepy).
  Big beside his words in chat, small top-left while you dance. Kit typography: Archivo + Plus Jakarta Sans.
- **FaceTime first:** callers through Chert get a portrait 720x1280 video (`DANCE_CANVAS=auto|portrait|landscape`) so a portrait
  iPhone shows everything. All HUD stays inside a safe area clear of FaceTime's controls and self-view tile.
- **Viral loop (the screenshot is the share button):**
  - At the end: "Strike your finish pose! 3-2-1". The scorecard becomes a photo of you with the kit Brodie sticker, grade, title,
    "NEW PERSONAL BEST" / "TOP N% TODAY", the code, a QR that opens FaceTime to the line, and `#BeatMyBrodie <code>`. The photo is
    never saved.
  - Brodie then says: "Screenshot this and send it to the friend who thinks they can out-dance you."
- **Chrome test page:** clean by default, with mic/camera dots and Pause / Repeat / Slower / Stop buttons (LiveKit data messages,
  state-gated like voice). `chrome.ps1 -Record` to save what Brodie shows (opt-in).

## Game layer, skills and tools (2026-10-11)

| Module | What it does |
|---|---|
| `choreo.py` | Generates routines from library phrases by style and difficulty (motion-complexity tertiles = beginner / intermediate / advanced); deterministic names `gen_<style>_<level>_<seed>` and `gen_daily_<yyyymmdd>` rebuild anywhere; combine runs (`with_combined`); Dance Cardio (fitness) style. |
| `feedback.py` | Directional, sided cues ("Lift your left arm higher on counts 5 to 7") from signed per-channel errors; mid-phrase tips after two red counts; `describe_phrase` for "explain". |
| `progression.py` | Grades S-D, titles ("Human Metronome", "Combo Menace"), XP, levels Rookie to Main Character, unlocks (intermediate at Groover, advanced at Mover), achievements. |
| `gestures.py` | Both hands up (hold 1 s) = ready/resume; arms crossed overhead and still (hold 1.2 s) = pause. Needed because the echo guard mutes the mic while music plays. |
| `multipose.py` | Up to 4 people per frame: a primary-dancer lock for solo calls (walk-ins are ignored), stable colour tags (VOLT, PINK, CYAN, AMBER) for squads; spectators in the background are filtered out. |
| `squad.py` | One engine per dancer on one music clock: per-dancer scores, Crew Sync %, MVP, squad title, one cue for the dancer furthest off. |
| `tools.py` | Gemini Live function tools: `control_session`, `set_preferences`, `check_camera`, `analyze_last_attempt`, `compare_with`, `generate_routine`, `get_profile`, `forget_me`, `get_squad_status`, `set_call_mode`, `assign_name`. Registered with `agent.update_tools`; `DANCE_TOOLS=0` falls back to keyword commands. |
| `telemetry.py` | One JSONL per session in `data/tmp/sessions/`: tool latency and errors, engine events, tracking rate, cue latency, a summary row posted to the backend (`GET /metrics`). |
| `text.py`, `fx.py`, `render.py` | Real type (Archivo Black, Space Grotesk; OFL), animations (judgment pops, ON FIRE glow, count-up reveal, grade stamp, confetti, level-up, achievement toasts, K.O.), recap card, captions, paused and safety cards, squad strip, battle HP bar, the scorecard with QR (`facetime:<DANCE_CALL_HANDLE>`). |

**Voice commands** (or the same through Gemini's tools): again, slower, faster, next/skip, pause/hold on, resume, stop, simplify/break
it down, explain/how do I, full routine, rematch/run it back, harder, easier, easy/medium/hard, hype/chill/strict coach, a style,
"left to right Maya, Dev" (squad names), a four-digit code (battle it), "forget me" twice (delete scores and progress). Pain words
(hurts, dizzy, cramp...) stop everything and show the safety card.

**Learn flow**: steps one at a time (demo before the first try only), then put together (1-2, then 1-3...). Two scores under 50 on
a step: automatic breakdown (slower, extra demo, arms first when the legs are the problem). Two passes: offer a full-speed run or
the next step. Up to three tries a step. Battle: warm-up, three rounds, ghost HP bar, K.O. at 285/300. Daily: same routine for
everyone, rank on today's board. After any run: "rematch", "harder", "easier", "battle", "daily" on the same call.

**Eval**: `python -m team.dance.eval.run_eval` runs scripted sessions (learn with lobby, battle, duel by code, safety stop,
pause/resume by gesture, forget me, walk-in, squad of two, daily) and writes `data/tmp/eval_report.json` and `data/tmp/cue_review.csv`.
`python -m team.dance.eval.snapshots [--landscape]` renders every HUD state to `data/tmp/hud/`.

## Test without a phone

```
.venv\Scripts\python -m pytest team\dance\tests -q                                   # synthetic scoring and engine tests
.venv\Scripts\python -m team.dance.offline_test selftest                              # 14 checks incl. every HUD state, QR, timing
.venv\Scripts\python -m team.dance.offline_test key --video ref.mp4 --name wave --bpm 100 --source "..." --licence "..."
.venv\Scripts\python -m team.dance.offline_test run --video attempt.mp4 --key wave --out data\tmp\run.mp4 --report data\tmp\run.json
.venv\Scripts\python -m team.dance.live --simulate attempt.mp4 --key wave --mode battle --out data\tmp\sim.mp4   # the live code path, no LiveKit
.venv\Scripts\python -m team.dance.fake_room_test                                      # the REAL worker job against a fake room
```
Any browser as the caller (Chrome on the laptop or an Android phone; no iPhone, no Chert line): run the worker, then
```
.venv\Scripts\python team\scripts\serve_web.py          # https on the LAN with a self-signed certificate (team/web/certs)
.\team\scripts\link_local.ps1 [-Facing user]              # prints https://<laptop-ip>:8443/phone.html?... and a LiveKit Meet fallback
```
Open the link in Chrome, accept the certificate warning once, tap Join: the page sends the camera and mic and shows Brodie's
composed video full size (ghost, timeline, score). The link carries a room token; do not post it. A laptop webcam sees the
upper body only: start the worker with `DANCE_MIN_VISIBLE=0.4` so arm-led keys (waack, pop) still score.
Automated browser run (no person): `.venv\Scripts\python team\scripts\chrome_test.py` launches headless Chrome through the same
page with a dance clip as its camera and reports PASS once the worker scores a phrase on the browser's frames.

Class mode in the simulator (one backend file `team/backend/team.db` is shared across runs, so the code carries over):
```
python -m team.dance.live --simulate teacher.mp4 --mode class --name Maya --out data/tmp/t.mp4        # records 2 phrases, prints a code
python -m team.dance.live --simulate student.mp4 --mode class --code 1234 --name Dev --out data/tmp/s.mp4
python -m team.dance.live --simulate any.mp4 --mode class --code 1234 --name Maya --out data/tmp/r.mp4  # the teacher's name opens the report
```
Validation so far: synthetic self-test 11/11; on the starter's recordings the key's own video scores 97, another recording of
the same exercise 74, a different exercise 26. Before filming a real move: three takes (clean, late, wrong arm) and confirm the
scorer separates them (`offline_test run` on each).

## Run live

```
copy .env.example .env.local        # LiveKit URL/key/secret from the team's project; Gemini token + relay URL from the team card
.\team\scripts\backend.ps1          # optional (events to SQLite)
.\team\scripts\dance.ps1 -Key wave -Mode learn -Debug data\tmp\dance_live.mp4       # Terminal B: the worker in dance mode
.\team\scripts\link.ps1             # Terminal C: a phone-page link (open the THIRD line on the phone)
```
Fake bridge instead of a phone: `python -m facetime.fake_bridge --room <room> --video attempt.mp4 --fps 15` with the worker running
under automatic dispatch (no agent name). On the Chert line: `.\team\scripts\dance.ps1 -AgentName <dispatch name>` then
`chert dev --agent <name>` from a Mac (the CLI wants Node 22.23.2 or 24.x).

Env for dance mode: `DANCE_KEY`, `DANCE_MODE=learn|battle|telephone`, `DANCE_TEMPO`, `DANCE_RTT` (round-trip lag, seconds;
measure with a clap), `DANCE_FPS`, `DANCE_POSE=full|lite`, `DANCE_DEBUG_MP4`, `DANCE_MAX_PHRASES`, `DANCE_CODE` (join a duel or
chain), `DANCE_MIN_VISIBLE` (fraction of joints that must be visible to score; 0.6 default, 0.4 for a desk webcam),
`DANCE_MIRROR_VIEW` (1 default: the caller sees a mirror; scoring always runs on the unflipped pose). Class mode:
`DANCE_CLASS_PHRASES` (2) phrases the teacher records, `DANCE_CLASS_DEMOS` (1; 2 = ask for a second demo and report agreement),
`DANCE_CLASS_ATTEMPTS` (2) tries a student gets per phrase, `DANCE_CLASS_ROLE=teacher` forces the report with a code.
Opening conversation: `DANCE_LOBBY` (1; 0 = start the music straight away), `DANCE_LOBBY_S` (25 s before starting with sensible
defaults). Game layer: `DANCE_CALL_HANDLE` (the Chert line's FaceTime number or Apple ID email; the scorecard QR opens a FaceTime call
to it, no QR when unset), `DANCE_DIFFICULTY` (beginner), `DANCE_TOOLS` (1), `DANCE_UNLOCK_ALL` (0; 1 opens every difficulty for demos),
`DANCE_MAX_POSES` (4), `DANCE_SESSION_LOG` (where the session log JSON goes). Pose: `DANCE_POSE_PX` (384, the short side the pose model sees), `DANCE_PARTIAL_ARMS` (0.7: arm channels visible
this much score the upper body alone when the legs are out of frame).

Modes: `learn` (ghost twice, count-in, attempt, cue, adaptive tempo), `battle` (two phrases plus a freestyle groove round, duel by
code), `telephone` (record your own move or imitate the previous link; chain code), `class` (teacher records phrases to a count-in
and gets a class code; students learn against the teacher's own skeleton by code; the teacher's next call opens with the class
report: the phrase where the class drops, the weak count, the joint group most off, a suggestion; heatmap page at
`/dance/class/<code>/page`).

## Library

Keys live in `team/dance/library/*.json`. Each carries `source` and `licence`. Sourcing order and licences:
`research/ideation/DANCE_AGENT_PLAN.md` section 1b. The test key `test_squats` is from the starter's own sample recording;
`cc_huntrix` and `cc_pm_dancer` are Wikimedia Commons clips (CC BY / BY-SA, see `data/dance_clips/SOURCES.md`).

AIST++ (`data/aist/keypoints3d`, 1,408 sequences of COCO-17 3D keypoints at 60 fps, research licence, demo only, never
redistribute): `python -m team.dance.ingest_aist --genres LH MH HO PO LO WA KR JS --per-genre 3 --music 3` builds the
`aist_<genre>_110_ch01..03` keys from the basic moves (sBM) at 110 bpm (BPM = 80 + 10 x music index). The 3D joints are mapped
to MediaPipe's 33-point layout in `ingest.py`, so the same 14 angle channels and the same scorer apply.

Mirror convention: keys are stored unmirrored (the reference's own left and right) and the display is flipped like a mirror
(`DANCE_MIRROR_VIEW=1`). The caller copies the ghost as they would a mirror, which means they raise the same arm the reference
raised. Build a key with `--mirror` only if you want the caller to use the opposite arm (facing a teacher who does not mirror).

## What the caller sees

The canvas follows the camera (landscape for a laptop webcam, portrait for a phone; nothing cropped). The caller's own skeleton
is drawn on them; the reference dancer is always in a panel on the right and, when enough of the caller is in view, overlaid on
their body (upper body only when the legs are out of frame). Chips show which body parts are in view with a framing hint.
The call opens with a conversation: name, mode (learn, battle, telephone, class), style, "ready". The simulator runs it with
`--lobby --say "4:I'm Maya" --say "10:let's learn waacking" --say "22:ready"`.

## Known limits

- Pose costs about 50 ms a frame on a 4-core laptop at 288 px: 12 fps live, or `DANCE_POSE=lite`.
- Two-seat Chert rooms: one caller per call. Group play is several dancers in one camera (squad) or asynchronous by code (crew
  boards). A multi-phone live room is not built. Squad colours follow position; dancers crossing fast can swap tags.
- Mid-phrase voice commands can't land (the starter's echo guard mutes the mic while music plays): use the gestures.
- Brodie never contacts anyone outside a call: no reminders, no outbound calls.
- Spoken lines are Windows speech until re-rendered with Gemini TTS.
- The Anaconda Python on this PC cannot load two native wheels; the venv must be uv's managed CPython 3.12.11 (see `team/PROGRESS.md`).
