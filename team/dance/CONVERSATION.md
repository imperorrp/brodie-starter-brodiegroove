# How Brodie thinks: prompt chains, models, blockers and edge cases

Brodie is a dance coach on a FaceTime call (Chert line; Chrome/LiveKit for testing). This page explains:
- what each model sees and produces
- the exact words we send to Gemini
- how the call moves from "hi" to a scored dance
- what can derail a conversation, and how each case is handled

Code: `team/dance/live.py` (flow), `tools.py` (Gemini tools), `persona.md` (system prompt), `setup.py` (camera check),
`scoring.py` / `feedback.py` (judging), `render.py` (what the caller sees).

**Two ground rules that everything below protects:**
1. **One voice.** Only Gemini speaks, in Brodie's voice. Our code never talks over it: it queues one line at a time and waits for silence.
2. **Nothing starts without the caller.** A routine needs a clear yes to an offer, a camera check, and a "ready" (voice or both hands up).

---

## 1. Who does what

| Part | Model / code | Input | Output |
|---|---|---|---|
| Voice, conversation | **Gemini Live** `gemini-3.8-live` (via the team relay) | caller audio (input transcribed in en-US); our text turns; tool results | Brodie's speech, transcripts of both sides, tool calls |
| Understanding answers | our `classify()` (regex) **and** Gemini's `caller_answered` tool | caller transcript / Gemini's reading of the audio | yes / no / hold / cancel / question / request |
| Body tracking | **MediaPipe Pose** (lite/full, multi-person, 384 px) | caller video frames, 15 fps | 33 landmarks per person; a primary dancer is locked |
| Camera check | `setup.py` rules | landmarks, brightness, frame shape | one issue at a time (too close, too far, off centre, dark, not facing) |
| Judging | `scoring.py`: banded DTW over 14 joint angles on a beat grid | live angles vs the reference routine | score 0–100 = 60 shape + 30 timing + 10 energy; a verdict per count |
| The one tip | `feedback.py` | per-count signed angle errors | e.g. "Lift your left arm higher on counts 5 to 7" |
| Routines | `choreo.py` | style + level (+ seed; the daily is date-seeded) | a routine from the move library (8-count steps, halves, combinations) |
| Gestures | `gestures.py` | landmarks | both hands up **held still** 1 s = yes/ready; arms crossed overhead, still = pause |

---

## 2. The prompt chain (exactly what Gemini receives)

| # | Turn type | When | Exact form |
|---|---|---|---|
| 1 | **System instruction** | once, at session start | `persona.md` (who Brodie is, the flow, tools, safety, privacy, "always speak English", "never read [bracketed] lines aloud") |
| 2 | **Background note** (user role; asks for no reply) | state changes, camera changes, profile, results | `[Background note for you only. Do not reply to it or read it out.] <facts>` |
| 3 | **Stage direction** (user role; Gemini speaks it) | every line our flow needs said | `[Stage direction from the system, not the caller. Do not answer it, mention it or read it out.] Say exactly this to the caller, nothing before or after: "<line>"` |
| 4 | **Caller audio** | whenever the mic is open (off while Brodie's audio plays) | Gemini replies on its own, transcribes it, may call a tool |
| 5 | **Tool results** | after a tool call | short JSON (`will_happen`, `already_offered`, `counted`, `note`…) |

**State rules** (sent as background notes on every change, plus the coaching style: HYPE / CHILL / COACH):

| State | Rule given to Gemini |
|---|---|
| EXPLORE | Chat, get to know them, recommend. You can't start dancing; call `propose_activity` only on a clear wish. |
| OFFER | A routine was suggested. Ask once if they want to start now (camera check first). Wait for a clear yes. |
| PREPARE | Camera setup: the system hands you one fix at a time; answer questions in one sentence. |
| CONFIRM | You asked if they're ready. Wait. Don't count in. |
| ACTIVITY | Music is playing. Stay silent unless directed. |
| REVIEW | Say only the short line you're given, then wait. |
| PAUSED / SAFETY | Say how to resume / tell them to rest. |

**Tools Gemini can call:**
- **Answers and offers:** `caller_answered(yes|no|ready)` (counts only while one of our questions is open), `propose_activity` (makes an offer, never a start; repeating the same offer changes nothing).
- **During a routine:** `control_session` (only for a running routine; "stop" also cancels setup), `confirm_start` (report only).
- **Information:** `check_camera` (before a yes it says "nothing has started"), `analyze_last_attempt`, `get_profile`.
- **Other:** `set_preferences`, `generate_routine`, `compare_with` (challenge code), `forget_me` (needs two asks), squad tools.

**Why user-role turns:** the LiveKit plugin sends `generate_reply(instructions=…)` as a *model* turn. Gemini took our instruction for its own words and continued them: lines said twice, "(Private instruction…)" read aloud (live calls on 2026-10-11). `live.py` patches `GeminiModel.generate` to send `user_input` instead. Checked against the relay.

---

## 3. One call, step by step (inputs → reasoning → outputs)

| Step | Brodie says (our line, or Gemini's own) | What counts as the answer | Exit if nothing happens |
|---|---|---|---|
| Hello | "Hey, I'm Brodie! What's your name, and what kind of dancing are you into?" | anything; name only from "my name is / call me" or a capitalised name | 14 s quiet → Brodie suggests a routine |
| Offer | Gemini asks, or after 5 s we do: "Want to start a beginner hip-hop routine now? We'll do a quick camera check first." | yes, a request-style yes ("all right let's start it"), Gemini's `caller_answered(yes)`, both hands up | re-asked up to 3 times; 60 s quiet → chat; "no" → chat, and that style isn't offered again right away |
| Camera check | one fix at a time ("Take a step back so I can see you down to your hips") | the camera showing the fix (smoothed: ≥65% good frames over 1.5 s) | same fix at most every 20 s; arms-only offered after 20 s with no hips; chat after 120 s without progress or 150 s total (with a spoken line) |
| Space | "…two big steps of clear space around you? Say yes, or raise both hands." | yes or hands up, even late | 2 nudges, then "I'll trust you've got room. Let's go!" |
| Ready | "You're all set. Say ready, or raise both hands." | "ready" / yes / hands up (not the space yes) | asked twice; 40 s quiet → chat, with a line |
| Demo | "First move: counts 1 to 4. Watch me, really slow." plus a spoken breakdown, then the 0.5× demo | — | — |
| Got it? | "Got it? Say again to see it once more, or ready to try it with me." | any non-no reply = yes; "again", "slower", "explain"; only "skip this move" skips | reprompt at 18 s; 25 s after that: practise with the ghost (unscored), then "Ready to try it on your own?" |
| With me → your turn | "Do it with me…", then "Now on your own" | — | — |
| Result | "<reaction> <score>. <one tip>." then "Once more?" / "A little faster?" / "Next move?" | again / faster / slower / next / simplify / explain / stop / yes | one reprompt, then waits quietly (never skips ahead) |
| Wrap | "That's a wrap! You're a <title>… Your code is 1 2 3 4…" plus a finish pose and score card | "rematch" | only if something was danced |

**Gemini's reasoning steps each turn:**
1. Hear and transcribe the caller.
2. Read the latest notes (state, camera, profile).
3. Reply in one or two sentences in the coaching style.
4. Optionally call a tool (an offer, `caller_answered`, a camera peek), then say what the tool reported.

**Judging pipeline per frame:**
1. Pose (33 landmarks), then the primary dancer is locked.
2. 14 joint angles and hip energy are computed on a 4-samples-per-beat grid.
3. A DTW within one beat of the reference.
4. A per-count verdict (green < 18°, amber < 32°).
5. The score.
6. The worst run of counts becomes one directional tip.
7. PERFECT / NICE / MISS flashes, the combo, the count tiles.

---

## 4. What can derail a conversation, and the guard for each

| Blocker | What used to happen | Guard now |
|---|---|---|
| Our line sent as a model turn | doubled lines, instructions read aloud | stage directions as user turns (§2) |
| A yes that sounds like a request ("let's start it") | offer re-made, yes erased, endless "want to start?" | the same offer is a no-op; a request in OFFER = yes |
| Gemini replies before the transcript arrives | the yes landed "before" the question | the offer counts from the *first* time it was asked |
| Answer given while the question is still playing | thrown away; re-ask loop | answers count from 1 s after the line *starts* |
| A question line dropped (state change, caller talking) | a wait that could never end | question lines are never dropped for age; every wait falls back after 12–14 s |
| Notes containing commands | Gemini answered the note, then our line (two answers) | notes state facts only, marked "no reply" |
| Wrong-language transcripts ("c'est quand", Hindi) | yes/no missed; Brodie once answered in German | input transcription pinned to en-US; foreign yeses (claro, ja, oui) accepted; `caller_answered` from Gemini's own hearing |
| Quiet caller 2 m from the mic | answers never transcribed | sensitive start-of-speech detection; "say yes **or raise both hands**" on every question |
| Framing flicker (hips at the frame edge) | silent setup or the same tip repeated | smoothed tracker; the most frequent issue is spoken if none leads; 20 s minimum between repeats; hard caps |
| Stale commands ("stop" at the wrap, "next" twice) | the next routine ended or skipped by itself | commands are timestamped; only those said after the current step count |
| "Hold on" | became a "no" or a pause, and the music restarted | a hold only waits; never a no |
| Dancing / stretching with arms up | read as "yes" | hands-up needs a 1 s *still* hold, only while a question is open |
| A crash in one frame or one routine | the lesson aborted, or the call froze | per-frame errors are skipped (30 in a row escalate); stages are guarded and return to chat with an apology |
| Gemini claims things ("camera's locked in") | false progress | tool results say "nothing has started" until it has; `control_session` reports what really happens |

---

## 5. Audio situations

| Situation | Handling |
|---|---|
| Clear speech | transcript → `classify`, plus Gemini's reply and tools |
| Quiet / far from the mic | sensitive start of speech; hands-up alternative; Gemini's `caller_answered` |
| Background talk, TV, music | the mic is muted while Brodie's audio plays; a yes only counts while a question is open (gap: Gemini may still reply to TV speech, see §8) |
| Caller talks over Brodie | their words are dropped while his audio plays (echo guard), so questions are short and the caller has the floor after |
| Garbled / other language | en-US transcription; foreign yes words; Gemini always answers in English (persona) |
| Silence | every wait has a nudge and an exit (§3); chat itself is caller-led after at most 3 suggestions |
| Slang / fillers ("bet", "I'm down", "oh yeah", "yesss") | accepted (test table in `tests/test_live.py`) |
| Mixed answers ("no worries", "can't wait", "don't stop") | not treated as no |
| Change of mind ("actually K-pop") | a new offer for the new style |
| Pain words | first person ("my knee hurts", "ow") → safety stop at once; unclear ("this song hurts") → "Are you okay?" |
| "Forget me" | asked twice → profile deleted; video is never stored |
| Off-topic chat | Gemini answers briefly, then steers back (persona: be proactive, suggest the next step) |
| Unsafe requests (flips, tricks) | persona refuses unsafe tricks and offers a safe build-up; never diagnoses |

## 6. Video situations

| Situation | Handling |
|---|---|
| Nobody in view | "Step into the picture…"; after 60 s with no video → back to chat with a line |
| Too close (laptop, sitting) | step-back tip; after 20 s with no hips → "say arms only"; upper-body mode needs only head and shoulders |
| Too far / off centre / dark / turned away | one tip each (facing is checked aspect-correct for 16:9 webcams) |
| Hips flicker at the edge | smoothing; a stable 65% is enough |
| Phone portrait vs laptop landscape | canvas follows the camera; FaceTime via Chert gets portrait 720×1280; everything stays in the FaceTime safe area |
| Someone walks in behind | primary-dancer lock; walk-ins never steal scoring |
| Two or more dancers | squad mode on request only; colour tags; nobody's looks described |
| Camera freezes or drops | in setup: "I've lost your video…", back to chat after 60 s; during a try: unscored → back to the camera check; while paused: ends after 2 min with nobody in view |
| Partial body during a try | scored on what is visible (hips needed for energy); "couldn't see enough" → back to the camera check |
| Caller leaves mid-pause | never auto-resumes; ends after 2 min |

## 7. Other edge cases

| Case | Handling |
|---|---|
| Says "next" at "Got it?" | means "try it with me", not skip (only "skip this move" skips) |
| Says "stop" during setup or the ready check | back to chat ("No problem. Tell me whenever you want to dance.") |
| A challenge code spoken ("4471") | loads that routine as a battle offer (times like "at 10:30" or years are ignored) |
| Nothing danced, then stop | no finish pose, XP or card; "No worries, we'll dance next time." |
| A routine can't be built for a style | falls back to hip-hop instead of failing |
| A stale second worker in the room | logged as a warning ("two voices"); stop the old worker |

---

## 8. What we can still improve

1. **Barge-in.** The echo guard drops the caller's audio while Brodie talks. Sending silence instead of dropping frames (a starter change in `facetime/voice.py`) would stop cut-off words.
2. **Voice during music.** "Stop" / "pause" can't be heard while the track plays; only the crossed-arms gesture works. A tiny on-device keyword spotter, or ducking the music on loud speech, would close this safety gap.
3. **A confidence score for answers.** Combine the regex, Gemini's `caller_answered` and gesture timing into one score; ask "Was that a yes?" when unsure instead of guessing.
4. **Measure live.** Log per call: time to first dance, re-asks per question, answers lost, silences over 20 s. Today these come from the simulator.
5. **Gemini's own questions.** We skip our "ready?" / "space?" if Gemini just asked it. Extending that to the offer and "Got it?" would remove the remaining double questions.
6. **Camera hints with pictures.** A small arrow or ghost outline showing where to stand would help callers who can't hear well.
7. **Upper-body scoring.** In arms-only mode, score and tip only the arms, and say so on screen.
8. **Other voices in the room.** Tell Gemini (persona) to ignore speech that isn't addressed to Brodie, and require the caller's own voice for yes (speaker check).
9. **Language.** If a caller clearly speaks another language, offer to continue in it (needs a per-call input-language switch).

## 9. How to test

- **Unit tests:** `.venv/Scripts/python -m pytest -q team/dance/tests` (125, including the phrasing table).
- **Eval:** `.venv/Scripts/python -m team.dance.eval.run_eval --jobs 6` (26 scenarios). The `real_*` scenarios use real-call timing and a caller who answers Brodie's actual lines (`--realistic --answer space:yes`).
- **Live on Chrome:** `.\team\scripts\chrome.ps1 -Restart` (restart so the new code loads), then follow §3.
