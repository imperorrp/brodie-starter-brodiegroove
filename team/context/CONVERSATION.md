# How Brodie thinks: prompt chains, models, blockers and edge cases

Brodie is a dance coach on a FaceTime call (Chert line; Chrome/LiveKit for testing). This page explains:
- what each model sees and produces
- the exact words we send to Gemini
- how the call moves from "hi" to a scored dance and back to the menu
- what can derail a conversation, and how each case is handled

Code: `team/dance/live.py` (flow), `modes.py` (the six mode cards), `tools.py` (Gemini tools), `persona.md` (system prompt),
`setup.py` (camera check), `orient.py` (depth: turn vs limb), `tracker.py` (who is who), `cues.py` + `style_vocab.py` (coaching
words), `addressed.py` / `answers.py` (was that said to Brodie, was that a yes), `audio_in.py` (noise cancellation), `watchdog.py`
(no dead air), `pacing.py` (breaks between moves), `ghost.py` (saved battle ghosts), `scoring.py` / `feedback.py` (judging),
`render.py` (what the caller sees).

**Ground rules everything below protects:**
1. **One voice.** Only Gemini speaks, in Brodie's voice. Our code never talks over it: it queues one line at a time and waits for silence.
2. **Nothing starts without the caller.** A mode is picked by the caller, a routine needs a clear yes, a camera check and a "ready" (voice or both hands up).
3. **No dead air.** Outside the music, someone speaks at least every 25-30 s.
4. **Every feature ends on the menu.** Score card, then the cards again with "What's next?".

---

## 1. Who does what

| Part | Model / code | Input | Output |
|---|---|---|---|
| Voice, conversation | **Gemini Live** `gemini-3.8-live` (team relay) | caller audio (en-US transcription, after Krisp BVC on LiveKit Cloud and gain control); our text turns; tool results | Brodie's speech, transcripts, tool calls |
| Was it said to Brodie? | `addressed.py` | transcript, whether a question is open, whether Brodie was talking | score 0-1 (≥ 0.5 counts) |
| Understanding answers | `classify()` (regex), Gemini's `caller_answered`, gestures, `answers.py` confidence | the above | yes / no / hold / cancel / question / request, or "Was that a yes?" |
| Body tracking | **MediaPipe Pose** (full, up to 4 people, 384 px) + `tracker.py` | frames at 15 fps | 33 landmarks per person, stable ids (motion + torso colours), a locked primary dancer |
| Depth | `orient.py` (calibrated in the camera check) | image + world landmarks | how far / which way the body is turned, facing away, a limb toward the camera, joints too ambiguous to judge |
| Camera check | `setup.py` + `orient.py` | landmarks, brightness | one issue at a time, with hysteresis; directional "turn to your left" |
| Judging | `scoring.py`: banded DTW over the angle channels on a beat grid | live angles vs the reference | 0-100 = 60 shape + 30 timing + 10 energy, a verdict per count |
| Tips | `feedback.py` | per-count signed errors, 3D body signals | up to three ranked corrections (shape, depth/facing, timing, size) |
| Coaching words | `cues.py` + `style_vocab.py` | the reference itself | per count: arm positions, level, bounce, weight, facing, hit vs flow; a move name; a feel line per style |
| Routines | `choreo.py` | style + level (beginners 2 steps, then 3 / 4); daily is date-seeded | stitched AIST++ phrases (no authored text) |
| Gestures | `gestures.py` | landmarks | both hands up held still 1 s = yes / pick the lit card; crossed arms 1.2 s = pause. A hold that already answered something must come down before it counts again |

## 2. The prompt chain (exactly what Gemini receives)

| # | Turn type | When | Exact form |
|---|---|---|---|
| 1 | **System instruction** | session start | `persona.md` (who Brodie is, the flow, tools, safety, privacy, "always speak English", "never read [bracketed] lines aloud", "ignore talk that isn't to you") |
| 2 | **Background note** | state changes, camera, profile, each move's card, results | `[Background note for you only. Do not reply to it or read it out.] <facts>` |
| 3 | **Stage direction** | every line our flow needs said | `[Stage direction from the system, not the caller. ...] Say exactly this to the caller, nothing before or after: "<line>"` |
| 4 | **Caller audio** | while the mic is open (off while Brodie's audio plays, except a 2.5 s barge-in window over the music) | Gemini replies, transcribes, may call a tool |
| 5 | **Tool results** | after a tool call | short JSON (`will_happen`, `already_offered`, `counted`, `note`…) |

**State rules** (a note on every change; lobby states also say: never claim the camera is set, how many people you see, or that anything is starting unless a `[camera now]` note or a tool says so):

| State | Rule given to Gemini |
|---|---|
| CHOOSE | Six cards are on screen. Answer questions in one sentence; call `propose_activity` only when the caller clearly names one. Never suggest or start. Stay silent for room talk. |
| OFFER | Ask once if they want to start now (camera check first). Wait for a clear yes. |
| PREPARE | Camera setup: one fix at a time; answer questions in one sentence. |
| CONFIRM | Wait for "ready". Don't count in. |
| ACTIVITY | Music is playing. Stay silent unless directed. |
| REVIEW | Say only the short line you're given, then wait. |
| PAUSED / SAFETY | Say how to resume / tell them to rest. |

**Tools:** `caller_answered` (counts only while one of our questions is open; weaker if the recent speech wasn't to Brodie),
`propose_activity` (an offer, never a start; refused in CHOOSE unless the caller spoke to Brodie in the last 6 s), `control_session`,
`confirm_start` (report only), `check_camera`, `analyze_last_attempt`, `get_profile`, `set_preferences`, `generate_routine`,
`compare_with`, `forget_me` (two asks), squad tools.

**Why user-role turns:** the plugin sent `generate_reply(instructions=…)` as a model turn; Gemini continued our instruction as its own
words (doubled lines, "(Private instruction…)" read aloud, live calls 2026-10-11). `live.py` patches it to `user_input`.

---

## 3. One call, step by step

| Step | Brodie says / shows | What counts as the answer | Exit if nothing happens |
|---|---|---|---|
| Hello | "Hey, I'm Brodie! What are we doing: learn a routine, a ghost battle, or the daily? With friends: squad battle, telephone or a class." Six mode cards on screen (site copy and tags) | a mode name said to Brodie, a code, "what can you do", "surprise me" | after 10 s quiet (or a noisy room: 2 unusable lines in 15 s) a highlight steps through the cards every 2.5 s and **both hands up picks the card lit when the hold began**; nudges every 25 s; nobody seen or heard for 5 min ends the call |
| Offer | "Ghost Battle! Want to start a 3-round hip-hop ghost battle? Say yes, or raise both hands again." | yes, request-style yes, Gemini's `caller_answered`, hands up (a fresh hold) | re-asked up to 3 times; "no" → back to the cards, no line |
| Camera check | one fix at a time; turned or facing away gets "Turn a little to your right to face me" / "Turn around to face me" | the camera showing the fix (≥ 65% good frames over 1.5 s; thresholds with hysteresis) | same fix at most every 20 s; arms-only offered; a flip-flop (A, B, A, B in a minute) → "Stay right there, that works"; 150 s per round, **180 s across all rounds** |
| Space | "...two big steps of clear space? Say yes, or raise both hands." Asked again for every new activity (not for a rematch within 10 min) | yes or hands up | 2 nudges, then trusted |
| Ready | "Say ready, or raise both hands." Squad: hands-up from most dancers within 3 s | "ready" / yes / hands up | back to setup only if nobody is in view or the hips are lost for 5 s |
| Move intro | "First move, counts 1 to 4: Bounce Reach. Start facing me, feet hip-width. Watch me, really slow." The full move card goes to Gemini as a note | — | — |
| Demo | pose by pose: each count held 3.5 s, one short cue per count ("Three: arms up in a V, sink low"); a pose never moves on while its line is still playing (up to +2 s) | — | — |
| Feel + Got it? | after 1 s: the style's feel line ("Stay low and loose. Let your knees bounce on every beat."), then "Got it? Say again, or ready to try it with me." | any non-no reply = yes; "again", "slower", "explain"; only "skip" skips | reprompt at 18 s; 25 s after that: practise with the ghost |
| Match it (beginners) | freeze-frame: hold each pose; a matched pose rests 0.8 s; skipped for a whole step whose halves were passed | — | each pose moves on after 4 s anyway |
| With me → break → your turn | "Do it with me", then a 3 s **BACK TO START** break showing the first pose, then your scored try | — | — |
| Result | "<reaction> <score>. <tip>." then "Once more?" / "A little faster?" / "Next move?" | again / faster / slower / next / loop / explain / stop / yes / **hands up = yes** | reprompt, then the dead-air nudge every 22 s (doubling after 3) |
| Next move | a 3.5 s **NEXT MOVE** break with a 3-2-1 ring and the next move's first pose | — | after 12 min: "full routine now, or keep practising?" |
| Wrap | finish pose, "That's a wrap! ..." and the card; battles: a code only on a win, then "Save your ghost for friends?" | "rematch" / hands up (a fresh hold, at least 1.5 s after the card) | after 6 s (and the wrap line), the cards return with **"WHAT'S NEXT?"** and "Nice work! What next? Pick any mode, or say bye." |
| Bye | "Thanks for dancing with me!" | "bye", "I'm done", "gotta go" | the call ends |

Between features everything activity-specific is reset (squad roster, battle HUD, gesture latches, pending commands, stale answers);
the name, profile, coach style, arms-only choice and today's daily stay. A mode named mid-activity ("let's do a ghost battle
instead") between moves leaves the current one and offers the new one; during the music it is ignored.

**Judging per frame:** pose → primary (or each squad dancer) → angles on a 4-per-count grid (joints the depth module marks ambiguous
leave the tip choice, and leave scoring with `DANCE_DEPTH_GATE=1`) → banded DTW → verdict per count (green < 18°, amber < 32°) → score →
up to three ranked tips → PERFECT / NICE / MISS, the combo, the count tiles.

---

## 4. What can derail a conversation, and the guard for each

| Blocker | What used to happen | Guard now |
|---|---|---|
| Our line sent as a model turn | doubled lines, instructions read aloud | stage directions as user turns |
| Brodie offered a routine by himself 14 s in | callers were dropped into hip-hop instead of choosing | CHOOSE: no auto-offers; cards, voice pick, hands-up picker |
| Background talk / TV taken as the caller | Brodie answered a conversation in the room; chatter kept the "quiet" clock from ever running out | `addressed.py`: talk not to Brodie never answers, offers, picks a mode or resets timers; a Gemini reply to it is cut; persona rule |
| A yes from the room, a mumbled yes | wrong starts, or re-ask loops | answer confidence: unsure → "Was that a yes?" once per question; a second medium yes is accepted; a medium no only waits |
| 50 s of silence after a result | the choice waited forever after one reprompt | dead-air watchdog per state (and hands-up answers there too) |
| Setup flip-flop for minutes | step back / come closer at a threshold; ready check bouncing back for small drifts | hysteresis bands, flip-flop breaker, 180 s total cap, ready check bounces only for a real break |
| A tick over Brodie's count line | the voice paused mid-word (clipped lines) | no tick while a line is pending or playing |
| Two replies glued together | a line placed while Gemini was still thinking | the speaker waits for `agent_state != thinking` |
| Gemini claims progress ("camera's locked in") | false progress | rule in every lobby state; tool results say "nothing has started"; logged as `false_progress` |
| "one two three four" heard as a code | a battle code offer from a count-in | outside the menu a code needs the word "code" or a short line, and counting runs are refused |
| Squad "ready" by hands up | needed every dancer's hold in the same frame (never happened) | a majority of dancers within 3 s |
| Space carried over | the next activity skipped the space question | asked again per activity (a rematch within 10 min keeps it) |
| A seated caller (no hips in view) told to face the camera | the turn estimate needs hips; without them it isn't guessed |
| Hands still up after picking a card / a finish pose | one gesture answered two questions | a fired hold is blocked until the hands come down; the card and the picker arm only once hands are seen down |
| Two dancers crossing; a dancer stepping out | ids (colours, scores) swapped; returners became strangers; a walk-in took a solo routine after 2 s | `tracker.py`: motion + torso colours, re-ID for 20 s (squad) / 6 s (solo), swap check after a crossing, primary held 8 s and never handed to a walk-in mid-routine |
| A turn read as a wrong arm | "lift your arm" when the caller had turned | `orient.py`: turn size from calibrated widths, direction from depth with hysteresis; tips skip joints the camera can't judge |
| Stale commands, "hold on", arms up while dancing, per-frame crashes | (as before) | timestamped commands; hold only waits; hands-up must be still; frame errors skipped, stages guarded (an error returns to the cards) |

## 5. Audio situations

| Situation | Handling |
|---|---|
| Background voices, TV, music | Krisp BVC removes other voices before Gemini on LiveKit Cloud (`DANCE_NC=auto`, off elsewhere with the reason logged); `addressed.py` filters what still gets through; Gemini replies to room talk are cut |
| Quiet / far from the mic | automatic gain control (`DANCE_AGC=1`); sensitive start of speech; "or raise both hands" on every question; Gemini's `caller_answered` |
| Speech with no transcript | the local VAD hears it: "Sorry, I didn't catch that. Was that a yes?" |
| Voice during the music | the local VAD on the caller's own track: speech above the echo level learned in the count-in opens the mic for 2.5 s and ducks the music ~18 dB, so "stop", "pause", "next" are heard; at most 20% of a phrase (singing along keeps it shut); crossed arms always works |
| Squads / classes | BVC keeps the nearest voice, so every question also offers hands up, and per-dancer readiness is by gesture |
| Garbled / other language, slang, mixed answers, pain words, "forget me", off-topic, unsafe requests | as before: en-US transcription and foreign yeses; phrasing table in `tests/test_live.py`; first-person pain stops at once; two asks to forget (it also deletes a saved ghost); persona refuses unsafe tricks |

## 6. Video situations

| Situation | Handling |
|---|---|
| Nobody in view / too close / too far / off centre / dark | one tip at a time, hysteresis, caps (§3) |
| Turned away or back to the camera | `orient.py`: "Turn a little to your right to face me" / "Turn around to face me"; Gemini's camera note says the turn |
| A limb toward the camera | told apart from a body turn (the forearm looks short while the shoulders stay wide); ambiguous joints skipped by tips |
| Phone portrait vs laptop landscape | canvas follows the camera; FaceTime via Chert is portrait 720×1280; everything in the FaceTime safe area |
| Someone walks in | the primary keeps the scoring through 8 s of absence; mid-routine nobody else takes over unless alone and centred |
| Two to four dancers | colour tags; re-identified by clothing colours after an occlusion; children and dancers further back count in squad mode; more than four: "I can score four at a time. The rest of you can cheer!" |
| Camera freezes / partial body / caller leaves mid-pause | video gone > 4 s mid-try: "Your video froze. Let's try that again." (unscored, back to the camera check); never auto-resumes |
| The phone moves or falls mid-try | the whole picture jumps (phase correlation on a tiny frame; a dancer moving only changes part of it): "Looks like the phone moved. Let's check the camera." (unscored) |
| A bright window behind the caller | the person is much darker than the picture: "There's bright light behind you. Turn so the window or lamp is in front of you." |
| Someone falls | torso flat and low for 2 s during the music: the routine pauses, "Whoa, everyone okay?" |
| Slow computer | frames over 55 ms for 5 s: the lite pose model, then 12 fps (`guards.FrameBudget`) |

## 7. Other edge cases

| Case | Handling |
|---|---|
| "next" at "Got it?" | means "try it with me"; "skip" skips; during a demo "next" ends it; during a scored try it waits for the score |
| "stop" in setup / ready check | back to the cards |
| A code at hello | straight to that code's offer (a friend's saved ghost if there is one) |
| Battle lost | no code ("So close! You needed 195 to beat the ghost and earn a code.") |
| Battle won | a code; "Save your ghost for friends?" (body points only, with a yes, 30 days, deleted by "forget me"); saving needs the one-time token only the winner's call has |
| Daily already played | the card shows DONE TODAY; picking it offers a practice run that isn't posted |
| "menu" between moves | the lesson ends with its card, then the cards |
| "bye" | goodbye, the call ends |
| Nothing danced, then stop | no finish pose, XP or card |
| Brodie's voice drops (session closed, relay limit) | a caption "Brodie lost his voice for a second. Hang on!"; after 30 s the call ends |
| A long call | a warning 2 minutes before `DANCE_CALL_MAX_MIN` (20); the call ends on the menu, never mid-routine |
| Squad dancer steps out for a round | that round is unscored for them (not zero); ranking uses the mean of the rounds each one danced |
| Coach style | "hype coach", "chill coach" or "strict coach" any time (or ask Brodie); kept for the call and saved for a named caller |

## 8. What we can still improve

1. **True barge-in while Brodie talks.** The echo guard still drops the caller's audio while Brodie speaks (only the music has a barge-in window). Sending silence instead of dropping frames is a starter change.
2. **Speaker check for yes.** Require the caller's own voice for a yes (needs a speaker model).
3. **Language.** Offer to continue in the caller's language (a per-call input-language switch).
4. **Noise cancellation per mode.** Switch BVC off for squads mid-call (the selector is fixed when the track is subscribed).
5. **Reference depth.** With AIST++ keypoints3d on disk, re-ingest the keys with per-count facing, travel and height for "step toward me" / "turn left" cues (today: facing size only).
6. **Measure live.** The per-call metrics (time to choose, re-asks, unused answers, non-addressed lines, longest silence, setup bounces, noise cancellation on/off) need a dashboard view.

## 9. How to test

- **Unit tests:** `.venv/Scripts/python -m pytest -q team/dance/tests` (163).
- **Eval:** `.venv/Scripts/python -m team.dance.eval.run_eval --jobs 6` (45 scenarios, plus the duel). `real_*` use real-call timing and a caller who answers Brodie's actual lines (`--realistic --answer key:text`); `choose_*` cover the cards; `battle_*` the win-only code and saved ghosts.
- **Live on Chrome:** `.\team\scripts\chrome.ps1 -Restart` (restart so the new code loads), then follow §3.
