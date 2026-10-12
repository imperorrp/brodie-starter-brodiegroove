# BrodieGroove codebase map

The one file to read before touching code. It says what every file is for, which functions matter, and how the pieces
hand off to each other. Maintained by hand: when you add, rename, or change the role of a file, fix its entry here in the
same commit (rules at the end).

Last full pass: **2026-10-11, late evening ET**, against commit `daa3e9e`. Line counts are from that pass.

This folder, `team/context/`, is the LLM context for the repo (the root `CLAUDE.md` points here). Siblings:
[`STATUS.md`](STATUS.md) (what works, what is verified, what shipped), [`TODO.md`](TODO.md) (what is next, ranked),
[`CONVERSATION.md`](CONVERSATION.md) (the call flow, prompt chain and every guard, in depth),
[`PLANNED_IMPROVEMENTS.md`](PLANNED_IMPROVEMENTS.md) (the longer plan). Elsewhere: [`../PROGRESS.md`](../PROGRESS.md)
(the dated build log; private, git-ignored), [`../dance/README.md`](../dance/README.md) (how to run and test the skill),
the root [`README.md`](../../README.md) (the public pitch).

---

## 0. The product, and the shape of the code

**Brodie Groove** is a dance coach you reach by FaceTime (through the organiser's Chert line) or from any browser (our own
LiveKit room). The caller sees their own video with a ghost of the move drawn on their body, gets a PERFECT / NICE / MISS
verdict on every count, one spoken tip, a grade and a four-digit challenge code. Six modes: Learn, Ghost Battle, Daily
Challenge, Squad Battle, Dance Telephone, Class.

The whole thing is **one Python process**: the organiser's LiveKit Agents worker (`facetime/worker.py`) with perception
switched off, loading our `team/dance/live.py` as its handlers file. That file registers a consumer coroutine which, once a
caller is bound, runs MediaPipe pose in a thread, drives the scoring engine, renders a video track, and talks to Gemini Live
through the worker's voice layer. A small FastAPI backend holds codes, boards, profiles and classes in Postgres (Supabase)
when deployed and in SQLite for tests and the simulator. Deployed, the worker and the backend run in one container on Railway
(`team/cloud/`, no laptop involved). Everything else in `team/` is tooling: run scripts, a Chrome stand-in for the phone, the
landing page, promo and pitch-video generators.

```
iPhone (FaceTime via Chert)  ──┐                     ┌─ MediaPipe pose (multipose.py, tracker.py)
Chrome (team/web/phone.html) ──┴─► LiveKit room ─► worker ─┤  engine.py + scoring.py  → verdicts, score, cue
                                      ▲   ▲    (live.py)   ├─ render.py → coach-video track (back to the caller)
                                      │   │                ├─ Gemini Live via facetime/voice.py (one voice, 14 tools)
                                      │   └─ coach-audio   └─ Backend (team/backend) ← codes, boards, profiles, classes
                                      └─ coach-video
```

Three folders matter:

| Folder | What | Who wrote it |
|---|---|---|
| `team/` | Everything we built (the skill, backend, web, scripts, promo, docs) | us |
| `facetime/`, `src/rtcoach/`, `starter/`, `scripts/`, `docs/`, `.github/` | The Brodie starter and the organiser's hackathon hub | Seraphim Labs (we changed two files, section 9.6) |
| `research/`, `notes/`, `team/docs/`, `team/PROGRESS*.md` | Private research and planning; on disk, never in git | us |

Size at the last pass: `team/dance` ≈ 11,900 lines (incl. 1,900 of tests), rest of `team/` ≈ 5,300, starter ≈ 6,700.

---

## 1. Where things live

```
ivyhacks-brodie/
├── README.md                 public pitch, run instructions, credits
├── requirements.txt          starter deps + pillow, segno (ours)
├── vercel.json               static hosting of team/web/groove (the landing page)
├── railway.json nixpacks.toml .python-version   the cloud box build and deploy (Railway, Nixpacks; section 6.1)
├── .env.example              LIVEKIT_URL/API_KEY/API_SECRET, AGENT_NAME, GEMINI_API_KEY, GEMINI_BASE_URL; cloud: DATABASE_URL, DANCE_LIBRARY_URL, JOIN_KEY ...
├── CLAUDE.md                 working rules: read team/context first, keep it current, push to origin only
├── team/                     ── ours ──
│   ├── context/              LLM context: CODEBASE.md (this), STATUS.md, TODO.md, CONVERSATION.md, PLANNED_IMPROVEMENTS.md
│   ├── README.md CHROME_TEST_GUIDE.md PROGRESS.md (private)
│   ├── dance/                the skill (section 3)
│   │   ├── live.py           the session: state machine, modes, Gemini queue, tools backend, simulator
│   │   ├── engine.py scoring.py features.py reference.py feedback.py choreo.py ...
│   │   ├── render.py text.py fx.py avatar.py screens.py       the video
│   │   ├── tools.py setup.py orient.py tracker.py multipose.py squad.py gestures.py ...
│   │   ├── persona.md plan.txt modes.py state.py run_worker.py
│   │   ├── library/*.json    reference keys (AIST++ ones are git-ignored)
│   │   ├── assets/brodie/    the official character kit + cream checker; assets/fonts/ (OFL)
│   │   ├── voice/*.pcm       dormant local TTS lines
│   │   ├── tests/            165 pytest cases;  eval/  scripted end-to-end sessions + HUD snapshots
│   ├── backend/              app.py (FastAPI routes, HTML pages, /join), db.py (Postgres or SQLite), dbcheck.py, import_sqlite.py, team.db (ignored)
│   ├── cloud/                the cloud box: start.sh (API + worker, restart loops), requirements.txt, fetch_models.py, fetch_library.py, pack_library.py, README.md (the runbook)
│   ├── web/                  phone.html (browser caller), groove/index.html (landing page), certs/ (ignored)
│   ├── scripts/              chrome.ps1, dance.ps1, dance_loop.ps1, chert.ps1 + shim, mint_link.py, serve_web.py, ...
│   ├── promo/                8-day poster/reel kit (render.py, clips.py, art.py, days/*.html, STRATEGY.md, CAPTIONS.md)
│   ├── video/                pitch video (make_pitch.py, music.py, voice.py, timeline.py) and film/ (Veo/Lyria film)
│   ├── media/                README screenshots
│   ├── skill/ tests/         legacy fitness-era handler and replay harness (section 8)
│   └── docs/                 private planning docs (git-ignored; section 12)
├── facetime/                 the starter worker, voice layer, cue player, perception (section 9)
├── src/rtcoach/              the starter's rep counter, classifier, cue tables (we use pose.py only)
├── starter/                  the starter's three editable files + meal-prep example (unused by us)
├── scripts/ docs/ .github/   organiser tests, hub pages, slot/submission bots
├── data/  models/            bundles, AIST++ keypoints, clips, cue bank, tmp outputs, pose models (all git-ignored)
├── research/ notes/          ideation, research plan, the deck and handbook (git-ignored)
└── research_papers_and_other_useful_code_and_dataset_ideas.md   private
```

---

## 2. How a call runs, end to end

1. **Launch.** `team/scripts/dance.ps1` (or `dance_loop.ps1`, `chrome.ps1`) runs
   `python -m team.dance.run_worker --perception off --voice gemini --persona team/dance/persona.md --handlers team/dance/live.py --plan-file team/dance/plan.txt --log-caller --audio-queue-ms 300 --agent-name=<name> dev`.
   `run_worker.py` imports the Google plugin on the main thread (a Windows requirement) and calls `facetime.worker.main()`.
2. **Dispatch.** An empty `--agent-name=` means automatic dispatch (every new room in our LiveKit project). A name means
   explicit dispatch: the Chert CLI (`chert dev --agent <name>`) or a token minted by `team/scripts/mint_link.py` with a
   `RoomAgentDispatch` carries the room to that worker.
3. **Bind.** The worker joins, publishes the silent `coach-audio` track, and binds to the first non-agent participant that
   publishes video (`chert-bridge-<call_id>` from FaceTime, `chrome-laptop` or similar from `phone.html`). It loads
   `live.py` by path (so `live.py` is re-executed per job; the session pointer lives in `state.py`), starts the Gemini Live
   session, and runs every registered consumer: ours is `live.dance_consumer`.
4. **Session start.** `DanceSession.run()` waits for the first frame, picks the canvas (portrait 720x1280 for Chert, else
   the camera's orientation), publishes `coach-video`, registers the 14 Gemini tools, starts the local VAD listener, sends
   the EXPLORE rules, then `play_loop()`.
5. **Per frame** (`_frame_step`): newest frame → pose in the one-worker thread (`MultiPose`, up to 4 people) → `PlayerTracker`
   assigns ids and the primary dancer → `Orient` (turn/depth) → `GestureDetector` per dancer → guards (frame budget, camera
   shift, backlight, fall, frozen video) → `DeadAir` watchdog → engine `on_frame`/`tick` → HUD dict → `render.compose` /
   `compose_idle` / `draw_card` → encode → `coach-video`.
6. **The flow.** CHOOSE (six mode cards) → OFFER (needs a clear yes) → PREPARE (camera check, one issue at a time) →
   CONFIRM ("ready" or both hands up) → REVIEW/ACTIVITY (the mode's coroutine runs `PhraseEngine`, `PoseStepEngine` or
   `SquadEngine`) → scorecard → back to CHOOSE. PAUSED and SAFETY sit beside. Every stage is inside `guard()`.
7. **Talking.** Only Gemini speaks. Our lines are queued by `ask()`/`say_line()` and spoken one at a time by
   `_speaker_loop` as user-role "stage directions"; facts go in as "background notes" (`context()`). The caller's
   transcript is parsed by `on_worker_event` (regex `classify()`, `addressed.score()`, `answers.decide()`), and Gemini's
   tool calls land on `tool_*` methods of the session.
8. **Results.** `finish_session()` POSTs `/dance/result`, gets a code and (for battles) a one-time ghost token, awards XP via
   `/dance/profile/{name}/award`, draws the scorecard with a QR (`facetime:<DANCE_CALL_HANDLE>`), and the telemetry summary
   goes to `/events` when the session closes.
9. **Without LiveKit.** `python -m team.dance.live --simulate clip.mp4 ...` runs the same `DanceSession` against a video with
   fake player/voice/job objects and scripted caller lines; `eval/run_eval.py` runs 46 such scenarios in parallel.

---

## 3. `team/dance/` — the skill

### 3.1 Engine and reference library (landmarks → features → score → cue)

**`features.py`** (199) — 33 MediaPipe landmarks → 14 angle channels in degrees, visibility mask, hip energy, a
hip-centred 2D skeleton for the ghost, beat-grid resampling, and 3D body-frame signals.
- `CHANNELS` (elbow_l/r, shoulder_l/r, hip_l/r, knee_l/r, torso_lean, head_tilt, arm_elev_l/r, leg_spread, wrist_dist),
  `GROUPS` (arms / legs / torso channel indices), `CHANNEL_JOINTS`, `MIRROR_PAIRS`, `SKELETON_EDGES`, landmark index constants.
- `features_from_landmarks(lm_img, lm_world=None)`, `visibility_mask(lm_img, thr=0.5)` (joint visible and inside the
  picture; ankle channels forced off when `IGNORE_FEET`), `mirror_landmarks`, `hip_energy(prev, cur, dt)`,
  `skeleton2d_normalized`, `resample_to_grid` / `resample_mask`, `body_frame_signals(lm_world)` (arm_fwd_l/r, torso_twist,
  facing_yaw), `arm_planarity`, `arm_direction_2d`.
- Env: `DANCE_IGNORE_FEET` (default on). Used by nearly everything below.

**`reference.py`** (172) — a **key** is one move as feature samples on a beat grid (4 per beat), with mask, energy, ghost
skeleton and provenance, stored as `library/<name>.json`.
- `extract_landmarks(video)` (MediaPipe offline via `rtcoach.pose.Pose`), `beat_grid(video|bpm)` (librosa or fixed BPM),
  `build_key(video, name, bpm, ..., mirror=True, source, licence)`, `key_from_arrays(name, bpm, F, M, E, S)` (ingesters,
  tests, ghosts, generated routines), `phrase_slice(key, i) -> (F, M, E, S)`, `save_key`, `load_key`, `list_keys`.
- Key JSON: `version, name, bpm, per_beat, n_beats, counts_per_phrase, beat_times, phrases[{start_beat, beats}], mirrored,
  style, difficulty, source, licence, video, features, mask, energy, skeleton, fps`; generated keys add `title, level,
  generated, seed, moves[], default_tempo`.

**`scoring.py`** (172) — stateless maths on two grids.
- `dtw_band(A, B, band, mask_a, mask_b)` Sakoe-Chiba banded DTW; `score_phrase(live, mask, energy, key, kmask, kenergy)` →
  `score = 100 * (0.6 spatial + 0.3 timing + 0.1 energy)`, per-beat verdicts (green err < 18° and |offset| ≤ 0.35 beat;
  amber < 32° and ≤ 0.6; else red), `weak_beats`, `max_combo`, `per_group`, `channel_signed`, `partial` (upper-body mode
  when legs are hidden but arms visible); `groove_score(...)` for the reference-free freestyle round (on-beat, variety, energy).
- Env: `DANCE_GREEN_DEG` 18, `DANCE_AMBER_DEG` 32, `DANCE_HALF_MARKS_DEG` 20, `DANCE_MIN_VISIBLE` 0.6, `DANCE_PARTIAL_ARMS` 0.7.

**`engine.py`** (276) — `PhraseEngine`: the phrase state machine `idle → show → countin → attempt → scored → feedback → done`
with an injected clock and lag compensation (`rtt_s`).
- `on_frame(t, lm_img, lm_world)` collects live samples during the attempt and emits per-beat `beat` verdicts without DTW;
  `tick(t)` advances states and calls `_score()` at the end of the window (resample → `scoring.score_phrase` or
  `groove_score`, plus `body3d` and `size` extras); `ghost_sample(t)` (one beat ahead), `hud(t)` (the dict the renderer
  reads), `abort`, `emit`. Options: `tempo`, `show_repeats`, `countin_beats`, `skip_show`, `mirror_live`, `freestyle`,
  `focus='arms'`, `demo_only`. `choose_cue(result)` → `feedback.structured_cue`.

**`posestep.py`** (151) — `PoseStepEngine`: the no-music sibling. `mode='watch'` holds each count's pose for
`POSE_HOLD_S` with an eased glide; `mode='freeze'` is freeze-frame practice (match within 25°, hold 0.5 s, hints from
`feedback`), never scored. Same surface as `PhraseEngine` where `live.py`/`render.py` touch it.

**`feedback.py`** (307) — scoring result → words.
- `structured_cue(result)` (visibility > timing > praise ≥ 85 > worst channel on its longest weak run), `channel_cue`,
  `live_tip(rows)` (mid-phrase, after two reds ≥ 15°), `describe_phrase(key, i)` ("counts 1 to 2: both arms go up"),
  `layered_cues(result, key, pi, arms_only, limit=3)` (adds `depth_cues` from 3D signals, `timing_cue`, `size_cue`),
  `progress_note(prev, cur, prev_cue)`. Deadbands: depth 35°, facing 30°.

**`cues.py`** (163) + **`style_vocab.py`** (68) — coaching text read off the reference arrays (no authored text per AIST
phrase): `count_cues`, `move_card(key, pi)` (name like "Bounce Reach", setup line, per-count cues, genre feel line),
`count_line`, `intro_line`, `card_note` (≤ 60 words for Gemini). `style_vocab` maps AIST genre codes to authored feel lines.

**`choreo.py`** (186) — routines stitched from library phrases by style and level; deterministic from their name.
- `generate(style, level, n_phrases, seed, name)` → `gen_<style>_<level>_<seed>`, `daily(date)` → `gen_daily_<yyyymmdd>`,
  `load_any(name)` (dispatches `gen_*` to a rebuild, else `load_key`), `with_combined`, `with_half`, `with_counts`,
  `routine_seconds`, `phrase_complexity` (tertiles = beginner / intermediate / advanced), `STYLE_POOLS`, `DAILY_ROTATION`,
  `STYLE_CODE`. Levels: 2 / 3 / 4 phrases at tempo 0.8 / 0.95 / 1.05.

**`ingest.py`** (104) + **`ingest_aist.py`** (65) — COCO-17 keypoint sequences (2D/3D; AIST++ pickles) → keys.
`coco_to_mediapipe`, `key_from_coco_sequence`, `load_aist_keypoints`; the AIST batch CLI walks `data/aist/keypoints3d`,
writes `aist_<genre>_<bpm>_ch<nn>` (BPM = 80 + 10 × music index). Run once: `python -m team.dance.ingest_aist --genres LH MH HO PO LO WA KR JS`.

**`ghost.py`** (60) — a battle winner's own attempt as arrays (never video): `live_to_arrays(engine)`, `ghost_key(code,
owner, rounds, bpm)`, `pack`/`unpack`, `valid` (delegates to `backend/db.valid_ghost`).

**`pacing.py`** (30) — teaching-pace constants, each overridable as `DANCE_<NAME>`: `POSE_HOLD_S` 3.5, `POSE_EASE_S` 0.8,
`POSE_SPEECH_EXTRA_S` 2.0, `FREEZE_MATCH_REST_S` 0.8, `DEMO_TO_ASK_S` 1.0, `TURN_GAP_S` 3.0, `MOVE_GAP_S` 3.5,
`ROUND_GAP_S` 4.0, `LESSON_SOFT_CAP_S` 720.

**`library/`** — 29 keys on disk: 24 `aist_*` (git-ignored, research terms), `cc_huntrix`, `cc_pm_dancer`,
`cc_street_girl` (Wikimedia Commons, tracked), `test_squats` (starter recording, tracked), `class_1771` (ignored).
`gen_*`, `ghost_*`, `chain_*` keys are never saved here.

### 3.2 The session: `live.py` (3862 lines)

The whole call. Module layout, in file order:

| Lines | What |
|---|---|
| 1–52 | docstring (worker command, env list, simulator usage), imports, `sys.path` setup |
| 53–156 | transcript vocabulary: `NAME_RE`, `CMD_WORDS` (phrase → command), `AFFIRM_RE`/`DENY_RE`/`CANCEL_RE`/`HOLD_RE`/`QUESTION_RE`/`REQUEST_RE`/`REMATCH_RE`, `classify(text)` → `{request, rematch, affirm, deny, cancel, hold, question}`, `pain_level`, `style_from` |
| 159–225 | tuning tables: setup caps, `WIN_TOTAL` (195/216/240 by difficulty), `MODES`, `PRIORITY`, `STYLE_*`, `PERSONA_PROMPTS` (hype/chill/coach), `RUNGS` (tempo ladder 0.5 → 0.6 → 0.75 → 0.9), `STATE_RULES` (the per-state note sent to Gemini), `LEVEL_CFG`, `CHOICES` |
| 228–297 | helpers: `parse_code` (digits or number words; strict outside the menu), `parse_commands`, `pick_command`, `key_title` |
| 300–363 | `class Backend` — HTTP via `urllib` (3 s, in an executor, never raises) or, with `BACKEND_URL=fake|off`, `team/backend/db.py` loaded by path and every route mapped in-process |
| 366–452 | Gemini turn shaping and the starter patches: `stage_direction(prompt)`, `tune_realtime()` (en-US transcription, high VAD sensitivity, context compression), `patch_voice_layer()` (applied at import unless `DANCE_VOICE_PATCH=0`): `GeminiModel.build` wrapped, `GeminiModel.generate` replaced with `generate_reply(user_input=stage_direction(...))`, `GeminiModel.start` wrapped by `audio_in.wrap_start`, `Floor.gate_input` wrapped for the barge-in window and 600 ms of silence on a mid-word cut |
| 455–484 | `no_power_throttling()` (Windows EcoQoS opt-out), `on_event(event, say_now, ask_voice)` — the `--handlers` hook; stores the verbs in `STATE` and forwards every event to `STATE['session'].on_worker_event()` |
| 487–3631 | `class DanceSession` (below) |
| 3634–3667 | `dance_consumer(job)` builds a session from the `DANCE_*` env and awaits `run()`; `_register_consumer()` appends it to `CONSUMERS` on both `sys.modules['__main__']` and `facetime.worker` |
| 3670–3845 | simulator fakes (`_FakePlayer`, `_FakeModel`, `_FakeVoice`, `_FakeJob`), `simulate(...)`, `main()` argparse |

**States** (`self.state`, strings): `EXPLORE` (initial; `explore()` moves to CHOOSE at once), `CHOOSE` (six cards),
`OFFER` (one suggestion, needs a yes), `PREPARE` (camera), `CONFIRM` (ready check), `ACTIVITY` (music), `REVIEW`
(between phrases), `PAUSED`, `SAFETY`. All changes go through `transition(state, why)`, which bumps `epoch` (stale queued
lines are dropped) and sends the new rules as a background note.

**`DanceSession` methods by area:**
- *Speech out:* `ask(prompt, key, ttl, urgent)` (the only path to Gemini's mouth), `say_line`, `pick` (variants, never the
  same twice), `context` (background note), `_speaker_loop` (one line at a time; waits for audio idle, model not thinking,
  caller quiet 1 s and silent 6 s; skips if Gemini already asked the same; up to 25 s for the line to end), `say`/`sfx`/
  `cut_audio`/`bank_say` (cue player), `wait_spoken`, `answer_after`, `wait_answer`, `wait_choice`, `wait_cmd`, `idle`.
- *Transcript in:* `on_worker_event` (coach lines → caption, `false_progress` counter; caller lines → `addressed.score`,
  names, codes, `classify`, style/persona/difficulty words, `parse_commands` → `_cmd`), `check_unclear`/`ask_unclear`
  ("Was that a yes?" once per question).
- *Commands:* `_cmd(cmd, source)` (dedupe across voice/tool/gesture/button, state-gated), `take_cmd`, `button_cmd`
  (data-channel buttons from `phone.html`), `_gestures`, `_arm_gesture`, `_hands_down`.
- *Lobby:* `explore(after)` (cards, "What's next?"), `say_menu`, `handle_code`, `request_activity(mode, style, diff, source)`
  (voice/code/tool/card/brodie), `offer_loop`, `welcome_back`, `recommendation`, `load_challenge`, `say_bye`.
- *Setup:* `_start_activity`, `setup_activity`, `prepare_and_confirm`, `_setup_camera` (a `SetupTracker`, `lines_for`,
  `facing_phrase`, `camera_note`; caps 150 s per round, 180 s total, flip-flop breaker), `_space_answer`, `ready_check`.
- *Modes:* `play_loop` dispatches on `self.mode`: `learn()` (plan items half/step/combine; `teach_item`: move card, first
  look pose-by-pose, "got it?", freeze practice for beginners, with-me run, scored tries up the `RUNGS` ladder; pass marks
  65/72/80; two misses simplify; `loop_counts` for weak bars; final full-speed challenge), `battle()` (warm-up, three
  rounds: phrase 0, phrase 1, phrase 0 at ×1.15; HP bar; win ≥ `WIN_TOTAL`; `offer_ghost_save`), `daily()` (one posted
  shot at 0.9×), `squad_battle()` (`SquadEngine`, two phrases + freestyle), `class_lead()` (Brodie leads a group),
  `telephone()` (`record_phrase` → chain, or copy the last link and `show_compare`), `class_mode()` → `class_teacher` /
  `class_student` / `class_report`.
- *Engines:* `run_phrase`, `run_pose_step`, `phrase_outcome`, `summarise` (tips via `feedback`), `xp_for_attempt`,
  `record_round`, `make_routine`, `pick_key`.
- *Finish:* `finish_session` (POST `/dance/result`, board, award, brag lines, `final_card`), `finish_pose` (3-2-1 photo),
  `award`, `brag_lines`, `show_board`, `offer_ghost_save`, `finish_squad`.
- *Safety and memory:* `safety_halt`/`hold_safety` (pain words), `paused_wait`, `request_forget`/`do_forget` (two asks),
  `apply_persona`, `apply_difficulty`.
- *Tool backends* (called by `tools.py`): `tool_command`, `tool_answer`, `tool_confirm`, `set_preferences`, `camera_status`,
  `last_attempt_summary`, `tool_compare`, `queue_routine`, `profile_summary`, `tool_forget`, `squad_status`,
  `set_call_mode`, `assign_name`.
- *Video:* `publish` (`coach-video`, no simulcast, 1.8 Mbit/s), `newest_frame` (reads `job.latest`), `prepare`/`_pose_sync`
  (pose thread), `render_push`/`_render_sync`/`_encode_sync`. `_close_pose_sync` closes the MediaPipe landmarkers on the pose
  thread: when the budget guard asks for `lite` (`pose_reload`) and when the call ends. Never leave them to the garbage
  collector: MediaPipe's `__del__` → `close()` inside its own executor `submit` deadlocked the whole worker on 2026-10-12.
- *Lifecycle:* `run`, `_frame_step`, `_guards`, `_dead_air_step`, `vad_listener`/`on_vad`/`barge_in` (2.5 s mic window over
  the music, ≤ 20 % of a phrase), `listen_buttons`, `register_tools`, `check_single_agent`, `reset_activity_state`, `guard`.

**Room-name switches:** a room starting `demo-` skips the chat/offer/ready gates (one routine after the camera check);
`desk-` anywhere means arms-only (`DANCE_UPPER=1` does the same). A bound identity starting `chert-bridge` selects the
portrait canvas.

**Simulator / CLI:** `python -m team.dance.live --simulate clip.mp4 --mode learn --lobby --say "3:I'm Maya" --say "9:teach me hip hop" --say "20:ready" --answer gotit:ready --gesture 40:ready:1.2 --realistic --players 2 --out data/tmp/sim.mp4`.
`--backend` defaults to `fake`; `--code`, `--name`, `--realtime`, tempo/rtt/fps/pose.

### 3.3 Conversation helpers

**`modes.py`** (37) — the six mode `CARDS` (title, tag, colour, icon, spoken line, body; copy mirrors the landing page),
`HELLO`, `NEXT_LINE`, `CYCLE_LINE`, `NOISY_LINE`, `by_mode`, `index_of`.

**`state.py`** (7) — `STATE = dict(say_now, ask_voice, session, registered)`. Exists because the worker re-imports the
handlers file per job; the running session must be found through a module cached in `sys.modules`.

**`run_worker.py`** (31) — Windows entry point: import `livekit.plugins.google` on the main thread, then `facetime.worker.main()`.

**`tools.py`** (157) — the 14 Gemini Live function tools (`livekit.agents.function_tool`), each a timed wrapper that
calls a session method and returns short JSON; failures return `{ok: false, error}`; every call is a telemetry `tool` row.
`propose_activity(kind, style, difficulty)` (an offer, never a start), `confirm_start()` (report only), `control_session(action)`
(12 actions), `caller_answered(answer)`, `set_preferences(personality, difficulty, style)`, `check_camera()`,
`analyze_last_attempt()`, `compare_with(code)`, `generate_routine(style, difficulty)`, `get_profile()`, `forget_me(confirm)`,
`get_squad_status()`, `set_call_mode(mode)`, `assign_name(colour, name)`. `register(session, agent)` → `agent.update_tools`.
Off with `DANCE_TOOLS=0`.

**`addressed.py`** (51) — `score(text, question_open, brodie_speaking, caller_name) -> 0..1`: was this line said to Brodie?
Answer-shaped short lines while a question is open score high; third-person stories, long lines without our words, lines
during Brodie's speech, garbled tokens score low. `ADDRESSED_AT` 0.5; below `IGNORE_AT` 0.2 the session cuts Gemini's reply.

**`answers.py`** (31) — `confidence(kind, addressed, agrees)` merges regex yes (strong 0.9 / weak 0.6 / foreign 0.6),
Gemini's `caller_answered` (0.75), hands up (1.0); `decide(conf, already_asked)` → accept ≥ 0.7, ask 0.4–0.7, else ignore.

**`watchdog.py`** (83) — `DeadAir`: per-state silence limits (CHOOSE/EXPLORE/OFFER 25 s, PREPARE/CONFIRM 30 s, REVIEW 22 s),
three nudge lines per state, gap doubling after three, `('end', ...)` after 300 s with nobody seen or heard.

**`audio_in.py`** (80) — Krisp background-voice cancellation + AGC on the caller's audio into Gemini, by swapping
`RoomOptions` during `GeminiModel.start` for `patched_options(...)`, a **subclass** (AgentSession.start calls
`RoomOptions._ensure_options` and checks `isinstance`; the plain function used until 2026-10-12 made Gemini fail to start on
every real call while unit tests and the simulator passed). `choose_nc` (auto only on `*.livekit.cloud`), `wrap_start`. Env: `DANCE_NC`
(auto|bvc|bvc_tel|off), `DANCE_AGC` (1). Needs `livekit-plugins-noise-cancellation` (`team/requirements.txt`).

**`persona.md`** (72) — Gemini's system instruction: the hard rule (speak only to the caller, never narrate or mention
tools), the flow (menu → chat → offer → camera → ready; never start anything yourself; never count in), answers via
`caller_answered`, bracketed lines are stage directions, the tools, coaching styles HYPE/CHILL/COACH, the modes with
honest durations, safety (stop on pain, never diagnose), privacy ("forget me" twice), English only, ≤ 2 sentences, stay
silent for room talk. Per-state rules and persona prompts are appended at runtime by `live.py`.

**`plan.txt`** (0) — intentionally empty so the starter's workout plan never reaches Gemini.

### 3.4 Perception, tracking, camera, gestures

**`multipose.py`** (100) — `MultiPose(variant, max_poses=4)`: MediaPipe PoseLandmarker with `num_poses`, returns
`[(lm_img (33,5), lm_world (33,3)), ...]`. Re-exports `PlayerTracker` and the geometry helpers from `tracker.py`.

**`tracker.py`** (283) — `PlayerTracker`: OC-SORT-style ids from motion prediction + torso colour histograms + Hungarian
assignment (`scipy`, greedy fallback). Candidates need a visible torso and ≥ 40 % (solo) / 25 % (squad) of the largest
torso. Lost tracks are re-identified by colour for 20 s (squad) / 6 s (solo); the primary is held 8 s. `_fix_swaps` undoes
an id swap after a crossing. **Primary lock:** kept while present or lost; otherwise the largest stable dancer, and
during an activity only if alone and centred (walk-ins never take a routine). Events: `player_joined`, `player_left`,
`player_back`, `id_swap_fixed`, `primary_locked`, `extra_person_ignored`, `crowd`. `assign_colours` (volt/pink/cyan/amber
left to right), `solo_pose`, `stable_players`, `pop_events`.

**`orient.py`** (141) + **`filters.py`** (32) — depth from one camera. `Calib` learns shoulder/hip width ratios and arm
segment lengths over 20 good frames in the camera check; `Orient.update` → `yaw`, `side`, `away` (face hidden while
shoulders seen), `lr_swapped`, `toward` (a limb pointing at the camera is not a body turn), `ambiguous` (arm channels too
foreshortened to judge, skipped by tips), `conf`. `facing_phrase(yaw, away)` → "Turn a little to your right to face me." /
"Turn around to face me." `OneEuro` smooths the yaw.

**`setup.py`** (183) — the camera check. `evaluate(lm, luma, upper, aspect, prev)` → issues in order (`no_person`,
`too_close`, `too_far`, `off_center`, `dark`, `not_facing`) with hysteresis (torso 0.42/0.38 close, 0.09/0.11 far,
centre band 0.25–0.75); `SetupTracker.update` reports one stable issue (held 2.5 s, led for 1 s) and `ok_stable` (65 %
good frames over 1.5 s); `lines_for(issue, upper, checkin)` the exact second-person lines; `camera_note` the
`[camera now: ...]` note for Gemini; `SHORT` on-screen labels. Feet are never required.

**`gestures.py`** (68) — `classify(lm)` → `ready` (both hands up, wide) / `pause` (arms crossed overhead); `GestureDetector`
fires once after a still hold (1.0 s / 1.2 s), 2 s cooldown, `reset()` blocks a hold until the hands come down, `progress`
for the HUD ring.

**`guards.py`** (59) — `FrameBudget` (5 s median over 55 ms → `lite` model, then 12 fps), `CameraShift` (phase correlation
on a 64×48 frame: the phone moved), `VideoFreeze` (1.5 s unjudged, 4 s abort).

**`squad.py`** (128) — `SquadEngine`: one `PhraseEngine` per dancer on one clock; `crew_sync` (pairwise angle error per
beat), `squad_result` (ranking, MVP, Crew Sync %, spread, `squad_title`, one cue for the dancer furthest off).

### 3.5 Video: what the caller sees

**`render.py`** (1357) — OpenCV on a BGR canvas. `W, H = 720, 1280`; `canvas_size(src_w, src_h)` picks landscape 1280×720
for a wide source; `to_canvas` crops only what the aspect forces; `safe_box` keeps everything clear of FaceTime's controls
(portrait: 9 % top, 15 % bottom, 5 % sides).
- Entry points: `compose(frame, lm, hud, ghost, size, squad_lms)` (a phrase in progress), `compose_idle(frame, lm, ...,
  state, avatar_state, menu, setup, recap, caption, compare, brk, board, players, hints)` (everything else),
  `draw_card(...)` (the end card: `draw_photo_card` finish-pose photo with grade ring, brag pills, XP, kit sticker, code
  tiles and QR; `_draw_classic_card` for squads/unscored), `visibility(lm)`, `framing_hint(parts)`, `mirror_view`.
- Elements: `draw_figure` (caller in paper with a head ring, never covering the face), `draw_ghost` (cyan, hip-anchored at
  torso scale, upper-only when legs are hidden; `ghost_placement`), `draw_panel` (the reference in a corner panel),
  `draw_badge` (Brodie's PiP tile), `draw_state_chip` (WATCH / WITH ME / YOUR TURN / SETUP / READY? ...), `draw_score_block`,
  `draw_battle` (HP bar), `draw_squad_strip`/`draw_squad_counts`, `draw_beat_dots` (count tiles), `draw_judgments`
  (PERFECT / NICE / MISS pops), combo edge glow, `draw_tip`, `draw_gesture_ring`, `draw_main` (5-6-7-8 count-in, reveals),
  `draw_pose_step`, `draw_reveal`/`draw_score_card`/`draw_squad_reveal`, `draw_recap`, `draw_overlays` (confetti, flash,
  K.O., level-up, achievements, MVP), `draw_chat` (speech bubble + hints), `draw_caption`, `draw_paused`, `draw_safety_card`,
  `draw_mode_cards` (the six cards with a lit/picked halo), `draw_setup_view` (head circle, body box, BODY/DISTANCE/LIGHT
  pills, arrows), `draw_break` (NEXT MOVE / BACK TO START ring with the next first pose), `draw_compare` (telephone
  before/you), `qr_payload`/`qr_matrix`/`draw_qr` (`segno`).
- Env: `DANCE_PUBLIC_URL` (QR fallback when no handle). Palette: volt `#D9FF3D`, pink `#FF3DA8`, cyan `#3DE0FF`, amber
  `#FFB13D`, red `#FF4D5E`, ink `#0A0A0B`, paper `#F4F4F0`.
- A legacy layer (`draw_topbar`, `draw_timeline`, `draw_chips`, `draw_history`, `draw_combo`, `draw_menu`,
  `draw_progress_row`, `draw_skeleton`, `to_portrait`) is still defined but no longer called.

**`text.py`** (118) — `put_text`, `text_size`: Pillow-rendered, cached sprites with Archivo Black / Space Grotesk / Plus
Jakarta Sans from `assets/fonts` (Hershey fallback).
**`fx.py`** (79) — `FxQueue` of timed effects (`judgment` 0.65 s, `tip`, `combo`, `confetti`, `levelup`, `achievement`,
`grade_stamp`, `flash`, `ko`, `mvp`), easing helpers, analytic `confetti_particles`.
**`avatar.py`** (172) — Brodie from the kit's `brodie-sit-wave.png` only: `grin` / `closed` / `sleepy` variants, states
idle, listening (volt ring), speaking (mouth alternates 7 Hz), thinking (dots), offline; `draw`, `draw_tile` (PiP).
**`screens.py`** (83) — `draw_code_tiles`, `draw_board` (daily top five), `draw_squad_tags` (colour pill over each head).
**`assets/brodie/`** — the official kit: `KIT_README.md` (cream `#EFE7DA` body, ink outline, locked features),
`brodie.json` (machine spec, QA thresholds), art PNGs, `tools/check_cream.py` (fails on any non-cream body; run by
`test_avatar_is_kit_compliant` and `promo/art.py`). **`assets/fonts/`** — three OFL fonts and licences.

### 3.6 Audio

**`audio.py`** (148) — synthetic 24 kHz PCM: `click`, `metronome(bpm, n_beats, accent_every, countin_beats)`,
`register_phrase_track(bank, bpm, show_beats, countin_beats, attempt_beats, voice={})` (the whole show/count-in/attempt
track under the cue string `[dance:track]`), `register_sfx(bank)` (`SFX`: levelup, stamp, ko, combo, pause, achievement,
mvp, tick), `register_clip` (writes into the starter's `CueBank` cache so `player.say(text)` plays it), `FakeBank`.
**`tts_local.py`** (91) + **`voice/*.pcm`** (12 lines) — OS speech renders of fixed lines. Dormant: only loaded with
`DANCE_LOCAL_VOICE=1` and not mixed into the track (two voices on one call confused callers).

### 3.7 Game layer and telemetry

**`progression.py`** (147) — pure functions: `grade` (S ≥ 92, A ≥ 80, B ≥ 65, C ≥ 50, D), `title_for` ("Human Metronome",
"Combo Menace", "Camera Shy (step back)"...), `squad_title`, `LEVELS` (Rookie 0 → Main Character 2500), `level_for`,
`xp_for_session` (score/4 × difficulty multiplier, bonuses for S/A, combos, battles, duel wins, first today), `UNLOCKS`,
`difficulty_unlocked`, `ACHIEVEMENTS` (12), `new_achievements`. Env: `DANCE_UNLOCK_ALL`. Also loaded by `backend/db.py`.

**`telemetry.py`** (109) — one JSONL per session in `data/tmp/sessions/`: `cue` (latency), `phrase_scored`, `tool`,
`error`, free `event(type, ...)` rows (state, speech, caller_said, nonaddressed, offer, command, gesture, dead_air,
setup_issue, award, tracker events), `summary(completed)` (tracking rate, scores, cue latency p50/p90, tool stats; caller
name hashed), `close`. The summary is POSTed to `/events` by `live.py`.

### 3.8 Tests, eval, offline tools

**`tests/`** — 107 test functions, **165 collected cases** (`python -m pytest team/dance/tests -q`):
`test_scoring` (8: selftest, DTW identity, degradations, engine states, freestyle), `test_ingest` (2), `test_handedness` (2:
mirror convention, cue names the caller's own side), `test_modules` (16: progression, choreo, feedback, gestures, telemetry,
tracker, text/fx), `test_modules2` (10: cues, style vocab, ghost, addressed, answers, noise cancellation, pacing, guards,
screens; the `RoomOptions` swap survives `_ensure_options`), `test_orient` (5), `test_tracker` (5: crossings, re-ID, walk-ins, children, crowd), `test_phase1` (9: codes vs
count-ins, squad ready window, space re-ask, hysteresis, dead air, ghost tokens, freed codes), `test_backend` (7: FastAPI
`TestClient` on a temp DB: codes/duels, chains, classes, profiles, boards/funnel, forget me, metrics), `test_live` (43
functions, 101 cases: `classify` phrasings, flow, speaker queue, offers, space/ready waits, tools guarded, buttons, portrait
canvas, cards, kit compliance, photo card + QR, modes, pose step, depth cues, layered cues, the lite-model swap closing the old
landmarker on the pose thread).

**`eval/run_eval.py`** (358) — 46 scripted scenarios (+ a runtime `duel`) through `live --simulate` in parallel with named
checks (`long_chat`, `choose_*`, `offer_needs_yes`, `learn_lobby`, `patient_*`, `space_*`, `real_*` with realistic timing,
`menu_pick_*`, `telephone_*`, `class_*`, `battle_*`, `learn_pose_step`, `loop_weak_bar`, `garbled_answers`, `setup_*`,
`desk_tracking`, `safety`, `pause_resume`, `forget`, `walk_in`, `squad`, `daily`). Writes `data/tmp/eval_report.json`,
`data/tmp/cue_review.csv`, per-scenario logs and MP4s in `data/tmp/eval/`. Flags `--clip`, `--only`, `--jobs`, `--no-duel`.
Run it alone; it does not finish beside a live worker on this laptop.
**`eval/snapshots.py`** (101) — renders all 20 HUD states to `data/tmp/hud/*.png` (`--landscape` variant) and times `compose`.
**`offline_test.py`** (207) — `selftest` (14 checks incl. QR decode and compose time), `key --video --name --bpm`, `run
--video --key --out` (annotated MP4 + JSON report), `landmarks`, `list`.
**`fake_room_test.py`** (149) — the real `facetime.worker.CoachJob` against a fake room with an MP4 as the caller; PASS =
bound, `coach-video` published, metronome through the cue player, a scored round, clean end, MP4 written.

---

## 4. `team/backend/` — FastAPI + Postgres (Supabase) or SQLite

**`app.py`** (336) — routes (all JSON unless noted; CORS open). `_startup()` creates the tables with a few retries and
refuses to start on SQLite when a cloud marker is in the env (`REQUIRE_DATABASE_URL`, `RAILWAY_PROJECT_ID`, `RENDER`,
`FLY_APP_NAME`, `VERCEL`), so a deployed box can never silently run on a file that the next restart wipes.

| Group | Routes |
|---|---|
| Health, events | `GET /health` → `{ok, db: postgres|sqlite, db_ok}`; `POST /events`; `GET /events/recent`; `GET /metrics` (session summaries + funnel); `GET /sessions/last` (legacy) |
| Browser caller | `GET /join?prefix=demo-desk&facing=user&k=` mints a one-hour token for a fresh room (dispatch to `AGENT_NAME`, same default as the worker) and redirects to `GET /phone.html` (serves `team/web/phone.html`). `JOIN_KEY` set → `k` required. Needs `LIVEKIT_*` on the backend. The no-laptop way to call Brodie from a browser |
| Results, codes | `POST /dance/result` → `{code, score100, ghost_token?, versus?, winner?, board?}`; `GET /dance/result/{code}`; `GET /dance/code/{code}` (what kind a spoken code is); `GET /dance/leaderboard?key&mode&since&limit`; `GET /dance/history?name`; `GET /dance/daily_played?name&key`; `GET /dance?key=` (HTML board) |
| Profiles | `GET /dance/profile/{name}`; `POST .../award`; `POST .../prefs`; `DELETE /dance/profile/{name}` ("forget me": profile, results, ghosts, codes; chains/classes anonymised) |
| Share funnel, boards | `POST /dance/share/{code}/start`; `GET /dance/share/{code}`; `GET /dance/funnel`; `GET /dance/board/{code}` (+ `/page` HTML) |
| Ghosts | `POST /dance/ghost/{code}` (needs the winner's one-time token); `GET /dance/ghost/{code}` |
| Telephone | `POST /dance/chain`; `POST /dance/chain/{code}/link`; `GET /dance/chain/{code}` (+ `/page` drift table) |
| Class | `POST /dance/class`; `GET /dance/class/{code}`; `POST .../student`; `GET .../report`; `GET .../page` (heatmap HTML) |

**`db.py`** (596) — tables `events`, `dance_results` (code UNIQUE, name, key, mode, total, rounds, score100, challenge),
`dance_chains`, `dance_classes`, `dance_profiles` (xp, level, streak, achievements, prefs), `dance_shares`, `dance_ghosts`,
`dance_ghost_tokens`. Codes are four digits, unique across all tables, freed after 30 days (`_free_old_codes`,
`_release_code` takes the ghost, token and share history with it). `post_result` is shared by HTTP and the in-process fake.
`valid_ghost` is a strict shape check (16–64 samples, ≤ 400 KB, 40–220 bpm). Loads `team/dance/progression.py` by path for
levels and achievements.

Two engines, one set of queries: `engine()` is `postgres` when `DATABASE_URL` is set (psycopg 3, one persistent autocommit
connection with an explicit transaction per call, pinged after 60 s idle and reopened when the pooler drops it,
`prepare_threshold=None` for the Supabase transaction pooler), else `sqlite` (`DB_PATH`: `TEAM_DB`, else `team/backend/team.db`;
tests that set `TEAM_DB` or reassign `DB_PATH` always get SQLite, so a test can never write to the real database). Every query
is written in the shared dialect: `?` placeholders (`_PgConn` rewrites to `%s`), `INSERT ... ON CONFLICT` upserts, JSON in
TEXT, epoch seconds in REAL; `_ddl()` turns `INTEGER PRIMARY KEY` into `BIGSERIAL` and `REAL` into `DOUBLE PRECISION` for
Postgres. `_Row` makes psycopg rows behave like `sqlite3.Row`. `ping()` backs `/health`. Rollback is one env var: unset
`DATABASE_URL` and everything is back on the SQLite file (nothing SQLite-side was removed; only the old Vercel `/tmp` default
went, and a cloud marker now refuses SQLite instead). Locally `.env.local` has no `DATABASE_URL`, so the laptop runs on SQLite.

**`dbcheck.py`** (33) — round trip against the configured engine (init, write, read, award, forget me). **`import_sqlite.py`**
(44) — one-time copy of a `team.db` into Postgres (`ON CONFLICT DO NOTHING`, serial counters moved past the imported ids).

Run locally: `team/scripts/backend.ps1` → uvicorn on `:8787` (SQLite unless `DATABASE_URL` is in `.env.local`). The session
defaults to `BACKEND_URL=http://127.0.0.1:8787`; `fake` runs the same `db.py` in-process (the simulator, eval, `chrome.ps1`
and `dance_loop.ps1` use this; with `DATABASE_URL` set the in-process copy also talks to Postgres). Deployed: `team/cloud/start.sh`
(section 6.1).

---

## 5. `team/web/`

- **`phone.html`** (133) — any browser as the caller. Loads `livekit-client` from jsdelivr; params `url`, `token`, `facing`,
  `autojoin=1`, `debug=1`. Publishes the camera as `facetime-remote-video` (1280×720, no simulcast) and the mic as
  `facetime-remote-audio` (the Chert bridge's names, so the worker binds to either), shows the worker's video full screen,
  plays its audio, and sends `{"type":"dance_cmd","cmd":...}` data messages from the Pause / Repeat / Slower / Stop buttons.
- **`groove/index.html`** (1237) — the landing page (https://brodiegroove.vercel.app/), self-contained: hero with a beat
  clock, demo phone mock, a 16-note rhythm mini-game, the six mode cards, coach styles, the 60/30/10 scoring section, the
  flow diagram, FAQ, final CTA, waitlist modal. `CALL_URL = ''` (every "Call" button opens the waitlist);
  `WAITLIST_URL` is the Google Form. `groove/assets/` holds the fonts and kit PNGs.
- **`index.html`** (22) — fitness-era placeholder that dumps `/sessions/last`. Unused.
- **`certs/`** — self-signed cert from `serve_web.py` (git-ignored).

---

## 6. `team/scripts/`

| Script | Does |
|---|---|
| `chrome.ps1` | The one-command Chrome test: starts `serve_web.py` (https :8443) and the worker if missing (under `AGENT_NAME` from `.env.local`, empty = automatic: the same name `mint_link.py` puts in the token), mints a room via `mint_link.py`, opens Chrome or prints the Android short link. Flags `-Demo` (demo room, no gates), `-Restart`, `-Record`, `-Android`, `-Key`, `-Mode`, `-Code`, `-MinVisible`, `-Rtt`. Defaults `DANCE_MAX_POSES=1`, `DANCE_POSE_PX=320`, `BACKEND_URL=fake` |
| `dance.ps1` / `dance.sh` | The dance worker (command in section 2). `-AgentName` for the Chert line, `-Key`, `-Mode`, `-Debug <mp4>`, `-Tempo`, `-Rtt`, `-NoVoice` |
| `dance_loop.ps1` | The same under a restart loop (LiveKit's native library can panic on Windows when a room is deleted mid-connect); `-MaxPoses 4` for squads; stop file `data/tmp/worker_stop` |
| `chert.ps1` / `chert.sh` + `chert_win_shim.js` | Run the Chert CLI on Windows (unsupported upstream): the shim reports `linux`, fakes the 0600/uid credential check, and skips directory fsync (EPERM on Windows). `chert.ps1 doctor`, `chert.ps1 dev --agent <name>` |
| `mint_link.py` | Mint a 1 h room token; with `AGENT_NAME`/`--agent` it adds a `RoomAgentDispatch` so a named worker gets the room; writes the page URL to `team/.go_url` (ignored) for `/go` |
| `serve_web.py` | Threaded https server for `team/web` (self-signed cert generated on first run); `GET /go` redirects to the last minted link |
| `chrome_test.py` | Headless Chrome through `phone.html` with a dance clip as a fake camera; PASS when a phrase scores or ACTIVITY is reached |
| `backend.ps1` / `.sh` | uvicorn on :8787 |
| `link.ps1` / `.sh`, `link_local.ps1` | Phone-page links via the starter's `facetime.mint_token` (automatic dispatch only) |
| `env.ps1` / `.sh` | Load `.env.local` into the shell, `PYTHONUTF8=1` |
| `submit_to_fork.ps1` / `.sh` | **Retired.** Used to squash `team/` onto the public fork while a private repo held the history. The public repo is now the only repo: commit on `main`, `git push origin main`. Do not run these; delete when convenient |
| `publish_site.ps1` / `.sh` | Force-push the landing page to `gh-pages` (Vercel is the live host; `.ps1` points at a folder that no longer exists, see section 13) |
| `worker.ps1` / `.sh` | Legacy: the starter worker with `team/skill/` (not the dance path) |

### 6.1 `team/cloud/` — the cloud box (no laptop)

One container runs the backend API and the worker; the database is Supabase Postgres; Railway builds the repo with Nixpacks
(`/railway.json` → builder NIXPACKS, start `bash team/cloud/start.sh`, restart ALWAYS, health `/health`; `/nixpacks.toml` →
Python 3.12, apt `libgl1 libglib2.0-0` for mediapipe's opencv, CPU-only torch from the PyTorch index, then
`team/cloud/requirements.txt`, then the pose-model download; `/.python-version` = 3.12). The runbook with the keys, the
Supabase and Railway steps, costs and troubleshooting is `team/cloud/README.md`.

| File | Does |
|---|---|
| `start.sh` (45) | env defaults (`PORT`, `BACKEND_URL=http://127.0.0.1:$PORT`, `REQUIRE_DATABASE_URL=1`, `DANCE_PUBLIC_URL` from `RAILWAY_PUBLIC_DOMAIN`, solo pose defaults), `fetch_models.py`, `fetch_library.py`, then uvicorn and `run_worker ... start` each in a restart loop with backoff (2 s doubling to 60 s while a process keeps dying inside 30 s). `WORKER=0` runs the API alone |
| `requirements.txt` | the box's deps: numpy, scipy, opencv-headless, mediapipe, av, google-genai, livekit + agents + google + noise-cancellation plugins, pillow, segno, fastapi, uvicorn, psycopg |
| `fetch_models.py` (28) | `models/pose_landmarker_{full,lite}.task` from Google's bucket when missing (build and boot) |
| `fetch_library.py` (43) | the git-ignored keys (`aist_*`, `class_*`) from `DANCE_LIBRARY_URL` (a zip; a private Supabase Storage signed URL) into `team/dance/library/`, skipping files present; without the URL it only reports how many AIST keys exist |
| `pack_library.py` (26) | the laptop side: zips those keys to `data/tmp/library.zip` for upload |

A VM runs the same `start.sh` under systemd (README, last section). The Chert line is unchanged: `chert dev --agent` still
runs on any machine during a slot and dispatches to the cloud worker by `AGENT_NAME`.

---

## 7. `team/promo/` and `team/video/` (build-time content; nothing at runtime imports them)

**Promo kit** — eight days of posters and reels driven by `config.json` (`call_line`, `call_url`, handle, hashtag,
`daily_style`, `recap`). `render.py` screenshots `days/*.html` with headless Edge/Chrome at 4:5, 9:16, 16:9 (inlines the
fonts, writes `config.js`, QR via `segno`, fills `CAPTIONS.md` → `data/tmp/promo/captions_filled.md`, contact sheet).
`clips.py` cuts the reels with ffmpeg from the pitch video, the film, `data/tmp/eval/*.mp4` and HUD stills (loudness −14 LUFS).
`art.py` generates extra Brodie poses with Gemini and cream-checks them. `promo.js` + `brand.css` are the shared runtime and
tokens. `STRATEGY.md` is the calendar (Day 0 teaser Sun 10/11 → Day 7 recap Sun 10/18) and audience notes; `CAPTIONS.md`
the per-platform copy. Output: `data/tmp/promo/<day>/`.

**Pitch video** (`team/video/`, 46 s of real footage): `scout.py` (contact sheets of eval captures) → `timeline.py` (the
edit as data, 100 BPM) → `voice.py` (Gemini TTS Puck/Kore with transcription re-takes, via `facetime.cues_tts`) →
`music.py` (the landing page groove in numpy, SFX, ducking) → `make_pitch.py` (frames → ffmpeg) → `data/tmp/pitch/brodie_pitch.mp4`.

**Film** (`team/video/film/`, ≈ 55 s "Maya: two left feet → S-rank", generated): `gen.py` (cached Gemini image / Lyria /
Veo calls, spend log, `VEO_BUDGET_S`) → `keyframes.py` + `shots.py` (14 shots) → `clips.py` → `peek.py` → `hud_insert.py`
(the real `render.compose` on a generated shot) → `score.py` → `voice.py` → `audio.py` → `make_film.py` →
`data/tmp/film/brodie_film.mp4`.

---

## 8. Legacy (fitness era, kept for reference)

`team/skill/handlers.py` (the starter's `on_event` contract, forwards events to `/events`), `persona.md`, `plan.txt`
(`squats:10,pushups:8`); run by `team/scripts/worker.*`. `team/tests/replay_handlers.py` feeds an events file through a
handlers module; `fixtures/sample_events.json` is one squat set. Nothing in the dance path uses these.

---

## 9. The starter (template) and what we rely on

### 9.1 `facetime/` — the worker and voice layer
- **`worker.py`** (653) — the LiveKit Agents worker. `main()` parses the starter flags, stores them in
  `RTCOACH_WORKER_ARGS`, and runs `livekit.agents.cli` (`dev | start | connect`). `prewarm` builds the pipeline (skipped with
  `--perception off`) and the `CueBank`; `entrypoint` builds `CoachJob` and awaits `run()`: connect, publish `coach-audio`,
  bind to the first video publisher, run the voice layer and every `@consumer`. Warm jobs (`chert_warm`) say nothing. Ends
  2 s after the caller leaves (`--leave-grace-s`), on room disconnect, `--max-s`, or an error. Flags we pass: `--perception
  off --voice gemini --persona --handlers --plan-file --log-caller --audio-queue-ms --agent-name`.
- **`voice.py`** (655) — Gemini Live in the same participant: `GeminiModel` (`gemini-3.8-live`, voice Puck, `en-US`,
  relay via `GEMINI_BASE_URL`; `build`, `start`, `generate`, `interrupt`, `add_context`), `Floor` (gating; `always_open`
  in our mode; `gate_input` is the echo guard we patch), `MixerAudioOutput` (model audio into the cue player's track),
  `VideoTee` (frames to the model with `--vision`; we do not use it), `VoiceLayer` (`run`, `say_now`, `ask_voice`,
  `load_handlers`). We monkey-patch `GeminiModel.build/generate/start` and `Floor.gate_input` from `live.py`.
- **`audio_out.py`** (341) — `CueBank` (manifest + PCM cache), `CuePlayer` (one `AudioSource`, priority slots pending /
  deferred / urgent, `say`, `say_now`, underrun counts), `attach_mixer` (model audio held while a clip plays). Our
  `audio.py` writes synthetic clips into the bank; `live.py` plays SFX with `say_now`.
- **`coach_agent.py`** — the older standalone agent; we import only `frame_to_rgb`.
- **`perception.py`**, `pipeline.py`, `session.py` — the rep-counting eyes and script (events schema v1). Off in our mode;
  `fake_room_test.py` imports `perception.Config`.
- **`cues_tts.py`** — renders the cue bank with Gemini TTS; `team/video/voice.py` and `film/gen.py` reuse its relay client.
- **`fake_bridge.py`** (a headless caller from an MP4), **`mint_token.py`** (plain room tokens; `link*.ps1`),
  **`replay_local.py`** (recording → events), `webcam_local.py`, `mirror.py`, `avatar.py`, `frame_stamp.py`,
  `latency_report.py`, `loopback_inproc.py`, `probe.py`, `visibility.py`, `phone_test.html` — starter tooling we do not use.

### 9.2 `src/rtcoach/`
The rep counter (`repcount2.py`), classifier (`model.py`), trigger, cue tables (`cues.json`, `cue_prior.json`,
`thresholds_e3c.json`). **We use only `pose.py`** (`Pose(variant)` → image + world landmarks; `multipose.py` mirrors its
construction). `cue_prior.json`, `thresholds_e3c.json`, `data/models/clf_pose.pt`, `trigger_lr.json` are QEVD-licensed:
keep private, never a public URL.

### 9.3 `starter/`
The three files the starter expects teams to edit (`persona.md`, `plan.txt`, `handlers.py`) and the meal-prep example.
We use our own copies under `team/dance/` instead.

### 9.4 `scripts/` (organiser)
`test_perception.py`, `test_voice.py`, `test_worker_shutdown.py` (starter tests), `e2e_from_recording.py` (latency from a
film), `gen_slots.py`, `slot-bot.js`, `submission-bot.js` (hackathon hub).

### 9.5 `docs/`, `.github/`, `assets/`, `models/`
`docs/STARTER_README.md`, `STARTER_EVENTS.md`, `BUILD_ON_BRODIE.md` (read these for the worker, dispatch, the Chert line,
the Gemini relay and the event schema); `docs/index.html`, `slots/`, `submit/` (the organiser's hub pages); `.github/`
(slot and submission bots). `assets/avatar/` is the starter's old puppet; `models/` the MediaPipe `.task` files (ignored).

### 9.6 What we changed in the starter (diff vs `upstream/main`)
1. `facetime/fake_bridge.py`: `add_signal_handler` wrapped in `try/except NotImplementedError` (Windows).
2. `facetime/voice.py`: the private-instruction prefix in `GeminiModel.generate` tightened ("no plan, no narration"). At
   runtime `live.py` replaces `generate` entirely, so this matters only with `DANCE_VOICE_PATCH=0`.
3. `requirements.txt`: `pillow==12.3.0`, `segno==1.6.6`.
Everything else the skill needs is done by monkey-patching from `live.py`, so upstream merges stay trivial.

---

## 10. Data, models, assets on disk (git-ignored)

| Path | What |
|---|---|
| `data/rtcoach_starter_part1.tgz`, `part3.tgz`, `STARTER_BUNDLE_README.txt` | the starter bundle |
| `data/models/clf_pose.pt`, `trigger_lr.json` | QEVD-licensed classifier and trigger (unused by us; keep private) |
| `data/cues_audio_gemini/` | the 379-clip Puck cue bank (`manifest.json`); our synthetic clips are added at runtime |
| `data/aist/keypoints3d/*.pkl` (1,408) | AIST++ 3D keypoints; `ingest_aist` builds the `aist_*` keys from the basic moves |
| `data/dance_clips/` + `SOURCES.md` | four Wikimedia Commons clips (CC BY / BY-SA) behind the `cc_*` keys |
| `data/tmp/sessions/*.jsonl` | per-call telemetry |
| `data/tmp/eval/`, `eval_report.json`, `cue_review.csv` | eval outputs |
| `data/tmp/hud/*.png` | HUD snapshots (used by the promo kit) |
| `data/tmp/pitch/`, `data/tmp/film/`, `data/tmp/promo/` | video and promo outputs |
| `data/tmp/worker_*.log`, `dance_session_log.json`, `*.mp4` | worker logs, the last session log, debug recordings |
| `models/pose_landmarker_{lite,full,heavy}.task` | MediaPipe pose models (`team/cloud/fetch_models.py` downloads full and lite) |
| `team/backend/team.db` | the local SQLite database (deployed: Supabase Postgres via `DATABASE_URL`) |
| `data/tmp/library.zip` | the AIST and class keys packed for the cloud box (`team/cloud/pack_library.py`; upload privately) |
| `team/.go_url` | the last minted phone link (carries a token) |
| `notes/` | the hackathon handbook PDF (prints an API key; never commit) and the deck |
| `research/` | the research plan, seven workstream reports, `SYNTHESIS_H0.md`, `ideation/` (276 ideas, `DANCE_AGENT_PLAN.md`, `STARTER_CAPABILITIES.md`) |

Python: uv-managed CPython 3.12.11 in `.venv` (Anaconda's 3.12.1 cannot load `livekit-local-inference` or torch).
`.venv_conda_broken` is the dead one; delete when convenient.

---

## 11. Environment variables

**Starter (`.env.local`):** `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `AGENT_NAME` (empty = automatic
dispatch; set = the Chert dispatch name), `GEMINI_API_KEY` (team token), `GEMINI_BASE_URL` (the relay). Also
`GEMINI_LIVE_MODEL`, `GEMINI_VOICE`, `GEMINI_LANGUAGE`, `CUE_AUDIO_DIR`, `RTCOACH_EVENT_LOG`, `RTCOACH_CUE_LOG`.

**Ours:** `BACKEND_URL` (`http://127.0.0.1:8787` | `fake` | `off` | the deployed `https://` host), `TEAM_DB`, `DANCE_LIBRARY`.

**Cloud (`team/cloud/`):** `DATABASE_URL` (Supabase transaction-pooler URI; set = Postgres), `REQUIRE_DATABASE_URL` (`start.sh`
sets 1: the API refuses SQLite), `DANCE_LIBRARY_URL` (zip of the private keys, fetched at boot), `JOIN_KEY` (gates `GET /join`),
`PORT` (the API's port, from Railway), `WORKER` (`0` = API only), `RAILWAY_PUBLIC_DOMAIN` (Railway-provided; becomes
`DANCE_PUBLIC_URL`).

| `DANCE_*` | Default | Meaning |
|---|---|---|
| `KEY`, `MODE`, `DIFFICULTY`, `CODE` | generated hip-hop, `learn`, `beginner`, none | what the session starts with |
| `TEMPO`, `RTT`, `FPS`, `POSE`, `POSE_PX`, `MAX_POSES` | 1.0, 0.3 s, 15, `full`, 384, 4 | engine clock and pose input |
| `CALL_HANDLE` | unset | the line's FaceTime address; the scorecard QR opens `facetime:<handle>` (no QR when unset) |
| `PUBLIC_URL` | unset | QR fallback to the board page |
| `LOBBY`, `UPPER`, `CANVAS` | 1, 0, `auto` | chat gates on/off; arms-only; portrait/landscape |
| `MIRROR_VIEW`, `IGNORE_FEET`, `DEPTH_GATE` | 1, 1, 0 | mirror the display; never require feet; mask ambiguous joints from scoring |
| `MIN_VISIBLE`, `PARTIAL_ARMS`, `GREEN_DEG`, `AMBER_DEG`, `HALF_MARKS_DEG` | 0.6, 0.7, 18, 32, 20 | scoring thresholds |
| `TOOLS`, `VOICE_PATCH`, `BARGE_IN`, `VAD`, `INPUT_LANG`, `IGNORE_CUT`, `LAYERED_CUES`, `POSE_STEP` | all on (`en-US`, `high`) | Gemini tools, starter patches, mic window over music, VAD sensitivity, cut replies to room talk, ranked tips, pose-by-pose teaching |
| `NC`, `AGC` | `auto`, 1 | Krisp noise cancellation, gain control |
| `CALL_MAX_MIN`, `MAX_S`, `MAX_PHRASES`, `WIN_TOTAL` | 20, 0, 0, by difficulty | caps |
| `CLASS_PHRASES`, `CLASS_DEMOS`, `CLASS_ATTEMPTS`, `CLASS_ROLE` | 2, 1, 2, unset | class mode |
| `POSE_HOLD_S`, `POSE_EASE_S`, `POSE_SPEECH_EXTRA_S`, `FREEZE_MATCH_REST_S`, `DEMO_TO_ASK_S`, `TURN_GAP_S`, `MOVE_GAP_S`, `ROUND_GAP_S`, `LESSON_SOFT_CAP_S` | see `pacing.py` | teaching pace |
| `DEBUG_MP4`, `SESSION_LOG`, `LOCAL_VOICE`, `UNLOCK_ALL`, `NO_THROTTLE` | unset, `data/tmp/dance_session_log.json`, 0, 0, 1 | developer switches |

---

## 12. Docs index

| Doc | Status | What |
|---|---|---|
| `README.md` (root) | public, current | pitch, modes, how it works, run it, honest limits, credits |
| `CLAUDE.md` (root) | current | working rules for Claude and teammates: read `team/context/` first, keep it true in the same commit, what never gets committed, push to origin only, how to verify, Windows notes |
| `team/context/CODEBASE.md` | this file | |
| `team/context/STATUS.md` | current | what is built, verified, shipped |
| `team/context/TODO.md` | current | ranked next steps |
| `team/context/CONVERSATION.md` | current | the prompt chain, the step-by-step call, every guard, audio/video situations, what to improve |
| `team/context/PLANNED_IMPROVEMENTS.md` | current | the ranked longer plan (supersedes `team/docs/make-it-better.md`) |
| `team/README.md` | partly stale | the team workspace layout; still describes the fitness-era `team/skill` path and the retired private-repo workflow |
| `team/dance/README.md` | mostly current | how the skill works, env vars, test commands; Render/Audio sections predate the current HUD |
| `team/CHROME_TEST_GUIDE.md` | current | Chrome and Chert testing, troubleshooting table |
| `team/PROGRESS.md` (private) | the dated log | status at submission time, milestones M0–M5, log newest first |
| `team/PROGRESS_REPORT_v1..v7.md` (private) | historical | the seven progress reports of build day |
| `team/docs/AUDIT_2026-10-11.md` (private) | current | the final-evening audit |
| `team/docs/criteria-status.md` (private) | current | the 11 criteria × built / check / status |
| `team/docs/fork-scan.md`, `other-team-ideas.md` (private) | snapshot | the other teams' forks |
| `team/docs/research-viral-agents.md`, `research.md` (private) | reference | what travelled, competitors |
| `team/docs/platform-notes.md`, `site-and-waitlist.md`, `waitlist-form.md`, `vercel-services.md` (private) | reference | platform facts, site and waitlist, the parked two-service Vercel layout |
| `team/docs/private-repo.md` (private) | retired | the old three-remote workflow; the public repo is the only repo now |
| `team/docs/decisions.md`, `ideas.md`, `demo-and-posting.md`, `make-it-better.md`, `enhance-brodie.md`, `biohacking-angle.md`, `gamification-and-combat.md`, `workout-pivots.md` (private) | stale or idea docs | pre-pivot shortlist, superseded plans, side angles |
| `team/promo/STRATEGY.md`, `CAPTIONS.md` | current | the posting calendar and copy |
| `team/dance/assets/brodie/KIT_README.md` | reference | the character kit rules |
| `docs/STARTER_README.md`, `STARTER_EVENTS.md`, `BUILD_ON_BRODIE.md` | organiser | the starter, dispatch, the Chert line, the relay, the event schema |
| `research/RESEARCH_PLAN.md`, `SYNTHESIS_H0.md`, `ideation/*` (private) | reference | the research that led to the dance agent (`ideation/DANCE_AGENT_PLAN.md` is the design) |

---

## 13. Known mismatches between docs and code (fix the doc or the code; list kept here until done)

- Root `README.md` says 13 Gemini tools and 53 tests; the code has **14 tools** and **165 collected tests**.
  `team/dance/README.md` lists 11 tools (misses `propose_activity`, `confirm_start`, `caller_answered`).
- Root `README.md` says the starter is unchanged except "a Windows signal guard, a Gemini instruction prefix, per-call
  recordings"; the diff is two files (guard, prefix) plus `requirements.txt`. Recordings are ours (`telemetry.py`, `DANCE_DEBUG_MP4`).
- Root `README.md` credits `multipose.py` with the primary-dancer lock; it lives in `tracker.py`.
- `team/dance/README.md` calls the lobby EXPLORE and says state changes "update Gemini's instructions"; the lobby is CHOOSE
  and the rules go in as background notes. Its Render and Audio sections describe the earlier HUD and the Windows voice lines.
- `scoring.py`'s docstring still quotes 15° half-marks and ±0.25 beat; the code uses 20°, 18/32° and 0.35/0.6 beats.
- `team/README.md` and `backend/app.py`'s docstring describe the `team/skill/handlers.py` path; the live path is `live.py`.
- `publish_site.ps1` copies `team\web\site` (gone); the `.sh` copies `team/web/groove`. Vercel is the live host anyway.
- `link_local.ps1` mints with `facetime.mint_token` (no dispatch); comments elsewhere say it uses `mint_link.py`.
- Two `vercel.json` files with different roots (root: `team/web/groove`; `team/`: `web/groove`) so either Vercel root works.
- `BACKEND_URL=off` means "in-process DB" in `live.py` but "no forwarding" in the legacy `skill/handlers.py`.
- `setup.py` carries a `feet_cut` issue that `evaluate()` never emits (feet are ignored by design).
- `live.py` defines but never uses `NUDGE_S`/`DANCE_NUDGE_S`, `MENU_LINE`, `MAX_ATTEMPTS`, `count_words()`.
- `ingest_aist` defaults to six genres; the shipped library and README use eight (add `KR JS`).
- Keys rebuilt now with `IGNORE_FEET` on would have empty knee mask columns, and `cues.count_cues` would stop reporting
  level/bounce; the shipped keys predate the flag and are fine.
- `promo/clips.py` Day 4 expects `data/tmp/promo/vo/{hype,chill,coach}.wav`; no script in the tree writes them.
- `scipy` is imported by `music.py`, `film/audio.py`, `film/score.py`, `tracker.py` (optional); it is in `team/cloud/requirements.txt`
  but still not in the root `requirements.txt`.
- The cloud box (`team/cloud/`) has not been deployed yet: `nixpacks.toml`, `railway.json` and `start.sh` are written from the
  Nixpacks and Railway docs, not from a green build. The first deploy may need a tweak; remove this line once `/health` answers
  from Railway. The Postgres engine itself is verified against a local Postgres 17 (2026-10-12: `dbcheck.py`, the 7
  `test_backend` cases, the ghost-token / forget-me / 30-day-expiry checks, `import_sqlite.py` run twice on a copy of `team.db`);
  Supabase's transaction pooler is not yet tried.
- `chrome_test.py` reads `--worker-log` as UTF-8; a log written by Windows PowerShell's `*>` redirect is UTF-16 and shows "(none)".
  `chrome.ps1`'s worker logs only to its own window, so the default `worker_dance5.log` is usually stale.
- `livekit-local-inference` is stubbed in this `.venv` (its native module crashed on load), so `live.vad_listener` dies at the
  start of every call (logged, not fatal): no local VAD barge-in over the music on this laptop.

---

## 14. Maintaining this file

- One entry per file, in the section where it belongs. New file → add it; renamed → fix every mention; removed → delete it.
- Keep line counts roughly right (`wc -l`); they tell a reader how big a thing is.
- When behaviour changes (a threshold, a state, a route, an env var), update sections 3, 4 and 11 in the same commit.
- Anything you notice that the docs get wrong goes in section 13 until fixed, then comes out.
- `STATUS.md` holds what works; `TODO.md` holds what is next; `CONVERSATION.md` holds what Brodie says and when; the
  dated narrative goes in `../PROGRESS.md`. Do not put those here. New context docs go in this folder.
