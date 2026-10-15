# BrodieGroove to-do

Ranked. `[ ]` open, `[~]` started, `[x]` done, `[!]` blocked. Tick things here; write the story of how in `../PROGRESS.md`;
when a tick changes what works, update `STATUS.md`; when it changes a file's role, update `CODEBASE.md`.
Sources folded in: `PLANNED_IMPROVEMENTS.md`, `team/docs/AUDIT_2026-10-11.md`, `CONVERSATION.md` §8,
`team/promo/STRATEGY.md`, the mismatches in `CODEBASE.md` §13. Updated 2026-10-15.

The judged window is **Mon Oct 12 to Sun Oct 18**: most traction wins. Every engineering item below should earn a post.

## 0. Now (before Monday's launch post)

- [x] Version the proven Railway compatibility fixes so GitHub builds retain the working runtime.
- [!] Enable `main`-push autodeploy after the repository owner authorizes the Railway GitHub App; write access is confirmed, but Railway reports `NO_INSTALLATION`.

- [x] Temporary deployment preparation (2026-10-14): separate checkout, official Mac CLI on Node 22, Railway service/domain; DB-dependent HTTP status, secret-safe access logging, and a Railpack translation of the legacy Nixpacks build.
- [x] Temporary Railway deployment: supplied credentials, PostgreSQL roundtrip, HTTPS health, keyed join dispatch and browser rendering, LiveKit registration/doctor dispatch, Gemini Live setup, and assigned Chert project confirmed.
- [x] Verify full/lite pose loading and blank-frame inference remotely after adding missing EGL/GLES native libraries.
- [x] Phone tester (2026-10-15): every mode said "Sorry, I glitched" after the camera-check offer. Cause: no AIST keys on Railway,
      so `choreo.generate` raised on the yes. Routines now fall back to the tracked `cc_*` keys. Redeploy Railway to ship it.
- [x] Phone tester (2026-10-15): both hands up was drawn on the skeleton but never counted as a yes. The detector needed a wide V
      and a perfect frame run; now hands straight up count, with jitter/dropout tolerance and an elbow stand-in. Needs a phone call to confirm.
- [!] Private AIST library unavailable on the cloud box: routines there use only the three Commons keys (one BPM group, so mostly
      `cc_pm_dancer`) until the team obtains/rebuilds the AIST keys.
- [ ] Supervised media test when requested: caller audio, agent audio, caller video, agent video, and cleanup separately. The user authorized continuous Chert connector operation; the operator login service is running, with clean no-call stop/release/restart verified.

- [x] Submission issue opened on `seraphimlabs-dev/brodie-starter` (2026-10-11).
- [x] Sunday LinkedIn post tagging Brodie and Chert (the entry ticket) (2026-10-11).
- [x] Public repo `imperorrp/ivyhacks-brodiegroove` up to date; it is the only repo from now on (2026-10-11 22:16 ET).
- [x] **Put the worker/backend in the cloud; prepare the operator-Mac Chert connector** (`team/cloud/README.md`). Temporary Railway Postgres is provisioned; the API and LiveKit agent are deployed. `JOIN_KEY` and the confirmed call handle are configured. The existing Vercel site remains unchanged. Generated-routine and supervised-call work remains above.
- [ ] **Log every post** at getbrodie.com/ivyhacks (resubmit with the same team name each time).
- [ ] **Retire `team/docs/demo-and-posting.md`**; its checklist is stale and its plan is "_tbd_". `team/promo/STRATEGY.md`
      is the plan now; keep one source.
- [~] **Set `DANCE_CALL_HANDLE`** (temporary Railway service configured; team laptop still separate) in `.env.local` and on Railway to the line's FaceTime address and document it in
      `.env.example` (keep the value out of git). It is the result page's Call Brodie button and, with no public URL, the QR.
      Zero code.
- [ ] **Book Chert slots for the week** and put the times on the site; set `call_line` / `call_url` in
      `team/promo/config.json`, re-run `python -m team.promo.render` and `python -m team.promo.clips`.
- [ ] **One real FaceTime call that lands.** Worker on `dance_loop.ps1 -AgentName dance-battle-agent-ivyhacks`, `chert.ps1
      dev --agent ...` (or a Mac), call the line, prop the phone, go through CHOOSE → scorecard. Film it with a second
      phone. This is the clip every post needs and the first time the portrait canvas, the QR and live tool calls are seen.

## 1. The posting week: one feature, one post, each day

Calendar from `promo/STRATEGY.md` (posters and reels exist for every day) with the feature that makes each post true,
from `PLANNED_IMPROVEMENTS.md` §4. Times: TikTok/Reels 6–9 PM, IG feed 11 AM–1 PM, X 12–3 PM, LinkedIn 8–10 AM.

| Day | Post (ready) | Ship so the post is real | |
|---|---|---|---|
| Mon 10/12 | Launch: the line is open; 46 s pitch video | Line booked, `call_line` set, handle set; the real-call clip | [ ] |
| Tue 10/13 | How it works carousel; 55 s Maya film | **Brodie's face with moods** (built 2026-10-13: happy / pumped flex / focus / encourage from kit art, driven by verdicts, scores, K.O., level-ups, coach style) and the **result link with a preview card** (built: QR → `/r/<code>`, `/r/<code>/og.png`). Left: deploy so the link is public, then paste one into iMessage / Instagram / X to see it unfurl, and see the moods on a live call | [~] |
| Wed 10/14 | "Real scores, not vibes" (builder day): 60/30/10, battle capture | **Start-offset search before the DTW** (§3.1) so the numbers deserve the post; the self-consistency table if it exists | [ ] |
| Thu 10/15 | Pick your Brodie: HYPE / CHILL / COACH | Record the three voice-overs `data/tmp/promo/vo/{hype,chill,coach}.wav` + `lines.json` (nothing writes them yet; `team/video/voice.py` has the shape) | [ ] |
| Fri 10/16 | Squad up, get tagged | A squad run with real people on one camera (`-MaxPoses 4`; a GPU box or a fast laptop: 4 poses cost 237 ms/frame here) | [ ] |
| Sat 10/17 | Which grade are you? #BeatMyBrodie weekend | Nothing new to build; push the fill-in template and reply to every tag | [ ] |
| Sun 10/18 | Week one on the line: stats + scorecard wall | Fill `recap` in `config.json` from `GET /metrics` and `GET /dance/funnel`; re-render Day 7 | [ ] |
| Every day | "Today's 8-count" story (`daily/daily_9x16_<day>.png`) | The daily key rotation (`choreo.DAILY_ROTATION`) matches `config.json` `daily_style` | [ ] |

Stretch posts if a day has room (each reuses existing routes): the **Break Room feed** (Brodie posts one line about each
call to a public page, names stripped; "my AI dance coach talks about me behind my back"), the **spectator link** (a page
that subscribes to `coach-video` in our own room and sends emoji on the data channel), the **teacher page** ("Brodie for
dance teachers": class mode's heatmap page as the product shot; a LinkedIn post), **creator upload** (URL or file →
`reference.build_key` → a code), the **telephone chain page** with a drift curve, a **week leaderboard** and a "Brodie
Wrapped" card. Keep battle, telephone and class as spoken words that work, not a menu.

## 2. Quick wins (hours each, ranked by value per minute)

- [x] **Responsive portrait phone caller**: portrait camera capture on narrow screens, safe-area-aware layout, and touch-sized
      controls in `team/web/phone.html` (2026-10-15); physical iOS/Android validation remains open below.
- [ ] **Validate the phone caller on real iOS Safari and Android Chrome**, including camera/mic permissions, controls, and
      visibility of the full portrait coach video.
- [ ] **"Ready" button on `team/web/phone.html`** mapped to the affirm path in `live.button_cmd` (about six lines);
      makes laptop filming reliable when Gemini mishears "ready".
- [ ] **Shorter waits in `demo-` rooms**: `wait_choice` reprompts after 18 s; use 6 s when `lobby_enabled` is off.
- [ ] **"It noticed"**: `--vision` during the lobby only, one persona rule (mention one visible thing, never guess personal
      facts); gate by Gemini minutes (240 per team on the relay).
- [ ] **Memory across calls**: store the last three weak counts and the last cue in the profile; open the next call with
      "last time count six got you".
- [x] **Fix `chrome.ps1` dispatch mismatch**: the worker now starts under `AGENT_NAME`, the name `mint_link.py` puts in the
      token (2026-10-12).
- [ ] **Install `livekit-plugins-noise-cancellation` in `.venv`** (`uv pip install -r team/requirements.txt`): every call logs
      "noise cancellation off (plugin not installed)", so Krisp has never run here.
- [ ] **`vad_listener` on a stubbed `livekit-local-inference`**: it raises at the start of every call ("Task exception was
      never retrieved"); either get the native module loading again or catch it once and log "barge-in off".
- [ ] **`chrome_test.py` log reading**: accept UTF-16 logs (PowerShell's `*>`), and have `chrome.ps1` tee its worker to
      `data/tmp/worker_chrome.log` so the test and a human can read the same log.
- [ ] **Postgres in the test run**: an opt-in `TEST_DATABASE_URL` that runs `test_backend` and the db checks of
      `test_phase1` on Postgres too (done by hand on 2026-10-12 against a local Postgres 17; keep it from regressing).
- [ ] **`publish_site.ps1`** copies `team\web\site` (gone); point it at `team\web\groove` like the `.sh`.
- [ ] **`link_local.ps1`** should mint through `mint_link.py` (named dispatch, `.go_url`) as the comments claim.
- [ ] **Install note** in both READMEs: `uv pip install -r requirements.txt -r team/requirements.txt` (pillow, segno and
      the noise-cancellation plugin were missing in this venv at one point).
- [ ] **Landing page copy**: the hero says "Call Brodie and dance." with no number; show the slot times and a `facetime:`
      link when the line is ours; make the six modes and three coach styles explicit; FAQ on privacy, the upper-body rule,
      codes and squad tagging; Open Graph and Twitter card tags; a kit favicon.
- [ ] **Live board embed** on the site (`/dance` leaderboard and the daily board, read-only) and the funnel on a dashboard.

## 3. Scorer: make the number mean something (the week's engineering core)

Today a cropped-webcam caller lands 10 to 15 points above random noise against their own key. In order:

- [ ] **3.1 Start-offset search** (±1 beat of constant offset) before the banded DTW (half a day). Fixes most "late on the
      beat" verdicts on honest attempts.
- [ ] **3.2 Three explicit losses**: aligned pose error on the DTW path, timing penalised against the original beat clock,
      and a dynamics term from body-relative joint velocities; each with its own threshold (one day).
- [ ] **3.3 Fix the energy feature**: world landmarks are hip-centred, so hip translation is meaningless; use body-relative
      joint velocities (two hours).
- [ ] **3.4 Camera-tilt normalisation** of the angle channels so a desk laptop and a floor phone score alike (half a day).
- [ ] **3.5 One-Euro smoothing** of landmarks, tuned so it does not smear rhythm; measure the added phase delay (two hours).
- [ ] **3.6 The self-consistency benchmark** (then permanent): same performer repeats the reference (target 90+), cropped
      copy (70+), slightly slower, opposite arm (localised error), turned toward the camera, partial occlusion (no
      confident false correction), intentionally late (timing error survives the DTW), freestyle. Put the table in the
      README; it is the strongest technical post.
- [ ] **3.7 `beat-this`** for uploads: beat phase and downbeats instead of a fixed BPM.
- [ ] **3.8 Re-tune thresholds** (`DANCE_GREEN_DEG`, `AMBER_DEG`, `HALF_MARKS_DEG`, timing windows) on the team's own
      clean / late / wrong-arm takes, not on synthetic data.

## 4. Conversation and coaching (from `CONVERSATION.md` §8 and `PLANNED_IMPROVEMENTS.md` §3)

- [ ] **True barge-in while Brodie talks**: the echo guard drops the caller's audio while Brodie speaks; sending silence
      instead of dropping frames is a starter change (`Floor.gate_input`).
- [ ] **Speaker check for yes** (require the caller's own voice; needs a speaker model).
- [ ] **Language**: offer to continue in the caller's language (per-call input-language switch).
- [ ] **Noise cancellation per mode**: switch BVC off for squads mid-call (the selector is fixed when the track is subscribed).
- [ ] **Reference depth**: re-ingest the AIST++ keys with per-count facing, travel and height for "step toward me" / "turn
      left" cues (today: facing size only).
- [ ] **Phase-aware structured feedback**: add the movement phase and the measured delay ("your right arm peaks half a beat
      late on counts 5 to 6"); pick the drill by error type (late → tempo down and loop; wrong limb → isolate; low energy → "bigger").
- [ ] **Intervention timing policy** on the live verdicts: one tip after two consecutive misses of the same kind, never
      twice in four counts, never over the count-in.
- [ ] **Agentic inspection of the worst segment**: the scorer picks the single worst run, Gemini gets those counts' numbers
      and a frame, explains, then one drill.
- [ ] **Counterfactual ghost**: draw the caller's own skeleton nudged toward the corrected pose instead of the reference dancer.
- [ ] **Metrics dashboard** for the per-call numbers already logged (time to choose, re-asks, unused answers, non-addressed
      lines, longest silence, setup bounces, noise cancellation on/off).

## 5. Verify with people (the things no simulator covers)

- [ ] A FaceTime call that lands on the Chert line (§0).
- [ ] A person through the full current flow in Chrome on a propped phone, whole body in frame (the last human runs
      predate the mode cards, tracker, depth and noise work).
- [ ] Squad with two to four real people; check tag swaps when crossing.
- [~] Gemini tool calls on a live call (watch the `tool` rows in the session JSONL); fallback `DANCE_TOOLS=0`.
      `propose_activity` and `confirm_start` seen ok on a human Chrome call (2026-10-12); the other 12 not yet.
- [ ] **Before any post or demo: `chrome_test.py` headless, then check the newest `data/tmp/sessions/*.jsonl` for
      `speech_dropped`.** Real-call-only bugs (Gemini start, the pose-model deadlock) passed every unit test and eval scenario.
- [ ] The QR in a FaceTime screenshot, opened from iPhone Photos, lands on `/r/<code>` and its Call Brodie button opens
      FaceTime to the line; the same link pasted into iMessage, Instagram DMs and X unfurls as the card.
- [ ] Krisp BVC with a TV on in the room; "Was that a yes?" with a mumbled answer.
- [ ] `DANCE_RTT` from a clap test on the first real call (default 0.3 s).
- [ ] The eval harness alone on the laptop after each engine change (`run_eval --jobs 6`; it did not finish beside a live worker).
- [ ] Pilot numbers during the week: first-timer completion, time to first dance, "would use again", cue relevance
      (human review of `data/tmp/cue_review.csv`), share-funnel conversion (`GET /dance/funnel`).

## 6. Content and data

- [ ] **Own clean solo keys** of three trending moves, filmed front-on, whole body, steady BPM, with clean / late /
      wrong-arm takes for §3.6. The single most valuable hour of filming.
- [ ] Class-mode teacher keys as the growth path (every teacher who records a class grows the library, with consent).
- [ ] PopDanceSet through `ingest.py` for younger-audience moves; AIST++ stays demo-only under its research terms.
- [ ] A Pexels ingestion queue for reference variety (credit and link terms).
- [ ] Library README with source and licence per key.

## 7. Platform

- [ ] A Mac for the line during the posting week; the Windows shim stays the fallback.
- [ ] Multi-pose on a GPU box for squads (237 ms a frame here); keep solo defaults on laptops (`DANCE_MAX_POSES=1`, 320 px).
- [ ] Live web room with browser-side pose (MediaPipe in each browser, landmarks over the data channel): live squads and
      live classes on separate phones; we own the room, so no seat limit. A few days.
- [ ] Process hygiene on Windows: worker and page server in their own windows, keep the restart loop, never run the agent
      dispatch test while the line is live, close spare Chrome tabs.
- [x] Decide on the backend's home: Supabase Postgres behind the same FastAPI, deployed with the worker on Railway
      (`team/cloud/`; the Vercel two-service idea in `team/docs/vercel-services.md` is dropped). Deployment itself: §0.
- [ ] Cloud follow-ups once the box is live: Realtime on the crew board, (the Open Graph card per code is built: `/r/<code>`), share
      links with a `source` on `dance_shares`, a `sessions` table (one row per call) for the traction chart, pg_cron for the
      30-day expiry, a second worker container when calls overlap (state is already off the disk).
- [ ] Chert outbound audio for reminders, if the project ever gets it; until then nothing leaves the call.

## 8. Doc and code hygiene (each is minutes; from `CODEBASE.md` §13)

- [ ] Root `README.md`: 14 tools (not 13), 165 tests (not 53), the starter diff is two files plus `requirements.txt`
      (drop "per-call recordings"), the primary-dancer lock is in `tracker.py`.
- [ ] `team/dance/README.md`: list all 14 tools; the lobby is CHOOSE; rules go to Gemini as background notes; rewrite the
      Render and Audio sections for the current HUD and the one-voice rule; add the install line.
- [ ] `team/README.md` and `backend/app.py` docstring: describe the `live.py` path, not `team/skill/handlers.py`; fill
      "Who owns what".
- [ ] `scoring.py` docstring: 20° half-marks, 18/32° verdicts, 0.35/0.6-beat windows.
- [ ] `team/dance/__init__.py` docstring: add `setup`, `orient`, `tracker`, `addressed`, `answers`, `watchdog`.
- [ ] `ingest_aist` default genres: add `KR JS` to match the shipped library.
- [ ] Remove dead code: `live.py` `NUDGE_S`/`DANCE_NUDGE_S`, `MENU_LINE`, `MAX_ATTEMPTS`, `count_words()`; `render.py`'s
      legacy layer (`draw_topbar`, `draw_timeline`, `draw_chips`, `draw_history`, `draw_combo`, `draw_menu`,
      `draw_progress_row`, `draw_skeleton`, `to_portrait`); `multipose.py`'s duplicate helpers; `setup.py`'s `feet_cut`.
- [ ] Add `scipy` to `requirements.txt` (used by `tracker.py`, `music.py`, `film/audio.py`, `film/score.py`).
- [ ] Rebuilding keys with `IGNORE_FEET` on would blank the knee columns and silence `cues.count_cues` level/bounce;
      either build keys with the flag off or make `count_cues` read the key's own mask.
- [ ] Retire or archive `team/skill/`, `team/tests/`, `team/web/index.html`, `scripts/worker.*` (fitness era) or mark
      them legacy in place.
- [ ] Delete `team/scripts/submit_to_fork.ps1` / `.sh` and `team/docs/private-repo.md`; rewrite the repo section of
      `team/README.md` for the single public repo.
- [ ] `team/docs/decisions.md` still says "Idea: not picked yet"; add the dance decision and the date.
- [ ] Delete `.venv_conda_broken` when convenient.

## 9. Research path (after the product week)

- [ ] GVHMR for world-grounded 3D motion recovery of uploads, then a 3D ghost.
- [ ] V-JEPA 2.1 clip embeddings as a second scorer, compared on the §3.6 benchmark against joint angles and a fusion.
- [ ] GMR humanoid retargeting demo ("learn a dance with Brodie, watch the robot attempt it").
- [ ] A concept bottleneck (late extension, insufficient leg movement, inconsistent timing) feeding the cue and the Break Room.
- [ ] Fitness and biohacking on the same engine (`MovementAssessment` interface, adaptive technique coach, Fitness Age /
      Balance Age "Brodie's Lab"; see `team/docs/biohacking-angle.md`, `workout-pivots.md`).
