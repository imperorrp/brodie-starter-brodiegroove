# Brodie Groove: planned improvements (updated Sun 2026-10-11, evening ET)

Supersedes the "what is next" lists in `PROGRESS_REPORT_v7.md`, the three-hour plan in `make-it-better.md` and the research
mapping in `research_papers_and_other_useful_code_and_dataset_ideas.md` (repo root, private). Everything below is ranked
within its section; effort tags are honest guesses for one person on this codebase. Nothing here is started unless marked.

## 0. Where we stand

Shipped and verified today: the dance engine, the v3 session (chat, offer, camera setup, ready check, teaching ladder), the
game layer (grades, titles, XP, combos, scorecard with QR), class mode, telephone, battle, daily, squad in the simulator, the
kit avatar, Gemini tools, telemetry, the Chrome demo path with named dispatch and a demo room mode, the Chert CLI on Windows up
to a held line. Not yet verified: a FaceTime call that lands, a person through the v3 flow end to end, squad with real people.

Honest limits that shape the list: the scorer puts a cropped-webcam caller only 10 to 15 points above random noise against
their own key; Windows is fragile (native panics, memory pressure, unsupported CLI); the shared line is one team at a time.

## 1. This week, before and between posts (hours each)

1. **Publish what exists**: commit the uncommitted demo fixes, run `submit_to_fork.ps1`, tick `demo-and-posting.md`.
2. **`DANCE_CALL_HANDLE`** set to the line's FaceTime address so the scorecard QR works (zero code).
3. **"Ready" button** on the phone page, mapped to the affirm path in `live.py` (six lines).
4. **Demo-room waits**: 6 s instead of 18 s before the reprompt in `demo-` rooms.
5. **README refresh** (`team/dance/README.md` Render and Audio sections are stale; add `uv pip install -r requirements.txt`).
6. **Brodie's face with moods** (the avatar exists; drive its expression from the last verdict and the persona's coach style:
   hype, unimpressed, "no nines"). The emotional loop `make-it-better.md` ranks first; one sprite layer plus a mood variable.
7. **Result page per code with an Open Graph image** so a pasted link unfurls into the scorecard (Pillow, one backend route;
   the share funnel endpoints already exist).
8. **"It noticed"**: `--vision` on during the lobby only, one persona rule (mention one visible thing, never guess personal
   facts). Cheapest viral moment; gate it by Gemini minutes.

## 2. Scorer: make the number mean something (the week's engineering core)

From the research file's "five changes" and the v7 measurement, in order:

1. **Start-offset search before alignment** (half a day). A caller never starts on the key's first beat; search ±1 beat of
   constant offset, then run the banded DTW. Fixes most of the "late on the beat" verdicts on honest attempts.
2. **Separate aligned pose from beat-phase timing from movement dynamics** (one day): score what happened on the DTW path,
   but penalise when it happened against the original beat clock, plus a dynamics term from body-relative joint and angular
   velocities. The current 60/30/10 blend becomes three explicit losses with their own thresholds.
3. **Fix the energy feature** (two hours): MediaPipe world landmarks are hip-centred, so hip translation is meaningless;
   use body-relative joint velocities (and image-space motion when needed).
4. **Camera-tilt normalisation of the angle channels** (half a day): a laptop at desk height and a phone on the floor should
   score alike. Later: viewpoint normalisation from a reconstruction model (GVHMR, VGGT-Omega) offline on uploads.
5. **One-Euro smoothing** of landmarks (two hours), tuned so it does not smear rhythm; measure the phase delay it adds.
6. **The self-consistency benchmark** (half a day, then permanent): same performer repeats the reference (90+); cropped copy
   of the same clip (70+); slightly slower (shape stays, timing moves); opposite arm (localised error); turned toward the
   camera (graceful degradation); partial occlusion (no confident false corrections); intentionally late (timing error survives
   DTW); freestyle (rhythm and variety only). Report the table in the README; it is the most convincing technical post we have.
7. **Beat This for uploads** (`pip install beat-this`; half a day): beat phase and downbeats instead of a fixed BPM when a
   creator uploads a clip; Danceba's beat-alignment metrics as an evaluation yardstick.

## 3. Coaching: proactive and specific

1. **Structured, phase-aware feedback** (BioCoach and ExpertAF as the model): the cue already names a side, a body part and
   counts; add the movement phase and the measured delay ("your right arm peaks half a beat late on counts 5 to 6"), and let
   the drill selection depend on the error type (late → tempo down and loop; wrong limb → isolate the arm; low energy → "bigger").
2. **Intervention timing** (Stream-VLM's question: when to speak): a small policy on the live per-count verdicts that fires one
   tip only after two consecutive misses of the same kind, never twice in four counts, never over the count-in.
3. **Agentic temporal inspection** (cheap, no new model): the scorer picks the single worst segment, the session hands Gemini
   those counts' numbers and a frame, Gemini explains or disambiguates, then one drill. Same shape as VideoITG without the model.
4. **Counterfactual ghost** (PoseGuru's idea; one to two days): instead of the reference dancer, draw the caller's own skeleton
   nudged toward a plausible corrected pose under joint-length and joint-limit constraints. Visually distinctive; also works for
   mobility and fitness.
5. **Memory across calls**: the profile already stores XP and prefs; add the last three weak counts and the last cue so the
   next call opens with "last time count six got you; let's start there".

## 4. Product and virality (one feature, one post)

| Day | Ship | Post hook |
|---|---|---|
| Mon | Face with moods, OG result card, the demo clip on the site | "I FaceTimed a number and an AI gave me a 6 for the Renegade. It doesn't give nines." |
| Tue | Break Room feed: after each call Brodie posts one or two lines about the session (names stripped, PII check), two other personas reply, humans only read | "My AI dance coach has a social life and talks about me." |
| Wed | Spectator link: a page that subscribes to the coach video in our room and sends emoji on the data channel that pop on the caller's HUD (we own the room, so a third participant is fine) | "My friends watched me get judged live." |
| Thu | Teacher page: class mode presented as "Brodie for dance teachers" (routes exist; needs the page polish and a 30-second clip) | LinkedIn: "The teacher records the choreo once. Brodie runs the class over FaceTime." |
| Fri | Creator upload: paste a URL or file, `reference.py` builds the key, returns a code; one workout key as the example (`workout-pivots.md`) | "Any video becomes a scored Brodie routine." |
| Sat | Telephone chain page with the drift curve; week leaderboard; Brodie Wrapped card | "We played dance telephone through an AI. Here's how the move mutated." |

Keep battle, telephone and class as spoken words that work, not a menu; the v3 flow already leads with one suggestion.

## 5. The site (`team/web/groove`)

1. **The demo section needs a real clip** (the `#demo` block): a 20-second phone recording of a call, scorecard at the end.
2. **"Call Brodie now" should be a `facetime:` link** to the line when it is ours, with the slot times shown; otherwise the
   waitlist form (already wired) plus a "try it in Chrome" link that mints a demo-room token server-side (a backend route that
   calls `mint_link.make`, rate-limited, no token in the page source).
3. **Live board embed**: the backend's `/dance` leaderboard and the daily board as a section, read-only, with the crew code
   explained; later the Break Room feed as its own section.
4. **A scorecard gallery**: the OG images of consenting callers (opt-in on the card), the strongest social proof we can show.
5. **Teacher and class page** linked from the modes section; the heatmap page is the product shot for the B2B story.
6. **FAQ additions**: privacy (video analysed live, never stored; "forget me" twice deletes everything), the upper-body
   rule (a desk webcam is fine, arms get scored), what a code is, how squad tagging works (colours, no faces).
7. **Open Graph and Twitter card tags**, a favicon from the kit, Lighthouse pass (the page is a single large HTML file), and
   analytics on the share funnel the backend already records (`/dance/funnel`).
8. **Copy**: the hero still says "Call Brodie and dance." with no number; the modes list shows "Beat Assassin" (a title) in the
   modes row; make the six modes and the three coach styles explicit, and add one line on the Chert line and FaceTime.

## 6. Fitness and biohacking on the same engine (next month)

The engine scores a body against a reference on a time grid; dance is the version with music. From the research file and
`biohacking-angle.md`, in the order they reuse code:

1. **A `MovementAssessment` interface** (extract features, segment, score, diagnose, suggest intervention) with the dance
   assessment and a fitness assessment (rep phases, range of motion, control, symmetry, consistency) behind it. Sports2D for
   joint and segment angles, Fitness-AQA and FLEX as research data (both non-commercial; verify terms).
2. **Adaptive technique coach**: recurring quality problem → one corrective drill → the next set is compared. BioCoach plus
   Stream-VLM plus ExpertAF as inspiration, the existing drill ladder as the mechanism.
3. **Personal intensity controller**: within-person baselines across sets, trend detection (reduced range, slowing), a rest or
   an easier variation, then verify. Never equate camera features with heart rate or fatigue.
4. **Explainable fitness battles**: multi-criterion scores (depth, stability, control, symmetry, timing) with a rule engine so
   the LLM never invents a quality number (NS-AQA's shape).
5. **Brodie's Lab, the 7-day n-of-1**: Fitness Age plus Balance Age in one two-minute call, the before/after ghost (you from
   day 1 drawn on today's video), test your parents (chair stand and balance for the 60+ branch), a readiness call each morning.
6. **Workout pivots that cost almost nothing**: weights in `scoring.py`, a metronome instead of music, short phrases, no
   mirroring for side views, follow-along keys from any video.

## 7. Content and data

1. **Own clean solo keys** of three trending moves, filmed front-on, whole body, steady BPM, with clean / late / wrong-arm
   takes for the benchmark in section 2. The single most valuable hour.
2. **Class-mode teacher keys** as the growth path: every teacher who records a class grows the library with consent.
3. **PopDanceSet** through `ingest.py` for younger-audience moves; AIST++ stays demo-only under its research terms.
4. **Pexels API** ingestion queue for reference variety (credit and link terms), never "search any TikTok by title".

## 8. Platform

1. **Live web room with browser-side pose** (MediaPipe in each browser, landmarks over the data channel): live squad battles
   and the live class with teacher and students on separate phones. We own the room, so no seat limit. A few days.
2. **Multi-pose on a GPU box** for squad in one camera (237 ms a frame here; fine on a desktop GPU). Keep solo defaults on laptops.
3. **A Mac for the line** during the posting week; the Windows shim stays the fallback, never the plan.
4. **Chert outbound (audio) for reminders** if the project gets it: the retest call for the Lab, the "your friend beat your
   score" nudge. Until then nothing leaves the call.
5. **Memory and process hygiene on Windows**: run the worker and the page server in their own windows, keep the restart loop,
   never run the agent dispatch test while the line is live.

## 9. Research path (after the product week)

1. **GVHMR** for world-grounded 3D motion recovery of uploads (static camera), then a 3D ghost; SAM 3D Body only for pretty
   offline highlights.
2. **V-JEPA 2.1 embeddings** of short clips as a second scorer, compared on the section 2 benchmark against joint angles and a
   fusion: do learned video features improve coaching decisions? A small, reportable result.
3. **GMR humanoid retargeting** (MIT licence): "learn a dance with Brodie, capture the corrected motion, retarget it to a
   simulated humanoid, watch the robot attempt it." The credible robotics extension; a demo, not a promise about data markets.
4. **Concept bottleneck** (DANCE, NeurIPS 2025 as inspiration): trajectories → named concepts (late extension, insufficient leg
   movement, inconsistent timing) feeding the cue and the Break Room.

## 10. Measurement

- Keep `telemetry.py` and `GET /metrics` on for every call; the eval harness runs alone on the laptop, not beside the worker.
- Pilot numbers to collect in the posting week: first-timer completion, time to first dance, "would use again", cue relevance
  (human review of `data/tmp/cue_review.csv`), share-funnel conversion (`/dance/funnel`).
- The section 2 benchmark table is the technical claim; everything else is experience.
