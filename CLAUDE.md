# Instructions for Claude (and anyone else working in this repo)

Brodie Groove: a dance coach you FaceTime, built by team BrodieGroove on the Brodie starter for IvyHacks 2026.
Everything we wrote lives in `team/`; the rest is the organiser's starter and must stay untouched (we monkey-patch
from `team/dance/live.py` instead, so upstream merges stay trivial).

## Read `team/context/` first

The LLM context for this repo lives in `team/context/`. Read these before touching code, in this order:

1. `team/context/CODEBASE.md`: the map of every file, function, env var and hand-off.
2. `team/context/STATUS.md`: what exists and how far it is verified.
3. `team/context/TODO.md`: the ranked work.
4. `team/context/CONVERSATION.md`: the call flow, the prompt chain and every guard (read before changing `live.py`,
   `persona.md`, `tools.py`, `setup.py` or anything the caller hears).
5. `team/context/PLANNED_IMPROVEMENTS.md`: the longer plan behind the to-do list.

For running and testing: `team/dance/README.md`, `team/CHROME_TEST_GUIDE.md`. For the starter: `docs/STARTER_README.md`,
`docs/STARTER_EVENTS.md`, `docs/BUILD_ON_BRODIE.md`.

## Keep `team/context/` true, in the same commit as the code change

- **`CODEBASE.md`**: add, rename or remove a file → fix its entry and every mention. Change a state, a threshold, an env
  var, a backend route, a Gemini tool, a script flag, a key format → update sections 3, 4, 6 and 11. Notice a doc that
  disagrees with the code → add it to section 13 until fixed, then remove it. Keep line counts roughly right.
- **`STATUS.md`**: when something reaches a new verification level (unit → sim → fake room → headless Chrome → human
  Chrome → Chert line → a landed FaceTime call), when a weakness is fixed or found, when something ships. Bump "As of".
- **`TODO.md`**: tick what you finish, add what you discover, keep the ranking honest.
- **`CONVERSATION.md`**: any change to what Brodie says, when he waits, what counts as an answer, or a guard.
- **`PLANNED_IMPROVEMENTS.md`**: when the plan changes, not for every tick (that is `TODO.md`).
- If a change alters the architecture diagram in `CODEBASE.md` §0 or the call steps in §2, update those too.

A change is not done until `team/context/` says what the code now does. New context docs go in that folder, nowhere else.

## Where things go

- Skill code: `team/dance/`. Backend: `team/backend/`. Pages: `team/web/`. Run helpers: `team/scripts/` (a `.ps1` and a
  `.sh` twin). Content tooling: `team/promo/`, `team/video/`. LLM context: `team/context/`.
- Never commit: `.env.local`, `data/`, `models/`, `notes/`, `research/`, `team/docs/` (private notes), `team/PROGRESS*.md`,
  `team/.go_url`, `team/web/certs/`, `*.db`, recordings. AIST++-derived keys (`team/dance/library/aist_*.json`) are
  research-licensed and git-ignored; rebuild with `python -m team.dance.ingest_aist`. The QEVD files in `src/rtcoach/`
  and `data/models/` are licensed to the starter: keep private.
- Only kit art (`team/dance/assets/brodie/`) may be used for Brodie's face; run `tools/check_cream.py` on any new image.

## Git: one repo, pushed directly

- The repo is the public `imperorrp/ivyhacks-brodiegroove` (`origin`). Commit on `main` and `git push origin main`.
- The private repo (`private` remote) and `team/scripts/submit_to_fork.*` are **retired**. Never push to `private`
  (its push URL is set to DISABLED) and never run `submit_to_fork`; it would squash a stray commit on top of real history.
- Pull before you push (`git pull --rebase origin main`); teammates push here too.
- Commit messages: one plain line saying what changed and why, like the existing history.

## Verify before claiming done

```
.venv\Scripts\python -m pytest -q team\dance\tests            # 164 cases
.venv\Scripts\python -m team.dance.offline_test selftest      # 14 checks
.venv\Scripts\python -m team.dance.eval.run_eval --jobs 6     # 46 scenarios; nothing else running
.venv\Scripts\python team\scripts\chrome_test.py              # headless Chrome through the live worker
```
State which of these you ran and what they printed. Simulator and tests are not a real call; say so.

## Windows notes

- Use `.venv\Scripts\python` (uv-managed CPython 3.12.11; Anaconda's Python cannot load the LiveKit and torch wheels).
- Start the worker with `python -m team.dance.run_worker ...` (the scripts do), never `python -m facetime.worker` directly.
- Never run `chert doctor --agent` while `chert dev` is up. Use `team\scripts\dance_loop.ps1` for the line.
- Long heredocs and backslashes do not survive the Bash tool here; write files with the editor tools.
