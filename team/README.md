# Team workspace (IvyHacks Brodie x Chert)

Everything **we** add lives in `team/`. The rest of the repo is the Brodie starter; we leave it alone where we can, so pulling
their updates (`git fetch upstream && git merge upstream/main`) stays painless.

**Deadline: Sun Oct 11, 6 PM ET.** Submission = a GitHub issue on seraphimlabs-dev/brodie-starter with this public fork, a link to our
first post, and 1-2 sentences on what our skill does. Checklist: [docs/demo-and-posting.md](docs/demo-and-posting.md).

## Layout

```
team/
  README.md              you are here
  docs/                  PRIVATE: git-ignored (our fork is public). Share with teammates outside git.
    ideas.md             shortlist + scoring (pick one, write it in decisions.md)
    decisions.md         what we decided and why, newest first
    platform-notes.md    what Brodie / Chert / LiveKit can and can't do (read before designing)
    research.md          competitors and sources
    demo-and-posting.md  submission checklist, demo shot list, posting plan
  skill/                 our Brodie skill (the worker loads these via flags; starter/ stays untouched)
    persona.md           who Brodie is in our product (Gemini system prompt)
    plan.txt             the workout plan (name:count,...)
    handlers.py          on_event(event, say_now, ask_voice) + forwards every event to our backend
  backend/               our server: FastAPI + SQLite (user state, scoring, anything after the call)
  web/                   our web pages (leaderboard, join links, ...)
  scripts/               run helpers for Windows (.ps1) and Mac/Linux (.sh)
  tests/                 replay an events file through our handlers, no call needed
  recordings/            our own test videos (git-ignored: *.mp4 never gets committed)
```

## How the pieces talk

```
phone (phone page or FaceTime via Chert) -> LiveKit room -> Brodie worker (python -m facetime.worker)
                                                              | loads team/skill/handlers.py
                                                              v
                                                   handlers.forward(event) --HTTP--> team/backend (port 8787)
```

The handler sends events from a background thread, so a slow or offline backend never stalls the call.

## Setup (once)

Follow the root `README.md` steps 1-4 (venv, requirements, pose models, data bundle, `.env.local`). Then:

```
uv pip install -r team/backend/requirements.txt
```

Windows notes: see "Windows" in [docs/platform-notes.md](docs/platform-notes.md). Short version: use the `.ps1` scripts, install
ffmpeg (`winget install Gyan.FFmpeg`), and never rely on `AGENT_NAME=` being empty in PowerShell (the scripts pass `--agent-name=`).

## Run (three terminals)

| | Windows (PowerShell, from the repo root) | Mac / Linux |
|---|---|---|
| A. backend | `.\team\scripts\backend.ps1` | `team/scripts/backend.sh` |
| B. coach | `.\team\scripts\worker.ps1` | `team/scripts/worker.sh` |
| C. phone link | `.\team\scripts\link.ps1` | `team/scripts/link.sh` |

Start A and B first, then C prints a fresh phone link (new room every time). Open the THIRD line on the phone.

On our Chert slot: run B with our dispatch name (`.\team\scripts\worker.ps1 -AgentName <name>` / `team/scripts/worker.sh <name>`),
then `chert dev --agent <name>` (root `docs/STARTER_README.md` section 5).

## Develop without a call

```
python -m facetime.replay_local --video data/tmp/mac_handoff/crops/run3.mp4 --plan "squats:10,pushups:8" --fast --events data/tmp/run3.events.jsonl
python team/tests/replay_handlers.py data/tmp/run3.events.jsonl
python team/tests/replay_handlers.py team/tests/fixtures/sample_events.json     # tiny built-in sample
```

## Env vars we add

| var | default | meaning |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8787` | where handlers send events; `off` disables forwarding |
| `TEAM_DB` | `team/backend/team.db` | SQLite file for the backend |

## Who owns what

| area | owner |
|---|---|
| Brodie side (skill, worker changes) | _tbd_ |
| backend + integrations | _tbd_ |
| web + share cards | _tbd_ |
| demo, Chert slot, posting | _tbd_ |
