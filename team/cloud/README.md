# The cloud box: Brodie worker and backend

One container runs the backend API and the Brodie worker (`start.sh`), the database is PostgreSQL, and the
landing page stays on Vercel. Browser calls need no laptop process; FaceTime calls also require the official Chert CLI
running on an authorized Mac or Linux host for the agreed test duration. Native Windows is unsupported. No Docker: Railway builds the
repo with Railpack from `railpack.json`; `railway.json` records the deploy settings. `nixpacks.toml` remains the legacy recipe; a plain Linux VM runs the same script
under systemd (last section). The temporary October 14 deployment uses Railway Postgres; the Supabase instructions below are an alternative.

```
caller (FaceTime via Chert, or /join in a browser) ─► LiveKit Cloud ─► worker ──► API (same box) ─► Supabase Postgres
                                                                        │           ▲
                                                                        └ Gemini    └ board pages, /join, /metrics (public https)
```

## What you need (keys and accounts)

| What | Where it comes from | Used by |
|---|---|---|
| `DATABASE_URL` | Supabase > your project > **Connect** > **Transaction pooler** URI (port 6543), password filled in. Add `?sslmode=require` | the API (`db.py`) |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | the LiveKit Cloud project you already use | worker (joins rooms), API (`/join` mints tokens) |
| `GEMINI_API_KEY`, `GEMINI_BASE_URL` | the team token and relay from the hackathon card (240 minutes per team; a Gemini key of your own with `GEMINI_BASE_URL` empty afterwards) | worker |
| `AGENT_NAME` | the Chert dispatch name (`dance-battle-agent-ivyhacks`); the API mints `/join` tokens for the same name | worker, API |
| `DANCE_LIBRARY_URL` | a signed URL to the zip from `pack_library.py` (see below) | worker boot |
| `DANCE_CALL_HANDLE` | the line's FaceTime address, for the scorecard QR | worker |
| `JOIN_KEY` (optional) | any string; then `/join?k=<it>` is required, so strangers cannot burn Gemini minutes | API |
| a Railway account | GitHub sign-in; Hobby plan ($5/month, usage-billed, no sleeping) | hosting |

Supabase gives you one more thing worth keeping: the dashboard **Table editor** shows every code, result and profile live.

## Set up, once

1. **Supabase.** New project (any region near the callers). Copy the Transaction pooler URI. Check it from the laptop:
   ```
   set DATABASE_URL=postgresql://postgres.<ref>:<password>@<host>.pooler.supabase.com:6543/postgres?sslmode=require
   .venv\Scripts\python team\backend\dbcheck.py              # creates the tables, writes and deletes a test row
   .venv\Scripts\python team\backend\import_sqlite.py        # optional: carry over team.db so old codes keep working
   ```
2. **The AIST keys.** They are research-licensed and never in git, and every generated routine needs them:
   ```
   .venv\Scripts\python team\cloud\pack_library.py           # -> data/tmp/library.zip (24 aist_* keys + class keys)
   ```
   Supabase > **Storage** > new **private** bucket `library` > upload the zip > `...` > **Get URL** > signed, expiry one year.
   That URL is `DANCE_LIBRARY_URL`. (Any private URL works; never a public one.)
3. **Railway.** New project > **Deploy from GitHub repo** > this repo, branch `main`. It builds with `railpack.json` (about five minutes the first time: CPU torch, mediapipe, the pose models). Then:
   - **Variables**: paste everything in the table above (the `.env.local` lines plus `DATABASE_URL`, `DANCE_LIBRARY_URL`,
     `DANCE_CALL_HANDLE`, `JOIN_KEY`).
   - **Settings > Networking > Generate domain** (port 8787 or whatever `PORT` Railway shows). `start.sh` turns that domain
     into `DANCE_PUBLIC_URL` on its own, so the scorecard QR and share links point at the box.
   - Deploy logs should end with `[backend] database: postgres` and the worker's `registered worker` line from LiveKit.
4. **Check.** `https://<domain>/health` → `{"ok": true, "db": "postgres", "db_ok": true}`. `https://<domain>/join?prefix=demo-desk`
   opens the browser caller with a fresh room that the cloud worker answers (this is the link to put on the site).
   `https://<domain>/dance` is the live board.
5. **The Chert line.** Verify the authenticated Chert project and its assigned line with `chert whoami` and `chert doctor`
   (without `--agent`). On an authorized Mac or Linux host, run `chert dev --agent <AGENT_NAME> --env-file <protected-env-file>`
   only for the agreed test duration. It must use the Railway worker's LiveKit project and dispatch name, and remain running
   throughout testing. Ctrl+C releases the line. Railway does not start this connection. Do not use a shared line without
   assignment confirmation; a real FaceTime call needs the test operator's go-ahead.

The temporary service uses native GitHub autodeploy from this repository's `main` branch (enabled 2026-10-15).
Railway recognizes the GitHub App installation and has an enabled `main` deployment trigger. The root build
config has no watch-path filter, so all commits on the connected branch can deploy. Check the deployment's
commit and successful health check after each push; an enabled trigger alone does not prove a build succeeded.
See [Railway GitHub autodeploy requirements](https://docs.railway.com/deployments/github-autodeploys).

`DANCE_LIBRARY_URL` is configured as a Railway service variable using the team's private signed Storage URL.
The supplied ZIP contains 24 AIST keys and one class key. The deployed service loaded 29 total keys and passed generation across
all eight styles and three levels, and daily generation inside Railway (2026-10-15). The signed URL expires 2027-10-15; replace the variable
before expiry. Keep both the URL and the research-licensed ZIP out of Git. Boot logs should confirm keys loaded.

The compatibility fixes are versioned so GitHub builds retain the working runtime recipe.
For an emergency manual deployment, use `railway up --service brodie` from the linked checkout.
Rollback: Railway > Deployments > an earlier one > **Redeploy**. Logs: Railway > the service > **Logs**.
Deployments preserve the Postgres volume. To stop this temporary setup, disable any active deployment trigger first,
then stop the `brodie` and `Postgres` Railway deployments and gracefully stop the operator-Mac Chert connector.

For new services, confirm the platform settings explicitly: start `bash team/cloud/start.sh`, health path `/health`, timeout
600 seconds, restart `ALWAYS`, sleeping disabled. The October 14 temporary deployment required these settings on the service
because its legacy Nixpacks config did not supply a start command to Railpack. See [Railway build configuration](https://docs.railway.com/builds/build-configuration).

## How it stays up on its own

- Railway restarts the container when it exits (`restartPolicyType: ALWAYS`) and only switches traffic to a new deploy
  once `/health` returns HTTP 200. A failed database probe returns HTTP 503 and `ok: false`. This checks the API/database,
  not LiveKit registration, the dance library, Gemini, or either direction of audio/video. Railway's health check gates
  deployment; it is not continuous media monitoring.
- Inside the container `start.sh` runs the API and the worker in separate restart loops, so a worker crash never takes
  the board pages down and vice versa; a process that keeps dying within 30 s backs off up to a minute between tries.
- The worker is a LiveKit Agents worker: it reconnects to LiveKit Cloud by itself, and every call runs inside the session's
  error boundaries (`live.py`). Backend calls from the worker time out after 3 s and fall back to "no backend" for that step.
- The API keeps one Postgres connection, pings it after a minute idle and reopens it when the pooler drops it; no prepared
  statements, so the Supabase transaction pooler is fine. The API refuses to start on SQLite in the cloud
  (`REQUIRE_DATABASE_URL=1`), so a misconfigured box fails loudly instead of losing codes.
- Pose models are fetched at build and re-checked at boot; the key library is re-fetched at boot. Both are idempotent.
- Uvicorn access logging is disabled because `/join` and `/phone.html` query strings contain a join key or room token.

## Sizing and cost

The worker is CPU-only: solo pose at 320 px measured 50 ms a frame on a laptop, four poses at 384 px 237 ms. Railway bills
actual use (about $20 per vCPU-month and $10 per GB-month, pro-rated), and an idle worker uses almost nothing, so a quiet
week is a few dollars plus the $5 Hobby base; a busy day of calls is still under a coffee. Supabase free tier covers this
scale (500 MB database, 1 GB storage). If calls drop frames, set `DANCE_POSE=lite` or `DANCE_FPS=10` in the variables
before paying for more CPU.

## Files

| File | Role |
|---|---|
| `start.sh` | the process: env defaults, model and library fetch, the two restart loops |
| `requirements.txt` | the box's Python deps (CPU torch comes from `railpack.json`) |
| `fetch_models.py` | MediaPipe pose models into `models/` (build and boot) |
| `fetch_library.py`, `pack_library.py` | the private key zip: fetched at boot from `DANCE_LIBRARY_URL`, packed on the laptop |
| `../backend/dbcheck.py`, `../backend/import_sqlite.py` | prove `DATABASE_URL` works; carry the old SQLite rows over |
| `/railpack.json`, `/railway.json`, `/.python-version` | the Railway build and deploy config (repo root) |

## A VM instead of Railway

Any Ubuntu 24.04 box with 2 vCPU and 4 GB (Hetzner CX22, Lightsail, Droplet) runs the same script:

```
sudo apt install -y git python3.12-venv libgl1 libegl1 libgles2 libglib2.0-0 && curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/imperorrp/ivyhacks-brodiegroove.git brodie && cd brodie
uv venv --python 3.12 .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r team/cloud/requirements.txt
cp .env.example .env.local   # fill it in; add DATABASE_URL, DANCE_LIBRARY_URL, PORT=8787
```

`/etc/systemd/system/brodie.service`:

```
[Service]
WorkingDirectory=/home/ubuntu/brodie
EnvironmentFile=/home/ubuntu/brodie/.env.local
ExecStart=/bin/bash -c '. .venv/bin/activate && bash team/cloud/start.sh'
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
```

`sudo systemctl enable --now brodie`; put Caddy or the provider's load balancer in front of port 8787 for https (the
browser caller needs https for the camera). Update: `git pull && sudo systemctl restart brodie`.

## Troubleshooting

| Symptom | Cause, fix |
|---|---|
| MediaPipe fails on `libGL.so.1`, `libEGL.so.1`, or `libGLESv2.so.2` | check the `deploy.aptPackages` in `railpack.json` and that Railway uses Railpack; worker registration alone does not load the pose model |
| `[backend] DATABASE_URL is not set` | the variable is missing on the service, or `TEAM_DB` is set there (remove it) |
| `db_ok: false` on `/health` | wrong password or host in `DATABASE_URL`; use the pooler URI, not the direct one (direct is IPv6-only on the free tier) |
| `no library phrases for style` in the worker log | `DANCE_LIBRARY_URL` unset or the signed URL expired; the log line `[library] ... keys added` tells you |
| `/join` says LIVEKIT not set | the three LiveKit variables are missing on the service |
| the worker registers but never gets a call | `AGENT_NAME` differs between `chert dev --agent` and the service, or `/join` was minted by a backend with a different name |
| choppy ghost video | one vCPU is saturated: `DANCE_POSE=lite`, `DANCE_FPS=10`, or more CPU |
