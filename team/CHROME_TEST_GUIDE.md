# Testing Brodie dance from Chrome (laptop or Android), no iPhone, no Chert

There is no "Chert room" until a FaceTime line is attached from a Mac. Brodie lives in a LiveKit room, and any browser can be
the caller in that room. Our page `team/web/phone.html` sends your camera and mic and shows Brodie's video (ghost, timeline,
score, end card) full size, with his audio. Three things have to be up: the dance worker (Brodie), a small https server for the
page, and a link with a room token. One script does all of it.

## The one command

Open PowerShell in the repo root (`ivyhacks-brodie`).

```
.\team\scripts\chrome.ps1                 # laptop: front camera, opens Chrome on the page
.\team\scripts\chrome.ps1 -Android        # phone on the same Wi-Fi: prints a short link to type on the phone
```

What it does, in order:
1. Starts the https server if it is not running (a minimised window).
2. Starts the dance worker if it is not running (a new window; wait for `registered worker` there, about 10 s). A worker that
   is already running keeps its key and mode; add `-Restart -Key <key> -Mode <mode>` to change them.
3. Mints a one-hour token for a fresh room, writes the link, prints the short link `https://<laptop-ip>:8443/go`, and on the
   laptop opens Chrome on the page.

Defaults: key `aist_wa_110_ch01` (AIST++ waack basic move, arm-led, good for a desk webcam), mode `learn`, visibility gate 0.4
(desk webcam sees the upper body only). Flags: `-Key`, `-Mode learn|battle|telephone|class`, `-Code 1234` (join a duel, chain or
class), `-Facing user|environment`, `-MinVisible 0.6` (whole-body phone test), `-Rtt 0.3`, `-NoOpen` (print only).

## Demo mode and the dispatch name (read this if the worker is on the Chert line)

While the worker runs under the Chert dispatch name (`dance_loop.ps1 -AgentName ...`), a Chrome room only reaches it when the
link carries that dispatch. `chrome.ps1` and `link_local.ps1` do this automatically through `team/scripts/mint_link.py` (it
reads `AGENT_NAME` from `.env.local`); the old `facetime/mint_token.py` links will not.

`.\team\scripts\chrome.ps1 -Demo` names the room `demo-desk-...`: Brodie skips the chat, the offer and the ready check and
goes straight from the camera check into one routine (good for filming), and the camera check is arms-only on the laptop.
Without `-Demo` the full flow runs: Brodie chats, suggests a routine, waits for your yes, checks the camera (head to hips),
asks about space, then waits for "ready" or both hands up. In both cases the teaching ladder waits for you to say **"ready"**
(or "again") after each demonstration.

## In Chrome

1. The first time, Chrome shows "Your connection is not private" (our certificate is self-signed): click **Advanced**, then
   **Proceed to ... (unsafe)**. Once per device.
2. Allow the camera and the microphone when asked. Tap **Join**.
3. The top bar should read `connected`, `coach video: yes`, `audio: playing`. If audio says "tap to enable", tap the button.
4. Step back until your head and hands are in frame (laptop) or prop the phone at knee height two to three metres away with the
   whole body in frame (Android). Volume up.

## What you should see and hear

- The picture fills the screen in your camera's own orientation (landscape on a laptop, portrait on a phone). Your skeleton
  is drawn on you in yellow-green so you can see what the tracker sees. Four chips at the bottom (head, arms, torso, legs)
  light up for the body parts in view, with a hint such as "Upper body only" if your legs are out of frame.
- "Hi, I'm Brodie", then he asks your name and what you'd like: **learn a move, a battle, dance telephone, or a class**. Answer
  in plain words ("I'm Maya, let's learn something"). He then asks the **style**: hip-hop, house, pop, lock, waack, krump,
  jazz, K-pop, or "surprise me". He confirms the move and you say **"ready"**. If you say nothing for about 50 s he starts
  with the preset move. You can also ask him questions; he answers in a sentence or two.
- The metronome starts. **WATCH**: the reference dancer appears in the panel on the right (whole body, always), and the same
  ghost is overlaid on your body when enough of you is in view (upper body only when your legs are out of frame).
- Four high clicks count you in with a big countdown. **GO**: dance the move. The timeline at the bottom turns green, amber or
  red per count; a running score and a combo counter appear top right.
- The score card: shape, timing, energy, the one cue. Gemini says the score and the cue in two sentences.
- Learn mode loops: under 50 it slows the tempo 15 percent, 80 or three tries moves to the next phrase; in between he asks
  "again, or the next one?". Say "again", "slower", "faster", "next" or "stop" between phrases.
- At a desk, only your arms are scored ("upper body scored" under the score). For legs, prop the phone or stand back.
- The display is a mirror: copy the ghost as you would copy a mirror, you end up using the same arm as the reference.

Modes to try with `-Restart -Mode ...`: `battle` (two phrases plus a freestyle round, a four-digit duel code on the end card),
`telephone` (record your own eight counts, get a chain code; with `-Code` imitate the previous dancer), `class` (no code: you are
the teacher, demonstrate two phrases to a count-in and get a class code; `-Code <class>`: you are a student; the teacher's name or
saying "report" opens the class report). The heatmap page for a class needs the backend server: `.\team\scripts\backend.ps1`
then `http://127.0.0.1:8787/dance/class/<code>/page` (the worker writes to the same SQLite file, so codes carry over).

## Where things land

| What | Where |
|---|---|
| What the caller saw (video) | `data\tmp\dance_chrome.mp4`, written when the call ends |
| Every scored phrase | the worker window, lines starting with `[dance] phrase` |
| The session log (asks, cues, results) | `data\tmp\dance_session_log.json` |
| Codes, results, chains, classes | `team\backend\team.db`; pages via `backend.ps1` at `http://127.0.0.1:8787/dance` |

## If something is off

| Symptom | Fix |
|---|---|
| Phone cannot open the short link | Same Wi-Fi as the laptop? Windows Firewall may block inbound 8443: in an **admin** PowerShell run `New-NetFirewallRule -DisplayName "Brodie dance https 8443" -Direction Inbound -Protocol TCP -LocalPort 8443 -Action Allow`. Or use the "Fallback" Meet link the script prints (no local server). |
| `coach video: no` after joining | The worker is not running or not yet registered: look for `registered worker` in its window, then rejoin (`.\team\scripts\chrome.ps1` again for a fresh room). |
| No camera prompt | Chrome blocked it earlier: `chrome://settings/content/camera`, remove the block for the laptop's address. |
| Two different voices | Fixed on 2026-10-11: the Windows voice lines are off by default (`DANCE_LOCAL_VOICE=1` brings them back). Everything spoken is Gemini live or Brodie's own recorded lines from the cue bank. |
| Music plays but Brodie never speaks | Gemini is down or the relay token expired: check `GEMINI_API_KEY` and `GEMINI_BASE_URL` in `.env.local`; the metronome, ghost and scoring work without him. |
| "No score" | Not enough joints in view: at least shoulders, elbows and wrists must be inside the picture for the arms to score; step back or raise the laptop. |
| The skeleton on you jumps around | Another person or a poster in the frame; plain background, good light, you centred. Pose runs at 384 px (`DANCE_POSE_PX`), `-Restart` with `DANCE_POSE=heavy` is slower but steadier. |
| Ghost feels early or late | Clap once on camera, count the delay until the overlay reacts, restart with `-Rtt <seconds>` (default 0.3). |
| Chrome says the page is not secure and refuses the camera | You opened `http://`; it must be `https://` (the script prints https). |
| Everything is slow or audio stutters | Close other heavy apps; pose runs at 12 fps on the CPU. `-Restart` also helps after a long run. |

## Stop or restart

Close the worker window and the server window, or `.\team\scripts\chrome.ps1 -Restart ...` to replace the worker. The link and
token expire after an hour; run the script again for a fresh room.

## The real FaceTime line, from this Windows laptop

The Chert CLI supports macOS and Linux only, so it runs here through a small preload (`team/scripts/chert.ps1`). Two terminals,
inside a booked slot (the line is shared, one team at a time):

```
.\team\scripts\dance_loop.ps1 -AgentName dance-battle-agent-ivyhacks    # Terminal A: the worker under the dispatch name, auto-restarted
.\team\scripts\chert.ps1 doctor                                          # read-only: session, project, line, LiveKit
.\team\scripts\chert.ps1 dev --agent dance-battle-agent-ivyhacks         # Terminal B: claims the line; wait for "Connected to Chert"
```
Do not run `chert doctor --agent ...` while `chert dev` is up: the test room it creates and deletes can crash the worker's
LiveKit library on Windows (the loop restarts it, but a call in that moment would be declined). Run the agent test before
`chert dev`, or not at all once the warm room shows up in Terminal A.
Then FaceTime video-call the line's address (on the assignment card) from a phone; Terminal A prints `bound to 'chert-bridge-...' 1280x720` when the call
lands. Hang up, Ctrl-C Terminal B and wait for "Stopped" to release the line. Signed in already (`chert whoami`); to sign in
again: `.\team\scripts\chert.ps1 login --device` and enter the code at https://facetime.trychert.com/cli. With the worker on
the dispatch name, the Chrome page's automatic dispatch does not pick it up: `.\team\scripts\chrome.ps1 -Restart` swaps back.

## The same path with nobody on camera

```
.venv\Scripts\python team\scripts\chrome_test.py
```
Headless Chrome joins through the same page with a dance clip as its camera and reports PASS once the worker scores a phrase on
the browser's frames (it did on 2026-10-11). Use it to confirm the plumbing before a person tries.
