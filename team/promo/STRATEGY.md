# BrodieGroove: 8-day launch kit

A teaser the day before launch (Sun 10/11), then 7 days of posts once the FaceTime line opens (Mon 10/12 to Sun 10/18).
Every day has a 4:5 poster, a 9:16 story/cover, a 16:9 landscape, a 9:16 reel, and captions per platform.

## Quick start

```bash
python -m team.promo.render            # all posters, stories, end cards  -> data/tmp/promo/<day>/*.png + sheet.jpg
python -m team.promo.clips             # all reels (~3.5 min)             -> data/tmp/promo/<day>/<day>_reel.mp4
python -m team.promo.render d3         # one day;  --clean = no CTA band;  --guides = show 9:16 UI safe zones
python -m team.promo.art hype=1        # re-roll one Brodie pose (Gemini, cached; checked against canon cream)
```

**When the line is ready:**
1. In `config.json`, set `call_line` (what people read, e.g. the number or address) and `call_url` (what the QR opens).
2. Re-run `render` and `clips`.

Every CTA band then switches from "THE LINE OPENS MON 10.12" to "FACETIME BRODIE · <line>" with a scannable QR. The reel end cards and `captions_filled.md` update too. Prefer to add the CTA yourself in Canva/CapCut? Use `--clean`.

**Before Day 7:** put the real numbers in `config.json` under `recap`. The poster shows "—" until you do.
**Edit any copy:** it's plain HTML in `days/`. Open the file in a browser after one `render` run (that writes `config.js` and `fonts.css`).

## Who we're talking to

| Segment | What hooks them | Where |
|---|---|---|
| "Two left feet" Gen Z, 16-26 (core) | self-deprecating humour, practising privately ("no one's watching but Brodie"), the glow-up arc | TikTok, Reels |
| Choreo learners / K-pop fans | the ghost one beat ahead beats pausing a mirrored tutorial | TikTok, Reels |
| Rhythm-game kids (Just Dance, DDR, osu!) | PERFECT/COMBO, S-rank, titles worn as identity | TikTok, X |
| Friend groups | Squad Battle, colour tags, "tag the friend who…", challenge codes | IG Stories, TikTok |
| Dance-cardio people | "a rhythm game, not a class", short sessions | Reels, IG feed |
| Builders / judges / recruiters | real scoring (MediaPipe, 14 joint angles, DTW), Gemini Live, FaceTime as the UI | X, LinkedIn |

**Levers:**
- An incoming FaceTime call is a scroll-stopper.
- Grade titles are built to be shared ("Combo Menace", "Certified Off-Beat (for now)").
- The scorecard *is* the share button: code + QR + #BeatMyBrodie.
- Humour over polish.

## Look and voice (from the landing page and the Brodie kit)

- **Palette:** ink `#0A0A0B` background, volt `#D9FF3D` main accent, pink `#FF3DA8`, cyan `#3DE0FF`, amber `#FFB13D`, red `#FF4D5E`, paper `#F4F4F0` text.
- **Type:** Archivo Black (display, uppercase), Space Grotesk (labels), Plus Jakarta Sans (body).
- **Effects:** grain, blurred blobs, volt grid floor, tilted ✦ marquees, outlined → filled headline letters, die-cut sticker Brodie.
- **Brodie rules:**
  - Body is exactly cream `#EFE7DA`, never pink. `art.py` runs `check_cream.py` on every pose.
  - Flat 2D with a thick outline.
  - Readable BRODIE tank.
  - One hat/band and one prop per hand.
- **9:16 safe area:** keep content out of the top 11.5%, the bottom 22% and the right 13% (TikTok/Reels UI). `--guides` shows it.
- **Voice:** short, punchy, light slang, never cringe. Canon lines: "Lock in.", "Chef's kiss.", "Hug the beat.", "Screenshot this."

## Calendar

| Day | Date | Theme | Poster | Reel |
|---|---|---|---|---|
| 0 | Sun 10/11 | Teaser: "Brodie is calling… tomorrow." (the IvyHacks first post: before 6 PM ET) | incoming-call screen | call → "Missed call. He'll call back MON 10.12." |
| 1 | Mon 10/12 | Launch: the line is open | BRODIE GROOVE hero + CTA | 46 s pitch video, new end card |
| 2 | Tue 10/13 | How it works | 5-slide carousel on real HUD shots | 55 s Maya glow-up film |
| 3 | Wed 10/14 | Real scores, not vibes (builder day) | 60/30/10 score bar + pose stats | battle capture, scored per count |
| 4 | Thu 10/15 | Pick your Brodie | HYPE / CHILL / COACH cards | same miss, three Brodie voices |
| 5 | Fri 10/16 | Squad up, get tagged | 4 colour-tagged dancers, Crew Sync | squad capture + results |
| 6 | Sat 10/17 | Which grade are you? #BeatMyBrodie weekend | S-D tier list + fill-in template | grade-reveal montage |
| 7 | Sun 10/18 | Week one on the line | stats + scorecard wall | best-of cut |

**Every day:** a "Today's 8-count" story (`daily/daily_9x16_<day>.png`). The style per day is in `config.json` under `daily_style`.

**Posting times (local; adjust after Day 1 analytics):**
- TikTok/Reels: 6-9 PM.
- IG feed: 11 AM-1 PM.
- X: 12-3 PM.
- LinkedIn: 8-10 AM, on Tue-Thu plus Days 1 and 7.

Copy for every post is in `CAPTIONS.md`. Post from `data/tmp/promo/captions_filled.md`.

## Files

- `config.json`: the CTA line/URL, handle, hashtag, daily styles, recap numbers.
- `brand.css`, `promo.js`: the shared look and the runtime (sizes, slides, CTA slot).
- `days/*.html`: one poster per day, plus `daily.html`, `endcard.html` and `reelframe.html` (transparent reel titles).
- `render.py`: headless Edge/Chrome screenshots, QR (segno), captions fill-in, contact sheet.
- `clips.py`: ffmpeg reels from the pitch video, the film, the eval call captures and HUD stills. Loudness -14 LUFS.
- `art.py`: new Brodie poses via `team/video/film/gen.py` (Gemini, cached), cut out and cream-checked.

Generated files are written under `data/tmp/promo/` and aren't tracked in git: the art in `art/`, the voice lines in `vo/`, and all the posters and reels.
