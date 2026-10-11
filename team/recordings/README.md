Our own test videos go here (never committed). Convert phone clips to 10 fps first, or the replay runs in slow motion:

```
ffmpeg -i IMG_1234.mov -vf "fps=10,scale='if(gt(iw,ih),-2,288)':'if(gt(iw,ih),288,-2)'" -an -c:v libx264 -preset fast -crf 23 team/recordings/mine.mp4
python -m facetime.replay_local --video team/recordings/mine.mp4 --plan "squats:10" --fast --events team/recordings/mine.events.jsonl
python team/tests/replay_handlers.py team/recordings/mine.events.jsonl
```

Film in portrait, phone propped at knee height 2-3 m away, whole body in frame, stand still a second before the set.
