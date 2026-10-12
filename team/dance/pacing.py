"""Teaching pace: the gaps between moves (3-4 s, so a beginner can reset, breathe and see what's next). Every value can be
overridden with an environment variable of the same name prefixed DANCE_ (e.g. DANCE_MOVE_GAP_S=4).

  POSE_HOLD_S          each count of a pose-by-pose demo is held this long (and never moves on while its line is still playing,
                       up to POSE_HOLD_S + POSE_SPEECH_EXTRA_S)
  POSE_EASE_S          the ghost glides into each pose over this long (inside the hold)
  FREEZE_MATCH_REST_S  after a matched pose in freeze-frame, a short rest before the next pose
  DEMO_TO_ASK_S        after a demo ends, before "Got it?"
  TURN_GAP_S           between "with me" and "your turn": back to the start pose
  MOVE_GAP_S           between moves (a half, a step, the combination): the "NEXT MOVE" break
  ROUND_GAP_S          between battle / squad rounds
  LESSON_SOFT_CAP_S    a lesson this long offers "full routine now, or keep practising?"
"""
import os


def _env(name, default):
    try: return float(os.environ.get('DANCE_' + name, default) or default)
    except ValueError: return float(default)


POSE_HOLD_S = _env('POSE_HOLD_S', 3.5)
POSE_SPEECH_EXTRA_S = _env('POSE_SPEECH_EXTRA_S', 2.0)
POSE_EASE_S = _env('POSE_EASE_S', 0.8)
FREEZE_MATCH_REST_S = _env('FREEZE_MATCH_REST_S', 0.8)
DEMO_TO_ASK_S = _env('DEMO_TO_ASK_S', 1.0)
TURN_GAP_S = _env('TURN_GAP_S', 3.0)
MOVE_GAP_S = _env('MOVE_GAP_S', 3.5)
ROUND_GAP_S = _env('ROUND_GAP_S', 4.0)
LESSON_SOFT_CAP_S = _env('LESSON_SOFT_CAP_S', 720.0)
