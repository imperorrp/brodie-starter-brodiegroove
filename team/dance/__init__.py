"""Dance engine for the Brodie FaceTime agent (team skill).

Pure-Python package, independent of the licensed classifier: MediaPipe landmarks in, view-invariant joint-angle features,
reference keys built from videos or motion files, beat-anchored scoring with a banded DTW for timing, a phrase state machine,
and an OpenCV renderer for the video track the caller sees. Offline harness: `python -m team.dance.offline_test`.

Layout: features.py (landmarks -> channels), reference.py (video -> key), scoring.py (key vs attempt), engine.py (phrase state
machine), render.py (overlay), live.py (worker integration), library/ (keys as JSON). Game layer: choreo.py (generated routines),
feedback.py (directional cues), progression.py (grades, titles, XP, levels, achievements), gestures.py (held-pose controls),
multipose.py (several people, primary lock), squad.py (squad scoring), tools.py (Gemini function tools), telemetry.py (session logs),
text.py and fx.py (type and animations), eval/ (end-to-end scenarios, HUD snapshots).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
LIBRARY_DIR = os.environ.get('DANCE_LIBRARY', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'library'))
MODELS_DIR = os.path.join(ROOT, 'models')
