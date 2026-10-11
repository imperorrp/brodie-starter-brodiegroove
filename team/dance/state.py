"""Shared state between every loaded copy of live.py.

The worker loads the handlers file by path for EVERY job (load_handlers -> a fresh module each time), while the consumer that
runs the session was registered by the first copy. Keeping the session pointer here, in a normally imported module that
sys.modules caches once, means the newest handler copy's on_event still finds the running session (bug seen 2026-10-11: the
second call's spoken commands never reached the lobby)."""
STATE = dict(say_now=None, ask_voice=None, session=None, registered=False)
