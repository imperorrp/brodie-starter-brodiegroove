"""Gemini Live function tools for the dance session: the model calls real analysis and control code instead of pretending.

Every tool returns at once with a short JSON string: it sets session state (the phrase loop acts on its next tick) or reads
analysis the session already computed, so a tool call never stalls the conversation or the music. Each call is timed and logged
(telemetry 'tool' rows); a failure returns {"ok": false, "error": ...} instead of raising. The regex command parser in live.py
stays on as a fallback (DANCE_TOOLS=0 turns the tools off); the session dedupes a command heard both ways.

  build_tools(session) -> [FunctionTool, ...]      register with: await agent.update_tools(tools)
"""
import asyncio, json, time

KINDS = ('learn', 'battle', 'daily', 'squad_battle', 'squad_learn', 'telephone', 'class', 'class_record')
KIND_ALIASES = dict(ghost_battle='battle', duel='battle', daily_challenge='daily', squad='squad_battle', dance_telephone='telephone', group_class='class',
                    lesson='learn', routine='learn', record_class='class_record')
ACTIONS = ('again', 'slower', 'faster', 'next', 'pause', 'resume', 'stop', 'simplify', 'explain', 'full', 'harder', 'easier')


def _ok(**f):
    return json.dumps(dict(ok=True, **f), default=str)


def _err(e):
    return json.dumps(dict(ok=False, error=str(e)[:200]))


def _run(session, name, fn, /, **args):
    """Call fn() with timing and error capture; supports plain and async results."""
    t = time.perf_counter(); ok = True; err = None
    try:
        out = fn()
        return out
    except Exception as e:
        ok = False; err = repr(e); return _err(e)
    finally:
        tm = getattr(session, 'telemetry', None)
        if tm is not None: tm.tool_result(name, (time.perf_counter() - t) * 1000.0, ok, err, **{k: v for k, v in args.items() if isinstance(v, (str, int, float, bool))})


async def _arun(session, name, coro_fn, /, **args):
    t = time.perf_counter(); ok = True; err = None
    try:
        return await asyncio.wait_for(coro_fn(), 4.0)
    except Exception as e:
        ok = False; err = repr(e); return _err(e)
    finally:
        tm = getattr(session, 'telemetry', None)
        if tm is not None: tm.tool_result(name, (time.perf_counter() - t) * 1000.0, ok, err, **{k: v for k, v in args.items() if isinstance(v, (str, int, float, bool))})


def build_tools(session):
    from livekit.agents import function_tool

    @function_tool
    async def propose_activity(kind: str = 'learn', style: str = '', difficulty: str = '') -> str:
        """Make an OFFER of one routine: call it when the caller clearly asks to dance, learn, battle or do the daily challenge, or
        when you suggest a specific routine yourself (not for questions or "maybe"). It never starts anything: the system then waits
        for the caller's clear yes, and only after that runs the camera check and a ready check. Calling it again for the same
        routine changes nothing. kind: learn, battle (3-round ghost battle), daily (today's daily challenge), squad_battle (friends in
        one camera, each scored), squad_learn (a lesson for the whole squad), telephone (dance telephone), class (Brodie leads a class,
        everyone follows), class_record (a teacher records a class for students). style: hip-hop, pop, k-pop, house, fitness (dance
        cardio), waack, krump, jazz, lock. difficulty: beginner, intermediate, advanced."""
        def go():
            k = (kind or 'learn').strip().lower().replace(' ', '_').replace('-', '_')
            k = KIND_ALIASES.get(k, k)
            if k not in KINDS: return _err(f'unknown kind {kind!r}; use one of {", ".join(KINDS)}')
            return json.dumps(session.request_activity(mode=k, style=style or None, difficulty=difficulty or None, source='tool'))
        return _run(session, 'propose_activity', go, kind=kind, style=style, difficulty=difficulty)

    @function_tool
    async def confirm_start() -> str:
        """Check the ready check. Dancing starts only on the caller's own clear yes or raised hands, heard by the system; this reports
        whether that has happened and what it is waiting for. Never tell the caller it has started unless this says started."""
        return _run(session, 'confirm_start', lambda: _ok(**session.tool_confirm()))

    @function_tool
    async def control_session(action: str) -> str:
        """Change a routine that is already running (or the results between moves): repeat, slow down, speed up, skip, pause,
        resume, stop, simplify, explain, the full routine, harder or easier. It cannot start a routine and does nothing during the
        camera check or ready check (except stop, which goes back to chat). The result says what will really happen; say only that.
        action: one of again, slower, faster, next, pause, resume, stop, simplify, explain, full, harder, easier."""
        def go():
            a = (action or '').strip().lower()
            if a not in ACTIONS: return _err(f'unknown action {a!r}; use one of {", ".join(ACTIONS)}')
            return _ok(**session.tool_command(a))
        return _run(session, 'control_session', go, action=action)

    @function_tool
    async def caller_answered(answer: str) -> str:
        """Call this when the caller answers a yes/no or ready question that you or the system just asked them (want to start? do
        you have space? ready? got it?), with what they meant: yes, no or ready. You hear them better than the transcript does, so
        this is how a quiet or unclear "yeah" still counts. Never call it for anything else, and never guess: only a clear answer."""
        return _run(session, 'caller_answered', lambda: _ok(**session.tool_answer(answer)), answer=answer)

    @function_tool
    async def set_preferences(personality: str = '', difficulty: str = '', style: str = '') -> str:
        """Set the caller's preferences. personality: hype, chill or coach (strict). difficulty: beginner, intermediate or advanced.
        style: hip-hop, pop, house, k-pop, waack, krump, jazz, lock or fitness (dance cardio). Leave a field empty to keep it."""
        return _run(session, 'set_preferences', lambda: _ok(**session.set_preferences(personality, difficulty, style)), personality=personality,
                    difficulty=difficulty, style=style)

    @function_tool
    async def check_camera() -> str:
        """What the camera sees right now: which body parts are visible, a framing hint, how many dancers, and the tracking rate.
        Before a routine has been accepted this is only a peek: the real camera check comes after the caller says yes."""
        return _run(session, 'check_camera', lambda: _ok(**session.camera_status()))

    @function_tool
    async def analyze_last_attempt() -> str:
        """The analysis of the caller's last attempt: score, grade, title, shape/timing/energy, and the one correction with its body
        part, direction and counts. Use it when the caller asks how they did or what to fix."""
        return _run(session, 'analyze_last_attempt', lambda: _ok(**session.last_attempt_summary()))

    @function_tool
    async def compare_with(code: str) -> str:
        """Look up a four-digit challenge code: who set it, their score and routine. Leads to camera setup for a battle (never straight to music)."""
        return await _arun(session, 'compare_with', lambda: session.tool_compare(code), code=code)

    @function_tool
    async def generate_routine(style: str, difficulty: str = 'beginner') -> str:
        """Create a new routine for the caller from the move library, by style and difficulty, and queue it as the next routine."""
        return _run(session, 'generate_routine', lambda: _ok(**session.queue_routine(style, difficulty)), style=style, difficulty=difficulty)

    @function_tool
    async def get_profile() -> str:
        """The caller's progress: level, XP, next level, streak, achievements and what is unlocked."""
        return _run(session, 'get_profile', lambda: _ok(**session.profile_summary()))

    @function_tool
    async def forget_me(confirm: bool = False) -> str:
        """Delete the caller's saved scores and progress. Only call with confirm=true after the caller clearly asked twice."""
        return await _arun(session, 'forget_me', lambda: session.tool_forget(bool(confirm)), confirm=bool(confirm))

    @function_tool
    async def get_squad_status() -> str:
        """In a squad (several dancers in one camera): each dancer's colour or name, scores, the MVP so far and crew sync."""
        return _run(session, 'get_squad_status', lambda: _ok(**session.squad_status()))

    @function_tool
    async def set_call_mode(mode: str) -> str:
        """Switch between solo and squad play. mode: solo, squad_battle or squad_learn."""
        return _run(session, 'set_call_mode', lambda: _ok(**session.set_call_mode(mode)), mode=mode)

    @function_tool
    async def assign_name(colour: str, name: str) -> str:
        """In a squad, give the dancer wearing a colour tag (volt, pink, cyan, amber) their first name."""
        return _run(session, 'assign_name', lambda: _ok(**session.assign_name(colour, name)), colour=colour, name=name)

    return [propose_activity, confirm_start, control_session, caller_answered, set_preferences, check_camera, analyze_last_attempt, compare_with, generate_routine, get_profile, forget_me,
            get_squad_status, set_call_mode, assign_name]


async def register(session, agent):
    """Give the running Gemini agent the tools. Returns the number registered (0 when unavailable)."""
    tools = build_tools(session)
    r = agent.update_tools(tools)
    if asyncio.iscoroutine(r): await r
    return len(tools)
