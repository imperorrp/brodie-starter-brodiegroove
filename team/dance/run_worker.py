"""Launch the Brodie worker the way Windows needs it: import the Gemini plugin on the MAIN thread before any job runs.

livekit-agents runs jobs in a thread on Windows (JobExecutorType.THREAD), and `livekit.plugins.google` refuses to register
off the main thread ("Plugins must be registered on the main thread"); the starter imports it lazily inside the job, which fails
here and, on first import, blocks the event loop for close to a minute. Importing it here first fixes both. On macOS and Linux
this launcher is harmless.

Use exactly like `python -m facetime.worker ...`:
  .venv/Scripts/python -m team.dance.run_worker --perception off --voice gemini --persona team/dance/persona.md
      --handlers team/dance/live.py --log-caller --agent-name= dev
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, 'src')):
    if p not in sys.path: sys.path.insert(0, p)


def main():
    t = time.time()
    try:
        from livekit.plugins import google  # noqa: F401  (registers the plugin on the main thread; slow on first import)
        print(f'[run_worker] livekit.plugins.google imported on the main thread in {time.time() - t:.1f} s', flush=True)
    except Exception as e:
        print(f'[run_worker] google plugin import failed ({e!r}); --voice gemini will not work', flush=True)
    from facetime import worker
    return worker.main()


if __name__ == '__main__':
    sys.exit(main())
