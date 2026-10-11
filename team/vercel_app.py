"""Vercel entrypoint for the backend service (service root: team/, see vercel.json).

The backend is written to run with `uvicorn app:app --app-dir team/backend`, and db.py loads ../dance/progression.py by path, so
the service root must be team/ (not team/backend/). This shim puts backend/ on sys.path and re-exports the FastAPI app.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))
from app import app  # noqa: E402,F401
