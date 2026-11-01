"""Vercel serverless entry point.

Vercel's Python runtime imports a WSGI callable named `app` from this file.

Note that vercel.json rewrites every request here and a rewrite replaces the
request path, so the function is reached as "/api/index/<original path>". The
prefix is stripped by middleware installed in `create_app()` (see app.py)
rather than here, so the behaviour is identical under gunicorn on Render.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402,F401
