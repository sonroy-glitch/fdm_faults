"""Vercel serverless entry point.

Vercel's Python runtime looks for a WSGI/ASGI callable named `app` in this file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402,F401
