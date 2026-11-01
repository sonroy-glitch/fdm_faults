"""Vercel serverless entry point.

Vercel's Python runtime imports a WSGI callable named `app` from this file.

Every path is rewritten here by vercel.json, which means the function sees the
*destination* path rather than the one the client asked for. The rewrite keeps
the original path as a suffix (`/health` -> `/api/index/health`) and the
middleware below puts it back before Flask routes it; without that, Flask sees
`/api/index` and 404s everything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app as flask_app  # noqa: E402

MOUNT_PREFIX = "/api/index"


class StripMountPrefix:
    """Undo the vercel.json rewrite so Flask sees the client's real path.

    A no-op when the prefix is absent, so the app behaves identically when run
    directly (locally, or under gunicorn on Render).
    """

    def __init__(self, wsgi_app, prefix: str):
        self.wsgi_app = wsgi_app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == self.prefix or path.startswith(self.prefix + "/"):
            # SCRIPT_NAME is deliberately left alone: the public URL carries no
            # /api/index prefix, so adding it would corrupt any generated URL.
            environ["PATH_INFO"] = path[len(self.prefix) :] or "/"
        return self.wsgi_app(environ, start_response)


flask_app.wsgi_app = StripMountPrefix(flask_app.wsgi_app, MOUNT_PREFIX)

app = flask_app
