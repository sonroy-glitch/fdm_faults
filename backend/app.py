import base64
import binascii
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

import config
import inference
from config import load_labels

ALLOWED_MIME_PREFIX = "image/"

# Vercel rewrites every request to the serverless function, and a rewrite
# REPLACES the path rather than passing the original through. vercel.json keeps
# the real path as a suffix ("/health" -> "/api/index/health"); this strips the
# mount prefix back off before routing. It lives in the factory rather than in
# api/index.py so it applies no matter which entrypoint loads the app.
MOUNT_PREFIXES = ("/api/index", "/api")


class StripMountPrefix:
    """Undo a serverless mount prefix so Flask sees the client's real path.

    A no-op when no prefix is present, so gunicorn and local runs are unaffected.
    """

    def __init__(self, wsgi_app, prefixes):
        self.wsgi_app = wsgi_app
        self.prefixes = prefixes

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        for prefix in self.prefixes:
            if path == prefix or path.startswith(prefix + "/"):
                # SCRIPT_NAME is deliberately left alone: the public URL has no
                # prefix, so setting it would corrupt any generated URL.
                environ["PATH_INFO"] = path[len(prefix):] or "/"
                environ["fdm.stripped_prefix"] = prefix
                break
        return self.wsgi_app(environ, start_response)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

    CORS(
        app,
        resources={r"/*": {"origins": config.ALLOWED_ORIGINS}},
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app.wsgi_app = StripMountPrefix(app.wsgi_app, MOUNT_PREFIXES)

    @app.get("/")
    def root():
        return jsonify(
            service="fdm-fault-classifier",
            endpoints=["/health", "/classes", "/predict"],
        )

    @app.errorhandler(404)
    def not_found(_):
        # Reports the path actually routed, plus any prefix the middleware
        # removed, so a serverless path mismatch is self-diagnosing.
        return (
            jsonify(
                error="no such route",
                routed_path=request.path,
                raw_path=request.environ.get("RAW_URI")
                or request.environ.get("REQUEST_URI")
                or request.path,
                stripped_prefix=request.environ.get("fdm.stripped_prefix"),
                endpoints=["/", "/health", "/classes", "/predict"],
            ),
            404,
        )

    @app.get("/health")
    def health():
        """Reports whether the model actually loaded, not just that the app is up."""
        try:
            engine = inference.get_engine()
        except Exception as exc:  # noqa: BLE001
            return (
                jsonify(status="degraded", model_loaded=False, error=str(exc)),
                503,
            )
        return jsonify(
            status="ok",
            model_loaded=True,
            engine=engine.name,
            n_estimators=len(getattr(engine.model, "estimators_", []) or []),
            n_features=getattr(engine.model, "n_features_in_", None),
            num_classes=len(inference.class_names()),
        )

    @app.get("/classes")
    def classes():
        meta = load_labels()
        return jsonify(
            classes=inference.class_names(),
            input_size=meta["input_size"],
        )

    @app.post("/predict")
    def predict():
        image_bytes, error = _extract_image(request)
        if error:
            return jsonify(error=error), 400

        if not image_bytes:
            return jsonify(error="empty image payload"), 400

        started = time.perf_counter()
        try:
            result = inference.predict(image_bytes)
        except RuntimeError as exc:
            return jsonify(error=f"model unavailable: {exc}"), 503
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"could not read image: {exc}"), 400

        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        limit_mb = config.MAX_UPLOAD_BYTES / (1024 * 1024)
        return jsonify(error=f"image exceeds the {limit_mb:.0f} MB limit"), 413

    return app


def _extract_image(req):
    """Accept a multipart upload under `file`/`image`, or JSON {image: dataURL|base64}."""
    for key in ("file", "image"):
        uploaded = req.files.get(key)
        if uploaded is not None:
            mime = uploaded.mimetype or ""
            if mime and not mime.startswith(ALLOWED_MIME_PREFIX):
                return None, f"expected an image, got {mime}"
            return uploaded.read(), None

    payload = req.get_json(silent=True) or {}
    encoded = payload.get("image") or payload.get("image_base64")
    if isinstance(encoded, str) and encoded:
        if encoded.startswith("data:"):
            _, _, encoded = encoded.partition(",")
        try:
            return base64.b64decode(encoded, validate=True), None
        except (binascii.Error, ValueError):
            return None, "image is not valid base64"

    if req.data and (req.mimetype or "").startswith(ALLOWED_MIME_PREFIX):
        return req.data, None

    return None, "no image provided (send multipart `file` or JSON `image`)"


app = create_app()

if __name__ == "__main__":
    import os

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
