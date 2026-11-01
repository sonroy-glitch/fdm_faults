import json
import os
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

# A bundled copy wins when present (Vercel only uploads files under the project
# root of the function), otherwise fall back to the repo-level models/ folder.
MODEL_SEARCH_PATHS = [
    Path(p) for p in (os.environ.get("MODEL_PATH", ""),) if p
] + [
    BASE_DIR / "models" / "model.pkl",
    REPO_DIR / "models" / "model.pkl",
]

ONNX_SEARCH_PATHS = [
    Path(p) for p in (os.environ.get("ONNX_MODEL_PATH", ""),) if p
] + [
    BASE_DIR / "models" / "model.onnx",
    REPO_DIR / "models" / "model.onnx",
]

# "torch" (the default: runs models/model.pkl directly) or "onnx" (runs the
# exported backend/models/model.onnx, needed only where torch does not fit).
# "auto" falls back to whichever runtime and model file are actually present.
ENGINE = os.environ.get("INFERENCE_ENGINE", "torch").lower()

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]


@lru_cache(maxsize=1)
def load_labels() -> dict:
    with open(BASE_DIR / "labels.json", encoding="utf-8") as f:
        data = json.load(f)
    classes = data["classes"]
    pre = data.get("preprocessing", {})
    return {
        "classes": classes,
        "input_size": int(pre.get("input_size", 128)),
        "mean": pre.get("mean", [0.0, 0.0, 0.0]),
        "std": pre.get("std", [1.0, 1.0, 1.0]),
    }


def first_existing(paths):
    for p in paths:
        if p.is_file():
            return p
    return None
