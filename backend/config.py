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

# Colour-channel order used when binning the colour histogram at training time.
# OpenCV reads images as BGR; PIL and skimage read RGB. Only the histogram block
# is affected. See features.py.
def _default_channel_order() -> str:
    try:
        return load_labels()["channel_order"]
    except Exception:
        return "bgr"

# Threads used by the forest at predict time. One image at a time does not
# benefit from fan-out, and the free tier is a shared single CPU.
N_JOBS = int(os.environ.get("N_JOBS", "1"))

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
    pre = data.get("preprocessing", {})
    return {
        "classes": data["classes"],
        "input_size": int(pre.get("input_size", 128)),
        "channel_order": str(pre.get("channel_order", "bgr")).lower(),
    }


# Env var wins so the histogram channel order can be flipped without a redeploy
# of the labels file; otherwise labels.json decides.
CHANNEL_ORDER = os.environ.get("CHANNEL_ORDER", "").lower() or _default_channel_order()


def first_existing(paths):
    for p in paths:
        if p.is_file():
            return p
    return None
