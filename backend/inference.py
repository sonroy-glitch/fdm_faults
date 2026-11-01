"""RandomForest inference.

models/model.pkl is a dict:

    {"model": RandomForestClassifier, "label_map": {name: index}, "img_size": 128}

The embedded `label_map` is authoritative for class ordering, so it is used to
build the label list and is cross-checked against labels.json at load time.
"""

import threading

import numpy as np

import config
import features
from config import load_labels


class RandomForestEngine:
    name = "randomforest"

    def __init__(self, model_path):
        import pickle

        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(
                f"expected a dict with a 'model' key in {model_path}, "
                f"got {type(payload).__name__}"
            )

        self.model = payload["model"]
        self.img_size = int(payload.get("img_size", load_labels()["input_size"]))
        self.model_path = str(model_path)

        # The forest was pickled with n_jobs=-1, which fans out over every core.
        # On a shared free-tier instance that costs memory and context switching
        # without helping a single-image predict, so pin it.
        self.model.n_jobs = config.N_JOBS

        # Order the names by the index the model was trained against.
        label_map = payload.get("label_map") or {}
        if label_map:
            self.classes = [
                name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])
            ]
        else:
            self.classes = load_labels()["classes"]

        expected = getattr(self.model, "n_features_in_", None)
        if expected is not None and expected != features.EXPECTED_LENGTH:
            raise ValueError(
                f"model expects {expected} features but the extractor produces "
                f"{features.EXPECTED_LENGTH}; feature settings do not match training"
            )

    def probabilities(self, image_bytes: bytes) -> np.ndarray:
        x = features.extract(
            image_bytes, size=self.img_size, channel_order=config.CHANNEL_ORDER
        )
        probs = self.model.predict_proba(x)[0]

        # predict_proba columns follow model.classes_, which here are the
        # integer indices; reorder defensively in case they are not 0..n-1.
        order = np.argsort(np.asarray(self.model.classes_))
        return probs[order]


_engine = None
_engine_error = None
_lock = threading.Lock()


def _build_engine():
    path = config.first_existing(config.MODEL_SEARCH_PATHS)
    if path is None:
        raise RuntimeError(
            "model.pkl not found in " + ", ".join(str(p) for p in config.MODEL_SEARCH_PATHS)
        )
    return RandomForestEngine(path)


def get_engine():
    """Build the engine once; cache a failure so it is not retried per request."""
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error is not None:
        raise _engine_error
    with _lock:
        if _engine is None and _engine_error is None:
            try:
                _engine = _build_engine()
            except Exception as exc:  # noqa: BLE001
                _engine_error = exc
    if _engine_error is not None:
        raise _engine_error
    return _engine


def class_names():
    """Display names, preferring labels.json when it lines up with the model."""
    engine = get_engine()
    configured = load_labels()["classes"]
    return configured if len(configured) == len(engine.classes) else engine.classes


def predict(image_bytes: bytes) -> dict:
    engine = get_engine()
    probs = engine.probabilities(image_bytes)
    names = class_names()

    ranked = sorted(
        ({"label": c, "confidence": float(p)} for c, p in zip(names, probs)),
        key=lambda d: d["confidence"],
        reverse=True,
    )
    return {
        "prediction": ranked[0]["label"],
        "confidence": ranked[0]["confidence"],
        "probabilities": ranked,
        "engine": engine.name,
    }
