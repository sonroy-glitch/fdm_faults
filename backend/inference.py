"""Image preprocessing plus the two interchangeable inference engines.

Preprocessing is deliberately numpy/Pillow only (no torchvision) so the ONNX
deployment stays small enough for a serverless function.
"""

import io
import threading

import numpy as np
from PIL import Image

import config
from config import load_labels


def preprocess(image_bytes: bytes) -> np.ndarray:
    """bytes -> float32 NCHW batch of 1, matching the training transform."""
    meta = load_labels()
    size = meta["input_size"]

    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0

    mean = np.asarray(meta["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(meta["std"], dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std

    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class TorchEngine:
    name = "torch"

    def __init__(self, model_path):
        import pickle

        import torch

        import model_def

        model_def.register_for_unpickling()
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        self._torch = torch
        self.model = model.eval()
        self.model_path = str(model_path)

    def logits(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            out = self.model(torch.from_numpy(batch))
        return out.detach().cpu().numpy()


class OnnxEngine:
    name = "onnx"

    def __init__(self, model_path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # One thread keeps memory and cold-start predictable on serverless.
        so.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = str(model_path)

    def logits(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch})[0]


_engine = None
_engine_error = None
_lock = threading.Lock()


def _build_engine():
    requested = config.ENGINE
    onnx_path = config.first_existing(config.ONNX_SEARCH_PATHS)
    torch_path = config.first_existing(config.MODEL_SEARCH_PATHS)

    if requested == "onnx":
        order = [("onnx", onnx_path)]
    elif requested == "torch":
        order = [("torch", torch_path)]
    else:
        order = [("onnx", onnx_path), ("torch", torch_path)]

    errors = []
    for kind, path in order:
        if path is None:
            errors.append(f"{kind}: model file not found")
            continue
        try:
            return (OnnxEngine if kind == "onnx" else TorchEngine)(path)
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            errors.append(f"{kind}: {type(exc).__name__}: {exc}")

    raise RuntimeError("no usable inference engine (" + "; ".join(errors) + ")")


def get_engine():
    """Lazily build the engine once; cache the failure so we don't retry per request."""
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


def predict(image_bytes: bytes) -> dict:
    engine = get_engine()
    probs = softmax(engine.logits(preprocess(image_bytes)))[0]
    classes = load_labels()["classes"]

    ranked = sorted(
        ({"label": c, "confidence": float(p)} for c, p in zip(classes, probs)),
        key=lambda d: d["confidence"],
        reverse=True,
    )
    return {
        "prediction": ranked[0]["label"],
        "confidence": ranked[0]["confidence"],
        "probabilities": ranked,
        "engine": engine.name,
    }
