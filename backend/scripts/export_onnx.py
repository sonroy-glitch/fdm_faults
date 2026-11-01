"""Convert models/model.pkl into models/model.onnx for the lightweight deployment.

Run once, locally, then commit the .onnx file:

    python backend/scripts/export_onnx.py

PyTorch is only needed here, never at serving time on the ONNX target.
"""

import pickle
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import model_def  # noqa: E402
from config import first_existing, load_labels, MODEL_SEARCH_PATHS  # noqa: E402


def main() -> int:
    src = first_existing(MODEL_SEARCH_PATHS)
    if src is None:
        print("model.pkl not found", file=sys.stderr)
        return 1

    # backend/models/ is inside the Vercel deploy root, so the graph gets bundled.
    out = BACKEND_DIR / "models" / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    size = load_labels()["input_size"]

    model_def.register_for_unpickling()
    with open(src, "rb") as f:
        model = pickle.load(f)
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        # torch >= 2.9 defaults to the dynamo exporter, which needs onnxscript.
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out), **kwargs)

    # Confirm the exported graph matches the original within float tolerance.
    import onnxruntime as ort

    with torch.inference_mode():
        expected = model(dummy).numpy()
    got = ort.InferenceSession(
        str(out), providers=["CPUExecutionProvider"]
    ).run(None, {"input": dummy.numpy()})[0]

    delta = float(np.abs(expected - got).max())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), max abs diff {delta:.2e}")
    if delta > 1e-3:
        print("WARNING: outputs diverge more than expected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
