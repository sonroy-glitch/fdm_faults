# FDM fault classifier — API

Flask service wrapping the CNN in `../models/model.pkl`.

## The model

Recovered from the checkpoint (it is a plain `pickle.dump` of an
`nn.Module`, so `model_def.py` redefines the class and aliases it onto
`__main__` to make it loadable):

| | |
|---|---|
| Input | RGB, **128 × 128**, scaled to 0–1 |
| Body | 4 × (Conv 3×3 `same` → ReLU → BatchNorm → MaxPool 2×2), 32 → 256 ch |
| Head | Flatten(16384) → 128 → 64 → **5** |

The 16384-unit flatten (256 × 8 × 8) after four halvings is what pins the
input size at 128.

## Inference engines

`inference.py` can run the model two ways, chosen by `INFERENCE_ENGINE`:

- **`onnx`** (default, used on both Render and Vercel) — `backend/models/model.onnx`,
  exported from the pickle and verified to match it to ~1e-6.
- **`torch`** — loads `models/model.pkl` directly. Needs `requirements-torch.txt`.
  Will **not** fit in a Vercel function (torch unpacks to ~800 MB vs. a 250 MB limit).

`auto` tries ONNX, then torch.

Re-export after retraining:

```bash
pip install -r requirements-dev.txt
python scripts/export_onnx.py   # writes backend/models/model.onnx, checks parity
```

`backend/models/model.onnx` is committed on purpose — it must be inside the
deploy root to be bundled.

## Class labels

`labels.json` is the single source of truth. **Index order must match the
training label order.** Edit `classes` there and both API and UI follow; no
code changes.

It also holds the preprocessing constants (`input_size`, `mean`, `std`). The
defaults assume plain 0–1 rescaling; if the model was trained with normalised
inputs, set `mean`/`std` to match or accuracy will quietly suffer.

## API

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | 200 only if the model actually loaded, else 503 |
| `/classes` | GET | Class list and expected input size |
| `/predict` | POST | Classify one image |

`/predict` accepts multipart (`file` or `image`), JSON (`{"image": "<base64 or data URL>"}`),
or a raw image body. Response:

```json
{
  "prediction": "Stringing",
  "confidence": 0.998,
  "probabilities": [{ "label": "Stringing", "confidence": 0.998 }],
  "engine": "onnx",
  "latency_ms": 7.2
}
```

## Run locally

```bash
pip install -r requirements.txt
gunicorn app:app --bind 127.0.0.1:5000
```

```bash
curl -F "file=@print.jpg" http://127.0.0.1:5000/predict
```

## Deploy

Both targets use **`backend/` as the project root**, which is what makes the
flat `requirements.txt` resolve.

### Render

`render.yaml` is a blueprint — point Render at the repo and it reads it. Or set
it manually: root directory `backend`, build `pip install -r requirements.txt`,
start `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`.
Free instances sleep, so the first request after idling can take ~50 s.

### Vercel

Root directory **must** be `backend`. `vercel.json` routes every path to
`api/index.py` and bundles `models/**` with the function.

Three things this setup gets right, each of which broke a deploy:

1. Dependencies are read from `requirements.txt` at the **deploy root**, not
   from one next to the entrypoint.
2. `.vercelignore` patterns are gitignore-style — a bare `requirements.txt`
   matches at *every* depth and will silently exclude the function's
   dependency file. The entries here are anchored with a leading `/`.
3. A rewrite **replaces** the request path, so a destination of `/api/index`
   hands Flask that literal path and every route 404s. The rewrite therefore
   keeps the real path as a suffix:

   ```json
   { "source": "/(.*)", "destination": "/api/index/$1" }
   ```

   and `create_app()` in `app.py` installs a small WSGI middleware that strips
   the `/api/index` prefix before routing. It lives in the factory, not in
   `api/index.py`, so it applies whichever entrypoint loads the app, and it is
   a no-op when no prefix is present — Render and local runs are unaffected.

   A 404 from this service reports `routed_path`, `raw_path` and
   `stripped_prefix`, so a path mismatch says what it saw rather than needing
   a log hunt.

Set `ALLOWED_ORIGINS` to the frontend origin on whichever host you keep.
