# FDM fault classifier — API

Flask service wrapping the RandomForest in `../models/model.pkl`.

## The model

`model.pkl` is a pickled dict:

```python
{"model": RandomForestClassifier(...), "label_map": {...}, "img_size": 128}
```

| | |
|---|---|
| Estimator | `RandomForestClassifier`, 300 trees, `class_weight="balanced"` |
| Input | RGB image, resized to **128 × 128** |
| Features | **2276** engineered features (not raw pixels) |
| Classes | 5, ordered by the embedded `label_map` |
| Trained with | scikit-learn **1.7.1** |

The `label_map` inside the pickle is treated as authoritative for class order
and is cross-checked against `labels.json` at load time.

## Feature extraction (`features.py`)

This is the part that must match training exactly — the forest takes a 2276-dim
vector, and 2276 decomposes uniquely:

| Slice | Length | Content |
|---|---|---|
| `[0:1764]` | 1764 | HOG — 9 orientations, 16×16 px/cell, 2×2 cells/block, L2-Hys, on 128×128 grayscale (7·7 blocks × 2·2 cells × 9) |
| `[1764:2276]` | 512 | 8×8×8 joint colour histogram, L2-normalised |

The split is this way round, not the reverse, because **all 258 of the model's
zero-importance features fall inside `[1764:2276]`** — which is what empty
colour-histogram bins look like. The HOG block has none.

### The one ambiguous setting

`CHANNEL_ORDER` decides how the colour histogram bins channels:

- `bgr` (default) — matches an OpenCV pipeline (`cv2.imread` → `cv2.calcHist`)
- `rgb` — matches a PIL / skimage pipeline

It affects only the 512-value histogram block; grayscale/HOG is unaffected
either way. If predictions look systematically wrong on known images, flip this
first — it is an env var, no code change.

## API

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | 200 only if the forest actually loaded, else 503; reports tree and feature counts |
| `/classes` | GET | Class list and expected input size |
| `/predict` | POST | Classify one image |

`/predict` accepts multipart (`file` or `image`), JSON (`{"image": "<base64 or data URL>"}`),
or a raw image body:

```json
{
  "prediction": "Warping",
  "confidence": 0.323,
  "probabilities": [{ "label": "Warping", "confidence": 0.323 }],
  "engine": "randomforest",
  "latency_ms": 10.2
}
```

## Run locally

```bash
pip install -r requirements.txt
```

```bash
gunicorn app:app --bind 127.0.0.1:5000
```

```bash
curl -F "file=@print.jpg" http://127.0.0.1:5000/predict
```

## Deploy to Render

`render.yaml` is a blueprint — point Render at the repo and it reads everything
below. To fill the dashboard by hand instead:

| Field | Value |
|---|---|
| Language | `Python 3` |
| Root directory | `backend` |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120` |
| Health check path | `/health` |
| Instance type | `Free` |

Environment variables:

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.11.9` |
| `N_JOBS` | `1` |
| `OMP_NUM_THREADS` | `1` |
| `CHANNEL_ORDER` | `bgr` |
| `ALLOWED_ORIGINS` | the frontend origin, or `*` |

`models/model.pkl` sits at the repo root, one level above `rootDir`. That is
fine — `config.py` resolves it from `__file__`, not the working directory.

Measured: ~180 MB resident against the 512 MB free-tier limit, ~12 ms per warm
prediction. The forest is pickled with `n_jobs=-1`, which would fan out across
every core; `N_JOBS=1` pins it, since a single image gains nothing from
fan-out on a shared CPU.

**Do not float the scikit-learn pin.** Unpickling under a different minor
version raises `InconsistentVersionWarning` and is not guaranteed to score
identically.
