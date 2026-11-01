# FDM print fault classifier

A RandomForest that scores a photo of a 3D print against five FDM failure
modes, with a Flask API and a Next.js frontend.

```
models/model.pkl     RandomForest + label_map + img_size (pickled dict)
backend/             Flask API  -> Render
frontend/            Next.js UI -> Vercel
```

## Classes

Index order is confirmed and matches the `label_map` embedded in the model:

| Index | Dataset folder | Displayed |
|---|---|---|
| 0 | `Cracking` | Cracking |
| 1 | `Layer_shifting` | Layer shifting |
| 2 | `Off_platform` | Off platform |
| 3 | `Stringing` | Stringing |
| 4 | `Warping` | Warping |

## Quick start

```bash
pip install -r backend/requirements.txt && gunicorn --chdir backend app:app --bind 127.0.0.1:5000
```

```bash
cd frontend && npm install && cp .env.example .env.local && npm run dev
```

## Deploying

Both projects set their **root directory** to the relevant subfolder.

| Piece | Host | Root | Notes |
|---|---|---|---|
| API | **Render** | `backend` | `render.yaml` blueprint; ~180 MB of the 512 MB free tier |
| UI | **Vercel** | `frontend` | set `BACKEND_URL` to the Render URL |

Deploy the API first, then give the frontend its URL via `BACKEND_URL`. Free
Render instances sleep after ~15 minutes idle, so the first request after idle
is slow; the frontend already allows for this.

## Before you trust the output

The forest takes **2276 engineered features**, not raw pixels, so serving it
means reproducing the training-time feature extraction. The layout is pinned
down by the feature count and by which features the model never uses
(`backend/README.md` has the reasoning), but one choice is a genuine
assumption:

**Colour-channel order** (`CHANNEL_ORDER`, default `bgr`) — whether the colour
histogram bins channels OpenCV-style (BGR) or PIL-style (RGB). It affects 512
of the 2276 features. If predictions are systematically wrong on images whose
fault you know, flip it to `rgb`; it is an environment variable, no code change.

Validating this needs only a handful of correctly-labelled images.
