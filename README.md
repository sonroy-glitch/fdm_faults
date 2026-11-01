# FDM print fault classifier

A CNN that scores a photo of a 3D print against five FDM failure modes, with a
Flask API and a Next.js frontend.

```
models/model.pkl     original PyTorch checkpoint
backend/             Flask API  -> Render and/or Vercel
frontend/            Next.js UI -> Vercel
```

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
| API | **Render** | `backend` | `render.yaml` blueprint; runs `model.pkl` under PyTorch |
| UI | **Vercel** | `frontend` | set `BACKEND_URL` to the Render URL |

Deploy the API first, then give the frontend its URL via `BACKEND_URL`.

The API serves `models/model.pkl` directly with PyTorch. Measured footprint is
~240 MB against Render's 512 MB free-tier limit. Free instances sleep, so the
first request after idle can take 30-60 s.

An ONNX path (`INFERENCE_ENGINE=onnx`, `requirements-onnx.txt`) is kept as a
fallback and is the only way to fit the API in a Vercel serverless function,
where torch is far past the 250 MB limit. See `backend/README.md`.

## Before you trust the output

Two values were not recoverable from the checkpoint and are **assumptions**:

1. **Class names and their order** — set in `backend/labels.json`. Index 0 must
   be the first class in the training label order. The current names are
   placeholders.
2. **Preprocessing** — assumed plain 0–1 rescaling at 128×128. If training used
   normalised inputs, set `mean`/`std` in the same file.

Both are data, not code; nothing else needs touching.
