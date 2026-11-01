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
| API | Render | `backend` | `render.yaml` blueprint; free tier sleeps |
| API | Vercel | `backend` | `vercel.json`; runs on ONNX Runtime |
| UI | Vercel | `frontend` | set `BACKEND_URL` |

Deploy the API first, then give the frontend its URL via `BACKEND_URL`.

PyTorch cannot run in a Vercel function (~800 MB unpacked against a 250 MB
limit), so the service runs the model through ONNX Runtime by default on both
hosts. The exported graph is verified against the original checkpoint to ~1e-6.
See `backend/README.md`.

## Before you trust the output

Two values were not recoverable from the checkpoint and are **assumptions**:

1. **Class names and their order** — set in `backend/labels.json`. Index 0 must
   be the first class in the training label order. The current names are
   placeholders.
2. **Preprocessing** — assumed plain 0–1 rescaling at 128×128. If training used
   normalised inputs, set `mean`/`std` in the same file.

Both are data, not code; nothing else needs touching.
