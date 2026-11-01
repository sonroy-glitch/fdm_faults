# FDM fault classifier — frontend

Next.js 15 (App Router, TypeScript, Tailwind v4). Deploys to Vercel.

## Talking to the API

The browser never calls the Flask service directly. `app/api/predict/route.ts`
proxies the upload server-side, which:

- keeps `BACKEND_URL` private,
- removes CORS from the picture entirely,
- lets the timeout and error copy account for Render's cold starts.

## Local

```bash
npm install
```

```bash
cp .env.example .env.local   # point BACKEND_URL at your Flask server
```

```bash
npm run dev
```

## Deploy

Root directory `frontend`. The one required setting:

| Variable | Value |
|---|---|
| `BACKEND_URL` | the deployed API origin, e.g. `https://fdm-fault-api.onrender.com` |

It is a server-side variable — deliberately **not** `NEXT_PUBLIC_`.
