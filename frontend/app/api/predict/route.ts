import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:5000";
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

// Render's free tier sleeps; a cold container can take ~50s to answer.
export const maxDuration = 60;

export async function POST(request: Request) {
  const form = await request.formData().catch(() => null);
  const file = form?.get("file");

  if (!(file instanceof File)) {
    return NextResponse.json(
      { error: "No image was attached to the request." },
      { status: 400 },
    );
  }
  if (file.size === 0) {
    return NextResponse.json({ error: "That file is empty." }, { status: 400 });
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { error: "Images must be 10 MB or smaller." },
      { status: 413 },
    );
  }

  const upstream = new FormData();
  upstream.append("file", file, file.name || "upload");

  try {
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: upstream,
      signal: AbortSignal.timeout(55_000),
    });
    const payload = await res.json().catch(() => ({
      error: `The model service returned ${res.status}.`,
    }));
    return NextResponse.json(payload, { status: res.status });
  } catch (err) {
    const timedOut = err instanceof Error && err.name === "TimeoutError";
    return NextResponse.json(
      {
        error: timedOut
          ? "The model service timed out. If it is hosted on a free tier it may be waking up — try again."
          : "Could not reach the model service.",
      },
      { status: 504 },
    );
  }
}
