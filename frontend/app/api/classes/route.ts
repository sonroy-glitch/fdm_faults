import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:5000";

export const revalidate = 300;

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/classes`, {
      signal: AbortSignal.timeout(10_000),
      next: { revalidate },
    });
    if (!res.ok) throw new Error(String(res.status));
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json(
      { error: "Could not reach the model service." },
      { status: 503 },
    );
  }
}
