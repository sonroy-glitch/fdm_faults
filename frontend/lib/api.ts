import type { Prediction } from "./types";

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

export async function classifyImage(file: File): Promise<Prediction> {
  const body = new FormData();
  body.append("file", file);

  const res = await fetch("/api/predict", { method: "POST", body });

  let payload: unknown;
  try {
    payload = await res.json();
  } catch {
    throw new Error(`Server returned ${res.status} with no readable body.`);
  }

  if (!res.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload
        ? String((payload as { error: unknown }).error)
        : `Request failed with status ${res.status}.`;
    throw new Error(message);
  }

  return payload as Prediction;
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
