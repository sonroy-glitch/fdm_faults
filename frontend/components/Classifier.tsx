"use client";

import { useCallback, useEffect, useState } from "react";

import Dropzone from "./Dropzone";
import ResultPanel from "./ResultPanel";
import { MAX_UPLOAD_BYTES, classifyImage } from "@/lib/api";
import type { Prediction } from "@/lib/types";

export default function Classifier() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Object URLs are only freed when the component unmounts or the preview is
  // replaced, so revoke the previous one each time.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleSelect = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("That file is not an image.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setError("Images must be 10 MB or smaller.");
      return;
    }

    setError(null);
    setResult(null);
    setFileName(file.name);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(file);
    });

    setBusy(true);
    try {
      setResult(await classifyImage(file));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Classification failed.");
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <div className="space-y-6">
      <Dropzone onSelect={handleSelect} disabled={busy} />

      {error && (
        <p
          role="alert"
          className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200"
        >
          {error}
        </p>
      )}

      {previewUrl && (
        <figure className="overflow-hidden rounded-xl border border-edge bg-panel/60">
          {/* A blob: URL cannot be optimised by next/image, so use a plain img. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl}
            alt={fileName ? `Uploaded print: ${fileName}` : "Uploaded print"}
            className="max-h-80 w-full bg-black/40 object-contain"
          />
          {fileName && (
            <figcaption className="truncate border-t border-edge px-4 py-2.5 text-xs text-muted">
              {fileName}
            </figcaption>
          )}
        </figure>
      )}

      {busy && (
        <p className="text-sm text-muted">
          Running the model… a cold free-tier server can take up to a minute.
        </p>
      )}

      {result && !busy && <ResultPanel result={result} />}
    </div>
  );
}
