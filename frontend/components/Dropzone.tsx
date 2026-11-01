"use client";

import { useRef, useState, type DragEvent } from "react";

type Props = {
  onSelect: (file: File) => void;
  disabled?: boolean;
};

export default function Dropzone({ onSelect, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = event.dataTransfer.files?.[0];
    if (file) onSelect(file);
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={[
        "rounded-xl border border-dashed p-8 text-center transition-colors",
        dragging ? "border-accent bg-accent/5" : "border-edge bg-panel/60",
        disabled ? "opacity-60" : "",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="sr-only"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onSelect(file);
          // Reset so picking the same file twice still fires a change event.
          e.target.value = "";
        }}
      />

      <p className="text-sm text-slate-200">
        Drag a photo here, or
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          className="ml-1 rounded font-medium text-accent underline decoration-accent/40 underline-offset-4 hover:decoration-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed"
        >
          browse your files
        </button>
      </p>
      <p className="mt-2 text-xs text-muted">JPEG, PNG or WebP · up to 10 MB</p>
    </div>
  );
}
