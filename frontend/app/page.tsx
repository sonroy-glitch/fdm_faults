import Classifier from "@/components/Classifier";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-12 sm:py-16">
      <header className="mb-10">
        <p className="text-xs font-medium uppercase tracking-[0.2em] text-accent">
          Fused deposition modelling
        </p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">
          Print fault classifier
        </h1>
        <p className="mt-3 max-w-xl text-sm leading-relaxed text-muted">
          Upload a photo of a print. A convolutional network scores it against
          five known failure modes and reports the confidence for each.
        </p>
      </header>

      <Classifier />

      <footer className="mt-12 border-t border-edge pt-6 text-xs text-muted">
        Predictions are a decision aid, not a guarantee — always confirm against
        the physical part.
      </footer>
    </main>
  );
}
