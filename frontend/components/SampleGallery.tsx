"use client";

// Abstract, procedurally-generated placeholder images committed under
// public/samples/ — deliberately not photographs of real people, so this
// gallery carries no likeness/rights concern in a public repo. Clicking one
// fetches it as a File and feeds the exact same upload path as a real file
// pick — no backend change needed.
const SAMPLES = [
  { path: "/samples/sample-a.jpg", label: "Synthetic sample A" },
  { path: "/samples/sample-b.jpg", label: "Synthetic sample B" },
  { path: "/samples/sample-c.jpg", label: "Synthetic sample C" },
];

export function SampleGallery({ onSelect, disabled }: { onSelect: (file: File) => void; disabled?: boolean }) {
  async function pick(sample: (typeof SAMPLES)[number]) {
    const response = await fetch(sample.path);
    const blob = await response.blob();
    const file = new File([blob], sample.path.split("/").pop() ?? "sample.jpg", { type: blob.type });
    onSelect(file);
  }

  return (
    <div>
      <div className="meta-label mb-2 text-[10px]">Or try a sample</div>
      <div className="flex gap-2">
        {SAMPLES.map((sample) => (
          <button
            key={sample.path}
            type="button"
            disabled={disabled}
            onClick={() => pick(sample)}
            title={sample.label}
            className="h-12 w-12 overflow-hidden rounded-md border border-border transition-opacity hover:opacity-80 disabled:opacity-40"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={sample.path} alt={sample.label} className="h-full w-full object-cover" />
          </button>
        ))}
      </div>
    </div>
  );
}
