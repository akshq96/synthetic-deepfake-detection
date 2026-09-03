"use client";

import { Button } from "@/components/ui/Button";
import { Search } from "lucide-react";
import { useState } from "react";

// Every /api/results/* endpoint and /api/models/compare is a read-only
// passthrough keyed by a location the caller supplies (a results_dir under
// artifacts/, or an MLflow experiment name) — see backend/app/api/results.py.
// There's no server-side index of "which results exist," so the dashboard
// asks for that location rather than pretending to auto-discover it.
export function ResultsLookupForm({
  label,
  placeholder,
  onSubmit,
  initialValue = "",
}: {
  label: string;
  placeholder: string;
  onSubmit: (value: string) => void;
  initialValue?: string;
}) {
  const [value, setValue] = useState(initialValue);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (value.trim()) onSubmit(value.trim());
      }}
      className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-end"
    >
      <label className="flex-1">
        <span className="mb-1.5 block text-xs font-medium text-muted-foreground">{label}</span>
        <input
          className="input"
          placeholder={placeholder}
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
      </label>
      <Button type="submit" variant="secondary">
        <Search className="h-4 w-4" />
        Load
      </Button>
    </form>
  );
}
