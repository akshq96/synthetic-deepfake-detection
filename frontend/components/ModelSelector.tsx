"use client";

import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, Cpu } from "lucide-react";

// Lets the Detect page run inference with a specific trained checkpoint
// instead of the server's configured default — reads the real registry of
// checkpoints under artifacts/checkpoints/ (backend/app/api/models.py), not
// a hardcoded list. Omitted/empty selection falls back to the default,
// exactly as if this component didn't exist.
export function ModelSelector({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (modelRunId: string | null) => void;
}) {
  const query = useQuery({ queryKey: ["models"], queryFn: () => api.listModels() });

  if (query.isLoading || !query.data || query.data.length === 0) {
    return null;
  }

  return (
    <div className="relative">
      <Cpu className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
        className="input mono-value appearance-none py-1.5 pl-8 pr-7 text-[12px]"
      >
        <option value="">Default model</option>
        {query.data.map((model) => (
          <option key={model.run_id} value={model.run_id}>
            {model.run_name} ({model.model_name})
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
    </div>
  );
}
